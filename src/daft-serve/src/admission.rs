//! Admission control bounding concurrent query execution.
//!
//! Queries beyond the configured concurrency wait in a queue; a waiter that
//! exceeds the queue timeout is rejected with an actionable capacity error so
//! clients can back off or retry elsewhere.

use std::{sync::Arc, time::Duration};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::error::{ServeError, ServeResult};

/// Bounds the number of concurrently executing queries.
#[derive(Debug, Clone)]
pub struct Admission {
    semaphore: Arc<Semaphore>,
    max_concurrent: usize,
    queue_timeout: Duration,
}

/// A held execution slot; dropping it frees the slot.
#[derive(Debug)]
pub struct AdmissionPermit {
    _permit: OwnedSemaphorePermit,
}

impl Admission {
    /// Creates admission control with `max_concurrent` execution slots and a
    /// bound on how long a query may wait for one.
    #[must_use]
    pub fn new(max_concurrent: usize, queue_timeout: Duration) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            max_concurrent,
            queue_timeout,
        }
    }

    /// Configured maximum number of concurrently executing queries.
    #[must_use]
    pub const fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }

    /// Acquires an execution slot, waiting up to the queue timeout.
    ///
    /// # Errors
    /// Returns [`ServeError::AtCapacity`] if no slot frees up in time, or if
    /// admission is shut down.
    pub async fn acquire(&self) -> ServeResult<AdmissionPermit> {
        let at_capacity = || ServeError::AtCapacity {
            max_concurrent: self.max_concurrent,
            waited_secs: self.queue_timeout.as_secs(),
        };
        let acquired =
            tokio::time::timeout(self.queue_timeout, self.semaphore.clone().acquire_owned())
                .await
                .map_err(|_| at_capacity())?;
        let permit = acquired.map_err(|_| at_capacity())?;
        Ok(AdmissionPermit { _permit: permit })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn grants_up_to_capacity() {
        let admission = Admission::new(2, Duration::from_millis(50));
        let a = admission.acquire().await.unwrap();
        let b = admission.acquire().await.unwrap();
        drop((a, b));
    }

    #[tokio::test]
    async fn times_out_when_full() {
        let admission = Admission::new(1, Duration::from_millis(20));
        let held = admission.acquire().await.unwrap();
        let err = admission.acquire().await.unwrap_err();
        assert!(matches!(
            err,
            ServeError::AtCapacity {
                max_concurrent: 1,
                ..
            }
        ));
        drop(held);
    }

    #[tokio::test]
    async fn freed_slot_admits_waiter() {
        let admission = Admission::new(1, Duration::from_secs(5));
        let held = admission.acquire().await.unwrap();
        let waiter = {
            let admission = admission.clone();
            tokio::spawn(async move { admission.acquire().await })
        };
        drop(held);
        let permit = waiter.await.unwrap();
        assert!(permit.is_ok());
    }

    #[tokio::test]
    async fn dropped_permit_restores_capacity() {
        let admission = Admission::new(1, Duration::from_millis(20));
        drop(admission.acquire().await.unwrap());
        assert!(admission.acquire().await.is_ok());
    }
}
