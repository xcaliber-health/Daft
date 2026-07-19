//! Admission control bounding concurrent query execution.
//!
//! Queries beyond the configured concurrency wait in a queue; a waiter that
//! exceeds the queue timeout is rejected with an actionable capacity error so
//! clients can back off or retry elsewhere.

use std::{collections::HashMap, sync::Arc, time::Duration};

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

/// Routes admission by caller identity: named tenants with their own slot
/// pools, and a shared default pool for anonymous callers or tenants
/// without an override.
///
/// Each tenant's pool is independent, so one tenant saturating its slots
/// never delays another tenant's queries.
#[derive(Debug)]
pub struct TenantAdmission {
    default: Admission,
    per_tenant: HashMap<String, Admission>,
}

impl TenantAdmission {
    /// Creates identity-routed admission from the shared default pool and
    /// per-tenant pools. `per_tenant` maps tenant names to their dedicated
    /// pools; tenants absent from the map share the default pool.
    #[must_use]
    pub fn new(default: Admission, per_tenant: HashMap<String, Admission>) -> Self {
        Self {
            default,
            per_tenant,
        }
    }

    /// The pool serving the given caller: the tenant's dedicated pool when
    /// one exists, otherwise the shared default.
    #[must_use]
    pub fn pool(&self, tenant: Option<&str>) -> &Admission {
        tenant
            .and_then(|name| self.per_tenant.get(name))
            .unwrap_or(&self.default)
    }

    /// Configured slot count of the shared default pool.
    #[must_use]
    pub const fn default_max_concurrent(&self) -> usize {
        self.default.max_concurrent()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn tenant_pools_are_independent() {
        let mut per_tenant = HashMap::new();
        per_tenant.insert(
            "alpha".to_string(),
            Admission::new(1, Duration::from_millis(20)),
        );
        let routed = TenantAdmission::new(Admission::new(4, Duration::from_millis(20)), per_tenant);

        // Saturate alpha's single slot.
        let held = routed.pool(Some("alpha")).acquire().await.unwrap();
        assert!(routed.pool(Some("alpha")).acquire().await.is_err());

        // Other identities are unaffected.
        assert!(routed.pool(Some("beta")).acquire().await.is_ok());
        assert!(routed.pool(None).acquire().await.is_ok());
        drop(held);
    }

    #[tokio::test]
    async fn unknown_tenant_uses_default_pool() {
        let routed =
            TenantAdmission::new(Admission::new(1, Duration::from_millis(20)), HashMap::new());
        let held = routed.pool(Some("anyone")).acquire().await.unwrap();
        // Default pool is shared, so anonymous callers contend with it.
        assert!(routed.pool(None).acquire().await.is_err());
        drop(held);
    }

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
