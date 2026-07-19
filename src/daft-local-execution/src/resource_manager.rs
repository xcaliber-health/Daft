use std::sync::{Arc, Mutex, OnceLock};

use common_error::{DaftError, DaftResult};
use common_system_info::SystemInfo;
use tokio::sync::Notify;

pub(crate) static MEMORY_MANAGER: OnceLock<Arc<MemoryManager>> = OnceLock::new();

fn custom_memory_limit() -> Option<u64> {
    let memory_limit_var_name = "DAFT_MEMORY_LIMIT";
    if let Ok(val) = std::env::var(memory_limit_var_name)
        && let Ok(val) = val.parse::<u64>()
    {
        return Some(val);
    }
    None
}

pub(crate) fn get_or_init_memory_manager() -> &'static Arc<MemoryManager> {
    MEMORY_MANAGER.get_or_init(|| Arc::new(MemoryManager::new()))
}

pub(crate) struct MemoryPermit<'a> {
    bytes: u64,
    manager: &'a MemoryManager,
}

impl Drop for MemoryPermit<'_> {
    fn drop(&mut self) {
        if self.bytes > 0 {
            {
                let mut state = self.manager.state.lock().unwrap();
                state.available_bytes += self.bytes;
            } // lock is released here
            self.manager.notify.notify_waiters();
        }
    }
}

struct MemoryState {
    available_bytes: u64,
}

pub(crate) struct MemoryManager {
    total_bytes: u64,
    state: Mutex<MemoryState>,
    notify: Notify,
}

impl Default for MemoryManager {
    fn default() -> Self {
        let system_info = SystemInfo::default();
        let total_mem = system_info.calculate_total_memory();
        Self {
            total_bytes: total_mem,
            state: Mutex::new(MemoryState {
                available_bytes: total_mem,
            }),
            notify: Notify::new(),
        }
    }
}

impl MemoryManager {
    pub fn new() -> Self {
        if let Some(custom_limit) = custom_memory_limit() {
            Self {
                total_bytes: custom_limit,
                state: Mutex::new(MemoryState {
                    available_bytes: custom_limit,
                }),
                notify: Notify::new(),
            }
        } else {
            Self::default()
        }
    }

    pub async fn request_bytes(&self, bytes: u64) -> DaftResult<MemoryPermit<'_>> {
        if bytes == 0 {
            return Ok(MemoryPermit {
                bytes: 0,
                manager: self,
            });
        }

        if bytes > self.total_bytes {
            return Err(DaftError::ComputeError(format!(
                "Cannot request {} bytes, only {} available",
                bytes, self.total_bytes
            )));
        }

        loop {
            if let Some(permit) = self.try_request_bytes(bytes) {
                return Ok(permit);
            }
            self.notify.notified().await;
        }
    }

    fn try_request_bytes(&self, bytes: u64) -> Option<MemoryPermit<'_>> {
        let mut state = self.state.lock().unwrap();
        if state.available_bytes >= bytes {
            state.available_bytes -= bytes;
            Some(MemoryPermit {
                bytes,
                manager: self,
            })
        } else {
            None
        }
    }

    /// Attempts to take `bytes` from the budget without waiting.
    ///
    /// Returns whether the reservation fit. Callers that buffer data and can
    /// shed it to disk use this to detect memory pressure: a denial is the
    /// signal to spill rather than grow.
    fn try_reserve(&self, bytes: u64) -> bool {
        let mut state = self.state.lock().unwrap();
        if state.available_bytes >= bytes {
            state.available_bytes -= bytes;
            true
        } else {
            false
        }
    }

    /// Returns `bytes` to the budget and wakes any waiters.
    fn release(&self, bytes: u64) {
        if bytes > 0 {
            {
                let mut state = self.state.lock().unwrap();
                state.available_bytes =
                    (state.available_bytes + bytes).min(self.total_bytes);
            }
            self.notify.notify_waiters();
        }
    }
}

/// Owned, growable share of the memory budget held by an operator that can
/// spill its buffered state to disk.
///
/// Growth is non-blocking: [`Self::try_grow`] either fits within the global
/// budget or reports pressure so the owner sheds buffered bytes and calls
/// [`Self::shrink`]. All held bytes return to the budget on drop, so an
/// abandoned or failed query cannot leak accounting.
pub(crate) struct SpillBudget {
    manager: Arc<MemoryManager>,
    held: u64,
}

impl SpillBudget {
    /// Creates an empty budget share against the given manager.
    pub(crate) fn new(manager: Arc<MemoryManager>) -> Self {
        Self { manager, held: 0 }
    }

    /// Attempts to grow the held share by `bytes`; returns whether it fit.
    ///
    /// A denial means the global budget is exhausted — the owner should
    /// spill buffered state (and `shrink`) before retrying or proceeding.
    #[must_use]
    pub(crate) fn try_grow(&mut self, bytes: u64) -> bool {
        if self.manager.try_reserve(bytes) {
            self.held += bytes;
            true
        } else {
            false
        }
    }

    /// Grows the held share by `bytes` without checking the budget.
    ///
    /// Used after spilling everything shed-able still leaves the growth
    /// unfunded: execution must proceed, so the share is recorded anyway to
    /// keep pressure on other operators.
    pub(crate) fn grow_unchecked(&mut self, bytes: u64) {
        let mut state = self.manager.state.lock().unwrap();
        state.available_bytes = state.available_bytes.saturating_sub(bytes);
        drop(state);
        self.held += bytes;
    }

    /// Returns `bytes` of the held share to the budget.
    pub(crate) fn shrink(&mut self, bytes: u64) {
        let returned = bytes.min(self.held);
        self.held -= returned;
        self.manager.release(returned);
    }
}

impl Drop for SpillBudget {
    fn drop(&mut self) {
        self.manager.release(self.held);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio::time;

    use super::*;

    #[test]
    fn spill_budget_grow_shrink_and_drop_release() {
        let manager = Arc::new(MemoryManager::new());
        let total = manager.total_bytes;

        let mut budget = SpillBudget::new(manager.clone());
        assert!(budget.try_grow(total / 2));
        assert!(!budget.try_grow(total)); // over budget → pressure signal
        budget.shrink(total / 4);
        {
            let state = manager.state.lock().unwrap();
            assert_eq!(state.available_bytes, total - total / 2 + total / 4);
        }
        drop(budget);
        {
            let state = manager.state.lock().unwrap();
            assert_eq!(state.available_bytes, total);
        }
    }

    #[test]
    fn spill_budget_unchecked_growth_saturates() {
        let manager = Arc::new(MemoryManager::new());
        let total = manager.total_bytes;
        let mut budget = SpillBudget::new(manager.clone());
        budget.grow_unchecked(total + 100);
        {
            let state = manager.state.lock().unwrap();
            assert_eq!(state.available_bytes, 0);
        }
        drop(budget);
        // Release is clamped to the configured total.
        {
            let state = manager.state.lock().unwrap();
            assert_eq!(state.available_bytes, total);
        }
    }

    #[test]
    fn shrink_beyond_held_is_clamped() {
        let manager = Arc::new(MemoryManager::new());
        let total = manager.total_bytes;
        let mut budget = SpillBudget::new(manager.clone());
        assert!(budget.try_grow(100));
        budget.shrink(1000);
        let state = manager.state.lock().unwrap();
        assert_eq!(state.available_bytes, total);
    }

    #[test]
    fn test_get_or_init_memory_manager() {
        let manager1 = get_or_init_memory_manager();
        let manager2 = get_or_init_memory_manager();

        // Verify we get the same instance
        assert!(Arc::ptr_eq(manager1, manager2));
    }

    #[tokio::test]
    async fn test_zero_byte_request() {
        let manager = MemoryManager::new();
        let permit = manager.request_bytes(0).await.unwrap();
        assert_eq!(permit.bytes, 0);
    }

    #[tokio::test]
    async fn test_excessive_memory_request() {
        let manager = MemoryManager::new();
        let result = manager.request_bytes(manager.total_bytes + 1).await;
        assert!(result.is_err());
        if let Err(DaftError::ComputeError(_)) = result {
            // Expected error type
        } else {
            panic!("Expected ComputeError");
        }
    }

    #[tokio::test]
    async fn test_successful_memory_request() {
        let manager = MemoryManager::new();
        let total = manager.total_bytes;
        let first_request_size = total / 2;
        let second_request_size = total - first_request_size;

        // First request should succeed
        let permit1 = manager.request_bytes(first_request_size).await.unwrap();
        assert_eq!(permit1.bytes, first_request_size);

        // Second request should succeed
        let permit2 = manager.request_bytes(second_request_size).await.unwrap();
        assert_eq!(permit2.bytes, second_request_size);

        // Third request should fail
        let result = manager.try_request_bytes(1);
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_memory_release() {
        let manager = MemoryManager::new();
        let request_size = 1;
        let permit = manager.request_bytes(request_size).await.unwrap();

        // Verify available memory is reduced
        {
            let state = manager.state.lock().unwrap();
            assert_eq!(state.available_bytes, manager.total_bytes - request_size);
        }

        // Drop the permit
        drop(permit);

        // Verify memory is released
        {
            let state = manager.state.lock().unwrap();
            assert_eq!(state.available_bytes, manager.total_bytes);
        }
    }

    #[tokio::test]
    async fn test_waiting_for_memory() {
        let manager = Arc::new(MemoryManager::new());
        let total = manager.total_bytes;

        // Request all available memory
        let permit = manager.request_bytes(total).await.unwrap();

        // Spawn a task that waits for memory
        let manager_clone = manager.clone();
        let wait_handle = tokio::spawn(async move {
            let _permit = manager_clone.request_bytes(total / 2).await.unwrap();
        });

        // Short delay to ensure the waiting task is actually waiting
        time::sleep(Duration::from_millis(50)).await;

        // Drop the original permit
        drop(permit);

        // The waiting task should now complete
        wait_handle.await.unwrap();
    }

    #[tokio::test]
    async fn test_concurrent_memory_requests() {
        let manager = Arc::new(MemoryManager::new());
        let total = manager.total_bytes;
        let mut task_set = tokio::task::JoinSet::new();

        // Four tasks that request all available memory
        for _ in 0..4 {
            let manager_clone = manager.clone();
            task_set.spawn(async move {
                let _permit = manager_clone.request_bytes(total).await.unwrap();
            });
        }

        // Four tasks that request half the available memory
        for _ in 0..4 {
            let manager_clone = manager.clone();
            task_set.spawn(async move {
                let _permit = manager_clone.request_bytes(total / 2).await.unwrap();
            });
        }

        // Four tasks that request a quarter of the available memory
        for _ in 0..4 {
            let manager_clone = manager.clone();
            task_set.spawn(async move {
                let _permit = manager_clone.request_bytes(total / 4).await.unwrap();
            });
        }

        task_set.join_all().await;
    }
}
