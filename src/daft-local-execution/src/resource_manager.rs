use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicU64, Ordering},
};

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
    holders: Vec<Arc<HolderSlot>>,
}

/// Registry entry for one budget holder, used to negotiate shares between
/// concurrent holders: when a grow is denied, holders above their fair
/// share are asked to shed the excess.
struct HolderSlot {
    /// Bytes currently held, mirrored under the manager lock.
    held: AtomicU64,
    /// Bytes this holder has been asked to shed; consumed cooperatively at
    /// the holder's next reconcile.
    shed_requested: AtomicU64,
    /// Whether the holder can shed state to disk. Holders that cannot
    /// (resident build tables) are exempt from shed requests, and their
    /// held bytes are excluded from the shareable pool.
    can_shed: bool,
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
                holders: Vec::new(),
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
                    holders: Vec::new(),
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

    /// Posts shed requests to holders exceeding their fair share of the
    /// shareable pool. Called when a grow is denied so concurrent holders
    /// converge toward equal shares instead of first-come-first-served.
    fn request_rebalance(&self) {
        let state = self.state.lock().unwrap();
        let shedable: Vec<&Arc<HolderSlot>> =
            state.holders.iter().filter(|slot| slot.can_shed).collect();
        if shedable.len() < 2 {
            // A single shed-able holder is already spilling on its own
            // denials; nothing to negotiate.
            return;
        }
        let reserved: u64 = state
            .holders
            .iter()
            .filter(|slot| !slot.can_shed)
            .map(|slot| slot.held.load(Ordering::Relaxed))
            .sum();
        let pool = self.total_bytes.saturating_sub(reserved);
        let fair_share = pool / shedable.len() as u64;
        for slot in shedable {
            let held = slot.held.load(Ordering::Relaxed);
            if held > fair_share {
                let excess = held - fair_share;
                // Keep the largest outstanding request; requests are
                // consumed (reset) by the holder when it sheds.
                slot.shed_requested.fetch_max(excess, Ordering::Relaxed);
            }
        }
    }

    /// Returns `bytes` to the budget and wakes any waiters.
    fn release(&self, bytes: u64) {
        if bytes > 0 {
            {
                let mut state = self.state.lock().unwrap();
                state.available_bytes = (state.available_bytes + bytes).min(self.total_bytes);
            }
            self.notify.notify_waiters();
        }
    }
}

/// Per-query view of the memory budget: the shared global manager plus an
/// optional ceiling on this query's combined holdings.
///
/// The ceiling is enforced before the global pool, so one query (or the
/// tenant it belongs to) cannot monopolize a shared process even when the
/// process-wide budget has room.
#[derive(Clone)]
pub(crate) struct QueryMemoryScope {
    manager: Arc<MemoryManager>,
    group: Option<Arc<QueryGroup>>,
}

/// Combined holdings of one query's budget holders against its ceiling.
struct QueryGroup {
    cap: u64,
    held: AtomicU64,
}

impl QueryMemoryScope {
    /// Creates a scope over the shared manager with an optional per-query
    /// ceiling; `None` (or zero) means only the global budget applies.
    pub(crate) fn new(manager: Arc<MemoryManager>, cap: Option<u64>) -> Self {
        Self {
            manager,
            group: cap.filter(|cap| *cap > 0).map(|cap| {
                Arc::new(QueryGroup {
                    cap,
                    held: AtomicU64::new(0),
                })
            }),
        }
    }

    /// The shared process-wide manager.
    pub(crate) fn manager(&self) -> &Arc<MemoryManager> {
        &self.manager
    }

    /// Attempts to charge `bytes` against the per-query ceiling.
    /// Returns whether it fit (always true without a ceiling).
    fn try_charge_group(&self, bytes: u64) -> bool {
        let Some(group) = &self.group else {
            return true;
        };
        group
            .held
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |held| {
                held.checked_add(bytes).filter(|new| *new <= group.cap)
            })
            .is_ok()
    }

    fn charge_group_unchecked(&self, bytes: u64) {
        if let Some(group) = &self.group {
            group.held.fetch_add(bytes, Ordering::Relaxed);
        }
    }

    fn uncharge_group(&self, bytes: u64) {
        if let Some(group) = &self.group {
            let _ = group
                .held
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |held| {
                    Some(held.saturating_sub(bytes))
                });
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
    scope: QueryMemoryScope,
    slot: Arc<HolderSlot>,
    held: u64,
}

impl SpillBudget {
    /// Creates an empty, shed-capable budget share within the scope.
    pub(crate) fn new(scope: QueryMemoryScope) -> Self {
        Self::with_shed_capability(scope, true)
    }

    /// Creates an empty budget share, declaring whether the holder can shed
    /// its state to disk when asked. Holders that cannot shed are exempt
    /// from rebalancing requests, and their held bytes shrink the pool the
    /// remaining holders divide.
    pub(crate) fn with_shed_capability(scope: QueryMemoryScope, can_shed: bool) -> Self {
        let slot = Arc::new(HolderSlot {
            held: AtomicU64::new(0),
            shed_requested: AtomicU64::new(0),
            can_shed,
        });
        scope
            .manager
            .state
            .lock()
            .unwrap()
            .holders
            .push(slot.clone());
        Self {
            scope,
            slot,
            held: 0,
        }
    }

    /// Attempts to grow the held share by `bytes`; returns whether it fit.
    ///
    /// A denial means the global budget is exhausted: the owner should
    /// spill buffered state (and `shrink`) before retrying or proceeding.
    /// The denial also asks concurrent holders above their fair share to
    /// shed, so sustained contention converges toward equal shares.
    #[must_use]
    pub(crate) fn try_grow(&mut self, bytes: u64) -> bool {
        // The per-query ceiling comes first: exceeding it must trigger this
        // query's own shedding regardless of global headroom, and asking
        // other queries to rebalance would not help.
        if !self.scope.try_charge_group(bytes) {
            return false;
        }
        if self.scope.manager.try_reserve(bytes) {
            self.held += bytes;
            self.slot.held.store(self.held, Ordering::Relaxed);
            true
        } else {
            self.scope.uncharge_group(bytes);
            self.scope.manager.request_rebalance();
            false
        }
    }

    /// Grows the held share by `bytes` without checking the budget.
    ///
    /// Used after spilling everything shed-able still leaves the growth
    /// unfunded: execution must proceed, so the share is recorded anyway to
    /// keep pressure on other operators.
    pub(crate) fn grow_unchecked(&mut self, bytes: u64) {
        self.scope.charge_group_unchecked(bytes);
        let mut state = self.scope.manager.state.lock().unwrap();
        state.available_bytes = state.available_bytes.saturating_sub(bytes);
        drop(state);
        self.held += bytes;
        self.slot.held.store(self.held, Ordering::Relaxed);
    }

    /// Returns `bytes` of the held share to the budget.
    pub(crate) fn shrink(&mut self, bytes: u64) {
        let returned = bytes.min(self.held);
        self.held -= returned;
        self.slot.held.store(self.held, Ordering::Relaxed);
        self.scope.uncharge_group(returned);
        self.scope.manager.release(returned);
    }

    /// Takes and clears any outstanding request for this holder to shed
    /// bytes, posted by concurrent holders that were denied growth.
    #[must_use]
    pub(crate) fn take_shed_request(&self) -> u64 {
        self.slot.shed_requested.swap(0, Ordering::Relaxed)
    }
}

impl Drop for SpillBudget {
    fn drop(&mut self) {
        {
            let mut state = self.scope.manager.state.lock().unwrap();
            state.holders.retain(|slot| !Arc::ptr_eq(slot, &self.slot));
        }
        self.scope.uncharge_group(self.held);
        self.scope.manager.release(self.held);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio::time;

    use super::*;

    fn scope(manager: &Arc<MemoryManager>) -> QueryMemoryScope {
        QueryMemoryScope::new(manager.clone(), None)
    }

    #[test]
    fn per_query_cap_denies_before_the_global_pool() {
        let manager = Arc::new(MemoryManager::new());
        let capped = QueryMemoryScope::new(manager, Some(100));

        let mut budget = SpillBudget::new(capped.clone());
        assert!(budget.try_grow(80));
        // Global pool has room, but the query ceiling does not.
        assert!(!budget.try_grow(30));
        budget.shrink(50);
        assert!(budget.try_grow(30));

        // A second holder in the same scope shares the ceiling.
        let mut sibling = SpillBudget::new(capped);
        assert!(!sibling.try_grow(50));
        assert!(sibling.try_grow(40));
    }

    #[test]
    fn cap_denial_leaves_global_accounting_untouched() {
        let manager = Arc::new(MemoryManager::new());
        let total = manager.total_bytes;
        let capped = QueryMemoryScope::new(manager.clone(), Some(10));
        let mut budget = SpillBudget::new(capped);
        assert!(!budget.try_grow(100));
        assert_eq!(manager.state.lock().unwrap().available_bytes, total);
    }

    #[test]
    fn dropped_budget_releases_its_group_charge() {
        let manager = Arc::new(MemoryManager::new());
        let capped = QueryMemoryScope::new(manager, Some(100));
        {
            let mut budget = SpillBudget::new(capped.clone());
            assert!(budget.try_grow(90));
        }
        let mut fresh = SpillBudget::new(capped);
        assert!(fresh.try_grow(90));
    }

    #[test]
    fn spill_budget_grow_shrink_and_drop_release() {
        let manager = Arc::new(MemoryManager::new());
        let total = manager.total_bytes;

        let mut budget = SpillBudget::new(scope(&manager));
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
        let mut budget = SpillBudget::new(scope(&manager));
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
        let mut budget = SpillBudget::new(scope(&manager));
        assert!(budget.try_grow(100));
        budget.shrink(1000);
        let state = manager.state.lock().unwrap();
        assert_eq!(state.available_bytes, total);
    }

    #[test]
    fn denial_posts_shed_requests_toward_fair_share() {
        let manager = Arc::new(MemoryManager::new());
        let total = manager.total_bytes;

        let mut hog = SpillBudget::new(scope(&manager));
        assert!(hog.try_grow(total * 8 / 10));

        let mut late = SpillBudget::new(scope(&manager));
        assert!(!late.try_grow(total / 2)); // denied → rebalance posted

        // The hog is asked to shed down to the fair share (half the pool).
        let requested = hog.take_shed_request();
        assert_eq!(requested, total * 8 / 10 - total / 2);
        // Requests are consumed once taken.
        assert_eq!(hog.take_shed_request(), 0);
        // The denied holder, at zero held, is not asked to shed.
        assert_eq!(late.take_shed_request(), 0);
    }

    #[test]
    fn non_shedable_holders_shrink_the_shareable_pool() {
        let manager = Arc::new(MemoryManager::new());
        let total = manager.total_bytes;

        let mut resident = SpillBudget::with_shed_capability(scope(&manager), false);
        resident.grow_unchecked(total / 2);

        let mut a = SpillBudget::new(scope(&manager));
        assert!(a.try_grow(total * 4 / 10));
        let mut b = SpillBudget::new(scope(&manager));
        assert!(!b.try_grow(total / 4)); // only ~10% left → denied

        // Fair share = (total - resident) / 2 = total/4; a holds 40%.
        assert_eq!(a.take_shed_request(), total * 4 / 10 - total / 4);
        // Non-shed-able holders are never asked to shed.
        assert_eq!(resident.take_shed_request(), 0);
    }

    #[test]
    fn single_holder_gets_no_shed_requests() {
        let manager = Arc::new(MemoryManager::new());
        let total = manager.total_bytes;
        let mut only = SpillBudget::new(scope(&manager));
        assert!(only.try_grow(total));
        assert!(!only.try_grow(1));
        assert_eq!(only.take_shed_request(), 0);
    }

    #[test]
    fn dropped_holders_leave_the_registry() {
        let manager = Arc::new(MemoryManager::new());
        {
            let _budget = SpillBudget::new(scope(&manager));
            assert_eq!(manager.state.lock().unwrap().holders.len(), 1);
        }
        assert_eq!(manager.state.lock().unwrap().holders.len(), 0);
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
