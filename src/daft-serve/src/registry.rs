//! Registry of running queries, enabling cross-connection cancellation.
//!
//! Each admitted query registers a cancellation token under its
//! client-supplied id. A `cancel_query` action from any connection trips the
//! token; the execution task observes it and aborts. Entries deregister on
//! completion via a drop guard, so the registry cannot leak entries even when
//! execution fails or the client disconnects mid-stream. Re-registering an id
//! (idempotent resubmission racing an unfinished attempt) replaces the entry;
//! generation numbers ensure the superseded attempt's guard cannot deregister
//! the newer attempt.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use tokio_util::sync::CancellationToken;

#[derive(Debug)]
struct Entry {
    token: CancellationToken,
    generation: u64,
    owner: Option<String>,
}

#[derive(Debug, Default)]
struct Inner {
    entries: HashMap<String, Entry>,
    next_generation: u64,
}

/// Tracks in-flight queries by id.
#[derive(Debug, Default, Clone)]
pub struct QueryRegistry {
    inner: Arc<Mutex<Inner>>,
}

/// Locks registry state, recovering from a poisoned lock.
///
/// The guarded map has no cross-field invariants, so state left by a
/// panicking holder is safe to keep using; recovering keeps cancellation and
/// cleanup working instead of cascading panics through request handlers.
fn lock_inner(inner: &Mutex<Inner>) -> std::sync::MutexGuard<'_, Inner> {
    inner
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Deregisters a query when dropped.
#[derive(Debug)]
pub struct QueryGuard {
    registry: QueryRegistry,
    query_id: String,
    token: CancellationToken,
    generation: u64,
}

impl QueryRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a query owned by the given caller and returns its guard.
    ///
    /// If the id is already registered, the previous entry is replaced and
    /// the superseded guard becomes a no-op on drop.
    #[must_use]
    pub fn register(&self, query_id: &str, owner: Option<&str>) -> QueryGuard {
        let token = CancellationToken::new();
        let generation = {
            let mut inner = lock_inner(&self.inner);
            let generation = inner.next_generation;
            inner.next_generation += 1;
            inner.entries.insert(
                query_id.to_string(),
                Entry {
                    token: token.clone(),
                    generation,
                    owner: owner.map(str::to_string),
                },
            );
            generation
        };
        QueryGuard {
            registry: self.clone(),
            query_id: query_id.to_string(),
            token,
            generation,
        }
    }

    /// Trips the cancellation token for `query_id` when the caller owns it.
    ///
    /// A query is cancellable only by the identity that registered it, so
    /// one tenant can never cancel another tenant's query; a mismatched
    /// caller sees the same "not found" outcome as for an unknown id.
    /// Cancelling an unknown or already-finished query is a no-op, making
    /// cancellation idempotent.
    ///
    /// Returns whether a running query with that id was found and owned by
    /// the caller.
    pub fn cancel(&self, query_id: &str, caller: Option<&str>) -> bool {
        let inner = lock_inner(&self.inner);
        inner.entries.get(query_id).is_some_and(|entry| {
            if entry.owner.as_deref() == caller {
                entry.token.cancel();
                true
            } else {
                false
            }
        })
    }

    /// Number of currently registered queries.
    #[must_use]
    pub fn len(&self) -> usize {
        lock_inner(&self.inner).entries.len()
    }

    /// Whether no queries are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Cancels every registered query; used when a shutdown drain times out.
    pub fn cancel_all(&self) {
        let inner = lock_inner(&self.inner);
        for entry in inner.entries.values() {
            entry.token.cancel();
        }
    }
}

impl QueryGuard {
    /// Cancellation token observed by this query's execution task.
    #[must_use]
    pub fn token(&self) -> CancellationToken {
        self.token.clone()
    }
}

impl Drop for QueryGuard {
    fn drop(&mut self) {
        let mut inner = lock_inner(&self.registry.inner);
        if inner
            .entries
            .get(&self.query_id)
            .is_some_and(|entry| entry.generation == self.generation)
        {
            inner.entries.remove(&self.query_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_drop_leaves_registry_empty() {
        let registry = QueryRegistry::new();
        {
            let _guard = registry.register("q1", None);
            assert_eq!(registry.len(), 1);
        }
        assert!(registry.is_empty());
    }

    #[test]
    fn cancel_trips_token_of_registered_query() {
        let registry = QueryRegistry::new();
        let guard = registry.register("q1", None);
        assert!(!guard.token().is_cancelled());
        assert!(registry.cancel("q1", None));
        assert!(guard.token().is_cancelled());
    }

    #[test]
    fn cancel_unknown_query_is_noop() {
        let registry = QueryRegistry::new();
        assert!(!registry.cancel("missing", None));
    }

    #[test]
    fn double_cancel_is_idempotent() {
        let registry = QueryRegistry::new();
        let guard = registry.register("q1", None);
        assert!(registry.cancel("q1", None));
        assert!(registry.cancel("q1", None));
        assert!(guard.token().is_cancelled());
    }

    #[test]
    fn cancel_after_completion_is_noop() {
        let registry = QueryRegistry::new();
        drop(registry.register("q1", None));
        assert!(!registry.cancel("q1", None));
    }

    #[test]
    fn resubmission_replaces_entry_and_old_guard_is_inert() {
        let registry = QueryRegistry::new();
        let old_guard = registry.register("q1", None);
        let new_guard = registry.register("q1", None);
        assert_eq!(registry.len(), 1);

        // Dropping the superseded guard must not deregister the new attempt.
        drop(old_guard);
        assert_eq!(registry.len(), 1);

        // Cancellation targets the new attempt's token.
        assert!(registry.cancel("q1", None));
        assert!(new_guard.token().is_cancelled());

        drop(new_guard);
        assert!(registry.is_empty());
    }

    #[test]
    fn cancel_requires_matching_owner() {
        let registry = QueryRegistry::new();
        let guard = registry.register("q1", Some("alpha"));

        // A different tenant or an anonymous caller sees "not found".
        assert!(!registry.cancel("q1", Some("beta")));
        assert!(!registry.cancel("q1", None));
        assert!(!guard.token().is_cancelled());

        // The owner can cancel.
        assert!(registry.cancel("q1", Some("alpha")));
        assert!(guard.token().is_cancelled());
    }

    #[test]
    fn anonymous_owner_scopes_to_anonymous_callers() {
        let registry = QueryRegistry::new();
        let guard = registry.register("q1", None);
        assert!(!registry.cancel("q1", Some("alpha")));
        assert!(registry.cancel("q1", None));
        assert!(guard.token().is_cancelled());
    }

    #[test]
    fn cancel_all_trips_every_token() {
        let registry = QueryRegistry::new();
        let g1 = registry.register("q1", None);
        let g2 = registry.register("q2", None);
        registry.cancel_all();
        assert!(g1.token().is_cancelled());
        assert!(g2.token().is_cancelled());
    }
}
