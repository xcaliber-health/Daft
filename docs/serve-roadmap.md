# Query-server performance & scale roadmap

Evaluated roadmap for the remaining engine and serving improvements,
inspired by proven columnar-database designs. Every engine change is
flag-gated (default off), benchmarked in isolation, and A/B-verified
against the native baseline before any default changes — the same
discipline used for row-group scan splitting.

## Where we are

- The query server executes each query on the single-node streaming engine;
  one round trip per query, results streamed as produced.
- Landed so far from the same design study: execution-time row-group scan
  splitting (`enable_scan_task_row_group_splitting`, up to 1.43x on
  scan-bound queries), per-query timeouts, referenced-only partition-set
  scoping, multi-tenant identity/quotas (see `tests/serve/README.md`).
- Measured remaining gaps vs a state-of-the-art single-node engine at SF1
  TPC-H: grouped aggregation (~10x), filter/decode interplay (~14x on
  filter-heavy microbenches), fixed per-query pipeline setup (~75ms vs ~4ms).

## Phase B — Grouped-aggregation hash table (design + gated prototype)

The single biggest gap. Important context: the engine **already has** a
single-pass scatter aggregation (`src/daft-recordbatch/src/ops/inline_agg.rs`)
with typed accumulators and specialized single-column key paths — but it is
limited to numeric/bool Count/Sum/Product/Min/Max/AnyValue/BoolAnd/BoolOr;
everything else (multi-column keys, other aggregates) falls back to
"build per-group index lists, then gather each group and run a kernel"
(`ops/agg.rs`, `dispatch_per_group`), which is the slow path.

Prototype design (extend the existing inline pattern, do not replace the
sink or public types):

1. **Row arena**: per-group state stored as one contiguous row
   `[key values | 64-bit hash | aggregate states]` in a growable arena.
   Updating a group = compute its row address once, mutate states in place.
   No group→row index vectors, no gathers.
2. **Salted pointer table**: table entries pack a 16-bit hash tag with the
   row reference; probing compares tags first (one cache line) and touches
   key bytes only on tag match. Probe stride derived from the salt to
   avoid clustering. (Safe-Rust variant: `(u16 tag, u32 arena index)`.)
3. **Dictionary fast path**: when a group column arrives dictionary-encoded,
   probe the unique dictionary values once and cache their row addresses
   per dictionary — subsequent morsels reuse the cache.
4. **Radix-partitioned merge**: thread-local tables partitioned by hash so
   the final merge is one independent task per partition (the sink's
   existing partition-parallel finalize already has this shape).
5. **Direct-address fast path** for small bounded integer keys (no probing).

Rollout: new accumulator behind `enable_experimental_agg_hash_table`;
micro-benchmark vs both existing paths; TPC-H A/B (q1 is the canary);
retire fallback coverage aggregate-by-aggregate, never wholesale.

Isolation note: the hot kernel lives at crate boundaries
(`daft-recordbatch`, `daft-groupby`, `daft-core` series aggs) shared with
the distributed path — changes must be additive dispatch arms, not type
rewrites.

## Phase C — Memory substrate: accounting, spilling, negotiation

Prerequisite for per-tenant memory caps and for large-query robustness.
Today the engine's memory manager is an advisory byte-semaphore and **no
operator spills** — sort, aggregation, and join builds grow unboundedly.

1. Convert the resource manager to accounted budgets (per-query
   registration of blocking-operator reservations).
2. Spill for grouped aggregation first: radix-partitioned state makes
   external processing natural (process partitions one at a time under a
   memory reservation), then sort (external merge) and join build.
3. Cross-query negotiation: a central coordinator re-distributes the global
   budget across concurrent queries' registered operators (grant increments
   where they help throughput most; force spilling elsewhere) — the model
   proven by embedded analytical databases for multi-tenant fairness.
4. Only then: per-tenant memory caps in the server, wired through the same
   registration.

## Phase D — Filter execution improvements (scoped)

The parquet reader already does three-tier pushdown (row-group stats,
page-level row selections, and two-phase late materialization that decodes
predicate columns first). Remaining, low-risk wins:

- **Adaptive predicate ordering**: track observed selectivity per conjunct
  and reorder so the most selective/cheapest predicate runs first.
- **Dictionary filter pushdown**: evaluate equality/range predicates on the
  dictionary values during decode instead of on every decoded row.

**Rejected**: a DuckDB-style selection-vector rewrite of the columnar core.
It would change `Series`/`RecordBatch` semantics across every kernel — an
unacceptable blast radius for the remaining gain given late
materialization already exists.

## Phase E — Multi-node serving (native, not Ray-hosted)

The distributed scheduler is already engine-generic: the worker contract is
one trait (`src/daft-distributed/src/scheduling/worker.rs`), a non-Ray
in-process implementation exists as a test-gated template
(`scheduling/local_worker.rs`), the shuffle transport
(`src/daft-shuffles/`) is already independent of the cluster runtime, and
each distributed worker runs the same engine the server hosts.

Path:
1. Un-gate the generic worker plumbing.
2. `ServeWorkerManager`: workers are N server processes discovered from
   static config; task submission over the existing serve transport;
   results returned inline; workers started with their shuffle endpoint
   enabled.
3. Retarget the server's query path to the distributed plan runner for
   distributed-eligible queries, streaming the incremental result stream
   into the existing encoder loop.
4. Autoscaling and fault tolerance remain with the Ray backend until
   ported; the gateway-vs-tenant-tiering decision is parked until the
   production scale shape is fixed.

Explicit non-goal: hosting the server on Ray Serve — it re-introduces the
head-actor and object-store machinery the one-round-trip design avoids
while adding no engine capability.

## Deferred

- Scheduler fairness beyond admission (weighted round-robin across tenant
  queues; cooperative yield of long tasks).
- Warm pipeline reuse across queries (needs per-input cancellation in the
  executor's plan-state map).
- Standard SQL transport compatibility layers.
