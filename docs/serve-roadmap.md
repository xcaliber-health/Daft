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

## Phase B — Grouped aggregation (REVISED after release-build measurement)

**The premise changed.** The "~10x aggregation gap" (and the 14x filter
gap) were measured through a debug-built extension: `make build` compiles
the engine at opt-level 0, while the comparison engine came from a release
wheel. Measured again on `make build-release`:

- The single-threaded aggregation kernel (`inline_agg.rs` — typed key
  paths, string symbolization, packed two-string keys) is roughly **at
  parity with the reference engine per core** (e.g. 2-string-key groupby
  over 10M rows: ~187ms vs ~190ms single-core-equivalent).
- End-to-end parquet scan + groupby at 10M rows: **~3x gap**
  (79ms vs 26.5ms), not 10-20x.
- A DuckDB-style hash-table rewrite is therefore **not** the next win and
  is dropped from the roadmap. The bench harness that proves this lives at
  `src/daft-recordbatch/src/ops/bench_agg.rs`
  (`cargo test -p daft-recordbatch --release -- bench_agg --nocapture --ignored`).

**Negative result (measured, July 2026): persistent per-worker streaming
aggregation state does not help.** A prototype that kept one group map and
one set of accumulators alive across a worker's morsels (avoiding
per-morsel table rebuild, group-key gather, and intermediate batches) was
implemented, verified correct, and benchmarked: −10% on random
high-cardinality keys (bigger table, worse probe locality, no dedup
available when groups ≈ rows), neutral on the clustered TPC-H shapes. The
per-morsel aggregation in those queries already runs at kernel speed — the
regime is *insert-heavy* (~4 rows/group; ~26ns/row, matching the
5M-distinct kernel bench). Beating it requires a faster **insert path**
(salted-tag arena table) combined with **radix-partitioned thread-local
shards** so the merge parallelizes without hash-repartitioning
materialization — a coherent redesign, not an incremental patch. Until
then the current strategies stand.

The real, measured targets:

1. **Single-chunk partitions aggregate single-threaded.** A large
   in-memory partition (cached DataFrame, shipped partition set) reaches
   the grouped-aggregate sink as ONE work unit — one worker aggregates 10M
   rows while other cores idle (observed: pipeline SLOWER than one thread
   running the kernel directly). Fix: split oversized morsels from
   in-memory sources (or in the blocking-sink dispatcher) so the existing
   per-worker partial-agg parallelism engages. Contained change; big win
   for the serving path where shipped psets are single-chunk.
2. **Residual ~3x on scan+agg**: profile the composition (decode, morsel
   handoff, partial→final merge) on the release build before choosing the
   next kernel-level move. Note: row-group splitting was *slower* than
   unsplit on a 10-row-group file (97ms vs 79ms) — the flag's benefit is
   file-count dependent; keep it opt-in.

### Tail attribution (measured, complete)

- **Hash join** (q12, q21, and part of q19): decomposed by varying probe
  size at a fixed 1.5M-row build side. Build ≈ 1.7x the reference engine
  (47ms vs 28.5ms); **probe ≈ 6.3x (22.1 vs 3.5 ms per 1M rows)** with a
  large build table (cache-resident small tables probe much faster). The
  probe-side output construction (gather/materialize of matched rows) is
  the join target, not the build.
- **q19** (152ms total): 76ms column decode (only ~1.4x the reference
  engine's 54ms — the scan itself is fine), ~30ms join, **~46ms compound
  OR-predicate evaluation over strings** — the direct payoff case for the
  Phase D items (dictionary filter pushdown + adaptive predicate
  ordering).
- **q18/q15/q21 aggregation**: insert-heavy regime; see the negative
  result above — waits on the arena + radix-shards redesign.

Any future kernel work must A/B on `make build-release` only.

## Phase C — Memory substrate: accounting, spilling, negotiation

Prerequisite for per-tenant memory caps and for large-query robustness.

**Slice 1 landed: accounting + grouped-aggregation spilling.**
- `SpillBudget` (resource_manager): a non-blocking, growable share of the
  existing global budget (`DAFT_MEMORY_LIMIT` or system memory). Growth
  denial is the spill signal; all held bytes release on drop.
- `spill` module: scratch directories with drop-based cleanup; runs written
  through the existing size-rotated, compressed columnar stream writer and
  read back whole-file on the IO pool. No new serialization invented.
- Grouped-aggregate sink: every worker accounts its buffered partition
  bytes after each morsel; on budget denial it sheds its largest buffered
  hash partition to disk (partial and raw buckets kept separate) and
  finalize restores shed runs into the same buckets before the existing
  per-partition merge. Config: `enable_spilling` (default on),
  `spill_dirs` (default `/tmp`); compression follows the shuffle setting.
- Verified: identical results at 30/60MB budgets vs unpressured on a
  working set several times larger; spill events observed at 30MB; scratch
  removed on completion; escape hatch produces no disk activity; full
  local + serving suites green; TPC-H unchanged (spilling never triggers
  at normal budgets — flagged deltas on the two shortest queries were
  noise with mixed signs under attribution).

**Slice 2 landed: external sort.**
- Streaming run writer and streaming run cursor on the spill scratch
  (bounded channel to the IO pool; batches read back one file at a time).
- Sorted-run merge (`sorted_merge`): pairwise two-cursor streaming merges
  using the engine's cross-batch row comparator; ties prefer the earlier
  run (stable); intermediate merged runs stream back to disk; peak memory
  is one batch per active side plus the emitted chunk.
- Sort sink: buffered morsels are accounted; on budget denial the entire
  buffer (including the triggering morsel) is sorted and shed as one
  sorted run; finalize merges spilled runs with the sorted in-memory
  remainder. The no-pressure path is byte-identical to before.
- Verified: 2M-row sort under a 30MB budget produces output identical to
  the unpressured run (fully sorted, spill events observed, scratch
  cleaned); merge unit tests cover three-way merges with an intermediate
  disk pass, descending + nulls-first, and empty/single-run edges.
- Known bound: the sink's output contract still materializes the final
  sorted result in memory; true end-to-end streaming needs a
  streaming-output sink API (future work).

**Slice 3 landed: join-build memory accounting.**
- The build side feeds each morsel incrementally into the growing probe
  table, and that table must stay fully resident through the streaming
  probe phase — so spilling raw build input buys nothing on its own. A
  real join spill is a partitioned (Grace) hash join: partition both
  sides to disk under pressure, then run per-partition build+probe
  passes. That requires re-driving the streaming probe side and is
  recorded below as its own future project, not a slice.
- What landed instead: the build side charges its bytes to the shared
  budget and holds them until its probe phase completes (the accounting
  travels with the finalized build state). A build side that cannot be
  funded records unfunded growth and logs one clear warning. Effect:
  when a join coexists with shed-able operators, those operators spill
  sooner and the process stays at the configured budget — verified with
  a join feeding a high-cardinality aggregation at an 8MB budget
  (aggregation spills continuously, results exact) and a build side
  alone exceeding the budget (warns, completes, results exact).

Remaining slices:
1. Partitioned (Grace) hash join — the true join spill; needs
   probe-side partitioning, probe-input spill, and per-partition
   replay. Substantial, design-first.
2. Cross-query negotiation: a central coordinator re-distributes the global
   budget across concurrent queries' registered operators (grant increments
   where they help throughput most; force spilling elsewhere) — the model
   proven by embedded analytical databases for multi-tenant fairness.
3. Only then: per-tenant memory caps in the server, wired through the same
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
