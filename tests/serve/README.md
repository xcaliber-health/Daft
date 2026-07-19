# Serve parity matrix

Every DataFrame API surface and how it behaves through the remote runner
(`daft.connect` / `RemoteRunner`), with the test that proves it. "Identical"
means the remote result is asserted equal to the in-process result.

| Surface | Status | Proven by |
|---|---|---|
| filter / select / with_columns | identical | `test_equality.py` (`filter_select`, `with_columns`) |
| groupby + aggregations | identical | `test_equality.py` (`groupby_agg`, `global_agg`, `extended_aggs`) |
| stddev, count_distinct, list/string agg, any_value | identical | `test_equality.py` (`extended_aggs`) |
| sort / limit / distinct | identical | `test_equality.py` (`sort_limit`, `distinct`) |
| joins (inner/left/right/outer/semi/anti) | identical | `test_equality.py::test_joins_match_native` |
| window functions (row_number, rank, dense_rank, agg-over) | identical | `test_equality.py` (`window_*`) |
| pivot / unpivot | identical | `test_equality.py` (`pivot`, `unpivot`) |
| explode / unnest / struct access | identical | `test_equality.py` (`explode`, `struct_get_and_unnest`) |
| expression namespaces (string, list, map, temporal) | identical | `test_equality.py` (`*_namespace`) |
| concat / into_partitions / repartition | identical | `test_equality.py` |
| every scalar/nested/decimal/tz dtype round trip | identical | `test_types.py` |
| scalar function UDFs (incl. captured closure state) | identical | `test_udfs.py` |
| class UDFs | identical | `test_udfs.py::test_class_udf` |
| async function UDFs | identical | `test_udfs.py::test_async_function_udf` |
| pooled UDFs (`concurrency=N`) | identical — pool runs on the server | `test_udfs.py::test_pooled_udf_with_concurrency` |
| process-isolated UDFs (`use_process=True`) | identical — processes spawn on the server | `test_udfs.py::test_process_isolated_udf` |
| GPU resource requests | mirrors local semantics (advisory; not enforced on a host without the resource) | `test_udfs.py::test_gpu_resource_request_behaves_like_native` |
| in-memory sources (`from_pydict` / `from_arrow`) | shipped with the plan; only referenced sets are materialized and sent | `test_equality.py::test_multiple_in_memory_sources_ship_together`, `test_auth_limits.py` |
| catalog table reads (lazy operator ships; scan planning server-side) | identical | `test_iceberg.py` |
| native file reads (`read_parquet` / `read_csv`) | identical; scan planning runs client-side (logged + noted in explain) because the operator holds process-local state; the server must reach the same paths | `test_equality.py::test_read_parquet_scans_execute_server_side`, `test_plan_shipping.py` |
| writes (parquet / csv / table-format) | execute through the connection when it is the process-global runner; snapshot commit stays client-side | `test_writes.py` |
| textual queries (`connection.sql`) | planned against server catalogs; version-skew tolerant | `test_sql_lane.py`, `test_lifecycle.py` |
| `explain` | server-side optimized plan; notes client-side planning fallback | `test_plan_shipping.py::test_explain_notes_client_side_planning` |
| streaming iteration / abandoned iterators | incremental; abandoning releases the server slot | `test_streaming_cancel.py`, `test_streaming_backpressure.py` |
| backpressure (small result buffer, slow consumer) | bounded buffer; no wedge | `test_streaming_backpressure.py` |
| cancellation (cross-connection, disconnect) | prompt abort, slot release | `test_streaming_cancel.py` |
| per-query execution timeout | typed error, slot release | `test_timeouts.py` |
| concurrent distinct clients | isolated results, no cross-client data visibility | `test_concurrent_clients.py` |
| auth (bearer token) | constant-time check; bad tokens rejected at connect | `test_auth_limits.py` |
| payload size limits | typed size error from ~4 MiB up to the configured cap | `test_auth_limits.py` |
| version mismatch | plan lane rejected with a clear error; text lane tolerant | `test_auth_limits.py::test_version_mismatch_rejects_plan_lane_end_to_end` |
| server lifecycle (subprocess, SIGTERM drain, crash) | graceful drain exits 0; crash surfaces an error, never hangs | `test_lifecycle.py` |

## Multi-tenancy

| Behavior | Status | Proven by |
|---|---|---|
| per-tenant bearer tokens resolve tenant identity | enforced, constant-time | `test_tenancy.py`, `auth.rs` unit tests |
| per-tenant admission slots (dedicated pool per tenant with a slot override) | one tenant's saturation never blocks another | `test_tenancy.py::test_saturated_tenant_*` |
| per-tenant execution timeout | typed error; other tenants unaffected | `test_tenancy.py::test_tenant_query_timeout_*` |
| per-tenant payload cap | typed size error; other tenants unaffected | `test_tenancy.py::test_tenant_payload_cap_*` |
| cancel scoped to owning tenant | cross-tenant cancel sees "not found" | `test_tenancy.py::test_cancel_is_scoped_*`, `registry.rs` unit tests |
| legacy single-token / insecure modes | unchanged | `test_auth_limits.py` |
| per-tenant memory caps | not yet — requires engine memory accounting + spilling (roadmap Phase C); use dedicated pods for hard memory isolation | `docs/serve-roadmap.md` |

## Known limitations (by design)

- **Plan lane requires identical client/server engine versions** — serialized
  plans are an internal format. Textual queries tolerate version skew.
- **UDF dependencies must be installed on the server.** Functions pickle into
  the plan; modules they reference resolve at execution time server-side.
- **Native file scans require shared storage visibility** — the client plans
  them (materializing pushdowns) and the server reads the enumerated files,
  so both sides must reach the same paths. Catalog-backed tables have no such
  requirement.
- **Eager write methods route through the process-global runner** — a script
  must use `daft.connect` (rather than a standalone `RemoteRunner`) for its
  writes to execute remotely.
