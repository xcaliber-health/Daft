# Daft Serve — User Guide

Daft Serve runs the Daft query engine as a network service. You start a server,
connect one or more clients, and the clients run ordinary Daft queries that
execute on the server — close to the data — with results streamed back.

This guide is arranged so you can read it top to bottom the first time, then come
back to any section as a reference.

**Getting started**
- [How it works, in one picture](#how-it-works-in-one-picture)
- [Your first server and client](#your-first-server-and-client)

**Day-to-day use**
- [Running a server](#running-a-server)
- [Connecting a client](#connecting-a-client)
- [Two ways to send a query](#two-ways-to-send-a-query)
- [Working with data catalogs](#working-with-data-catalogs)

**Operating a server**
- [Authentication](#authentication)
- [Serving multiple teams](#serving-multiple-teams)
- [Memory limits and spilling](#memory-limits-and-spilling)
- [Query lifecycle: streaming, cancellation, timeouts, shutdown](#query-lifecycle)
- [Deployment](#deployment)

**Reference**
- [Configuration reference](#configuration-reference)
- [Error reference](#error-reference)
- [Measuring performance](#measuring-performance)
- [Testing](#testing)
- [Limitations to know about](#limitations-to-know-about)
- [Troubleshooting](#troubleshooting)
- [Under the hood](#under-the-hood)

---

## How it works, in one picture

```
┌────────────┐    your query      ┌──────────────────────────┐
│   Client   │ ─────────────────► │        Daft Serve        │
│            │                    │  ┌────────────────────┐  │
│ daft.connect                    │  │    Daft engine     │  │──► object store
│            │ ◄───────────────── │  └────────────────────┘  │    / data catalog
└────────────┘   result stream    │  auth · query slots ·    │    / files
                                   │  memory limits · spill   │
                                   └──────────────────────────┘
```

- The **client** builds a query and sends it.
- The **server** plans it, reads the data, does the computation, and streams the
  results back as they're ready.
- Your client machine never needs to touch the data or hold a large amount of
  memory.

Each query is a single request and response — there is no cluster coordinator or
extra data shuffling in the request path.

---

## Your first server and client

This walks through a complete local setup. It assumes you've run
`make build-release` once.

**Step 1 — Start a server.** In one terminal:

```bash
export DAFT_SERVE_TOKEN=secret
python -m daft.serve --host 127.0.0.1 --port 9494
```

Wait for the log line `serving on grpc://127.0.0.1:9494`.

**Step 2 — Connect from Python.** In another terminal:

```python
import daft

conn = daft.connect("daft://localhost:9494", token="secret")
```

**Step 3 — Run a query.** After connecting, every Daft query in this process runs
on the server:

```python
df = daft.read_parquet("s3://bucket/events/*.parquet")
df.groupby("country").agg(daft.col("clicks").sum()).collect()
```

**Step 4 — Check on the server.** The connection object gives you server-scoped
tools:

```python
conn.server_info()      # version, limits, and available catalogs
print(conn.explain(df)) # the plan the server will run
```

**Step 5 — Stop the server.** Press `Ctrl-C` in the server terminal. It finishes
in-flight queries, then exits cleanly.

You now have the full loop: start, connect, query, inspect, stop. The rest of
this guide fills in the details.

---

## Running a server

### From the command line

```bash
export DAFT_SERVE_TOKEN=secret
python -m daft.serve --host 0.0.0.0 --port 9494
```

The most common options:

| Option | Default | What it does |
|---|---|---|
| `--host` | `127.0.0.1` | Which network interface to listen on. Use `0.0.0.0` to accept remote clients. |
| `--port` | `9494` | Which port to listen on. `0` picks a free one automatically. |
| `--config` | — | Path to a config file (see [Configuration reference](#configuration-reference)). |
| `--token-env` | `DAFT_SERVE_TOKEN` | Which environment variable holds the access token. |
| `--max-concurrent-queries` | `4` | How many queries run at once. Extra queries wait in line. |
| `--queue-timeout-secs` | `60` | How long a waiting query holds on before giving up. |
| `--max-pset-bytes` | `256 MiB` | Largest chunk of in-memory data a client may send with a query. |
| `--query-timeout` *(config)* | off | Cut off any query that runs longer than this. |
| `--drain-timeout-secs` | `120` | On shutdown, how long to let running queries finish. |
| `--disable-plan-payload` | off | Accept only text queries (see [Two ways to send a query](#two-ways-to-send-a-query)). |
| `--allow-insecure-remote` | off | Allow a remote server with no token. Development only. |

Command-line options override anything set in a config file.

### From Python

Useful for embedding a server in a larger program or in tests:

```python
import daft.serve

server = daft.serve.start_server(
    daft.serve.ServeSettings(
        host="0.0.0.0",
        port=9494,
        token="secret",
        max_concurrent_queries=8,
    )
)

print(server.address())              # e.g. grpc://0.0.0.0:9494
# ... let it serve ...
server.shutdown(drain_timeout_secs=30)
```

The returned handle has `.address()`, `.port()`, `.active_queries()`, `.wait()`
(block until stopped), and `.shutdown(...)`.

---

## Connecting a client

### The simple way: `daft.connect`

```python
import daft

conn = daft.connect("daft://analytics.internal:9494", token="secret")
```

This does two things:

1. Redirects **every** Daft query in the current process to the server.
2. Returns a connection object for server-scoped actions.

You can only call `daft.connect` once per process. The address can be written any
of these ways (all default to port 9494 if you leave it off):

```
daft://host:port      grpc://host:port      host:port      host
```

The connection object gives you:

| Method | What it does |
|---|---|
| `conn.address` | The normalized server address you're connected to. |
| `conn.server_info()` | The server's version, limits, and available catalogs. |
| `conn.sql(query)` | Run a text query against the server's catalogs, returns a DataFrame. |
| `conn.explain(df)` | Show the plan the server would run for a DataFrame. |
| `conn.cancel_query(query_id)` | Stop a running query by its id. |

### Running queries without redirecting the whole process

If you want to keep local execution as your default and only send *some* work to
a server, use the runner directly instead of `daft.connect`:

```python
from daft.runners.remote_runner import RemoteRunner

runner = RemoteRunner("grpc://localhost:9494", token="secret")
for table in runner.run_iter_tables(df._builder):
    ...
```

> **Note on writes.** Methods that write data out (`df.write_parquet(...)` and
> friends) run through whatever runner is set for the whole process. To make
> writes happen on the server, connect with `daft.connect` rather than a
> standalone `RemoteRunner`.

---

## Two ways to send a query

A query reaches the server in one of two forms. You usually don't choose
explicitly — it follows from how you wrote the query — but it helps to know the
difference.

| | **DataFrame query** | **Text query** |
|---|---|---|
| You write | `df.filter(...).collect()` after `daft.connect` | `conn.sql("SELECT ...")` |
| Client and server versions | Must be the **same** | Can differ |
| Needs the client to have data access | Only for direct file reads | Never |
| Best for | Programmatic pipelines | Ad-hoc queries, thin clients, mixed versions |

**DataFrame queries** send the query's plan to the server, which optimizes and
runs it. Because the plan is an internal format, the client and server must run
the same engine version.

**Text queries** send a string that the server plans from scratch against its own
catalogs. They tolerate version differences and work even when the client has no
data access at all — ideal for lightweight clients and dashboards.

If you run a server with `--disable-plan-payload`, it accepts text queries only.

---

## Working with data catalogs

A **catalog** tells the server where your tables live. Attach catalogs when you
start the server, and clients can query those tables by name — without needing
any storage credentials themselves.

```yaml
# serve.yaml
catalogs:
  - name: lake
    properties:
      type: rest
      uri: http://catalog:8181
      warehouse: s3://warehouse/
      s3.endpoint: http://storage:9000
```

```bash
python -m daft.serve --config serve.yaml
```

Now a client can query the `lake` catalog's tables:

```python
conn.sql("SELECT * FROM lake.sales WHERE year = 2026").collect()
```

Using a network-reachable catalog (rather than local files) is the recommended
setup: the server plans and reads everything itself, so the client stays thin.
See [Limitations to know about](#limitations-to-know-about) for the one case —
direct file reads — where the client also needs data access.

---

## Authentication

Clients prove who they are with a **token** — a shared secret string sent with
every request.

- Set the server's token in the `DAFT_SERVE_TOKEN` environment variable (or
  point `--token-env` at a different variable name).
- Clients pass it as `daft.connect(address, token="...")`.
- A wrong or missing token is rejected immediately, before any query runs.

A token is **required** whenever the server listens on anything other than
`127.0.0.1`. The server refuses to start on a public interface without one,
unless you explicitly pass `--allow-insecure-remote` (which you should only do
for local development).

For serving several teams with *different* tokens and limits, see the next
section.

---

## Serving multiple teams

You can host many independent **tenants** on one server. Each tenant has its own
token and, optionally, its own limits — so one busy team can't slow down or crowd
out another.

```yaml
# serve.yaml
max_concurrent_queries: 8          # the shared default
tenants:
  - name: analytics
    max_concurrent_queries: 4      # its own private set of query slots
    query_timeout_secs: 120
    memory_cap_bytes: 2147483648   # 2 GiB per query, then spill to disk
  - name: reporting
    max_pset_bytes: 134217728
```

Each tenant's token comes from an environment variable named
`DAFT_SERVE_TOKEN_<NAME>` — the tenant name in capitals, with dashes turned into
underscores:

```bash
export DAFT_SERVE_TOKEN_ANALYTICS=analytics-secret
export DAFT_SERVE_TOKEN_REPORTING=reporting-secret
```

When any tenant is configured, **every** request must present a tenant token, and
the plain server-wide token is ignored.

Each tenant can override these (anything left out inherits the server default):

| Setting | Effect |
|---|---|
| `max_concurrent_queries` | Gives the tenant its **own** pool of query slots. Filling it never delays other tenants. |
| `queue_timeout_secs` | How long this tenant's queries wait for a slot. |
| `query_timeout_secs` | Time limit for this tenant's queries. |
| `max_pset_bytes` | Largest in-memory chunk this tenant may send. |
| `memory_cap_bytes` | Per-query memory ceiling before this tenant's queries spill to disk. |

What you get for free:

- One tenant filling its slots never blocks another.
- A tenant's timeout, size, or memory errors don't touch other tenants.
- A tenant can only cancel **its own** queries.
- A tenant's total memory use is capped at `slots × memory_cap_bytes`.

---

## Memory limits and spilling

Big queries can need more memory than you want any one query — or the whole
server — to use. Daft Serve handles this by **spilling**: when a query hits its
memory limit, it writes intermediate data to disk and keeps going, still
producing the exact correct result instead of failing.

There are two limits, checked in this order:

1. **Per-query limit** — `query_memory_cap_bytes` (server-wide) or a tenant's
   `memory_cap_bytes`. This is checked first, so a single query can't hog a
   shared server even when the server as a whole has room to spare.
2. **Whole-server limit** — the total memory budget for the process (defaults to
   the machine's memory; override with the `DAFT_MEMORY_LIMIT` environment
   variable).

The operations that spill when they hit a limit are the memory-hungry ones:

- **Grouping and aggregation** — writes the largest in-progress group to disk and
  restores it before producing final results.
- **Sorting** — sorts in chunks, saves them to disk, and merges them back
  streaming.
- **Joins** — splits both sides into matching pieces and processes them one pair
  at a time, so only one piece is in memory at once.

When several queries compete for memory at the same time, the server shares it
fairly — asking the queries using the most to spill first — rather than letting
whoever arrived first take everything.

You control this per-query memory limit centrally, on the server. Clients cannot
raise their own limit.

---

## Query lifecycle

### Results stream as they're ready

You don't wait for the whole query to finish before you see anything — results
arrive in pieces as the server produces them. If your client consumes results
slowly, the server automatically slows down to match instead of racing ahead and
buffering everything. You can bound this buffer with `results_buffer_size` on
`conn.sql(query, results_buffer_size=...)`.

### Cancelling a query

You can stop a running query two ways:

- **Explicitly:** `conn.cancel_query(query_id)`.
- **Automatically:** just disconnect. If your client goes away, the server
  notices, stops the query, and frees the slot.

You can only cancel queries you started.

### Time limits

If `query_timeout_secs` is set (server-wide or per tenant), any query that runs
too long is stopped and the client gets a clear timeout error. It's off by
default.

### Shutting down cleanly

When the server receives a stop signal (`Ctrl-C`, or `SIGTERM` from an
orchestrator), it:

1. Stops accepting new queries (new attempts get a "server is shutting down"
   error).
2. Lets in-flight queries finish, for up to `drain_timeout_secs`.
3. Cancels anything still running past that window and exits.

This makes rolling restarts and Kubernetes deployments safe.

---

## Deployment

### Docker

The repository ships a ready-to-use image build at `deploy/Dockerfile.serve`. It
builds the engine from source and produces a small, non-root runtime image.

```bash
# Build the image (run from the repository root):
docker build -f deploy/Dockerfile.serve -t daft-serve:dev .

# Run it:
docker run -p 9494:9494 -e DAFT_SERVE_TOKEN=secret daft-serve:dev \
    --host 0.0.0.0 --port 9494
```

Anything after the image name is passed straight to the server as command-line
options.

### Kubernetes

A Helm chart lives at `deploy/helm/daft-serve`.

```bash
helm install daft-serve deploy/helm/daft-serve \
    --set image.repository=my-registry/daft-serve \
    --set image.tag=1.0.0 \
    --set auth.existingSecret=daft-serve-token
```

The main settings in `values.yaml`:

| Setting | Purpose |
|---|---|
| `replicas` | How many server copies to run behind one address. Each query runs on one server, so more replicas serve more clients. |
| `service` | Network address and port (default 9494). |
| `auth.existingSecret` / `auth.value` | Where the access token comes from. |
| `server.*` | Query limits: concurrency, queue timeout, max data size, drain timeout. |
| `config.catalogs` | The data catalogs to attach at startup. |
| `extraEnv` | Storage credentials (for example, an access key from a Secret). |
| `resources` | CPU and memory requests/limits (default 2–8 CPU, 4–16 GiB). |
| `probes` | Built-in health checks the cluster uses to know the server is alive. |
| `ingress`, `autoscaling` | Optional external access and automatic scaling on CPU load. |

The chart wires up health checks, mounts your config, injects the token from a
Secret, and gives the server enough shutdown time to drain in-flight queries
before the pod is removed.

**A short checklist before going live**

- [ ] Set an access token — don't run in insecure mode outside development.
- [ ] Give the **server** the storage and catalog credentials (the client
      doesn't need them).
- [ ] Set a per-query memory limit so a large query spills to disk instead of
      getting killed by the container.
- [ ] Prefer a network-reachable catalog so all data reading happens on the
      server.

---

## Configuration reference

Every server setting, whether written in a config file (JSON or YAML) or passed
to `ServeSettings(...)` in Python:

| Setting | Default | Meaning |
|---|---|---|
| `host` | `127.0.0.1` | Network interface to bind. |
| `port` | `9494` | Port to bind (`0` = pick a free one). |
| `token` | — | Access token (comes from the environment, never the file). |
| `allow_insecure_remote` | `false` | Allow a remote bind with no token. |
| `max_concurrent_queries` | `4` | Queries running at once; the rest wait. |
| `queue_timeout_secs` | `60` | How long a waiting query holds on. |
| `max_pset_bytes` | `268435456` (256 MiB) | Largest in-memory chunk sent per query. |
| `disable_plan_payload` | `false` | Accept text queries only. |
| `query_timeout_secs` | `0` (off) | Per-query time limit. |
| `query_memory_cap_bytes` | `0` (off) | Per-query memory ceiling, then spill. |
| `tenants` | none | Named tenants with their own tokens and limits. |
| `catalogs` | none | Data catalogs to attach at startup. |

A catalog entry has a `name` (the alias clients use in text queries) and
`properties` (the connection details — type, address, warehouse, credentials).

> **Tokens are never read from the config file.** The server-wide token comes
> from the `DAFT_SERVE_TOKEN` environment variable (or the one named by
> `--token-env`); each tenant's token comes from `DAFT_SERVE_TOKEN_<NAME>`. This
> keeps secrets out of files that might be checked into version control.

**Relevant environment variables**

| Variable | Effect |
|---|---|
| `DAFT_SERVE_TOKEN` | The server-wide access token. |
| `DAFT_SERVE_TOKEN_<NAME>` | A specific tenant's access token. |
| `DAFT_MEMORY_LIMIT` | Total memory budget for the whole server process. |

---

## Error reference

When something goes wrong, the client receives a specific, named error so you can
handle it precisely. The most common ones:

| Error | Means | What to do |
|---|---|---|
| `Unauthenticated` | Wrong or missing token. | Check the token you passed to `daft.connect`. |
| `VersionMismatch` | Client and server engine versions differ (DataFrame query). | Match versions, or use `conn.sql(...)`. |
| `PlanPayloadDisabled` | Server accepts text queries only. | Use `conn.sql(...)`. |
| `PsetTooLarge` | Sent more in-memory data than `max_pset_bytes` allows. | Raise the limit, or read from a catalog instead. |
| `AtCapacity` | All query slots busy and the wait timed out. | Add slots or replicas, or retry later. |
| `Timeout` | Query ran past its time limit. | Raise the limit or simplify the query. |
| `Draining` | Server is shutting down. | Reconnect to a running server. |
| `Cancelled` | Query was cancelled (by you or a disconnect). | Expected after a cancel. |

---

## Measuring performance

The suite under `benchmarking/serve/` measures Daft Serve on a standard set of
analytical queries over data catalog tables, and guards against slowdowns.

```bash
# Always build the optimized engine first.
make build-release

# A quick run: small data, checks correctness and prints timings.
python -m benchmarking.serve.bench --warehouse /tmp/data-small --scale-factor 0.01

# A real run: more data, more iterations, and save a performance baseline.
python -m benchmarking.serve.bench --warehouse /tmp/data --scale-factor 1 \
    --iterations 5 --record-baseline

# Many clients at once, to measure throughput and latency under load.
python -m benchmarking.serve.concurrency --warehouse /tmp/data-small \
    --clients 1 4 16 --duration-secs 30
```

The runner compares in-process execution against served execution so you can see
exactly what the server adds, checks that every result matches, and **fails if
any query gets more than 15% slower** than a saved baseline — so a performance
regression can't slip through unnoticed.

> **Always build with `make build-release` before measuring.** The plain
> `make build` produces a debug engine that is far slower, so its timings are
> meaningless as real numbers.

Full details, including the containerized environment, are in
[`benchmarking/serve/README.md`](https://github.com/xcaliber-health/Daft/blob/release/v0.7.21-platform/benchmarking/serve/README.md).

---

## Testing

The test suite under `tests/serve/` proves that queries run through a server
produce the *same results* as running locally, and that all the operational
behavior — authentication, multiple tenants, streaming, cancellation, timeouts,
size limits, and clean startup and shutdown — works as documented.

```bash
make test EXTRA_ARGS="tests/serve"
```

The complete list of what's verified is in
[`tests/serve/README.md`](https://github.com/xcaliber-health/Daft/blob/release/v0.7.21-platform/tests/serve/README.md).

---

## Limitations to know about

These are intentional, and worth knowing before you design around Daft Serve:

- **DataFrame queries require matching client and server versions.** The plan
  format is internal. For clients that might be on a different version, use text
  queries (`conn.sql`), which tolerate version differences.
- **Custom function dependencies must be installed on the server.** A user-defined
  function travels to the server as part of the query, but any libraries it
  imports must already be present there.
- **Reading files directly needs both sides to see the same storage.** With
  `read_parquet` / `read_csv`, the client figures out which files to read and the
  server reads them, so both must reach the same paths. Tables served through a
  data catalog have no such requirement — that's the recommended setup.
- **Writes need `daft.connect`.** To make a write run on the server, connect with
  `daft.connect` rather than a standalone `RemoteRunner`.

Daft Serve today runs on a single server per address. Scaling out to multiple
cooperating servers is on the roadmap (see [`docs/serve-roadmap.md`](../serve-roadmap.md)).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Server won't start on a public address | No token set | Set `DAFT_SERVE_TOKEN` (or `--allow-insecure-remote` for local dev). |
| `Unauthenticated` when connecting | Wrong or missing token | Pass the correct `token=` to `daft.connect`. |
| `VersionMismatch` when running a query | Client and server versions differ | Match versions, or switch to `conn.sql(...)`. |
| `PsetTooLarge` | Query sends too much in-memory data | Raise `max_pset_bytes`, or read from a catalog instead of shipping data. |
| `AtCapacity` | All query slots are busy | Add slots (`max_concurrent_queries`), add replicas, or give the tenant its own slots. |
| Query fails partway with a loud error | The server stopped mid-query (often out of memory) | Check server logs; set a `query_memory_cap_bytes` so large queries spill instead of crashing. |
| Query is much slower than expected | Debug build, or the client is reading files itself | Rebuild with `make build-release`; use a catalog so reading happens on the server (check `conn.explain(df)`). |
| A write ran locally instead of on the server | Used a standalone `RemoteRunner` | Connect with `daft.connect`. |

---

## Under the hood

You don't need any of this to use Daft Serve — it's here for the curious and for
anyone debugging network-level behavior.

- **Transport.** Clients and servers talk over a standard network protocol
  (gRPC). Data moves in Arrow's columnar format, so column types — including
  timestamps, decimals, and nested types — survive the trip exactly.
- **One request per query.** A whole query travels in a single request, and the
  results come back as a single stream. There's no back-and-forth chatter and no
  separate upload step — any in-memory data your query needs rides along inside
  the request, and only the parts the query actually uses are sent.
- **Result stream.** Results arrive as a schema, then a series of data batches,
  then a final summary (row and byte counts, and the plan that ran). If that
  final summary never arrives, the client raises an error rather than pretending
  the truncated result is complete.
- **Where the work happens.** For catalog tables, the server does all the
  planning and reading. For direct file reads, the client works out which files
  match and the server reads them — which is why both sides need to see the same
  storage in that one case.
