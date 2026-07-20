# Daft Serve

**Run the Daft query engine as a shared network service.**

Normally, every program that uses Daft loads the engine into its own process and
reads data directly. Daft Serve lets you run the engine *once*, as a long-lived
server, and have many lightweight clients send it queries over the network. The
server does the planning, the data reading, and the heavy computation — then
streams the results back.

You keep writing ordinary Daft code. You just point it at a server first.

```python
import daft

daft.connect("daft://my-server:9494", token="secret")

# From here on, this runs on the server — not on your machine:
daft.read_iceberg(sales).filter(daft.col("amount") > 100).collect()
```

---

## Why you might want this

- **Put the computation next to the data.** The server reads from your data lake
  and does the work there. Your client needs no storage access and no
  credentials — just a network connection.
- **Share one warm engine.** Instead of every script starting its own engine and
  claiming its own memory, many clients share a single, resource-managed server.
- **Everything Daft can do, unchanged.** Filters, joins, aggregations, window
  functions, and custom functions all behave exactly as they do locally.
- **Safe for many teams at once.** Give each team its own access token, query
  slots, and memory limit so no one team can slow down or crowd out another.
- **Never runs out of memory.** When a query gets too big for its memory budget,
  it automatically writes intermediate data to disk and still returns the correct
  result.

---

## Quick start

You need three things: build the engine, start a server, connect a client.

### 1. Build the engine

```bash
make build-release
```

> Use `make build-release`, not `make build`. The plain `build` produces a slow
> debug engine meant only for development.

### 2. Start a server

```bash
export DAFT_SERVE_TOKEN=secret          # the password clients must present
python -m daft.serve --host 127.0.0.1 --port 9494
```

You'll see a line like `serving on grpc://127.0.0.1:9494`. Leave it running.

### 3. Connect and run a query

In another terminal or script:

```python
import daft

# Point this process at the server. Every query now runs there.
conn = daft.connect("daft://localhost:9494", token="secret")

# Ordinary Daft code:
df = daft.read_iceberg(table).filter(daft.col("value") > 100)
df.groupby("region").agg(daft.col("value").sum()).collect()

# Or send a text query, run against the server's data catalogs:
conn.sql("SELECT region, SUM(value) FROM lake.sales GROUP BY region").collect()
```

That's a working setup. The **[User Guide](user-guide.md)** takes it from here:
running a real server, connecting clients, configuring limits, serving multiple
teams, deploying to Docker and Kubernetes, and measuring performance.

---

## Where to go next

| If you want to… | Read |
|---|---|
| Learn Daft Serve step by step | **[User Guide](user-guide.md)** |
| See every configuration option | [User Guide → Configuration reference](user-guide.md#configuration-reference) |
| Deploy to Kubernetes | [User Guide → Deployment](user-guide.md#deployment) |
| Confirm a feature works the same as local | [Feature parity matrix](https://github.com/xcaliber-health/Daft/blob/release/v0.7.21-platform/tests/serve/README.md) |
| Measure performance | [Benchmark suite](https://github.com/xcaliber-health/Daft/blob/release/v0.7.21-platform/benchmarking/serve/README.md) |
