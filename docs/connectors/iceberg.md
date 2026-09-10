# Reading from and Writing to Apache Iceberg

[Apache Iceberg](https://iceberg.apache.org/) is an open-source table format originally developed at Netflix for large-scale analytical tables and datasets. It provides a way to manage and organize data files (like Parquet and ORC) as tables, offering benefits like ACID transactions, time travel, and schema evolution.

Daft currently natively supports:

1. **Distributed Reads:** Daft will fully distribute the I/O of reads over your compute resources (whether Ray or on local multithreading)
2. **Skipping Filtered Data:** Daft uses [`df.where()`][daft.DataFrame.where] filter calls to only read data that matches your predicates
3. **All Catalogs From PyIceberg:** Daft is natively integrated with PyIceberg, and supports all the catalogs that PyIceberg does

A detailed Iceberg roadmap for Daft can be found on [our GitHub issues](https://github.com/Eventual-Inc/Daft/issues/2458). For the overall Daft development plan, see [Daft Roadmap](../roadmap.md).

## Tutorial

### Reading a Table

To read from the Apache Iceberg table format, use the [`daft.read_iceberg`][daft.read_iceberg] function.

We integrate closely with [PyIceberg](https://py.iceberg.apache.org/) (the official Python implementation for Apache Iceberg) and allow the reading of DataFrames easily from PyIceberg's Table objects. The following is an example snippet of loading an example table, but for more information please consult the [PyIceberg Table loading documentation](https://py.iceberg.apache.org/api/#load-a-table).

=== "🐍 Python"

    ```python
    # Access a PyIceberg table as per normal
    from pyiceberg.catalog import load_catalog

    catalog = load_catalog("my_iceberg_catalog")
    table = catalog.load_table("my_namespace.my_table")
    ```

After a table is loaded as the `table` object, reading it into a DataFrame is extremely easy.

=== "🐍 Python"

    ```python
    # Create a DataFrame
    import daft

    df = daft.read_iceberg(table)
    ```

Daft can also read a named Iceberg branch or tag. Branch, tag, and snapshot ID
reads are mutually exclusive.

=== "🐍 Python"

    ```python
    # Create branch/tag references with PyIceberg
    table.refresh()
    snapshot = table.current_snapshot()

    with table.manage_snapshots() as snapshot_manager:
        snapshot_manager.create_branch(snapshot.snapshot_id, "audit")
        snapshot_manager.create_tag(snapshot.snapshot_id, "v1")

    # Read the branch or tag with Daft
    branch_df = daft.read_iceberg(table, branch="audit")
    tag_df = daft.read_iceberg(table, tag="v1")
    ```

Any subsequent filter operations on the Daft `df` DataFrame object will be correctly optimized to take advantage of Iceberg features such as hidden partitioning and file-level statistics for efficient reads.

=== "🐍 Python"

    ```python
    # Filter which takes advantage of partition pruning capabilities of Iceberg
    df = df.where(df["partition_key"] < 1000)
    df.show()
    ```

### Writing to a Table

To write to an Apache Iceberg table, use the [`df.write_iceberg()`][daft.DataFrame.write_iceberg] method.

The following is an example of appending data to an Iceberg table:

=== "🐍 Python"

    ```python
    written_df = df.write_iceberg(table, mode="append")
    written_df.show()
    ```

This call will then return a DataFrame containing the operations that were performed on the Iceberg table, like so:

``` {title="Output"}

╭───────────┬───────┬───────────┬────────────────────────────────╮
│ operation ┆ rows  ┆ file_size ┆ file_name                      │
│ ---       ┆ ---   ┆ ---       ┆ ---                            │
│ Utf8      ┆ Int64 ┆ Int64     ┆ Utf8                           │
╞═══════════╪═══════╪═══════════╪════════════════════════════════╡
│ ADD       ┆ 5     ┆ 707       ┆ 2f1a2bb1-3e64-49da-accd-1074e… │
╰───────────┴───────┴───────────┴────────────────────────────────╯
```

## Checkpointing

Daft supports idempotent writes to Iceberg via the `checkpoint=` parameter on [`df.write_iceberg()`][daft.DataFrame.write_iceberg]. Retries of the same logical commit — after a crash, a transient catalog error, or a deliberate re-invocation — produce the same Iceberg state without duplicate snapshots. See the [Checkpointing user guide](../use-case/checkpointing.md) for concepts; the sections below cover Iceberg-specific behavior.

### Example

The pattern: one `CheckpointStore` paired into both the source (via `CheckpointConfig`) and the sink (via `IdempotentCommit`). The source records which inputs were processed; the sink stamps the resulting Iceberg snapshot with the idempotence key.

=== "🐍 Python"

    ```python
    import daft

    # One store, paired into both the source and the sink.
    store = daft.CheckpointStore("s3://my-bucket/ckpt/")

    df = daft.read_parquet(
        "s3://input/",
        checkpoint=daft.CheckpointConfig(store, on="file_id"),
    )

    # Any map-only operations work here: filter, project, UDF, explode, ...
    df = df.where(df["status"] == "active")

    written = df.write_iceberg(
        table,
        checkpoint=daft.IdempotentCommit(store, idempotence_key="job-2026-05-21-001"),
    )
    ```

A fresh run produces one new Iceberg snapshot tagged with `daft.idempotence-key=job-2026-05-21-001`. Retries with the same `idempotence_key` recognize the prior commit and exit cleanly — no duplicate snapshot, no reprocessing of inputs already handled.

### Inspecting the Marker

Every idempotent commit tags its Iceberg snapshot summary with `daft.idempotence-key`. To verify which logical commit produced the current state of a table:

=== "🐍 Python"

    ```python
    # Inspect via pyiceberg's Table API — `table` is the same pyiceberg.table.Table
    # loaded in the "Reading a Table" section above.
    summary = table.current_snapshot().summary
    print(summary.get("daft.idempotence-key"))
    # → "job-2026-05-21-001"
    ```

For older commits, walk `table.metadata.snapshots` (order isn't guaranteed — sort by `timestamp_ms` if you need chronological order) and check each snapshot's `summary`.

### Constraints

These are constraints specific to the Iceberg connector. Daft-wide constraints (Ray runner, map-only pipelines, single-writer concurrency, etc.) are in the [user guide's Limitations section](../use-case/checkpointing.md#limitations).

- **`mode='append'` only.** `mode='overwrite'` with `checkpoint=` raises `NotImplementedError`.
- **Reserved `daft.idempotence-*` property prefix.** Keys in `snapshot_properties` that start with this prefix raise `ValueError` — Daft uses this namespace internally.

### Idempotence-Key Contract

The user picks the `idempotence_key`. Daft uses it as the marker that identifies a logical commit. The key must be stable across retries and unique across distinct logical commits — both halves matter:

- **Same key, different inputs → silent no-op (data loss).** Daft sees the existing marker and skips the commit. The new data is dropped without an error.
- **Different key on a retry of the same logical commit → duplicate snapshot.** Daft doesn't recognize the prior attempt and lands a second snapshot.

### Recovery

The [user guide's Recovery section](../use-case/checkpointing.md#recovery) walks through the full chronology. The Iceberg-specific steps:

- **Commit-from-store.** When a run crashes after the pipeline finishes but before the snapshot commits, Daft reads the staged file references from the store on rerun and commits them as a single new Iceberg snapshot tagged with `daft.idempotence-key`.

- **Marker recognition.** When a run crashes after the snapshot commits but before the store is marked done — or when the user deliberately re-runs with the same key — Daft walks `table.metadata.snapshots`, finds the marker, marks the store, and exits. No second snapshot. Returned DataFrame is empty.

**Orphan files on crash.** A worker that crashed mid-task may leave parquet files unreferenced by any snapshot. Daft doesn't clean these up as part of the write — reclaim them with [`remove_orphan_files`](#table-maintenance) once they are older than any write that could still be in flight.

### Iceberg-Specific Notes

- **In-memory history walk.** Marker recognition scans `table.metadata.snapshots` directly — fast even on tables with many snapshots, since the metadata is already in memory after refresh.

- **`commit.manifest-merge.enabled` is honored.** If the table is configured for manifest merging, Daft uses `merge_append`; otherwise `fast_append`. Same behavior as a non-checkpoint write.

- **Partitioned tables include a `partitioning` struct column in the result.** Whether `checkpoint=` is set or not — callers don't see schema drift when they toggle the flag.

- **Transient retry on `CommitFailedException`.** Daft retries up to twice on this exception (concurrent-writer conflicts, lock contention). Other exceptions — REST 5xx, network errors, auth — propagate immediately. Wrap the call in your own retry policy if you need broader coverage.

## Table Maintenance

An Iceberg table accumulates small files, manifests, and history as it is written. Daft maintains a table through five operations on a catalog table handle, each following the semantics of Apache Iceberg's own maintenance actions so that a table maintained by Daft looks the same to every reader as one maintained by Spark.

```python
import daft
from daft.catalog import Table

table = Table.from_iceberg(pyiceberg_table)

table.rewrite_data_files("binpack", where="region = 'us'")
table.rewrite_position_delete_files()
table.rewrite_manifests()
table.expire_snapshots(older_than=cutoff, retain_last=100)
table.remove_orphan_files(older_than=three_days_ago, dry_run=True)
```

Every operation (`rewrite_data_files`, `rewrite_position_delete_files`, `rewrite_manifests`, `expire_snapshots`, `remove_orphan_files`) retries its commit on conflict with exponential backoff bounded by the table's `commit.retry.*` properties, refuses to run when `gc.enabled` is `false` where it would delete files, and returns a result dataclass with the counts it changed.

### `rewrite_data_files`

Reads the data files that match `where`, groups them by partition, and rewrites each group into files close to the target size, committing a `replace` snapshot that swaps the inputs for the outputs. Rows are never changed; only their layout is.

| Strategy  | What it does                                                                                       |
|-----------|----------------------------------------------------------------------------------------------------|
| `binpack` | Merges small files and splits oversized ones. The default.                                          |
| `sort`    | Also orders each output by `sort_order`, recording the table's matching sort order id on the files. |
| `zorder`  | Also clusters each output along an interleaved-bits curve over `zorder_by`.                         |

Which files are selected follows Iceberg's size-based planner: a file is a candidate when it is smaller than `min-file-size-bytes` (75% of target), larger than `max-file-size-bytes` (180% of target), carries at least `delete-file-threshold` delete files, or has at least `delete-ratio-threshold` of its rows deleted. Candidates are packed into groups of at most `max-file-group-size-bytes`, and a group is rewritten when it holds at least `min-input-files` files, more than a target's worth of content, or a file selected for its deletes. `rewrite-all` rewrites everything.

| Option                                | Default                     | Meaning                                                                                                                                       |
|---------------------------------------|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| `target-file-size-bytes`              | table `write.target-file-size-bytes`, else 512 MiB | Size each output aims at.                                                                                              |
| `min-input-files`                     | 5                           | Files a group needs before it is worth rewriting on count alone.                                                                              |
| `max-file-group-size-bytes`           | 100 GiB                     | Largest working set one group may hold.                                                                                                       |
| `max-files-to-rewrite`                | unset                       | Cap on the files one call rewrites; the group straddling it is cut rather than dropped.                                                       |
| `rewrite-job-order`                   | `none`                      | Order groups are committed in: `none`, `bytes-asc`, `bytes-desc`, `files-asc`, `files-desc`.                                                   |
| `partial-progress.enabled`            | `false`                     | Commit batches of groups as they finish rather than the whole plan at once.                                                                   |
| `partial-progress.max-commits`        | 10                          | Most commits one call makes; groups are cut into that many batches by plan position.                                                          |
| `partial-progress.max-failed-commits` | `max-commits`               | Failed batches tolerated before the call raises.                                                                                              |
| `max-concurrent-file-group-rewrites`  | 5                           | Groups rewritten at once, on any runner; the single-node engine shares one memory pool among them.                                              |
| `use-starting-sequence-number`        | `true`                      | Stamp outputs with the plan snapshot's sequence number so existing row-level deletes keep applying.                                           |
| `remove-dangling-deletes`             | `false`                     | After the rewrite, drop delete files that no live data file in their partition can be covered by, in a snapshot of their own.                   |
| `conflict-isolation`                  | `snapshot`                  | `snapshot` refuses only when an input file was removed or a new row-level delete covers one; `serializable` also refuses any file added to a touched partition. |
| `output-spec-id`                      | current spec                | Partition spec the outputs are written under.                                                                                                 |
| `rewrite-id`                          | derived from the plan       | Identity for an idempotent replay: a call that finds its own snapshots returns their result without rewriting.                                |
| `compression-factor`                  | measured                    | Expansion of sorted or z-ordered rows from disk into memory, seeding how the writer rolls files.                                              |
| `shuffle-partitions-per-file`         | 1                           | Sort and z-order only: ordered output partitions per target file.                                                                             |
| `max-output-size`, `var-length-contribution` | all bytes, 8         | Z-order only: cap on the key length, and bytes each string or binary column contributes.                                                      |

What the commit guarantees:

- **Existing deletes keep applying.** Outputs carry the plan snapshot's sequence number, so a position delete written before the rewrite still covers the rows it named, and a delete that no live data file can be covered by is dropped in the same commit, as any Iceberg commit drops it.
- **Concurrent writers are tolerated, not trusted.** A commit is refused when one of its inputs is no longer live or when another writer landed a row-level delete on a file being replaced; committing over such a delete would bring the removed rows back. Appends to the same partition coexist with the rewrite under the default isolation.
- **Nothing is left behind.** A batch whose commit is refused or exhausts its retries has its outputs deleted before the call goes on. Without partial progress any failure removes every output written so far before raising. Only a commit that fails for a reason other than a conflict keeps its files, because it may in fact have landed; orphan cleanup reclaims them.
- **Replays are recognized.** Every snapshot carries the rewrite's identity, so the same call again returns the committed result rather than repeating the work.

The returned `RewriteResult` reports `rewritten_files`, `added_files`, `bytes_rewritten`, `bytes_added`, `removed_delete_files`, `failed_groups`, `failed_data_files`, `commits`, `snapshot_ids`, and `rewrite_id`. Position and equality deletes are both applied while the inputs are read, following the specification's scope rules (an equality delete covers strictly older data files in its partition, or every partition when written under an unpartitioned spec), so a rewritten file carries no rows a delete had removed. Equality deletes count toward `delete-file-threshold` like position deletes.

### `rewrite_position_delete_files`

Packs the live position delete files of each partition into files of `write.delete.target-file-size-bytes` (64 MiB when absent) under the same size thresholds and options as `rewrite_data_files`: `target-file-size-bytes`, `min-file-size-bytes`, `max-file-size-bytes`, `min-input-files`, `rewrite-all`, `max-file-group-size-bytes`, `rewrite-job-order`, `partial-progress.enabled`, `partial-progress.max-commits`, `max-concurrent-file-group-rewrites` and `rewrite-id`. Rows naming a data file no longer live are dropped, the rest are written sorted by file and position, one packed file per data file (`write.delete.granularity=file`, the default) or per partition (`partition`), and the packed files replace the old ones at the highest sequence number of the files they replace, so the same rows stay deleted. A merge-on-read table that receives row-level deletes every commit needs this beside the data rewrite, which only removes the deletes of the files it rewrites.

```python
result = table.rewrite_position_delete_files()
result.rewritten_delete_files, result.added_delete_files
```

### `rewrite_manifests`

Repacks the current snapshot's data manifests and delete manifests for one partition spec, each kind on its own, into manifests close to `manifest-target-size-bytes` (table `commit.manifest.target-size-bytes`, else 8 MiB), clustered by partition or by the fields named in `sort-by`, and commits a `replace` snapshot that keeps every data file and delete file exactly as it was. A kind whose manifests are already as many as the target size calls for, none larger than the roll size and none without a live entry, is left alone. The result reports `rewritten_manifests_count`, `added_manifests_count`, `bytes_rewritten`, `bytes_added`, `rewrite_id`, and `snapshot_id`.

### `expire_snapshots`

Removes history following Iceberg's retention rules. Every branch keeps its head and walks its ancestry, keeping each snapshot while fewer than `retain_last` have been kept or the snapshot is at or after the `older_than` cutoff, and stopping at the first that is neither; a branch's own `max-snapshot-age-ms` and `min-snapshots-to-keep` take precedence. A tag keeps its snapshot until the tag ages past its `max-ref-age-ms`. A snapshot no branch or tag reaches is kept only while it is newer than the cutoff. `older_than` defaults to now less the table's `history.expire.max-snapshot-age-ms` (5 days), so `retain_last` on its own only floors what age would expire. Snapshots named in `snapshot_ids` expire regardless.

Files that only the expired snapshots referenced are deleted when `clean_expired_files` is set, computed as the difference between the files reachable before and after the commit; `clean_expired_metadata` also deletes superseded table-metadata files. The result counts the data, delete, manifest, manifest-list, statistics, and metadata files removed.

### `remove_orphan_files`

Lists the table location and deletes every file that no snapshot, manifest, manifest list, statistics file, or metadata-log entry references and that is older than `older_than`. The cutoff must be at least 24 hours in the past unless `options={"allow-recent": True}`, because a file a running write has not yet committed looks exactly like an orphan. `location` narrows the listing to a subpath of the table, `dry_run` reports without deleting, and `prefix_mismatch_mode` decides what happens to files whose location has an unexpected scheme or authority (`error`, `delete`, or `ignore`). The result reports `orphan_files_count` and `deleted_files_count`.

### Memory, spilling and the two runners

Every operation streams. A bin-pack rewrite reads each file group through the scan and writes through a size-rolling writer. A sort or z-order holds the rows it orders under the runtime's memory budget, which is the machine or cgroup limit unless `DAFT_MEMORY_LIMIT` pins it; under pressure the sort spills sorted runs to `spill_dirs` and merges them back. The maintenance operations set no execution configuration of their own, so the runtime uses every core and all the memory it is given.

The budget bounds the rows a sort buffers; the rest of a process's peak is the scan tasks in flight (one decoded input file each, up to `scantask_max_parallel`), the chunk being sorted for a spill (about twice the budget while it is concatenated and sorted), and the writer's open row group (`parquet_target_row_group_size`). Size a container from that sum rather than by lowering the runtime's defaults: a pod limit of about four times `DAFT_MEMORY_LIMIT` plus the scan parallelism times the largest decoded input leaves the defaults intact, and `spill_dirs` should point at a volume. Only where a pod cannot be sized that way does lowering `scantask_max_parallel` or the row-group size trade throughput for a lower peak.

On the distributed runner each file group is one plan and groups run one at a time; within a plan the scan, the range shuffle and the writes are spread across workers, and per-task memory is bounded by the object store, which spills on its own. A clustering rewrite sorts over every input partition and then joins adjacent sorted ranges into as many partitions as the planner expects output files, so the files are even and a range query opens few of them.

## Type System

| Iceberg                             | Daft                                                                                         |
|-------------------------------------|----------------------------------------------------------------------------------------------|
| `BOOLEAN`                           | [`daft.DataType.bool()`][daft.datatype.DataType.bool]                                        |
| `INT`                               | [`daft.DataType.int32()`][daft.datatype.DataType.int32]                                      |
| `LONG`                              | [`daft.DataType.int64()`][daft.datatype.DataType.int64]                                      |
| `FLOAT`                             | [`daft.DataType.float32()`][daft.datatype.DataType.float32]                                  |
| `DOUBLE`                            | [`daft.DataType.float64()`][daft.datatype.DataType.float64]                                  |
| `DECIMAL(precision, scale)`         | [`daft.DataType.decimal128(precision, scale)`][daft.datatype.DataType.decimal128]            |
| `DATE`                              | [`daft.DataType.date()`][daft.datatype.DataType.date]                                        |
| `TIME`                              | [`daft.DataType.int64()`][daft.datatype.DataType.int64]                                      |
| `TIMESTAMP`                         | [`daft.DataType.timestamp(timeunit="us", timezone=None)`][daft.datatype.DataType.timestamp]  |
| `TIMESTAMPZ`                        | [`daft.DataType.timestamp(timeunit="us", timezone="UTC")`][daft.datatype.DataType.timestamp] |
| `STRING`                            | [`daft.DataType.string()`][daft.datatype.DataType.string]                                    |
| `UUID`                              | [`daft.DataType.binary()`][daft.datatype.DataType.binary]                                    |
| `FIXED(size)`                       | [`daft.DataType.fixed_size_binary(size)`][daft.datatype.DataType.fixed_size_binary]          |
| `BINARY`                            | [`daft.DataType.binary()`][daft.datatype.DataType.binary]                                    |
| `STRUCT<[field_name: field_type,]>` | [`daft.DataType.struct(fields)`][daft.datatype.DataType.struct]                              |
| `LIST<element_type>`                | [`daft.DataType.list(element_type)`][daft.datatype.DataType.list]                            |
| `MAP<key_type, value_type>`         | [`daft.DataType.map(key_type, value_type)`][daft.datatype.DataType.map]                      |

See also [Iceberg Schemas and Data Types](https://iceberg.apache.org/spec/#schemas-and-data-types).

## Reference

Daft has high-level [Session](../api/sessions.md) and [Catalog](../api/catalogs_tables.md) APIs
to read and write Iceberg tables; however it is the [`daft.read_iceberg`][daft.read_iceberg] and
[`df.write_iceberg`][daft.DataFrame.write_iceberg] API which is ultimately the entry-point to Iceberg reads and
writes respectively. This section gives a short reference on those APIs and how
they relate to both DataFrames and Iceberg.

Daft's DataFrames are an abstraction over relational algebra operators like
filter, project, and join. DataFrames start with a *data source* and are built
upwards, via composition with additional operators, to form a tree with sources
as the leaves. We typically call these leaves *tables* or *sources* and their
algebraic operator is called a *scan*.

### [`read_iceberg`][daft.read_iceberg]

Daft's [`daft.read_iceberg`][daft.read_iceberg] method creates a DataFrame from the given PyIceberg
table. It produces rows by traversing the table's metadata tree to locate all
the data files for the given snapshot which is handled by our
`IcebergScanOperator`.

Daft's `IcebergScanOperator` initializes itself by fetching the latest schema,
or the schema of the given snapshot, along with setting up the partition key
metadata. The scan operator's primary method, `to_scan_tasks`, accepts pushdowns
(projections, predicates, partition filters) and returns an iterator of
`ScanTasks`. Each `ScanTask` object holds a data file, optional delete files,
and the associated pushdowns. Finally, we read each data file's parquet to
produce a stream of record batches which later operators consume and transform.

### [`write_iceberg`][daft.DataFrame.write_iceberg]

Daft's [`write_iceberg`][daft.DataFrame.write_iceberg] method writes the DataFrame's contents to the given PyIceberg table.
It works by creating a special *sink* operator which consumes all inputs and writes
data files to the table's location.

Daft's sink operator will apply the Iceberg partition transform and distribute
records to an appropriate data file writer. Each writer is responsible for
actually writing the parquet to storage and keeping track of metadata like total
bytes written. Once the sink has exhausted its input, it will close all open
writers.

Finally, we update the Iceberg table's metadata to include these new data files,
and use a transaction to update the latest metadata pointer.

The writer reads the table's `write.*` properties: `write.target-file-size-bytes`, `write.parquet.compression-codec` (zstd when absent) and `compression-level`, `row-group-size-bytes`, `page-size-bytes`, `page-row-limit` (20 000 rows when absent), `dict-size-bytes` (2 MiB when absent), `dict-encoding-enabled.column.<name>`, the bloom-filter settings, `write.data.path`, and `write.object-storage.enabled` with `write.object-storage.partitioned-paths`, which places each file under a hashed prefix so an object store spreads a hot table's requests. A column whose dictionary would cost more than the values it indexes, such as a unique key, is written plain unless the table says otherwise. Packed delete files follow `write.delete.parquet.*` where set, else the data settings. A `write.format.default` other than parquet is refused.

### Iceberg Architecture

!!! note "Note"

    Please see the
    [Iceberg Table Specification](https://iceberg.apache.org/spec/) for full
    details.

Iceberg tables are a tree where metadata files are the inner nodes, and data
files are the leaves. The data files are typically parquet, but are not
*necessarily* so. To write data, the new data files are inserted into the tree
by creating the necessary inner nodes (metadata files) and re-rooting the
parent. To read data, we choose a root (table version) and use the metadata
files to collect all relevant data files (their children).

The *data layer* is composed of data files and delete files; where as the
*metadata layer* is composed of manifest files, manifest lists, and metadata
files.

#### Manifest Files

Manifest files (avro) keep track of data files, delete files, and statistics.
These lists track the leaves of the iceberg tree. While all manifest files use
the same schema, a manifest file contains either exclusively data files or
exclusively delete files.

#### Manifest List

Manifests lists (snapshots) contain all manifest file locations along with their
partitions, and partition columns upper and lower bounds. Each entry in the
manifest list has the form,

| Field             | Description                                          |
|-------------------|------------------------------------------------------|
| `manifest_path`   | location                                             |
| `manifest_length` | file length in bytes                                 |
| `content`         | flag where `0=data` and `1=deletes`                  |
| `partitions`      | array of field summaries e.g. nullable, upper, lower |

These are not all of the fields, but gives us an idea of what a manifest
list looks like.

#### Metadata Files

Metadata files store the table schema, partition information, and a list of all
snapshots including which one is the current. Each time an Iceberg table is
modified, a new metadata file is created; this is what is meant earlier by
"re-rooting" the tree. The *catalog* is responsible for atomically updating the
current metadata file.

#### Puffin Files

Puffin files store arbitrary metadata as blobs along with the necessary metadata
to use these blobs.

### Table Writes

Iceberg can efficiently insert (append) and update rows. To insert data, the new
data files (parquet) are written to object storage along with a new manifest
file. Then a new manifest list, along with all existing manifest files, is added
to the new metadata file. The catalog then marks this new metadata file as the
current one.

An upsert / merge into query is a bit more complicated because its action is
conditional. If a record already exists, then it is updated. Otherwise, a new
record is inserted. This write operation is accomplished by reading all matched
records into memory, then updating each match by either copying with updates
applied (COW) or deleting (MOR) then including the updated in the insert. All
records which were not matched (did not exist) are inserted like a normal
insert.

### Table Reads

Iceberg reads begin by fetching the latest metadata file to then locate the
"current snapshot id". The current snapshot is a manifest list which has the
relevant manifest files which ultimately gives us a list of all data files
pertaining to the query. For each metadata layer, we can leverage the statistics
to prune both manifest files and data files.

### Other

#### COW vs. MOR

When data is modified or deleted, we can either rewrite the relevant portion
(copy-on-write) or save the modifications for later reconciliation
(merge-on-read). Copy-on-write optimizes for read performance; however,
modifications are more expensive. Merge-on-read optimizes for write performance;
however, reads are more expensive.

#### Delete File

Delete files are used for the merge-on-read strategy in which tables updates are
written to a "delete file" which is applied or "merged" with the data files
*while* reading. A delete file can contain either *positional* or *equality*
deletes. Positional deletes denote which rows (filepath+row) have been deleted,
whereas equality deletes have an equality condition i.e. `WHERE x = 100` to
filter rows.

#### Partitioning

When a table is partitioned on some field, Iceberg will write separate data files
for each record, grouped by the partitioned field's value. That is, each data file
will contain records for a single partition. This enables efficient scanning of
a partition because all other data files can be ignored. You may partition by a
column's value (identity) or use a *partition transform* to derive a partition value.

##### [Apache Iceberg Partition Transforms](https://iceberg.apache.org/spec/#partition-transforms)

| Transform     | Description                 |
|---------------|-----------------------------|
| `identity`    | Column value unmodified.    |
| `bucket(n)`   | Hash of value, mod `n`.     |
| `truncate(w)` | Truncated value, width `w`. |
| `year`        | Timestamp year value.       |
| `month`       | Timestamp month value.      |
| `day`         | Timestamp day value.        |
| `hour`        | Timestamp hour value.       |

## FAQs

1. **How does Daft read Iceberg tables?**

    *Daft reads Iceberg tables by reading a snapshot's data files into its arrow-based record batches. For more detail, please see the [`read_iceberg`](#read_iceberg) reference.*

2. **How does Daft write Iceberg tables?**

    *Daft writes Iceberg tables by writing the new, possibly partitioned, data files to storage then atomically committing an updated Iceberg metadata file. For more detail, please see [`write_iceberg`](#write_iceberg)*.

3. **How do Daft's data types compare to Iceberg's data types?**

    *The type systems are quite similar because they are both based around Apache Arrow's types. Please see our comprehensive type comparison table.*

4. **Does Daft support Iceberg REST catalog implementations?**

    *Yes! Daft uses PyIceberg to interface with Iceberg catalogs which has extensive Iceberg REST catalog support.*

5. **How does Daft handle positional deletes vs equality deletes?**

    *Table reads apply positional deletes; equality deletes are applied by the maintenance operations (`rewrite_data_files`), and support for them in reads is on the roadmap.*

6. **Can Daft leverage Iceberg's metadata for predicate pushdown?**

    *Yes, uses min/max statistics from manifest files for partition pruning.*

7. **Does Daft support time travel queries?**

    *Daft supports reading by snapshot id, and snapshot slices are on the roadmap*.

8. **Which complex data types does Daft support in Iceberg tables?**

    *Daft supports arrays, structs, and maps.*

9. **How does Daft handle schema evolution?**

    *Daft currently does not expose any data definition operators beyond [create_table][daft.session.Session.create_table].*

10. **Does Daft support reading table metadata like snapshot information?**

    *Daft does not have native APIs for this, and we recommend using PyIceberg for reading metadata.*

11. **How does Daft handle Iceberg's hidden partitioning?**

    *Transparently leverages partition information without user specification.*

12. **Can Daft read and write partition transforms like truncate, bucket, or hour?**

    *Daft can read and write all Iceberg partition transform types: identity, bucket, date transforms (year/month/day), and truncate.*

13. **How does Daft optimize queries against partitioned data?**

    *Daft will shuffle partitioned data when possible and based upon your execution environment.*

14. **Which writes operations does Daft support?**

    *Daft supports basic overwrite and append, it does not support upserts like copy-on-write updates.*
