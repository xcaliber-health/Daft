use std::{collections::HashMap, sync::Arc};

use common_error::DaftResult;
use daft_csv::{CsvConvertOptions, CsvParseOptions, CsvReadOptions};
use daft_dsl::{AggExpr, Expr};
use daft_io::{GetRange, IOStatsRef};
use daft_json::{JsonConvertOptions, JsonParseOptions, JsonReadOptions};
use daft_parquet::read::{ParquetReadOptions, ParquetSchemaInferenceOptions};
use daft_recordbatch::RecordBatch;
use daft_scan::{
    ChunkSpec, CsvSourceConfig, FileFormatConfig, JsonSourceConfig, ParquetSourceConfig, ScanTask,
    SourceConfig, TextSourceConfig,
};
use daft_text::{TextConvertOptions, TextReadOptions};
use daft_warc::WarcConvertOptions;
use futures::stream::BoxStream;

type SkippedCorruptFilesCollector = Option<Arc<std::sync::Mutex<Vec<(String, String, bool)>>>>;

/// Dispatches a ScanTask to the appropriate reader based on its SourceConfig,
/// returning a stream of RecordBatches.
///
/// Post-processing (schema conformance, partition fill) is handled by the caller.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn read_scan_task(
    scan_task: &Arc<ScanTask>,
    url: &str,
    file_column_names: Option<Vec<String>>,
    io_client: Arc<daft_io::IOClient>,
    io_stats: IOStatsRef,
    delete_map: Option<Arc<HashMap<String, Vec<i64>>>>,
    maintain_order: bool,
    chunk_size: usize,
    skipped_corrupt_files: SkippedCorruptFilesCollector,
) -> DaftResult<BoxStream<'static, DaftResult<RecordBatch>>> {
    match scan_task.source_config.as_ref() {
        SourceConfig::File(ffc) => match ffc {
            FileFormatConfig::Parquet(cfg) => {
                read_parquet(
                    scan_task,
                    cfg,
                    url,
                    file_column_names,
                    io_client,
                    io_stats,
                    delete_map,
                    maintain_order,
                    chunk_size,
                    skipped_corrupt_files,
                )
                .await
            }
            FileFormatConfig::Csv(cfg) => {
                read_csv(
                    scan_task,
                    cfg,
                    url,
                    file_column_names,
                    io_client,
                    io_stats,
                    chunk_size,
                    skipped_corrupt_files,
                )
                .await
            }
            FileFormatConfig::Json(cfg) => {
                read_json(
                    scan_task,
                    cfg,
                    url,
                    file_column_names,
                    io_client,
                    io_stats,
                    chunk_size,
                )
                .await
            }
            FileFormatConfig::Warc(_) => read_warc(scan_task, url, io_client, io_stats).await,
            FileFormatConfig::Text(cfg) => {
                read_text(scan_task, cfg, url, io_client, io_stats, chunk_size).await
            }
        },
        #[cfg(feature = "python")]
        SourceConfig::Database(cfg) => Ok(read_database(scan_task, cfg)),
        #[cfg(feature = "python")]
        SourceConfig::PythonFunction { .. } => Ok(read_python_function(scan_task)),
    }
}

#[allow(clippy::too_many_arguments)]
async fn read_parquet(
    scan_task: &Arc<ScanTask>,
    cfg: &ParquetSourceConfig,
    url: &str,
    file_column_names: Option<Vec<String>>,
    io_client: Arc<daft_io::IOClient>,
    io_stats: IOStatsRef,
    delete_map: Option<Arc<HashMap<String, Vec<i64>>>>,
    // Unused: the arrow-rs reader is always file-ordered within a single file
    // (see `build_rg_stream` in daft-parquet's reader). `maintain_order=false`
    // at the scan layer reorders only BETWEEN scan tasks, which is handled by
    // the caller, not here.
    _maintain_order: bool,
    chunk_size: usize,
    skipped_corrupt_files: SkippedCorruptFilesCollector,
) -> DaftResult<BoxStream<'static, DaftResult<RecordBatch>>> {
    let source = scan_task.sources.first().unwrap();

    if let Some(aggregation) = &scan_task.pushdowns.aggregation
        && let Expr::Agg(AggExpr::Count(_, _)) = aggregation.as_ref()
    {
        return count_pushdown_stream(
            url,
            io_client,
            io_stats,
            cfg.field_id_mapping.clone(),
            aggregation,
        )
        .await;
    }

    let row_groups = match source.get_chunk_spec() {
        Some(ChunkSpec::Parquet(rgs)) => Some(rgs.clone()),
        _ => None,
    };
    let opts = ParquetReadOptions {
        columns: file_column_names,
        num_rows: scan_task.pushdowns.limit,
        row_groups,
        predicate: scan_task.pushdowns.filters.clone(),
        schema_infer: ParquetSchemaInferenceOptions::new(Some(cfg.coerce_int96_timestamp_unit)),
        field_id_mapping: cfg.field_id_mapping.clone(),
        delete_rows: delete_map.as_ref().and_then(|m| m.get(url).cloned()),
        batch_size: cfg.chunk_size.or(Some(chunk_size)),
        metadata: scan_task
            .sources
            .first()
            .and_then(|s| s.get_parquet_metadata().cloned()),
        ignore_corrupt_files: cfg.ignore_corrupt_files,
        skipped_corrupt_files: skipped_corrupt_files.clone(),
        ..Default::default()
    };
    // Box::pin: setup future is large (~20KB) due to many tuning args.
    Box::pin(daft_parquet::read::read_parquet(
        url,
        io_client,
        Some(io_stats),
        opts,
    ))
    .await
}

async fn count_pushdown_stream(
    url: &str,
    io_client: Arc<daft_io::IOClient>,
    io_stats: IOStatsRef,
    field_id_mapping: Option<Arc<std::collections::BTreeMap<i32, daft_core::prelude::Field>>>,
    aggregation: &daft_dsl::ExprRef,
) -> DaftResult<BoxStream<'static, DaftResult<RecordBatch>>> {
    daft_parquet::read::stream_parquet_count_pushdown(
        url,
        io_client,
        Some(io_stats),
        field_id_mapping,
        aggregation,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn read_csv(
    scan_task: &ScanTask,
    cfg: &CsvSourceConfig,
    url: &str,
    file_column_names: Option<Vec<String>>,
    io_client: Arc<daft_io::IOClient>,
    io_stats: IOStatsRef,
    chunk_size: usize,
    skipped_corrupt_files: SkippedCorruptFilesCollector,
) -> DaftResult<BoxStream<'static, DaftResult<RecordBatch>>> {
    let schema_of_file = scan_task.schema.clone();
    let col_names = if !cfg.has_headers {
        Some(schema_of_file.field_names().collect::<Vec<_>>())
    } else {
        None
    };
    let convert_options = CsvConvertOptions::new_internal(
        scan_task.pushdowns.limit,
        file_column_names
            .as_ref()
            .map(|cols| cols.iter().map(|col| (*col).clone()).collect()),
        col_names
            .as_ref()
            .map(|cols| cols.iter().map(|col| (*col).to_string()).collect()),
        Some(schema_of_file),
        scan_task.pushdowns.filters.clone(),
    );
    let parse_options = CsvParseOptions::new_with_defaults(
        cfg.has_headers,
        cfg.delimiter,
        cfg.double_quote,
        cfg.quote,
        cfg.allow_variable_columns,
        cfg.escape_char,
        cfg.comment,
    )?;
    let csv_chunk_size = cfg.chunk_size.or(Some(chunk_size));
    let read_options = CsvReadOptions::new_internal(cfg.buffer_size, csv_chunk_size);
    daft_csv::stream_csv(
        url.to_string(),
        Some(convert_options),
        Some(parse_options),
        Some(read_options),
        io_client,
        Some(io_stats),
        None,
        cfg.ignore_corrupt_files,
        skipped_corrupt_files,
    )
    .await
}

async fn read_json(
    scan_task: &ScanTask,
    cfg: &JsonSourceConfig,
    url: &str,
    file_column_names: Option<Vec<String>>,
    io_client: Arc<daft_io::IOClient>,
    io_stats: IOStatsRef,
    chunk_size: usize,
) -> DaftResult<BoxStream<'static, DaftResult<RecordBatch>>> {
    let source = scan_task.sources.first().unwrap();
    let schema_of_file = scan_task.schema.clone();
    let convert_options = JsonConvertOptions::new_internal(
        scan_task.pushdowns.limit,
        file_column_names
            .as_ref()
            .map(|cols| cols.iter().map(|col| (*col).clone()).collect()),
        Some(schema_of_file),
        scan_task.pushdowns.filters.clone(),
    );
    let parse_options = JsonParseOptions::new_internal(cfg.skip_empty_files);
    let json_chunk_size = cfg.chunk_size.or(Some(chunk_size));
    let read_options = JsonReadOptions::new_internal(cfg.buffer_size, json_chunk_size);

    let range = source.get_chunk_spec().and_then(|spec| match spec {
        daft_scan::ChunkSpec::Bytes { start, end } => Some(GetRange::Bounded(*start..*end)),
        _ => None,
    });
    daft_json::read::stream_json(
        url.to_string(),
        Some(convert_options),
        Some(parse_options),
        Some(read_options),
        io_client,
        Some(io_stats),
        None,
        range,
    )
    .await
}

async fn read_warc(
    scan_task: &ScanTask,
    url: &str,
    io_client: Arc<daft_io::IOClient>,
    io_stats: IOStatsRef,
) -> DaftResult<BoxStream<'static, DaftResult<RecordBatch>>> {
    let convert_options = WarcConvertOptions {
        limit: scan_task.pushdowns.limit,
        include_columns: None,
        schema: scan_task.schema.clone(),
        predicate: scan_task.pushdowns.filters.clone(),
    };
    daft_warc::stream_warc(url, io_client, io_stats, convert_options, None).await
}

async fn read_text(
    scan_task: &ScanTask,
    cfg: &TextSourceConfig,
    url: &str,
    io_client: Arc<daft_io::IOClient>,
    io_stats: IOStatsRef,
    chunk_size: usize,
) -> DaftResult<BoxStream<'static, DaftResult<RecordBatch>>> {
    let schema_of_file = scan_task.schema.clone();
    let convert_options = TextConvertOptions::new(
        &cfg.encoding,
        cfg.skip_blank_lines,
        cfg.whole_text,
        Some(schema_of_file),
        scan_task.pushdowns.limit,
    );
    let text_chunk_size = cfg.chunk_size.or(Some(chunk_size));
    let read_options = TextReadOptions::new(cfg.buffer_size, text_chunk_size);
    daft_text::read::stream_text(
        url.to_string(),
        convert_options,
        read_options,
        io_client,
        Some(io_stats),
    )
    .await
}

/// Batches a blocking source may read ahead of its reader. Enough that a source
/// and its reader rarely wait on each other batch by batch: handing over one at a
/// time cost about 3 µs a batch more, measured on small batches.
#[cfg(any(feature = "python", test))]
const BATCHES_AHEAD: usize = 4;

/// Reads a source whose reading blocks, such as a call into Python, on a thread
/// of the blocking pool rather than on a worker of the runtime it is read from.
///
/// A source waiting for its next batch would otherwise hold the worker it is
/// polled on. With every worker held, no other read moves, nor the network
/// traffic a waiting source may itself be waiting for. `open` and each batch it
/// yields run on that thread, and the source reads at most [`BATCHES_AHEAD`]
/// batches ahead of its reader. Once the reader is dropped, reading stops at the
/// source's next batch. A panic while reading
/// ends the stream with an error, never as if the source had ended.
///
/// Must be called within a Tokio runtime, whose blocking pool is used.
#[cfg(any(feature = "python", test))]
fn read_on_blocking_thread<I>(
    open: impl FnOnce() -> DaftResult<I> + Send + 'static,
) -> BoxStream<'static, DaftResult<RecordBatch>>
where
    I: Iterator<Item = DaftResult<RecordBatch>>,
{
    use futures::StreamExt;

    let (tx, rx) = tokio::sync::mpsc::channel(BATCHES_AHEAD);
    let reading = tokio::task::spawn_blocking(move || {
        let batches = match open() {
            Ok(batches) => batches,
            Err(e) => {
                // The reader may be gone already, so the error has no one to reach.
                let _ = tx.blocking_send(Err(e));
                return;
            }
        };
        for batch in batches {
            if tx.blocking_send(batch).is_err() {
                return;
            }
        }
    });
    // Once the batches end, the thread has returned or panicked: a panic is the
    // stream's last item.
    let ending = futures::stream::once(async move {
        reading.await.err().map(|panicked| {
            Err(common_error::DaftError::External(
                format!("reading a source failed: {panicked}").into(),
            ))
        })
    })
    .filter_map(futures::future::ready);
    Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx).chain(ending))
}

#[cfg(feature = "python")]
fn read_database(
    scan_task: &Arc<ScanTask>,
    cfg: &daft_scan::DatabaseSourceConfig,
) -> BoxStream<'static, DaftResult<RecordBatch>> {
    use pyo3::Python;
    use snafu::ResultExt;

    use crate::PyIOSnafu;

    let scan_task = scan_task.clone();
    let cfg = cfg.clone();
    read_on_blocking_thread(move || {
        let predicate = scan_task
            .pushdowns
            .filters
            .as_ref()
            .map(|p| (*p.as_ref()).clone().into());
        let table: RecordBatch = Python::attach(|py| {
            daft_micropartition::python::read_sql_into_py_table(
                py,
                &cfg.sql,
                &cfg.conn,
                predicate,
                scan_task.schema.clone().into(),
                scan_task
                    .pushdowns
                    .columns
                    .as_ref()
                    .map(|cols| cols.as_ref().clone()),
                scan_task.pushdowns.limit,
            )
            .map(|t| t.into())
            .context(PyIOSnafu)
        })?;
        Ok(std::iter::once(Ok(table)))
    })
}

#[cfg(feature = "python")]
fn read_python_function(scan_task: &Arc<ScanTask>) -> BoxStream<'static, DaftResult<RecordBatch>> {
    let scan_task = scan_task.clone();
    read_on_blocking_thread(move || {
        let batches = daft_micropartition::python::read_pyfunc_into_table_iter(scan_task)?;
        Ok(batches.map(|batch| batch.map_err(Into::into)))
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use common_error::{DaftError, DaftResult};
    use daft_core::{prelude::Schema, series::Series};
    use daft_recordbatch::RecordBatch;
    use futures::StreamExt;

    use super::read_on_blocking_thread;

    /// A batch of no columns whose row count names it.
    fn batch(rows: usize) -> DaftResult<RecordBatch> {
        RecordBatch::new_with_size(Schema::empty(), Vec::<Series>::new(), rows)
    }

    fn rows(read: &[DaftResult<RecordBatch>]) -> Vec<Option<usize>> {
        read.iter()
            .map(|batch| batch.as_ref().ok().map(RecordBatch::len))
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn batches_arrive_in_order() {
        let read: Vec<_> = read_on_blocking_thread(|| Ok((1..=3).map(batch)))
            .collect()
            .await;

        assert_eq!(rows(&read), vec![Some(1), Some(2), Some(3)]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn a_source_that_cannot_open_ends_with_its_error() {
        let read: Vec<_> = read_on_blocking_thread(|| {
            Err::<std::iter::Empty<_>, _>(DaftError::ValueError("no source".to_string()))
        })
        .collect()
        .await;

        assert_eq!(read.len(), 1);
        assert!(matches!(&read[0], Err(DaftError::ValueError(m)) if m == "no source"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn a_panic_while_reading_ends_the_stream_with_an_error() {
        let read: Vec<_> = read_on_blocking_thread(|| {
            Ok((1..=3).map(|n| {
                assert!(n < 3, "the source broke");
                batch(n)
            }))
        })
        .collect()
        .await;

        assert_eq!(rows(&read[..2]), vec![Some(1), Some(2)]);
        assert_eq!(read.len(), 3);
        assert!(
            read[2]
                .as_ref()
                .is_err_and(|e| e.to_string().contains("the source broke"))
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn reading_stops_once_the_reader_is_gone() {
        let taken = Arc::new(AtomicUsize::new(0));
        let counted = taken.clone();
        let mut read = read_on_blocking_thread(move || {
            Ok((1..).map(move |n| {
                counted.fetch_add(1, Ordering::SeqCst);
                batch(n)
            }))
        });

        assert!(read.next().await.is_some());
        drop(read);
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // One batch was read, the channel holds as many as it may, and one was
        // being sent when the reader went.
        assert!(taken.load(Ordering::SeqCst) <= super::BATCHES_AHEAD + 2);
    }
}
