//! Disk spill for buffered operator state.
//!
//! Operators that accumulate unbounded state (aggregations, sorts, join
//! builds) shed buffered partitions to columnar stream files under memory
//! pressure and read them back during finalization. Files are written and
//! read on the IO pool, compressed, size-rotated, and removed when the
//! owning scratch directory is dropped.

use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

use common_error::{DaftError, DaftResult};
use common_runtime::get_io_runtime;
use daft_micropartition::MicroPartition;
use daft_writers::make_ipc_writer;

/// Target on-disk size of one spill file before rotating to the next.
const SPILL_TARGET_FILE_SIZE: usize = 64 * 1024 * 1024;

/// Spill configuration and lazily created scratch space shared by every
/// worker of one operator.
pub(crate) struct SpillContext {
    spill_dirs: Vec<String>,
    compression: Option<String>,
    scratch: std::sync::Mutex<Option<std::sync::Arc<SpillScratch>>>,
}

impl SpillContext {
    /// Builds a context from the execution configuration; `None` when
    /// spilling is disabled.
    pub(crate) fn from_config(cfg: &common_daft_config::DaftExecutionConfig) -> Option<Self> {
        cfg.enable_spilling.then(|| Self {
            spill_dirs: cfg.spill_dirs.clone(),
            compression: cfg.flight_shuffle_compression.clone(),
            scratch: std::sync::Mutex::new(None),
        })
    }

    /// Compression applied to spill files.
    pub(crate) fn compression(&self) -> Option<String> {
        self.compression.clone()
    }

    /// Returns the shared scratch directory, creating it on first use so
    /// queries that never spill touch no disk.
    pub(crate) fn scratch(&self) -> DaftResult<std::sync::Arc<SpillScratch>> {
        let mut guard = self
            .scratch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(scratch) = guard.as_ref() {
            return Ok(scratch.clone());
        }
        let scratch = std::sync::Arc::new(SpillScratch::try_new(&self.spill_dirs)?);
        *guard = Some(scratch.clone());
        Ok(scratch)
    }
}

/// A scratch directory owning every spill file written under it.
///
/// The directory is created eagerly and removed (best effort) on drop, so
/// completed, failed, and cancelled queries all clean up after themselves.
pub(crate) struct SpillScratch {
    root: PathBuf,
    next_run: AtomicUsize,
}

impl SpillScratch {
    /// Creates a unique scratch directory under the first configured spill
    /// location.
    ///
    /// # Errors
    /// Returns an error if no spill directory is configured or the scratch
    /// directory cannot be created.
    pub(crate) fn try_new(spill_dirs: &[String]) -> DaftResult<Self> {
        let base = spill_dirs.first().ok_or_else(|| {
            DaftError::ValueError("spilling requires at least one spill directory".to_string())
        })?;
        let base = base.strip_prefix("file://").unwrap_or(base);
        let root = PathBuf::from(base)
            .join("daft-spill")
            .join(uuid::Uuid::new_v4().to_string());
        std::fs::create_dir_all(&root)?;
        Ok(Self {
            root,
            next_run: AtomicUsize::new(0),
        })
    }

    /// Reserves a fresh subdirectory for one spill run.
    fn next_run_dir(&self) -> DaftResult<PathBuf> {
        let run = self.next_run.fetch_add(1, Ordering::Relaxed);
        let dir = self.root.join(format!("run-{run}"));
        std::fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    /// Writes the given partitions as one spill run and returns its handle.
    ///
    /// Runs on the IO pool; the caller's partitions are consumed. Partition
    /// order within the run is preserved on read-back.
    ///
    /// # Errors
    /// Returns an error if the files cannot be written.
    pub(crate) async fn spill(
        &self,
        parts: Vec<MicroPartition>,
        compression: Option<String>,
    ) -> DaftResult<SpilledRun> {
        let dir = self.next_run_dir()?;
        let num_rows: usize = parts.iter().map(MicroPartition::len).sum();
        let size_bytes: usize = parts.iter().map(|p| p.size_bytes()).sum();
        let dir_str = dir
            .to_str()
            .ok_or_else(|| {
                DaftError::ValueError(format!(
                    "spill directory path is not valid UTF-8: {}",
                    dir.display()
                ))
            })?
            .to_string();

        let file_paths = get_io_runtime(true)
            .spawn(async move {
                let mut writer =
                    make_ipc_writer(&dir_str, SPILL_TARGET_FILE_SIZE, compression.as_deref())?;
                for part in parts {
                    writer.write(part).await?;
                }
                let path_batches = writer.close().await?;
                let mut file_paths = Vec::with_capacity(path_batches.len());
                for batch in path_batches {
                    // The writer reports one single-row path column per
                    // rotated file.
                    let path = batch.get_column(0).utf8()?.get(0).ok_or_else(|| {
                        DaftError::InternalError(
                            "spill writer reported a file without a path".to_string(),
                        )
                    })?;
                    file_paths.push(path.to_string());
                }
                Ok::<_, DaftError>(file_paths)
            })
            .await??;

        log::info!(
            "spilled {num_rows} buffered rows ({size_bytes} bytes) to {} files",
            file_paths.len()
        );
        Ok(SpilledRun {
            file_paths,
            num_rows,
            size_bytes,
        })
    }
}

impl SpillScratch {
    /// Opens a streaming writer for one spill run, so a run larger than
    /// memory can be produced incrementally.
    ///
    /// # Errors
    /// Returns an error if the run directory cannot be created.
    pub(crate) fn start_run(&self, compression: Option<String>) -> DaftResult<RunWriter> {
        let dir = self.next_run_dir()?;
        let dir_str = dir
            .to_str()
            .ok_or_else(|| {
                DaftError::ValueError(format!(
                    "spill directory path is not valid UTF-8: {}",
                    dir.display()
                ))
            })?
            .to_string();
        let (tx, mut rx) = crate::channel::create_channel::<MicroPartition>(2);
        let task = get_io_runtime(true).spawn(async move {
            let mut writer =
                make_ipc_writer(&dir_str, SPILL_TARGET_FILE_SIZE, compression.as_deref())?;
            let mut num_rows = 0usize;
            let mut size_bytes = 0usize;
            while let Some(part) = rx.recv().await {
                num_rows += part.len();
                size_bytes += part.size_bytes();
                writer.write(part).await?;
            }
            let path_batches = writer.close().await?;
            let mut file_paths = Vec::with_capacity(path_batches.len());
            for batch in path_batches {
                let path = batch.get_column(0).utf8()?.get(0).ok_or_else(|| {
                    DaftError::InternalError(
                        "spill writer reported a file without a path".to_string(),
                    )
                })?;
                file_paths.push(path.to_string());
            }
            Ok(SpilledRun {
                file_paths,
                num_rows,
                size_bytes,
            })
        });
        Ok(RunWriter {
            tx: Some(tx),
            task: Some(task),
        })
    }
}

impl Drop for SpillScratch {
    fn drop(&mut self) {
        if let Err(e) = std::fs::remove_dir_all(&self.root) {
            log::warn!(
                "failed to remove spill scratch {}: {e}",
                self.root.display()
            );
        }
    }
}

/// Incremental writer for one spill run; partitions stream to disk with
/// bounded buffering between the producer and the file writer.
pub(crate) struct RunWriter {
    tx: Option<crate::channel::Sender<MicroPartition>>,
    task: Option<common_runtime::RuntimeTask<DaftResult<SpilledRun>>>,
}

impl RunWriter {
    /// Appends one partition to the run.
    ///
    /// # Errors
    /// Returns an error if the writer task has already failed.
    pub(crate) async fn push(&self, part: MicroPartition) -> DaftResult<()> {
        let tx = self.tx.as_ref().ok_or_else(|| {
            DaftError::InternalError("spill run writer used after finish".to_string())
        })?;
        tx.send(part)
            .await
            .map_err(|_| DaftError::InternalError("spill run writer terminated early".to_string()))
    }

    /// Completes the run and returns its handle.
    ///
    /// # Errors
    /// Returns an error if any write failed.
    pub(crate) async fn finish(mut self) -> DaftResult<SpilledRun> {
        drop(self.tx.take());
        let task = self.task.take().ok_or_else(|| {
            DaftError::InternalError("spill run writer finished twice".to_string())
        })?;
        task.await?
    }
}

/// Handle to one written spill run: its files plus the in-memory footprint
/// it displaced.
#[derive(Debug)]
pub(crate) struct SpilledRun {
    file_paths: Vec<String>,
    num_rows: usize,
    size_bytes: usize,
}

impl SpilledRun {
    /// Rows contained in this run.
    #[cfg(test)]
    pub(crate) fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Reads the run back into memory, one partition per file, in write
    /// order. Runs on the IO pool.
    ///
    /// # Errors
    /// Returns an error if a file cannot be read or decoded.
    pub(crate) async fn read_back(self) -> DaftResult<Vec<MicroPartition>> {
        let file_paths = self.file_paths;
        get_io_runtime(true)
            .spawn(async move {
                let mut parts = Vec::with_capacity(file_paths.len());
                for path in file_paths {
                    let bytes = std::fs::read(&path)?;
                    parts.push(MicroPartition::read_from_ipc_stream(&bytes)?);
                }
                Ok::<_, DaftError>(parts)
            })
            .await?
    }

    /// Opens a streaming cursor over the run, yielding record batches in
    /// write order one file at a time, so a run never needs to be resident
    /// in memory all at once.
    pub(crate) fn cursor(self) -> RunCursor {
        RunCursor {
            file_paths: self.file_paths.into_iter().collect(),
            pending: std::collections::VecDeque::new(),
        }
    }
}

/// Streaming reader over one spill run: batches come back in write order,
/// loading one file at a time on the IO pool.
pub(crate) struct RunCursor {
    file_paths: std::collections::VecDeque<String>,
    pending: std::collections::VecDeque<daft_recordbatch::RecordBatch>,
}

impl RunCursor {
    /// Returns the next batch of the run, or `None` when exhausted.
    ///
    /// # Errors
    /// Returns an error if a file cannot be read or decoded.
    pub(crate) async fn next_batch(&mut self) -> DaftResult<Option<daft_recordbatch::RecordBatch>> {
        loop {
            if let Some(batch) = self.pending.pop_front() {
                if batch.is_empty() {
                    continue;
                }
                return Ok(Some(batch));
            }
            let Some(path) = self.file_paths.pop_front() else {
                return Ok(None);
            };
            let part = get_io_runtime(true)
                .spawn(async move {
                    let bytes = std::fs::read(&path)?;
                    MicroPartition::read_from_ipc_stream(&bytes)
                })
                .await??;
            self.pending.extend(part.record_batches().iter().cloned());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use daft_core::{
        datatypes::{DataType, Field, Int64Array},
        series::IntoSeries,
    };
    use daft_recordbatch::RecordBatch;

    use super::*;

    fn part(values: Vec<i64>) -> MicroPartition {
        let series = Int64Array::from_vec("v", values).into_series();
        let batch = RecordBatch::from_nonempty_columns(vec![series]).unwrap();
        MicroPartition::new_loaded(batch.schema.clone(), Arc::new(vec![batch]), None)
    }

    #[tokio::test]
    async fn spill_and_read_back_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        let scratch = SpillScratch::try_new(&[tmp.path().to_str().unwrap().to_string()]).unwrap();

        let parts = vec![part(vec![1, 2, 3]), part(vec![4, 5])];
        let run = scratch.spill(parts, Some("lz4".to_string())).await.unwrap();
        assert_eq!(run.num_rows(), 5);

        let back = run.read_back().await.unwrap();
        let total: usize = back.iter().map(MicroPartition::len).sum();
        assert_eq!(total, 5);
        let mut values: Vec<i64> = Vec::new();
        for p in &back {
            for batch in p.record_batches() {
                values.extend(batch.get_column(0).i64().unwrap().values().iter().copied());
            }
        }
        assert_eq!(values, vec![1, 2, 3, 4, 5]);
    }

    #[tokio::test]
    async fn scratch_drop_removes_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let root = {
            let scratch =
                SpillScratch::try_new(&[tmp.path().to_str().unwrap().to_string()]).unwrap();
            let run = scratch.spill(vec![part(vec![7])], None).await.unwrap();
            assert_eq!(run.num_rows(), 1);
            scratch.root.clone()
        };
        assert!(!root.exists());
    }

    #[test]
    fn empty_spill_dirs_is_an_error() {
        assert!(SpillScratch::try_new(&[]).is_err());
    }
}
