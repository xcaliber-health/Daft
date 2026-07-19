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

impl Drop for SpillScratch {
    fn drop(&mut self) {
        if let Err(e) = std::fs::remove_dir_all(&self.root) {
            log::warn!("failed to remove spill scratch {}: {e}", self.root.display());
        }
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
        let series =
            Int64Array::from_vec("v", values).into_series();
        let batch = RecordBatch::from_nonempty_columns(vec![series]).unwrap();
        MicroPartition::new_loaded(batch.schema.clone(), Arc::new(vec![batch]), None)
    }

    #[tokio::test]
    async fn spill_and_read_back_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        let scratch =
            SpillScratch::try_new(&[tmp.path().to_str().unwrap().to_string()]).unwrap();

        let parts = vec![part(vec![1, 2, 3]), part(vec![4, 5])];
        let run = scratch
            .spill(parts, Some("lz4".to_string()))
            .await
            .unwrap();
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
            let run = scratch
                .spill(vec![part(vec![7])], None)
                .await
                .unwrap();
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
