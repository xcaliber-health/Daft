use std::{collections::HashMap, pin::Pin, sync::Arc};

use bytes::Bytes;
use common_error::{DaftError, DaftResult};
use common_runtime::{RuntimeTask, get_io_runtime};
use daft_dsl::optimization::get_required_columns;
use parquet::{
    arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions},
    errors::Result as ParquetResult,
    file::{
        metadata::{PageIndexPolicy, ParquetMetaData},
        reader::{ChunkReader, Length},
    },
};
use snafu::{OptionExt, ResultExt};

use crate::{
    ParquetMetadataSnafu, ReaderInternalSnafu,
    footer_cache::{self, FooterKey},
    metadata::apply_field_ids_to_arrowrs_parquet_metadata,
    read::ParquetReadOptions,
    task_err,
};

/// The largest gap between two page ranges fetched as one request. Smaller
/// than the whole-chunk coalescing gap: merging page ranges across a wide gap
/// fetches back the very bytes skipping the pages saves.
const PAGE_COALESCE_GAP: u64 = 64 * 1024;

fn coalesce_ranges(mut leaf_ranges: Vec<LeafRange>, max_gap: u64) -> Vec<RangeGroup> {
    leaf_ranges.sort_by_key(|r| r.start);
    let mut groups: Vec<RangeGroup> = Vec::new();
    for entry in leaf_ranges {
        let entry_end = entry.start + entry.len;
        if let Some(group) = groups.last_mut()
            && entry.start <= group.end + max_gap
        {
            group.end = group.end.max(entry_end);
            group.members.push(entry);
            continue;
        }
        groups.push(RangeGroup {
            start: entry.start,
            end: entry_end,
            members: vec![entry],
        });
    }
    groups
}

async fn drive_group(slot: &GroupSlot, path: &str) -> GroupResult {
    let mut guard = slot.state.lock().await;
    if let RangeState::Ready(r) = &*guard {
        return r.clone();
    }
    // Drive InFlight → Ready. Holding the lock across `.await` serializes
    // concurrent waiters, but they'd have had to wait for the same spawned
    // task to finish either way — no extra latency.
    //
    // `Pin::new(task).await` polls the task to completion without consuming
    // it (RuntimeTask is Unpin), so the borrow ends with the inner block and
    // we can then write the Ready result back through the same guard.
    let res: GroupResult = {
        let RangeState::InFlight(task) = &mut *guard else {
            unreachable!("Ready branch returned above")
        };
        match Pin::new(task).await {
            Ok(Ok(bytes)) => Ok(bytes),
            Ok(Err(e)) => Err(Arc::new(e)),
            Err(daft_err) => Err(Arc::new(task_err(path.to_string())(daft_err))),
        }
    };
    *guard = RangeState::Ready(res.clone());
    res
}

/// A local file read through for its footer, recording each byte read.
struct CountedFile<'a> {
    file: &'a std::fs::File,
    io_stats: Option<daft_io::IOStatsRef>,
}

impl Length for CountedFile<'_> {
    fn len(&self) -> u64 {
        self.file.len()
    }
}

impl ChunkReader for CountedFile<'_> {
    type T = std::io::BufReader<daft_io::CountingReader<std::fs::File>>;

    fn get_read(&self, start: u64) -> ParquetResult<Self::T> {
        use std::io::Seek;
        let mut file = self.file.try_clone()?;
        file.seek(std::io::SeekFrom::Start(start))?;
        Ok(std::io::BufReader::new(daft_io::CountingReader::new(
            file,
            self.io_stats.clone(),
        )))
    }

    fn get_bytes(&self, start: u64, length: usize) -> ParquetResult<Bytes> {
        let bytes = self.file.get_bytes(start, length)?;
        mark_bytes_read(self.io_stats.as_ref(), bytes.len());
        Ok(bytes)
    }
}

fn mark_bytes_read(io_stats: Option<&daft_io::IOStatsRef>, bytes: usize) {
    if let Some(io_stats) = io_stats {
        io_stats.mark_bytes_read(bytes);
    }
}

/// Opens a local Parquet file and loads its footer, recording the bytes the
/// footer took to read in `io_stats`.
pub(super) async fn open_local_file(
    path: &str,
    page_index: bool,
    io_stats: Option<daft_io::IOStatsRef>,
) -> crate::Result<(Arc<std::fs::File>, u64, ArrowReaderMetadata)> {
    let path_owned = path.to_string();
    let path_for_join = path.to_string();
    get_io_runtime(true)
        .spawn_blocking(move || {
            let file = std::fs::File::open(&path_owned).map_err(|e| crate::Error::LocalIO {
                path: path_owned.clone(),
                source: e,
            })?;
            let file_len = file
                .metadata()
                .map_err(|e| crate::Error::LocalIO {
                    path: path_owned.clone(),
                    source: e,
                })?
                .len();
            let options = ArrowReaderOptions::new().with_page_index_policy(if page_index {
                PageIndexPolicy::Optional
            } else {
                PageIndexPolicy::Skip
            });
            let counted = CountedFile {
                file: &file,
                io_stats,
            };
            let meta = ArrowReaderMetadata::load(&counted, options).with_context(|_| {
                ParquetMetadataSnafu {
                    path: path_owned.clone(),
                }
            })?;
            crate::Result::Ok((Arc::new(file), file_len, meta))
        })
        .await
        .map_err(task_err(path_for_join))?
}

/// Where a remote read's footer came from.
enum FooterSource {
    /// Handed over by the caller, which read it at planning.
    Caller,
    /// Taken from the process-wide footer cache.
    Cache,
    /// Fetched now, with the bytes read from the end of the file.
    Fetched(Bytes),
}

/// The file's raw footer, where it came from, and the file's size, asking the
/// store only for what the read does not already know.
async fn footer_and_size(
    uri: &str,
    known: Option<(Arc<ParquetMetaData>, FooterSource)>,
    known_size: Option<usize>,
    io_client: &Arc<daft_io::IOClient>,
    io_stats: Option<&daft_io::IOStatsRef>,
) -> crate::Result<(Arc<ParquetMetaData>, FooterSource, usize)> {
    match (known, known_size) {
        (Some((metadata, source)), Some(size)) => Ok((metadata, source, size)),
        (Some((metadata, source)), None) => {
            let size = io_client
                .single_url_get_size(uri.to_string(), io_stats.cloned())
                .await?;
            Ok((metadata, source, size))
        }
        (None, Some(size)) => {
            let (metadata, tail) = crate::metadata::read_parquet_metadata_and_tail(
                uri,
                Some(size),
                io_client.clone(),
                io_stats.cloned(),
                None,
                None,
            )
            .await?;
            Ok((metadata, FooterSource::Fetched(tail), size))
        }
        (None, None) => {
            let (footer, size) = Box::pin(futures::future::join(
                crate::metadata::read_parquet_metadata_and_tail(
                    uri,
                    None,
                    io_client.clone(),
                    io_stats.cloned(),
                    None,
                    None,
                ),
                io_client.single_url_get_size(uri.to_string(), io_stats.cloned()),
            ))
            .await;
            let (metadata, tail) = footer?;
            Ok((metadata, FooterSource::Fetched(tail), size?))
        }
    }
}

/// `metadata` with the read's field-id mapping applied, if it has one.
fn mapped(
    metadata: &Arc<ParquetMetaData>,
    opts: &ParquetReadOptions,
    uri: &str,
) -> crate::Result<Arc<ParquetMetaData>> {
    match opts.field_id_mapping.as_deref() {
        Some(mapping) => {
            apply_field_ids_to_arrowrs_parquet_metadata(metadata.clone(), mapping, uri)
        }
        None => Ok(metadata.clone()),
    }
}

pub(super) async fn prepare_remote_chunk_source(
    uri: &str,
    io_client: Arc<daft_io::IOClient>,
    io_stats: Option<daft_io::IOStatsRef>,
    opts: &ParquetReadOptions,
) -> crate::Result<(ChunkSourceBuilder, ArrowReaderMetadata)> {
    let cache_key = FooterKey::for_read(
        io_client.config(),
        uri,
        opts.file_version,
        opts.footer_cache,
    );
    let known = opts
        .metadata
        .as_deref()
        .and_then(crate::DaftParquetMetadata::as_whole_file)
        .map(|metadata| (metadata.clone(), FooterSource::Caller))
        .or_else(|| {
            cache_key
                .as_ref()
                .and_then(footer_cache::get)
                .map(|metadata| (metadata, FooterSource::Cache))
        });
    let known_size = opts
        .file_version
        .and_then(|version| usize::try_from(version.size).ok());
    let (mut raw_metadata, footer_source, file_size) =
        footer_and_size(uri, known, known_size, &io_client, io_stats.as_ref()).await?;

    // Apply Iceberg field-id mapping before filtering by column name —
    // otherwise the prefetch matches pre-rename names against post-rename
    // user-supplied names and fetches zero leaves. The cache keeps the raw
    // footer, since a table's mapping can change while its files do not.
    let mut parquet_metadata = mapped(&raw_metadata, opts, uri)?;

    // Prefetch needs the union of user-requested columns and predicate columns.
    let prefetch_col_names: Option<std::collections::HashSet<String>> =
        opts.columns.as_ref().map(|cols| {
            let mut acc: std::collections::HashSet<String> = cols.iter().cloned().collect();
            if let Some(pred) = opts.predicate.as_ref() {
                acc.extend(get_required_columns(pred));
            }
            acc
        });

    let schema_descr = parquet_metadata.file_metadata().schema_descr();
    let num_cols_total = schema_descr.num_columns();
    let active_col_indices: Vec<usize> = match &prefetch_col_names {
        None => (0..num_cols_total).collect(),
        Some(want) => (0..num_cols_total)
            .filter(|&i| {
                schema_descr
                    .column(i)
                    .path()
                    .parts()
                    .first()
                    .map(|n| want.contains(n.as_str()))
                    .unwrap_or(false)
            })
            .collect(),
    };

    // A filtered read fetches the page index, one range beside the footer, so
    // it can skip pages; a plain read never needs it. A cached footer may
    // already hold it.
    let index_loaded = if opts.page_index
        && opts.predicate.is_some()
        && raw_metadata.offset_index().is_none()
        && super::page_index::worth_reading(&parquet_metadata, &active_col_indices)
        && let Some(range) = super::page_index::index_range(&raw_metadata)
    {
        // The footer read usually reaches back over the index, which sits just
        // before the footer; fetch it only when it does not.
        let tail = match &footer_source {
            FooterSource::Fetched(tail) => Some(tail),
            FooterSource::Caller | FooterSource::Cache => None,
        };
        let tail_start = tail.map(|tail| (file_size as u64).saturating_sub(tail.len() as u64));
        let index = match (tail, tail_start) {
            (Some(tail), Some(tail_start)) if range.start >= tail_start => {
                OffsetBytes::whole(tail_start, tail.clone(), file_size as u64)
            }
            _ => {
                let got = io_client
                    .single_url_get(
                        uri.to_string(),
                        Some(daft_io::range::GetRange::Bounded(
                            range.start as usize..range.end as usize,
                        )),
                        io_stats.clone(),
                    )
                    .await?;
                OffsetBytes::whole(range.start, got.bytes().await?, file_size as u64)
            }
        };
        raw_metadata = Arc::new(
            super::page_index::with_page_index(Arc::unwrap_or_clone(raw_metadata), &index)
                .with_context(|_| ParquetMetadataSnafu {
                    path: uri.to_string(),
                })?,
        );
        parquet_metadata = mapped(&raw_metadata, opts, uri)?;
        true
    } else {
        false
    };

    // Keep what was fetched for the next read of this version of the file;
    // a footer handed over by the caller is the caller's, not the store's.
    if let Some(key) = cache_key {
        let fetched = matches!(footer_source, FooterSource::Fetched(_));
        let grown = matches!(footer_source, FooterSource::Cache) && index_loaded;
        if fetched || grown {
            footer_cache::insert(key, raw_metadata, opts.footer_cache.max_bytes);
        }
    }

    // A cached footer may hold a page index a read with the index turned off
    // must not use.
    if !opts.page_index && parquet_metadata.offset_index().is_some() {
        parquet_metadata = Arc::new(
            Arc::unwrap_or_clone(parquet_metadata)
                .into_builder()
                .set_column_index(None)
                .set_offset_index(None)
                .build(),
        );
    }

    let path: Arc<str> = Arc::from(uri);
    let meta = ArrowReaderMetadata::try_new(parquet_metadata, ArrowReaderOptions::new())
        .with_context(|_| ParquetMetadataSnafu {
            path: uri.to_string(),
        })?;
    let builder = ChunkSourceBuilder::Remote(RemoteChunkSourcePrep {
        path,
        uri: uri.to_string(),
        file_size,
        active_col_indices,
        io_client,
        io_stats,
    });
    Ok((builder, meta))
}

#[derive(Copy, Clone)]
struct LeafRange {
    leaf: usize,
    start: u64,
    len: u64,
}

struct RangeGroup {
    start: u64,
    end: u64,
    members: Vec<LeafRange>,
}

/// `ChunkReader` over the fetched bytes of one column chunk: the whole chunk,
/// or only the byte ranges of the pages a read needs. Reports the file's total
/// length but translates absolute offsets to the fetched bytes, so
/// `SerializedPageReader` works without holding the whole file in memory. A
/// read outside the fetched ranges is an error, never a silent gap.
#[derive(Clone)]
pub(crate) struct OffsetBytes {
    file_len: u64,
    /// Fetched byte ranges, as (absolute start, bytes), sorted and disjoint.
    segments: Arc<[(u64, Bytes)]>,
}

impl OffsetBytes {
    /// The bytes of a whole range starting at absolute offset `base`.
    pub(crate) fn whole(base: u64, bytes: Bytes, file_len: u64) -> Self {
        Self {
            file_len,
            segments: Arc::from([(base, bytes)]),
        }
    }

    /// The bytes of several ranges, as (absolute start, bytes), in any order.
    pub(crate) fn sparse(mut segments: Vec<(u64, Bytes)>, file_len: u64) -> Self {
        segments.sort_by_key(|(start, _)| *start);
        Self {
            file_len,
            segments: segments.into(),
        }
    }

    /// The fetched segment holding absolute offset `start`, and `start`'s
    /// offset within it.
    fn segment_at(&self, start: u64, call: &str) -> ParquetResult<(&Bytes, usize)> {
        let found = self
            .segments
            .partition_point(|(base, _)| *base <= start)
            .checked_sub(1)
            .map(|index| &self.segments[index]);
        match found {
            Some((base, bytes)) if start - base <= bytes.len() as u64 => {
                Ok((bytes, (start - base) as usize))
            }
            _ => Err(parquet::errors::ParquetError::General(format!(
                "OffsetBytes::{call}: offset {start} lies outside the fetched ranges"
            ))),
        }
    }
}

impl Length for OffsetBytes {
    fn len(&self) -> u64 {
        self.file_len
    }
}

impl ChunkReader for OffsetBytes {
    type T = bytes::buf::Reader<Bytes>;

    fn get_read(&self, start: u64) -> ParquetResult<Self::T> {
        use bytes::Buf;
        let (bytes, local) = self.segment_at(start, "get_read")?;
        Ok(bytes.slice(local..).reader())
    }

    fn get_bytes(&self, start: u64, length: usize) -> ParquetResult<Bytes> {
        let (bytes, local) = self.segment_at(start, "get_bytes")?;
        let end = local.checked_add(length).ok_or_else(|| {
            parquet::errors::ParquetError::General("OffsetBytes::get_bytes: offset overflow".into())
        })?;
        if end > bytes.len() {
            return Err(parquet::errors::ParquetError::General(format!(
                "OffsetBytes::get_bytes: range {start}+{length} runs past its fetched range"
            )));
        }
        Ok(bytes.slice(local..end))
    }
}

/// Per-(rg, leaf) factory for column-chunk byte slices. Enum-dispatched (rather
/// than `dyn`) so `read_rg_chunks` can be a plain async fn — no per-call
/// `BoxFuture` allocation.
pub(crate) enum ChunkSource {
    Local(LocalChunkSource),
    Remote(RemoteChunkSource),
}

/// Deferred construction of a `ChunkSource`. For `Local` the source is already
/// open (`open_local_file` is a cheap syscall + footer read). For `Remote` we
/// hold the bits needed to spawn byte-range fetches but have NOT spawned them
/// yet — caller passes a final, predicate-pruned RG set to [`Self::build`].
pub(crate) enum ChunkSourceBuilder {
    Local(LocalChunkSource),
    Remote(RemoteChunkSourcePrep),
}

pub(crate) struct RemoteChunkSourcePrep {
    pub(super) path: Arc<str>,
    pub(super) uri: String,
    pub(super) file_size: usize,
    pub(super) active_col_indices: Vec<usize>,
    pub(super) io_client: Arc<daft_io::IOClient>,
    pub(super) io_stats: Option<daft_io::IOStatsRef>,
}

impl ChunkSourceBuilder {
    pub(super) fn path(&self) -> &Arc<str> {
        match self {
            Self::Local(s) => &s.path,
            Self::Remote(p) => &p.path,
        }
    }

    /// Read `len` bytes starting at absolute file `offset`.
    ///
    /// Used for out-of-band reads (e.g. bloom filter bitsets) before the
    /// row-group byte fetches are spawned, reusing the already-open local file
    /// handle or the remote client.
    pub(crate) async fn read_range(&self, offset: u64, len: usize) -> DaftResult<Bytes> {
        match self {
            Self::Local(s) => {
                let file = s.file.clone();
                let io_stats = s.io_stats.clone();
                get_io_runtime(true)
                    .spawn_blocking(move || -> DaftResult<Bytes> {
                        let mut buf = vec![0u8; len];
                        #[cfg(unix)]
                        {
                            use std::os::unix::fs::FileExt;
                            file.read_exact_at(&mut buf, offset)?;
                        }
                        #[cfg(windows)]
                        {
                            use std::os::windows::fs::FileExt;
                            let mut filled = 0usize;
                            while filled < len {
                                let n =
                                    file.seek_read(&mut buf[filled..], offset + filled as u64)?;
                                if n == 0 {
                                    return Err(DaftError::IoError(std::io::Error::new(
                                        std::io::ErrorKind::UnexpectedEof,
                                        "short read while loading bloom filter",
                                    )));
                                }
                                filled += n;
                            }
                        }
                        mark_bytes_read(io_stats.as_ref(), len);
                        Ok(Bytes::from(buf))
                    })
                    .await?
            }
            Self::Remote(prep) => {
                let end = offset
                    .checked_add(len as u64)
                    .ok_or_else(|| DaftError::ValueError("byte range overflow".to_string()))?;
                let range = daft_io::range::GetRange::Bounded(
                    usize::try_from(offset).map_err(|_| {
                        DaftError::ValueError("byte offset exceeds usize".to_string())
                    })?..usize::try_from(end).map_err(|_| {
                        DaftError::ValueError("byte range end exceeds usize".to_string())
                    })?,
                );
                let result = prep
                    .io_client
                    .single_url_get(prep.uri.clone(), Some(range), prep.io_stats.clone())
                    .await?;
                Ok(result.bytes().await?)
            }
        }
    }

    /// Finalize the chunk source. For `Remote`, this is when byte-range fetches
    /// for `rg_indices` are spawned — call only after predicate-based pruning.
    /// A row group left out is fetched only as asked, through
    /// [`ChunkSource::read_ranges`].
    pub(super) fn build(
        self,
        parquet_metadata: Arc<ParquetMetaData>,
        rg_indices: &[usize],
    ) -> ChunkSource {
        match self {
            Self::Local(cs) => ChunkSource::Local(cs),
            Self::Remote(prep) => ChunkSource::Remote(RemoteChunkSource::from_ranged(
                prep.path,
                parquet_metadata,
                prep.file_size,
                &prep.active_col_indices,
                rg_indices,
                prep.io_client,
                prep.io_stats,
                prep.uri,
            )),
        }
    }
}

#[cfg(test)]
impl ChunkSourceBuilder {
    /// Open a local Parquet file as a `Local` chunk source for tests, returning
    /// the builder alongside the parsed metadata used for row-group pruning.
    pub(crate) fn local_for_test(path: &str) -> crate::Result<(Self, Arc<ParquetMetaData>)> {
        let file = std::fs::File::open(path).map_err(|e| crate::Error::LocalIO {
            path: path.to_string(),
            source: e,
        })?;
        let file_len = file
            .metadata()
            .map_err(|e| crate::Error::LocalIO {
                path: path.to_string(),
                source: e,
            })?
            .len();
        let meta =
            ArrowReaderMetadata::load(&file, ArrowReaderOptions::new()).with_context(|_| {
                ParquetMetadataSnafu {
                    path: path.to_string(),
                }
            })?;
        let metadata = meta.metadata().clone();
        Ok((
            Self::Local(LocalChunkSource {
                path: Arc::from(path),
                file: Arc::new(file),
                file_len,
                metadata: metadata.clone(),
                io_stats: None,
            }),
            metadata,
        ))
    }
}

impl ChunkSource {
    pub(super) async fn read_rg_chunks(
        &self,
        rg_idx: usize,
        leaves: Arc<[usize]>,
    ) -> crate::Result<HashMap<usize, OffsetBytes>> {
        match self {
            // Local pread is sync; offload to the IO runtime's blocking pool so
            // we don't park a tokio compute worker on syscalls under heavy
            // concurrency.
            Self::Local(s) => {
                let s = s.clone();
                let path = s.path.clone();
                get_io_runtime(true)
                    .spawn_blocking(move || s.read_rg_chunks_sync(rg_idx, &leaves))
                    .await
                    .map_err(task_err(path.to_string()))?
            }
            Self::Remote(s) => s.read_rg_chunks(rg_idx, &leaves).await,
        }
    }

    /// Fetches only `wanted`'s byte ranges, as (leaf, absolute ranges), and
    /// returns each leaf's bytes, which hold those ranges and nothing else.
    ///
    /// Ranges nearer than [`PAGE_COALESCE_GAP`] are fetched together.
    pub(super) async fn read_ranges(
        &self,
        wanted: Vec<(usize, Vec<std::ops::Range<u64>>)>,
    ) -> crate::Result<HashMap<usize, OffsetBytes>> {
        let leaf_ranges: Vec<LeafRange> = wanted
            .iter()
            .flat_map(|(leaf, ranges)| {
                ranges.iter().map(|range| LeafRange {
                    leaf: *leaf,
                    start: range.start,
                    len: range.end - range.start,
                })
            })
            .collect();
        let groups = coalesce_ranges(leaf_ranges, PAGE_COALESCE_GAP);
        let spans: Vec<std::ops::Range<u64>> =
            groups.iter().map(|group| group.start..group.end).collect();
        let fetched = match self {
            Self::Local(s) => {
                let s = s.clone();
                let path = s.path.clone();
                get_io_runtime(true)
                    .spawn_blocking(move || s.read_spans_sync(&spans))
                    .await
                    .map_err(task_err(path.to_string()))??
            }
            Self::Remote(s) => s.read_spans(&spans).await?,
        };
        let file_len = match self {
            Self::Local(s) => s.file_len,
            Self::Remote(s) => s.file_len,
        };
        let mut segments: HashMap<usize, Vec<(u64, Bytes)>> = HashMap::with_capacity(wanted.len());
        for (group, bytes) in groups.into_iter().zip(fetched) {
            for LeafRange { leaf, start, len } in group.members {
                let local = (start - group.start) as usize;
                segments
                    .entry(leaf)
                    .or_default()
                    .push((start, bytes.slice(local..local + len as usize)));
            }
        }
        Ok(segments
            .into_iter()
            .map(|(leaf, segments)| (leaf, OffsetBytes::sparse(segments, file_len)))
            .collect())
    }

    /// One per RG, called by the decoder dispatch. Lets each `ChunkSource`
    /// variant choose its own access pattern without leaking the dispatch into
    /// callers:
    ///
    /// - `Local`: one batched `spawn_blocking` for *all* requested leaves, then
    ///   every column decoder reads slices from the same `Arc<HashMap>`. The
    ///   alternative (per-column reads) would multiply `spawn_blocking` calls
    ///   by `num_cols`, dominating wide-schema runtimes.
    /// - `Remote`: returns a lazy handle. Each decoder later asks for its own
    ///   leaves and awaits only the coalesced byte-range groups covering them,
    ///   so fast columns start streaming while slower groups are still in
    ///   flight.
    pub(super) async fn open_rg(
        self: Arc<Self>,
        rg_idx: usize,
        all_leaves: Arc<[usize]>,
    ) -> crate::Result<RgReader> {
        match self.as_ref() {
            Self::Local(_) => {
                let chunks = self.read_rg_chunks(rg_idx, all_leaves).await?;
                Ok(RgReader::PreFetched(Arc::new(chunks)))
            }
            Self::Remote(_) => Ok(RgReader::Lazy {
                chunk_source: self,
                rg_idx,
            }),
        }
    }
}

/// Decoder-facing handle returned by [`ChunkSource::open_rg`]. Per-column
/// decoders call [`Self::read_col`] without caring whether the bytes are
/// pre-fetched or fetched on demand.
#[derive(Clone)]
pub(crate) enum RgReader {
    PreFetched(Arc<HashMap<usize, OffsetBytes>>),
    Lazy {
        chunk_source: Arc<ChunkSource>,
        rg_idx: usize,
    },
}

impl RgReader {
    pub(super) async fn read_col(
        &self,
        col_leaves: Arc<[usize]>,
    ) -> crate::Result<Arc<HashMap<usize, OffsetBytes>>> {
        match self {
            Self::PreFetched(map) => Ok(map.clone()),
            Self::Lazy {
                chunk_source,
                rg_idx,
            } => Ok(Arc::new(
                chunk_source.read_rg_chunks(*rg_idx, col_leaves).await?,
            )),
        }
    }
}

#[derive(Clone)]
pub(crate) struct LocalChunkSource {
    pub(super) path: Arc<str>,
    pub(super) file: Arc<std::fs::File>,
    pub(super) file_len: u64,
    pub(super) metadata: Arc<ParquetMetaData>,
    /// Where the bytes read from the file are recorded.
    pub(super) io_stats: Option<daft_io::IOStatsRef>,
}

#[cfg(not(any(unix, windows)))]
compile_error!(
    "LocalChunkSource needs FileExt::read_at (unix) or seek_read (windows); \
     no implementation for this target."
);

impl LocalChunkSource {
    const MAX_COALESCE_GAP: u64 = 64 * 1024;

    /// Reads each absolute byte span of the file.
    fn read_spans_sync(&self, spans: &[std::ops::Range<u64>]) -> crate::Result<Vec<Bytes>> {
        spans
            .iter()
            .map(|span| {
                let mut buf = vec![0u8; (span.end - span.start) as usize];
                #[cfg(unix)]
                let read = {
                    use std::os::unix::fs::FileExt;
                    self.file.read_exact_at(&mut buf, span.start)
                };
                #[cfg(windows)]
                let read = {
                    use std::os::windows::fs::FileExt;
                    let mut filled = 0usize;
                    let mut result = Ok(());
                    while filled < buf.len() {
                        match self
                            .file
                            .seek_read(&mut buf[filled..], span.start + filled as u64)
                        {
                            Ok(0) => {
                                result = Err(std::io::Error::new(
                                    std::io::ErrorKind::UnexpectedEof,
                                    "short read of a page range",
                                ));
                                break;
                            }
                            Ok(read) => filled += read,
                            Err(e) => {
                                result = Err(e);
                                break;
                            }
                        }
                    }
                    result
                };
                read.map_err(|e| crate::Error::LocalIO {
                    path: self.path.to_string(),
                    source: std::io::Error::new(
                        e.kind(),
                        format!("pread of page range {}..{}: {e}", span.start, span.end),
                    ),
                })?;
                mark_bytes_read(self.io_stats.as_ref(), buf.len());
                Ok(Bytes::from(buf))
            })
            .collect()
    }

    fn read_rg_chunks_sync(
        &self,
        rg_idx: usize,
        leaves: &[usize],
    ) -> crate::Result<HashMap<usize, OffsetBytes>> {
        if leaves.is_empty() {
            return Ok(HashMap::new());
        }
        let rg = self.metadata.row_group(rg_idx);
        let leaf_ranges: Vec<LeafRange> = leaves
            .iter()
            .map(|&l| {
                let (start, len) = rg.column(l).byte_range();
                LeafRange {
                    leaf: l,
                    start,
                    len,
                }
            })
            .collect();
        let groups = coalesce_ranges(leaf_ranges, Self::MAX_COALESCE_GAP);
        let mut out = HashMap::with_capacity(leaves.len());
        for RangeGroup {
            start: group_start,
            end: group_end,
            members,
        } in groups
        {
            let group_len = (group_end - group_start) as usize;
            let mut buf = vec![0u8; group_len];
            #[cfg(unix)]
            {
                use std::os::unix::fs::FileExt;
                self.file
                    .read_exact_at(&mut buf, group_start)
                    .map_err(|e| crate::Error::LocalIO {
                        path: self.path.to_string(),
                        source: std::io::Error::new(
                            e.kind(),
                            format!(
                                "pread for rg={} coalesced range {}..{}: {}",
                                rg_idx, group_start, group_end, e
                            ),
                        ),
                    })?;
            }
            #[cfg(windows)]
            {
                use std::os::windows::fs::FileExt;
                self.file
                    .seek_read(&mut buf, group_start)
                    .map_err(|e| crate::Error::LocalIO {
                        path: self.path.to_string(),
                        source: std::io::Error::new(
                            e.kind(),
                            format!(
                                "seek_read for rg={} coalesced range {}..{}: {}",
                                rg_idx, group_start, group_end, e
                            ),
                        ),
                    })?;
            }
            mark_bytes_read(self.io_stats.as_ref(), group_len);
            let group_bytes = Bytes::from(buf);
            for LeafRange { leaf, start, len } in members {
                let local_start = (start - group_start) as usize;
                let local_end = local_start + len as usize;
                let slice = group_bytes.slice(local_start..local_end);
                out.insert(leaf, OffsetBytes::whole(start, slice, self.file_len));
            }
        }
        Ok(out)
    }
}

type SharedErr = Arc<crate::Error>;
type GroupResult = Result<Bytes, SharedErr>;

/// Per-coalesced-byte-range fetch state. Starts as `InFlight(task)`; the
/// first awaiter drives the spawned task to completion and stores the
/// (cloneable) result in `Ready`. Subsequent awaiters clone the cached
/// bytes/error instead of re-fetching.
enum RangeState {
    InFlight(RuntimeTask<crate::Result<Bytes>>),
    Ready(GroupResult),
}

/// One coalesced byte-range within a row group: the absolute file offset its
/// bytes start at, plus the lazily-resolved fetch state.
struct GroupSlot {
    group_start: u64,
    state: tokio::sync::Mutex<RangeState>,
}

/// Where a leaf's bytes live: which coalesced group, and its absolute slice
/// within that group's bytes. We slice on demand at read time rather than
/// pre-building a per-leaf `(start, Bytes)` map.
struct LeafLoc {
    group_idx: usize,
    leaf_start: u64,
    leaf_len: u64,
}

struct RgState {
    groups: Vec<GroupSlot>,
    leaves: HashMap<usize, LeafLoc>,
}

/// Per-row-group coalesced byte-range GETs (merge ≤1MB gaps, split >24MB at
/// chunk boundaries into ~16MB pieces). Each coalesced group has its own
/// spawned fetch task with `InFlight → Ready` state transitions, so a
/// `read_rg_chunks` call only awaits the groups containing its requested
/// leaves — never the whole RG's bundle. Once driven, the cached `Ready`
/// bytes let a later call for a different leaf in the same group skip
/// re-fetching. Errors are wrapped in `Arc` so they can be cloned to every
/// subsequent awaiter.
pub(crate) struct RemoteChunkSource {
    path: Arc<str>,
    rgs: HashMap<usize, RgState>,
    file_len: u64,
    uri: String,
    io_client: Arc<daft_io::IOClient>,
    io_stats: Option<daft_io::IOStatsRef>,
}

impl RemoteChunkSource {
    const MAX_COALESCE_GAP: u64 = 1024 * 1024;
    const SPLIT_THRESHOLD: u64 = 24 * 1024 * 1024;
    const MAX_REQUEST_SIZE: u64 = 16 * 1024 * 1024;

    #[allow(clippy::too_many_arguments)]
    fn from_ranged(
        path: Arc<str>,
        parquet_metadata: Arc<ParquetMetaData>,
        file_size: usize,
        active_col_indices: &[usize],
        active_rg_indices: &[usize],
        io_client: Arc<daft_io::IOClient>,
        io_stats: Option<daft_io::IOStatsRef>,
        uri: String,
    ) -> Self {
        let file_len = file_size as u64;
        let io_runtime = get_io_runtime(true);
        let mut rgs = HashMap::with_capacity(active_rg_indices.len());

        for &rg_idx in active_rg_indices {
            let rg = parquet_metadata.row_group(rg_idx);
            let mut leaf_ranges: Vec<LeafRange> = Vec::with_capacity(active_col_indices.len());
            for &col_idx in active_col_indices {
                let (start, len) = rg.column(col_idx).byte_range();
                leaf_ranges.push(LeafRange {
                    leaf: col_idx,
                    start,
                    len,
                });
            }
            let groups = Self::coalesce_and_split(leaf_ranges);

            let mut group_slots: Vec<GroupSlot> = Vec::with_capacity(groups.len());
            let mut leaves: HashMap<usize, LeafLoc> =
                HashMap::with_capacity(active_col_indices.len());

            for (
                group_idx,
                RangeGroup {
                    start: group_start,
                    end: group_end,
                    members,
                },
            ) in groups.into_iter().enumerate()
            {
                let io_client = io_client.clone();
                let io_stats = io_stats.clone();
                let uri = uri.clone();
                let range = group_start as usize..group_end as usize;
                let task = io_runtime.spawn(async move {
                    let get_result = io_client
                        .single_url_get(
                            uri,
                            Some(daft_io::range::GetRange::Bounded(range)),
                            io_stats,
                        )
                        .await?;
                    let bytes = get_result.bytes().await?;
                    crate::Result::Ok(bytes)
                });
                group_slots.push(GroupSlot {
                    group_start,
                    state: tokio::sync::Mutex::new(RangeState::InFlight(task)),
                });
                for LeafRange { leaf, start, len } in members {
                    leaves.insert(
                        leaf,
                        LeafLoc {
                            group_idx,
                            leaf_start: start,
                            leaf_len: len,
                        },
                    );
                }
            }

            rgs.insert(
                rg_idx,
                RgState {
                    groups: group_slots,
                    leaves,
                },
            );
        }

        Self {
            path,
            rgs,
            file_len,
            uri,
            io_client,
            io_stats,
        }
    }

    /// Fetches each absolute byte span of the object, concurrently.
    async fn read_spans(&self, spans: &[std::ops::Range<u64>]) -> crate::Result<Vec<Bytes>> {
        let io_runtime = get_io_runtime(true);
        let fetches = spans.iter().map(|span| {
            let io_client = self.io_client.clone();
            let io_stats = self.io_stats.clone();
            let uri = self.uri.clone();
            let range = span.start as usize..span.end as usize;
            io_runtime.spawn(async move {
                let got = io_client
                    .single_url_get(
                        uri,
                        Some(daft_io::range::GetRange::Bounded(range)),
                        io_stats,
                    )
                    .await?;
                crate::Result::Ok(got.bytes().await?)
            })
        });
        let path = self.path.to_string();
        futures::future::join_all(fetches)
            .await
            .into_iter()
            .map(|joined| joined.map_err(task_err(path.clone()))?)
            .collect()
    }

    fn coalesce_and_split(leaf_ranges: Vec<LeafRange>) -> Vec<RangeGroup> {
        let mut groups = coalesce_ranges(leaf_ranges, Self::MAX_COALESCE_GAP);
        let mut split_groups: Vec<RangeGroup> = Vec::with_capacity(groups.len());
        for RangeGroup {
            start: group_start,
            end: group_end,
            mut members,
        } in groups.drain(..)
        {
            if group_end - group_start <= Self::SPLIT_THRESHOLD {
                split_groups.push(RangeGroup {
                    start: group_start,
                    end: group_end,
                    members,
                });
                continue;
            }
            members.sort_by_key(|r| r.start);
            let mut piece_start = group_start;
            let mut piece_members: Vec<LeafRange> = Vec::new();
            let mut piece_end = piece_start;
            for entry in members {
                let entry_end = entry.start + entry.len;
                let would_be_size = entry_end - piece_start;
                if !piece_members.is_empty() && would_be_size > Self::MAX_REQUEST_SIZE {
                    split_groups.push(RangeGroup {
                        start: piece_start,
                        end: piece_end,
                        members: std::mem::take(&mut piece_members),
                    });
                    piece_start = entry.start;
                }
                piece_end = entry_end;
                piece_members.push(entry);
            }
            if !piece_members.is_empty() {
                split_groups.push(RangeGroup {
                    start: piece_start,
                    end: piece_end,
                    members: piece_members,
                });
            }
        }
        split_groups
    }

    async fn read_rg_chunks(
        &self,
        rg_idx: usize,
        leaves: &[usize],
    ) -> crate::Result<HashMap<usize, OffsetBytes>> {
        if leaves.is_empty() {
            return Ok(HashMap::new());
        }
        let rg = self.rgs.get(&rg_idx).with_context(|| ReaderInternalSnafu {
            path: self.path.to_string(),
            message: format!("RemoteChunkSource: no pre-spawned fetch for rg={}", rg_idx),
        })?;

        // Map each leaf to its enclosing coalesced group, then dedup so we
        // only drive each group's fetch once even when several requested
        // leaves share a group.
        let mut needed_groups: Vec<usize> = Vec::with_capacity(leaves.len());
        for &leaf in leaves {
            let loc = rg.leaves.get(&leaf).with_context(|| ReaderInternalSnafu {
                path: self.path.to_string(),
                message: format!(
                    "RemoteChunkSource: chunk not pre-fetched for rg={}, leaf={}",
                    rg_idx, leaf
                ),
            })?;
            needed_groups.push(loc.group_idx);
        }
        needed_groups.sort_unstable();
        needed_groups.dedup();

        // Drive only the groups containing requested leaves, in parallel.
        // Unrelated groups in this RG stay in-flight (or untouched) and we
        // never park on their completion.
        let path = self.path.as_ref();
        let futs = needed_groups.iter().map(|&gi| {
            let slot = &rg.groups[gi];
            async move { (gi, drive_group(slot, path).await) }
        });
        let results = futures::future::join_all(futs).await;

        let mut group_bytes: HashMap<usize, Bytes> = HashMap::with_capacity(results.len());
        for (gi, res) in results {
            let bytes = res.map_err(|source| crate::Error::RemoteFetchFailed {
                path: self.path.to_string(),
                source,
            })?;
            group_bytes.insert(gi, bytes);
        }

        let mut out = HashMap::with_capacity(leaves.len());
        for &leaf in leaves {
            let loc = rg.leaves.get(&leaf).expect("validated above");
            let bytes = group_bytes.get(&loc.group_idx).expect("driven above");
            let group_start = rg.groups[loc.group_idx].group_start;
            let local_start = (loc.leaf_start - group_start) as usize;
            let local_end = local_start + loc.leaf_len as usize;
            let slice = bytes.slice(local_start..local_end);
            out.insert(
                leaf,
                OffsetBytes::whole(loc.leaf_start, slice, self.file_len),
            );
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use parquet::file::reader::ChunkReader;

    use super::OffsetBytes;

    fn pages() -> OffsetBytes {
        OffsetBytes::sparse(
            vec![
                (500, Bytes::from_static(b"second")),
                (100, Bytes::from_static(b"first")),
            ],
            1_000,
        )
    }

    #[test]
    fn sparse_bytes_read_within_each_fetched_range() {
        let bytes = pages();
        assert_eq!(
            bytes.get_bytes(100, 5).unwrap(),
            Bytes::from_static(b"first")
        );
        assert_eq!(
            bytes.get_bytes(502, 4).unwrap(),
            Bytes::from_static(b"cond")
        );
        assert_eq!(parquet::file::reader::Length::len(&bytes), 1_000);
    }

    #[test]
    fn sparse_bytes_refuse_a_read_outside_the_fetched_ranges() {
        let bytes = pages();
        for (start, length) in [(50, 1), (200, 1), (103, 5), (900, 1)] {
            assert!(bytes.get_bytes(start, length).is_err(), "{start}+{length}");
        }
    }
}
