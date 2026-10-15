//! A process-wide cache of decoded Parquet footers.
//!
//! A read that opens a file it has opened before, while the file is
//! unchanged, takes the file's metadata from here instead of fetching its
//! footer again. An entry is found only under the IO configuration that
//! fetched it, so a read never sees metadata its own credentials did not
//! fetch, and only for the same version of the file: its exact size, and its
//! last-modified time where the file is not immutable.
//!
//! The cache holds at most a caller-given number of bytes of decoded
//! metadata, evicting the least recently used entries first.

use std::sync::{Arc, LazyLock, Mutex};

use daft_io::IOConfig;
use lru::LruCache;
use parquet::file::metadata::ParquetMetaData;
use serde::{Deserialize, Serialize};

/// What a read knows about the version of the file it opens, from the file's
/// listing or its table's manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FileVersion {
    /// The file's exact length in bytes.
    pub size: u64,
    /// When the file was last written, in epoch milliseconds, if known.
    pub last_modified: Option<i64>,
    /// Whether the file is never rewritten in place, as a table format's data
    /// files are, so that its path and size alone name its content.
    pub immutable: bool,
}

/// How a read may use the footer cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FooterCacheSettings {
    /// The most decoded metadata the cache holds, in bytes; 0 turns it off.
    pub max_bytes: usize,
    /// Whether files that are not immutable are cached, by size and
    /// last-modified time.
    pub mutable_files: bool,
}

/// One file's version under one IO configuration.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct FooterKey {
    config: Arc<IOConfig>,
    path: String,
    size: u64,
    last_modified: Option<i64>,
}

impl FooterKey {
    /// The key a read of `path` caches under, or `None` when the read may
    /// not use the cache: it is off, the file's version is unknown, or the
    /// file may change in place and mutable files are not cached (or its
    /// last-modified time is unknown).
    pub(crate) fn for_read(
        config: &Arc<IOConfig>,
        path: &str,
        version: Option<FileVersion>,
        settings: FooterCacheSettings,
    ) -> Option<Self> {
        let version = version?;
        let cacheable = settings.max_bytes > 0
            && (version.immutable || (settings.mutable_files && version.last_modified.is_some()));
        cacheable.then(|| Self {
            config: config.clone(),
            path: path.to_string(),
            size: version.size,
            last_modified: if version.immutable {
                None
            } else {
                version.last_modified
            },
        })
    }
}

struct Entry {
    metadata: Arc<ParquetMetaData>,
    bytes: usize,
}

struct FooterCache {
    entries: LruCache<FooterKey, Entry>,
    bytes: usize,
}

impl FooterCache {
    fn new() -> Self {
        Self {
            entries: LruCache::unbounded(),
            bytes: 0,
        }
    }

    fn get(&mut self, key: &FooterKey) -> Option<Arc<ParquetMetaData>> {
        self.entries.get(key).map(|entry| entry.metadata.clone())
    }

    fn insert(&mut self, key: FooterKey, metadata: Arc<ParquetMetaData>, max_bytes: usize) {
        let bytes = metadata.memory_size();
        if let Some(old) = self.entries.pop(&key) {
            self.bytes -= old.bytes;
        }
        if bytes > max_bytes {
            self.evict_to(max_bytes);
            return;
        }
        self.evict_to(max_bytes - bytes);
        self.entries.put(key, Entry { metadata, bytes });
        self.bytes += bytes;
    }

    /// Evicts least recently used entries until at most `limit` bytes remain.
    fn evict_to(&mut self, limit: usize) {
        while self.bytes > limit {
            match self.entries.pop_lru() {
                Some((_, entry)) => self.bytes -= entry.bytes,
                None => break,
            }
        }
    }
}

static FOOTERS: LazyLock<Mutex<FooterCache>> = LazyLock::new(|| Mutex::new(FooterCache::new()));

/// The cached metadata for `key`, marking it most recently used.
pub(crate) fn get(key: &FooterKey) -> Option<Arc<ParquetMetaData>> {
    FOOTERS.lock().ok()?.get(key)
}

/// Caches `metadata` under `key`, within `max_bytes` in total. Metadata
/// larger than the whole bound is not cached.
pub(crate) fn insert(key: FooterKey, metadata: Arc<ParquetMetaData>, max_bytes: usize) {
    if let Ok(mut cache) = FOOTERS.lock() {
        cache.insert(key, metadata, max_bytes);
    }
}

#[cfg(test)]
mod tests {
    use parquet::{
        basic::Type as PhysicalType,
        file::metadata::{FileMetaData, ParquetMetaData},
        schema::types::{SchemaDescriptor, Type},
    };

    use super::*;

    fn metadata(num_rows: i64) -> Arc<ParquetMetaData> {
        let leaf = Type::primitive_type_builder("k", PhysicalType::INT64)
            .build()
            .unwrap();
        let root = Type::group_type_builder("schema")
            .with_fields(vec![Arc::new(leaf)])
            .build()
            .unwrap();
        let schema = Arc::new(SchemaDescriptor::new(Arc::new(root)));
        Arc::new(ParquetMetaData::new(
            FileMetaData::new(1, num_rows, None, None, schema, None),
            vec![],
        ))
    }

    fn version(immutable: bool, last_modified: Option<i64>) -> Option<FileVersion> {
        Some(FileVersion {
            size: 100,
            last_modified,
            immutable,
        })
    }

    const ON: FooterCacheSettings = FooterCacheSettings {
        max_bytes: 1 << 20,
        mutable_files: false,
    };

    #[test]
    fn which_reads_may_use_the_cache() {
        let config = Arc::new(IOConfig::default());
        let with_mutable = FooterCacheSettings {
            mutable_files: true,
            ..ON
        };
        let off = FooterCacheSettings {
            max_bytes: 0,
            mutable_files: true,
        };
        let cases = [
            ("immutable", version(true, None), ON, true),
            ("version unknown", None, with_mutable, false),
            ("mutable, not cached", version(false, Some(1)), ON, false),
            (
                "mutable, cached",
                version(false, Some(1)),
                with_mutable,
                true,
            ),
            (
                "mutable, no last-modified",
                version(false, None),
                with_mutable,
                false,
            ),
            ("cache off", version(true, None), off, false),
        ];
        for (name, version, settings, expected) in cases {
            assert_eq!(
                FooterKey::for_read(&config, "s3://b/f", version, settings).is_some(),
                expected,
                "{name}"
            );
        }
    }

    #[test]
    fn a_key_names_the_configuration_and_the_files_version() {
        let config = Arc::new(IOConfig::default());
        let mut other = IOConfig::default();
        other.s3.key_id = Some("another".to_string());
        let other = Arc::new(other);
        let mutable = FooterCacheSettings {
            mutable_files: true,
            ..ON
        };
        let key = |config: &Arc<IOConfig>, size: u64, last_modified: Option<i64>| {
            FooterKey::for_read(
                config,
                "s3://b/f",
                Some(FileVersion {
                    size,
                    last_modified,
                    immutable: false,
                }),
                mutable,
            )
            .unwrap()
        };

        assert_ne!(key(&config, 100, Some(1)), key(&other, 100, Some(1)));
        assert_ne!(key(&config, 100, Some(1)), key(&config, 101, Some(1)));
        assert_ne!(key(&config, 100, Some(1)), key(&config, 100, Some(2)));
    }

    #[test]
    fn the_least_recently_used_entries_leave_first_within_the_bound() {
        let config = Arc::new(IOConfig::default());
        let key = |path: &str| FooterKey::for_read(&config, path, version(true, None), ON).unwrap();
        let entry_bytes = metadata(1).memory_size();
        let mut cache = FooterCache::new();

        cache.insert(key("a"), metadata(1), 2 * entry_bytes);
        cache.insert(key("b"), metadata(2), 2 * entry_bytes);
        assert!(cache.get(&key("a")).is_some());
        cache.insert(key("c"), metadata(3), 2 * entry_bytes);

        assert!(cache.get(&key("a")).is_some());
        assert!(cache.get(&key("b")).is_none());
        assert_eq!(cache.get(&key("c")).unwrap().file_metadata().num_rows(), 3);
        assert_eq!(cache.bytes, 2 * entry_bytes);
    }

    #[test]
    fn metadata_larger_than_the_bound_is_not_cached() {
        let config = Arc::new(IOConfig::default());
        let key = FooterKey::for_read(&config, "a", version(true, None), ON).unwrap();
        let mut cache = FooterCache::new();

        cache.insert(key.clone(), metadata(1), metadata(1).memory_size() - 1);

        assert!(cache.get(&key).is_none());
        assert_eq!(cache.bytes, 0);
    }

    #[test]
    fn replacing_an_entry_counts_its_bytes_once() {
        let config = Arc::new(IOConfig::default());
        let key = FooterKey::for_read(&config, "a", version(true, None), ON).unwrap();
        let mut cache = FooterCache::new();

        cache.insert(key.clone(), metadata(1), 1 << 20);
        cache.insert(key.clone(), metadata(2), 1 << 20);

        assert_eq!(cache.bytes, metadata(2).memory_size());
        assert_eq!(cache.get(&key).unwrap().file_metadata().num_rows(), 2);
    }
}
