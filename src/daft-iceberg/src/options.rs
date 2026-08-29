use serde::{Deserialize, Serialize};

use crate::errors::IcebergRewriteError;

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

pub const DEFAULT_TARGET_FILE_SIZE_BYTES: u64 = 512 * MIB;
pub const MIN_TARGET_FILE_SIZE_BYTES: u64 = MIB;
pub const MAX_TARGET_FILE_SIZE_BYTES: u64 = 5 * GIB;
pub const DEFAULT_MIN_INPUT_FILES: u32 = 5;
pub const DEFAULT_MAX_GROUP_BYTES: u64 = 100 * GIB;
pub const DEFAULT_DELETE_FILE_THRESHOLD: u32 = u32::MAX;
/// Deleted fraction of a file at which it is rewritten on that ground alone.
pub const DEFAULT_DELETE_RATIO_THRESHOLD: f64 = 0.3;
/// Per-file open cost allowed for when sizing a split.
pub const SPLIT_OVERHEAD: u64 = 5 * 1024;
pub const DEFAULT_MAX_COMMITS: u32 = 10;
pub const DEFAULT_MAX_CONCURRENT: u32 = 5;
pub const DEFAULT_ZORDER_VAR_LEN_CONTRIBUTION: u32 = 8;
// Per-row interleaved key cap. Primitive columns contribute 8 bytes each and
// string columns contribute up to `var_length_contribution`; 4096 is generous
// enough for ~hundreds of columns and keeps the per-row allocation bounded.
pub const DEFAULT_ZORDER_MAX_OUTPUT_SIZE: u64 = 4096;
pub const MAX_ZORDER_MAX_OUTPUT_SIZE: u64 = MIB;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobOrder {
    BytesAsc,
    BytesDesc,
    FilesAsc,
    FilesDesc,
    None,
}

impl JobOrder {
    pub fn parse(s: &str) -> Result<Self, IcebergRewriteError> {
        match s {
            "bytes-asc" => Ok(Self::BytesAsc),
            "bytes-desc" => Ok(Self::BytesDesc),
            "files-asc" => Ok(Self::FilesAsc),
            "files-desc" => Ok(Self::FilesDesc),
            "none" => Ok(Self::None),
            other => Err(IcebergRewriteError::InvalidOption {
                name: "rewrite-job-order".into(),
                reason: format!(
                    "expected one of bytes-asc|bytes-desc|files-asc|files-desc|none, got `{other}`"
                ),
            }),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewriteOptions {
    pub target_file_size_bytes: u64,
    pub min_input_files: u32,
    pub max_file_group_size_bytes: u64,
    pub delete_file_threshold: u32,
    pub delete_ratio_threshold: f64,
    pub rewrite_all: bool,
    pub partial_progress_enabled: bool,
    pub partial_progress_max_commits: u32,
    pub partial_progress_max_failed_commits: Option<u32>,
    pub max_concurrent_file_group_rewrites: u32,
    pub output_spec_id: Option<i32>,
    pub use_starting_sequence_number: bool,
    pub remove_dangling_deletes: bool,
    pub max_files_to_rewrite: Option<u32>,
    pub min_file_size_bytes: Option<u64>,
    pub max_file_size_bytes: Option<u64>,
    pub job_order: JobOrder,
    pub compression_factor: f64,
    pub zorder_max_output_size: u64,
    pub zorder_var_length_contribution: u32,
    /// Number of sorted output partitions per target file for the sort and
    /// z-order strategies. Higher values produce more, smaller, contiguously
    /// ordered files; `1` produces one file per target size.
    pub shuffle_partitions_per_file: u32,
}

impl Default for RewriteOptions {
    fn default() -> Self {
        Self {
            target_file_size_bytes: DEFAULT_TARGET_FILE_SIZE_BYTES,
            min_input_files: DEFAULT_MIN_INPUT_FILES,
            max_file_group_size_bytes: DEFAULT_MAX_GROUP_BYTES,
            delete_file_threshold: DEFAULT_DELETE_FILE_THRESHOLD,
            delete_ratio_threshold: DEFAULT_DELETE_RATIO_THRESHOLD,
            rewrite_all: false,
            partial_progress_enabled: false,
            partial_progress_max_commits: DEFAULT_MAX_COMMITS,
            partial_progress_max_failed_commits: None,
            max_concurrent_file_group_rewrites: DEFAULT_MAX_CONCURRENT,
            output_spec_id: None,
            use_starting_sequence_number: true,
            remove_dangling_deletes: false,
            max_files_to_rewrite: None,
            min_file_size_bytes: None,
            max_file_size_bytes: None,
            job_order: JobOrder::BytesDesc,
            compression_factor: 1.0,
            zorder_max_output_size: DEFAULT_ZORDER_MAX_OUTPUT_SIZE,
            zorder_var_length_contribution: DEFAULT_ZORDER_VAR_LEN_CONTRIBUTION,
            shuffle_partitions_per_file: 1,
        }
    }
}

impl RewriteOptions {
    /// Lower size threshold for rewrite eligibility. Defaults to 75% of target.
    pub fn effective_min_file_size_bytes(&self) -> u64 {
        match self.min_file_size_bytes {
            Some(v) => v,
            None => (self.target_file_size_bytes as f64 * 0.75) as u64,
        }
    }

    /// Upper size threshold for rewrite eligibility. Defaults to 180% of target.
    pub fn effective_max_file_size_bytes(&self) -> u64 {
        match self.max_file_size_bytes {
            Some(v) => v,
            None => (self.target_file_size_bytes as f64 * 1.80) as u64,
        }
    }

    /// Upper size a file may reach while writing. Halfway between target and
    /// max, so an uneven remainder is absorbed rather than left undersized.
    #[must_use]
    pub fn write_max_file_size_bytes(&self) -> u64 {
        let target = self.target_file_size_bytes;
        let max = self.effective_max_file_size_bytes();
        target + (max.saturating_sub(target)) / 2
    }

    /// Number of files a group of `input_bytes` is written as. Rounds down when
    /// spreading the remainder keeps the average within 10% of target, else up.
    #[must_use]
    pub fn expected_output_files(&self, input_bytes: u64) -> u64 {
        let target = self.target_file_size_bytes;
        if input_bytes < target {
            return 1;
        }
        let with_remainder = input_bytes.div_ceil(target);
        let without_remainder = input_bytes / target;
        if input_bytes % target > self.effective_min_file_size_bytes() {
            return with_remainder;
        }
        let average = input_bytes / without_remainder;
        let ceiling = (1.1 * target as f64).min(self.write_max_file_size_bytes() as f64);
        if (average as f64) < ceiling {
            without_remainder
        } else {
            with_remainder
        }
    }

    /// How much input each output file is read from. Floored at the target and
    /// capped at [`Self::write_max_file_size_bytes`].
    #[must_use]
    pub fn input_split_size(&self, input_bytes: u64) -> u64 {
        let estimated = input_bytes / self.expected_output_files(input_bytes) + SPLIT_OVERHEAD;
        if estimated < self.target_file_size_bytes {
            return self.target_file_size_bytes;
        }
        estimated.min(self.write_max_file_size_bytes())
    }

    /// Failed-commit budget under partial-progress. Defaults to `partial_progress_max_commits`.
    pub fn effective_max_failed_commits(&self) -> u32 {
        self.partial_progress_max_failed_commits
            .unwrap_or(self.partial_progress_max_commits)
    }
}

impl RewriteOptions {
    pub fn validate(&self) -> Result<(), IcebergRewriteError> {
        let invalid = |name: &str, reason: String| IcebergRewriteError::InvalidOption {
            name: name.into(),
            reason,
        };

        if !(MIN_TARGET_FILE_SIZE_BYTES..=MAX_TARGET_FILE_SIZE_BYTES)
            .contains(&self.target_file_size_bytes)
        {
            return Err(invalid(
                "target-file-size-bytes",
                format!(
                    "must be in [{MIN_TARGET_FILE_SIZE_BYTES}, {MAX_TARGET_FILE_SIZE_BYTES}], got {}",
                    self.target_file_size_bytes
                ),
            ));
        }
        if self.min_input_files == 0 {
            return Err(invalid(
                "min-input-files",
                format!("must be > 0, got {}", self.min_input_files),
            ));
        }
        if self.max_file_group_size_bytes == 0 {
            return Err(invalid("max-file-group-size-bytes", "must be > 0".into()));
        }
        if self.partial_progress_enabled && self.partial_progress_max_commits == 0 {
            return Err(invalid(
                "partial-progress.max-commits",
                "must be >= 1 when partial-progress.enabled = true".into(),
            ));
        }
        if self.max_concurrent_file_group_rewrites == 0 {
            return Err(invalid(
                "max-concurrent-file-group-rewrites",
                "must be >= 1".into(),
            ));
        }
        if self.shuffle_partitions_per_file == 0 {
            return Err(invalid(
                "shuffle-partitions-per-file",
                "must be >= 1".into(),
            ));
        }
        if !(self.delete_ratio_threshold > 0.0 && self.delete_ratio_threshold <= 1.0) {
            return Err(invalid(
                "delete-ratio-threshold",
                format!("must be > 0 and <= 1, got {}", self.delete_ratio_threshold),
            ));
        }
        if !(self.compression_factor > 0.0 && self.compression_factor.is_finite()) {
            return Err(invalid(
                "compression-factor",
                format!("must be > 0 and finite, got {}", self.compression_factor),
            ));
        }
        if !(8..=MAX_ZORDER_MAX_OUTPUT_SIZE).contains(&self.zorder_max_output_size) {
            return Err(invalid(
                "max-output-size",
                format!(
                    "must be in [8, {MAX_ZORDER_MAX_OUTPUT_SIZE}] (per-row key cap), got {}",
                    self.zorder_max_output_size
                ),
            ));
        }
        if !(1..=64).contains(&self.zorder_var_length_contribution) {
            return Err(invalid(
                "var-length-contribution",
                format!(
                    "must be in [1, 64], got {}",
                    self.zorder_var_length_contribution
                ),
            ));
        }
        let lower = self.effective_min_file_size_bytes();
        let upper = self.effective_max_file_size_bytes();
        if lower >= self.target_file_size_bytes {
            return Err(invalid(
                "min-file-size-bytes",
                format!(
                    "must be < target-file-size-bytes ({}), got {}",
                    self.target_file_size_bytes, lower
                ),
            ));
        }
        if upper <= self.target_file_size_bytes {
            return Err(invalid(
                "max-file-size-bytes",
                format!(
                    "must be > target-file-size-bytes ({}), got {}",
                    self.target_file_size_bytes, upper
                ),
            ));
        }
        if let Some(cap) = self.max_files_to_rewrite
            && cap == 0
        {
            return Err(invalid(
                "max-files-to-rewrite",
                "must be >= 1 when set".into(),
            ));
        }
        if self.partial_progress_enabled
            && let Some(mfc) = self.partial_progress_max_failed_commits
            && mfc > self.partial_progress_max_commits
        {
            return Err(invalid(
                "partial-progress.max-failed-commits",
                format!(
                    "must be <= partial-progress.max-commits ({}), got {}",
                    self.partial_progress_max_commits, mfc
                ),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SortDirection {
    Asc,
    Desc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NullOrder {
    NullsFirst,
    NullsLast,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortColumn {
    pub name: String,
    pub direction: SortDirection,
    pub null_order: NullOrder,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZOrderKey {
    pub columns: Vec<String>,
    pub var_length_contribution: u32,
    pub max_output_size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Strategy {
    BinPack,
    Sort { columns: Vec<SortColumn> },
    ZOrder(ZOrderKey),
}

#[cfg(test)]
mod tests {

    /// Cross-checked against Iceberg's `SizeBasedFileRewritePlanner`.
    #[test]
    fn size_arithmetic_matches_the_reference() {
        // (target, input bytes, write max, expected output files, input split size)
        let cases: &[(u64, u64, u64, u64, u64)] = &[
            (2097152, 1048576, 2936012, 1, 2097152),
            (2097152, 1887436, 2936012, 1, 2097152),
            (2097152, 2097152, 2936012, 1, 2102272),
            (2097152, 3145728, 2936012, 2, 2097152),
            (2097152, 4194304, 2936012, 2, 2102272),
            (2097152, 5662310, 2936012, 3, 2097152),
            (2097152, 6291456, 2936012, 3, 2102272),
            (2097152, 11534336, 2936012, 5, 2311987),
            (2097152, 20971520, 2936012, 10, 2102272),
            (2097152, 36280729, 2936012, 17, 2139280),
            (8388608, 4194304, 11744051, 1, 8388608),
            (8388608, 7549747, 11744051, 1, 8388608),
            (8388608, 8388608, 11744051, 1, 8393728),
            (8388608, 12582912, 11744051, 2, 8388608),
            (8388608, 16777216, 11744051, 2, 8393728),
            (8388608, 22649241, 11744051, 3, 8388608),
            (8388608, 25165824, 11744051, 3, 8393728),
            (8388608, 46137344, 11744051, 5, 9232588),
            (8388608, 83886080, 11744051, 10, 8393728),
            (8388608, 145122918, 11744051, 17, 8541762),
            (67108864, 33554432, 93952409, 1, 67108864),
            (67108864, 60397977, 93952409, 1, 67108864),
            (67108864, 67108864, 93952409, 1, 67113984),
            (67108864, 100663296, 93952409, 2, 67108864),
            (67108864, 134217728, 93952409, 2, 67113984),
            (67108864, 181193932, 93952409, 3, 67108864),
            (67108864, 201326592, 93952409, 3, 67113984),
            (67108864, 369098752, 93952409, 5, 73824870),
            (67108864, 671088640, 93952409, 10, 67113984),
            (67108864, 1160983347, 93952409, 17, 68298258),
            (536870912, 268435456, 751619276, 1, 536870912),
            (536870912, 483183820, 751619276, 1, 536870912),
            (536870912, 536870912, 751619276, 1, 536876032),
            (536870912, 805306368, 751619276, 2, 536870912),
            (536870912, 1073741824, 751619276, 2, 536876032),
            (536870912, 1449551462, 751619276, 3, 536870912),
            (536870912, 1610612736, 751619276, 3, 536876032),
            (536870912, 2952790016, 751619276, 5, 590563123),
            (536870912, 5368709120, 751619276, 10, 536876032),
            (536870912, 9287866777, 751619276, 17, 546350224),
        ];
        for &(target, input_bytes, write_max, files, split) in cases {
            let o = RewriteOptions {
                target_file_size_bytes: target,
                ..RewriteOptions::default()
            };
            assert_eq!(
                o.write_max_file_size_bytes(),
                write_max,
                "write max for {target}"
            );
            assert_eq!(
                o.expected_output_files(input_bytes),
                files,
                "expected output files for {input_bytes} at target {target}"
            );
            assert_eq!(
                o.input_split_size(input_bytes),
                split,
                "input split size for {input_bytes} at target {target}"
            );
        }
    }

    use super::*;

    #[test]
    fn default_options_validate() {
        RewriteOptions::default().validate().unwrap();
    }

    #[test]
    fn rejects_undersized_target() {
        let o = RewriteOptions {
            target_file_size_bytes: 1024,
            ..Default::default()
        };
        assert!(o.validate().is_err());
    }

    #[test]
    fn accepts_min_input_files_of_one() {
        let o = RewriteOptions {
            min_input_files: 1,
            ..Default::default()
        };
        assert!(
            o.validate().is_ok(),
            "the reference requires only that it be positive"
        );
    }

    #[test]
    fn rejects_min_input_files_of_zero() {
        let o = RewriteOptions {
            min_input_files: 0,
            ..Default::default()
        };
        assert!(o.validate().is_err());
    }

    #[test]
    fn accepts_a_group_cap_below_the_target() {
        let o = RewriteOptions {
            target_file_size_bytes: 128 * 1024 * 1024,
            max_file_group_size_bytes: 4 * 1024 * 1024,
            ..Default::default()
        };
        assert!(
            o.validate().is_ok(),
            "a small cap bounds a group rather than being invalid"
        );
    }

    #[test]
    fn rejects_a_group_cap_of_zero() {
        let o = RewriteOptions {
            max_file_group_size_bytes: 0,
            ..Default::default()
        };
        assert!(o.validate().is_err());
    }

    #[test]
    fn job_order_parse_round_trip() {
        for s in ["bytes-asc", "bytes-desc", "files-asc", "files-desc", "none"] {
            JobOrder::parse(s).unwrap();
        }
        assert!(JobOrder::parse("garbage").is_err());
    }
}
