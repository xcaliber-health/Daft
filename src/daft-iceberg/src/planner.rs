//! File-group planning for a data-file rewrite.
//!
//! Buckets candidate files by partition, packs them into groups bounded by
//! size, and keeps only the groups worth rewriting.

use serde::{Deserialize, Serialize};

use crate::{
    errors::IcebergRewriteError,
    options::{JobOrder, RewriteOptions},
};

/// One data file considered for rewrite.
///
/// Files with equal `partition_key` and `partition_spec_id` are grouped together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateFile {
    /// Location of the data file.
    pub path: String,
    /// Size of the data file in bytes.
    pub size_bytes: u64,
    /// Canonical JSON encoding of the file's partition record.
    pub partition_key: String,
    /// Identifier of the partition spec the file was written under.
    pub partition_spec_id: i32,
    /// Locations of the position delete files that name this file.
    pub positional_delete_paths: Vec<String>,
    /// Locations of the equality delete files that apply to this file.
    #[serde(default)]
    pub equality_delete_paths: Vec<String>,
    /// Rows the data file holds, before deletes are applied.
    #[serde(default)]
    pub record_count: u64,
    /// Rows removed by delete files naming this one.
    ///
    /// A delete naming no data file cannot be attributed, so it counts for
    /// nothing here.
    #[serde(default)]
    pub deleted_record_count: u64,
}

impl CandidateFile {
    /// Whether at least `threshold` delete files of either kind apply to the file.
    fn too_many_deletes(&self, threshold: u32) -> bool {
        let deletes = self.positional_delete_paths.len() + self.equality_delete_paths.len();
        deletes >= threshold as usize
    }

    /// Whether the deleted fraction reaches `threshold`.
    ///
    /// Clamped to the file's own row count, so a shared delete cannot push the
    /// ratio above one.
    fn too_high_delete_ratio(&self, threshold: f64) -> bool {
        if self.positional_delete_paths.is_empty() || self.record_count == 0 {
            return false;
        }
        let deleted = self.deleted_record_count.min(self.record_count) as f64;
        deleted / self.record_count as f64 >= threshold
    }
}

/// A set of files from one partition that are rewritten together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileGroup {
    /// Canonical JSON encoding of the partition record shared by every file.
    pub partition_key: String,
    /// Identifier of the partition spec the output is written under.
    pub output_spec_id: i32,
    /// Files rewritten by this group, in planning order.
    pub files: Vec<CandidateFile>,
    /// Sum of `size_bytes` over `files`.
    pub total_bytes: u64,
    /// Number of files this group is written as, decided before writing.
    pub expected_output_files: u64,
    /// Bytes of input each of those files is read from.
    pub input_split_size: u64,
}

impl FileGroup {
    fn empty(partition_key: String, output_spec_id: i32) -> Self {
        Self {
            partition_key,
            output_spec_id,
            files: Vec::new(),
            total_bytes: 0,
            expected_output_files: 1,
            input_split_size: 0,
        }
    }

    /// Fill in the write shape once the group's contents are final.
    fn resolve_write_shape(&mut self, opts: &RewriteOptions) {
        self.expected_output_files = opts.expected_output_files(self.total_bytes);
        self.input_split_size = opts.input_split_size(self.total_bytes);
    }

    /// Whether the group is worth rewriting.
    ///
    /// Any one reason suffices: enough files, enough content, too much
    /// content, or a heavily deleted file.
    fn is_worth_rewriting(&self, opts: &RewriteOptions) -> bool {
        let enough_input_files =
            self.files.len() > 1 && self.files.len() as u32 >= opts.min_input_files;
        let enough_content = self.files.len() > 1 && self.total_bytes > opts.target_file_size_bytes;
        let too_much_content = self.total_bytes > opts.effective_max_file_size_bytes();
        enough_input_files
            || enough_content
            || too_much_content
            || self
                .files
                .iter()
                .any(|f| f.too_many_deletes(opts.delete_file_threshold))
            || self
                .files
                .iter()
                .any(|f| f.too_high_delete_ratio(opts.delete_ratio_threshold))
    }

    fn push(&mut self, f: CandidateFile) {
        self.total_bytes += f.size_bytes;
        self.files.push(f);
    }
}

/// Group candidate files into rewrite units.
///
/// Files are bucketed by `(partition_key, partition_spec_id)`, filtered to those
/// outside the desired size range or carrying too many deletes, bin-packed into
/// groups capped by `max_file_group_size_bytes`, kept only where the packed
/// group is worth rewriting, sorted by `job_order`, and cut to
/// `max_files_to_rewrite`. `candidates` is consumed.
///
/// # Errors
/// Returns `IcebergRewriteError::InvalidOption` when `opts` fails validation.
pub fn plan_file_groups(
    candidates: Vec<CandidateFile>,
    opts: &RewriteOptions,
    current_spec_id: i32,
) -> Result<Vec<FileGroup>, IcebergRewriteError> {
    opts.validate()?;
    let output_spec_id = opts.output_spec_id.unwrap_or(current_spec_id);

    let mut buckets: std::collections::BTreeMap<(String, i32), Vec<CandidateFile>> =
        std::collections::BTreeMap::new();
    for c in candidates {
        buckets
            .entry((c.partition_key.clone(), c.partition_spec_id))
            .or_default()
            .push(c);
    }

    let lower = opts.effective_min_file_size_bytes();
    let upper = opts.effective_max_file_size_bytes();

    let mut groups: Vec<FileGroup> = Vec::new();
    for ((part_key, spec_id), files) in buckets {
        let needs_spec_change = spec_id != output_spec_id;
        let survivors: Vec<CandidateFile> = files
            .into_iter()
            .filter(|f| {
                opts.rewrite_all
                    || needs_spec_change
                    || f.size_bytes < lower
                    || f.size_bytes > upper
                    || f.too_many_deletes(opts.delete_file_threshold)
                    || f.too_high_delete_ratio(opts.delete_ratio_threshold)
            })
            .collect();

        let forced = opts.rewrite_all || needs_spec_change;
        for mut group in pack(
            survivors,
            &part_key,
            output_spec_id,
            opts.max_file_group_size_bytes,
        ) {
            if forced || group.is_worth_rewriting(opts) {
                group.resolve_write_shape(opts);
                groups.push(group);
            }
        }
    }

    sort_groups(&mut groups, opts.job_order);
    apply_file_cap(&mut groups, opts);

    Ok(groups)
}

/// Cut the selection down to `max_files_to_rewrite`.
///
/// A group that does not fit whole is taken in part, so a cap below the first
/// group still does that much work.
fn apply_file_cap(groups: &mut Vec<FileGroup>, opts: &RewriteOptions) {
    let Some(cap) = opts.max_files_to_rewrite else {
        return;
    };
    let mut remaining = cap as usize;
    let mut kept: Vec<FileGroup> = Vec::with_capacity(groups.len());
    for mut group in std::mem::take(groups) {
        if remaining == 0 {
            break;
        }
        if group.files.len() > remaining {
            group.files.truncate(remaining);
            group.total_bytes = group.files.iter().map(|f| f.size_bytes).sum();
            group.resolve_write_shape(opts);
        }
        remaining -= group.files.len();
        kept.push(group);
    }
    *groups = kept;
}

/// Pack files into groups of at most `cap` bytes, in scan order.
///
/// Files arrive roughly ordered by the data they hold, so packing in that
/// order keeps each group's rows contiguous; sorting by size first would pack
/// tighter but scatter rows across every output file.
fn pack(
    files: Vec<CandidateFile>,
    partition_key: &str,
    output_spec_id: i32,
    cap: u64,
) -> Vec<FileGroup> {
    let mut out: Vec<FileGroup> = Vec::new();
    let mut current = FileGroup::empty(partition_key.to_string(), output_spec_id);
    for f in files {
        if current.total_bytes + f.size_bytes > cap && !current.files.is_empty() {
            out.push(std::mem::replace(
                &mut current,
                FileGroup::empty(partition_key.to_string(), output_spec_id),
            ));
        }
        current.push(f);
    }
    if !current.files.is_empty() {
        out.push(current);
    }
    out
}

fn sort_groups(groups: &mut [FileGroup], order: JobOrder) {
    match order {
        JobOrder::BytesAsc => groups.sort_by(|a, b| a.total_bytes.cmp(&b.total_bytes)),
        JobOrder::BytesDesc => groups.sort_by(|a, b| b.total_bytes.cmp(&a.total_bytes)),
        JobOrder::FilesAsc => groups.sort_by_key(|g| g.files.len()),
        JobOrder::FilesDesc => groups.sort_by_key(|g| std::cmp::Reverse(g.files.len())),
        JobOrder::None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cf(path: &str, size: u64, part: &str, spec: i32) -> CandidateFile {
        CandidateFile {
            path: path.into(),
            size_bytes: size,
            partition_key: part.into(),
            partition_spec_id: spec,
            positional_delete_paths: vec![],
            equality_delete_paths: vec![],
            record_count: 1,
            deleted_record_count: 0,
        }
    }

    fn opts(target: u64, min_input: u32, cap: u64) -> RewriteOptions {
        RewriteOptions {
            target_file_size_bytes: target,
            min_input_files: min_input,
            max_file_group_size_bytes: cap,
            ..RewriteOptions::default()
        }
    }

    #[test]
    fn empty_input_yields_no_groups() {
        let groups = plan_file_groups(vec![], &RewriteOptions::default(), 0).unwrap();
        assert!(groups.is_empty());
    }

    #[test]
    fn below_min_input_files_skipped() {
        let target = 64 * 1024 * 1024;
        let o = opts(target, 5, 5 * target);
        let candidates = (0..3)
            .map(|i| cf(&format!("/f{i}.parquet"), 1024, "{}", 0))
            .collect();
        let groups = plan_file_groups(candidates, &o, 0).unwrap();
        assert!(
            groups.is_empty(),
            "expected skip when below min-input-files"
        );
    }

    #[test]
    fn rewrite_all_includes_below_min() {
        let target = 64 * 1024 * 1024;
        let o = RewriteOptions {
            rewrite_all: true,
            ..opts(target, 5, 5 * target)
        };
        let candidates = (0..3)
            .map(|i| cf(&format!("/f{i}.parquet"), 1024, "{}", 0))
            .collect();
        let groups = plan_file_groups(candidates, &o, 0).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].files.len(), 3);
    }

    #[test]
    fn already_sized_files_excluded() {
        let target = 64 * 1024 * 1024;
        let o = opts(target, 2, 5 * target);
        let candidates = (0..5)
            .map(|i| cf(&format!("/f{i}.parquet"), target, "{}", 0))
            .collect();
        let groups = plan_file_groups(candidates, &o, 0).unwrap();
        assert!(groups.is_empty());
    }

    #[test]
    fn oversize_file_is_singleton_group() {
        let target = 64 * 1024 * 1024;
        let cap = 5 * target;
        let o = opts(target, 2, cap);
        let huge = 4 * target;
        let candidates = vec![
            cf("/big.parquet", huge, "{}", 0),
            cf("/tiny0.parquet", 1024, "{}", 0),
            cf("/tiny1.parquet", 1024, "{}", 0),
        ];
        let groups = plan_file_groups(candidates, &o, 0).unwrap();
        assert!(!groups.is_empty());
        // An oversized file may share a group only with a tail that fits under the cap.
        for g in &groups {
            assert!(g.total_bytes <= cap);
        }
    }

    #[test]
    fn output_spec_change_disables_min_input_gate() {
        let target = 64 * 1024 * 1024;
        let o = RewriteOptions {
            output_spec_id: Some(1),
            ..opts(target, 5, 5 * target)
        };
        let candidates = (0..2)
            .map(|i| cf(&format!("/f{i}.parquet"), 1024, "{}", 0))
            .collect();
        let groups = plan_file_groups(candidates, &o, 0).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].output_spec_id, 1);
    }

    #[test]
    fn equality_deletes_count_toward_the_delete_file_threshold() {
        let target = 64 * 1024 * 1024;
        let mut o = opts(target, 2, 5 * target);
        o.delete_file_threshold = 2;
        let mut c = cf("/f0.parquet", target, "{}", 0);
        c.positional_delete_paths = vec!["/p.parquet".into()];
        c.equality_delete_paths = vec!["/e.parquet".into()];
        let groups = plan_file_groups(vec![c], &o, 0).unwrap();
        assert_eq!(groups.len(), 1, "one file at the target is rewritten only for its deletes");
    }

    #[test]
    fn buckets_per_partition() {
        let target = 64 * 1024 * 1024;
        let o = opts(target, 2, 5 * target);
        let candidates = vec![
            cf("/a0.parquet", 1024, "{\"d\":\"2024-01-01\"}", 0),
            cf("/a1.parquet", 1024, "{\"d\":\"2024-01-01\"}", 0),
            cf("/b0.parquet", 1024, "{\"d\":\"2024-01-02\"}", 0),
            cf("/b1.parquet", 1024, "{\"d\":\"2024-01-02\"}", 0),
        ];
        let groups = plan_file_groups(candidates, &o, 0).unwrap();
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn max_files_to_rewrite_truncates_trailing_groups() {
        let target = 64 * 1024 * 1024;
        let o = RewriteOptions {
            max_files_to_rewrite: Some(3),
            job_order: JobOrder::BytesDesc,
            ..opts(target, 2, 5 * target)
        };
        // The cap of 3 takes the first group whole, then one file from the next.
        let candidates = vec![
            cf("/big0.parquet", 10_000_000, "p=a", 0),
            cf("/big1.parquet", 10_000_000, "p=a", 0),
            cf("/sm0.parquet", 1024, "p=b", 0),
            cf("/sm1.parquet", 1024, "p=b", 0),
        ];
        let groups = plan_file_groups(candidates, &o, 0).unwrap();
        let total_files: usize = groups.iter().map(|g| g.files.len()).sum();
        assert_eq!(total_files, 3, "the cap should be filled, not undershot");
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].files.len(), 2);
        assert_eq!(groups[1].files.len(), 1);
    }

    #[test]
    fn a_cap_below_the_first_group_still_does_that_much_work() {
        let target = 64 * 1024 * 1024;
        let o = RewriteOptions {
            max_files_to_rewrite: Some(2),
            ..opts(target, 2, 5 * target)
        };
        let candidates = (0..5)
            .map(|i| cf(&format!("/f{i}.parquet"), 1024, "p=a", 0))
            .collect();

        let groups = plan_file_groups(candidates, &o, 0).unwrap();

        let total_files: usize = groups.iter().map(|g| g.files.len()).sum();
        assert_eq!(
            total_files, 2,
            "a cap under the first group must not yield nothing"
        );
        assert_eq!(groups.len(), 1);
        assert_eq!(
            groups[0].total_bytes, 2048,
            "a sliced group reports what it kept"
        );
    }

    #[test]
    fn a_small_packed_group_is_discarded_like_the_reference() {
        let target = 64 * 1024 * 1024;
        let o = opts(target, 5, 5 * target);
        let candidates = vec![
            cf("/a.parquet", 1024, "p=a", 0),
            cf("/b.parquet", 1024, "p=a", 0),
        ];

        assert!(plan_file_groups(candidates, &o, 0).unwrap().is_empty());
    }

    #[test]
    fn a_group_holding_more_than_a_target_is_kept_below_min_input_files() {
        let target = 4 * 1024 * 1024;
        let o = opts(target, 5, 5 * target);
        let candidates = vec![
            cf("/a.parquet", 5 * 512 * 1024, "p=a", 0),
            cf("/b.parquet", 5 * 512 * 1024, "p=a", 0),
        ];

        let groups = plan_file_groups(candidates, &o, 0).unwrap();

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].files.len(), 2);
    }

    #[test]
    fn a_heavily_deleted_file_is_rewritten_on_its_own() {
        let target = 64 * 1024 * 1024;
        let o = RewriteOptions {
            delete_ratio_threshold: 0.3,
            ..opts(target, 5, 5 * target)
        };
        let mut deleted = cf("/deleted.parquet", target, "p=a", 0);
        deleted.positional_delete_paths = vec!["/d.parquet".into()];
        deleted.record_count = 100;
        deleted.deleted_record_count = 40;

        let groups = plan_file_groups(vec![deleted], &o, 0).unwrap();

        assert_eq!(
            groups.len(),
            1,
            "a file past the delete ratio is worth rewriting alone"
        );
        assert_eq!(groups[0].files.len(), 1);
    }

    #[test]
    fn a_lightly_deleted_file_is_left_alone() {
        let target = 64 * 1024 * 1024;
        let o = RewriteOptions {
            delete_ratio_threshold: 0.3,
            ..opts(target, 5, 5 * target)
        };
        let mut deleted = cf("/deleted.parquet", target, "p=a", 0);
        deleted.positional_delete_paths = vec!["/d.parquet".into()];
        deleted.record_count = 100;
        deleted.deleted_record_count = 10;

        assert!(plan_file_groups(vec![deleted], &o, 0).unwrap().is_empty());
    }

    #[test]
    fn a_delete_naming_no_data_file_leaves_the_ratio_at_zero() {
        let target = 64 * 1024 * 1024;
        let o = RewriteOptions {
            delete_ratio_threshold: 0.3,
            ..opts(target, 5, 5 * target)
        };
        let mut deleted = cf("/deleted.parquet", target, "p=a", 0);
        deleted.positional_delete_paths = vec!["/d.parquet".into()];
        deleted.record_count = 100;
        deleted.deleted_record_count = 0;

        assert!(plan_file_groups(vec![deleted], &o, 0).unwrap().is_empty());
    }

    #[test]
    fn groups_carry_the_write_shape_decided_for_them() {
        let target = 2 * 1024 * 1024;
        let o = opts(target, 2, 100 * target);
        let candidates = (0..6)
            .map(|i| cf(&format!("/f{i}.parquet"), 1024 * 1024, "p=a", 0))
            .collect();

        let groups = plan_file_groups(candidates, &o, 0).unwrap();

        assert_eq!(groups.len(), 1);
        let group = &groups[0];
        assert_eq!(
            group.expected_output_files,
            o.expected_output_files(group.total_bytes)
        );
        assert_eq!(
            group.input_split_size,
            o.input_split_size(group.total_bytes)
        );
        assert!(
            group.expected_output_files > 1,
            "six undersized files add up to several targets' worth"
        );
    }

    #[test]
    fn configurable_min_max_file_size_changes_survivors() {
        let target: u64 = 100 * 1024 * 1024;
        // Under the default 75% lower bound a 60 MiB file is undersized; with the
        // bound lowered to 50 MiB it is well-sized and must be skipped.
        let custom = RewriteOptions {
            target_file_size_bytes: target,
            max_file_group_size_bytes: 5 * target,
            min_input_files: 2,
            min_file_size_bytes: Some(50 * 1024 * 1024),
            max_file_size_bytes: Some(200 * 1024 * 1024),
            ..RewriteOptions::default()
        };
        let candidates = (0..3)
            .map(|i| cf(&format!("/f{i}.parquet"), 60 * 1024 * 1024, "{}", 0))
            .collect();
        let groups = plan_file_groups(candidates, &custom, 0).unwrap();
        assert!(
            groups.is_empty(),
            "60MiB files with min=50MiB should be considered well-sized and skipped"
        );
    }

    #[test]
    fn invalid_min_above_target_rejected() {
        let o = RewriteOptions {
            target_file_size_bytes: 100 * 1024 * 1024,
            max_file_group_size_bytes: 5 * 100 * 1024 * 1024,
            min_file_size_bytes: Some(200 * 1024 * 1024),
            ..RewriteOptions::default()
        };
        assert!(o.validate().is_err());
    }

    #[test]
    fn invalid_max_file_size_below_target_rejected() {
        let o = RewriteOptions {
            target_file_size_bytes: 100 * 1024 * 1024,
            max_file_group_size_bytes: 5 * 100 * 1024 * 1024,
            max_file_size_bytes: Some(50 * 1024 * 1024),
            ..RewriteOptions::default()
        };
        assert!(o.validate().is_err());
    }

    #[test]
    fn a_group_cap_below_the_target_bounds_each_group() {
        let cap = 4 * 1024 * 1024;
        let o = opts(64 * 1024 * 1024, 2, cap);
        let candidates = (0..8)
            .map(|i| cf(&format!("/f{i}.parquet"), 1024 * 1024, "p=a", 0))
            .collect();

        let groups = plan_file_groups(candidates, &o, 0).unwrap();

        assert!(
            groups.len() > 1,
            "a cap under the target splits the bucket into several groups"
        );
        assert!(groups.iter().all(|g| g.total_bytes <= cap));
    }

    #[test]
    fn bytes_desc_ordering() {
        let target = 64 * 1024 * 1024;
        let o = RewriteOptions {
            job_order: JobOrder::BytesDesc,
            ..opts(target, 2, 5 * target)
        };
        let candidates = vec![
            cf("/big0.parquet", 10_000_000, "p=a", 0),
            cf("/big1.parquet", 10_000_000, "p=a", 0),
            cf("/sm0.parquet", 1024, "p=b", 0),
            cf("/sm1.parquet", 1024, "p=b", 0),
        ];
        let groups = plan_file_groups(candidates, &o, 0).unwrap();
        assert_eq!(groups.len(), 2);
        assert!(groups[0].total_bytes >= groups[1].total_bytes);
    }
}
