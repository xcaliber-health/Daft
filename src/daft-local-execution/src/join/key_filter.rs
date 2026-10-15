//! Filters on a join's probe-side scan, made from the keys its build side holds.
//!
//! When a hash join's build side is small, the scan feeding its probe side can
//! skip what no build key matches: a file whose partition value holds none of
//! them, a row group whose statistics or bloom filter rule them out, and rows.
//! The scan waits for the build side to finish before it opens a file, and is
//! then handed, for each key, the range and (when few) the values the build
//! side holds.
//!
//! A key filter is redundant with its join: a scan that applies none of it
//! reads what it read before, and the join drops the same rows. It therefore
//! never changes an answer, only how much is read.

use std::{
    collections::{HashMap, HashSet, hash_map::Entry},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use arrow::{array::Array, buffer::BooleanBuffer};
use common_daft_config::DaftExecutionConfig;
use common_error::DaftResult;
use common_treenode::{TreeNode, TreeNodeRecursion};
use daft_core::{lit::Literal, prelude::*};
use daft_dsl::{
    Column, Expr, ExprRef,
    expr::{BoundColumn, bound_expr::BoundExpr},
    lit, resolved_col,
};
use daft_local_plan::{
    Concat, Dedup, Filter, HashAggregate, IntoBatches, LocalPhysicalPlan, LocalPhysicalPlanRef,
    PhysicalScan, Project,
    Sort, SourceId, WindowPartitionAndDynamicFrame, WindowPartitionAndOrderBy,
    WindowPartitionOnly,
};
use daft_logical_plan::{JoinType, stats::StatsState};
use daft_micropartition::MicroPartition;
use daft_recordbatch::RecordBatch;
use daft_scan::{FileFormatConfig, SourceConfig};
use tokio::sync::oneshot;

use crate::pipeline::InputId;

/// Whether a key of this type can filter a scan.
///
/// Floating-point keys are excluded: NaN orders above every number but equals
/// nothing, so a range taken from them is not a sound bound. Nested, binary
/// and extension types are excluded because neither statistics nor bloom
/// filters prune by them.
pub(crate) fn filters_by(dtype: &DataType) -> bool {
    dtype.is_integer()
        || matches!(
            dtype,
            DataType::Utf8 | DataType::Date | DataType::Timestamp(..) | DataType::Decimal128(..)
        )
}

/// What the build side holds of one key.
#[derive(Debug, PartialEq, Eq)]
enum KeyValues {
    /// No build row has a non-null key: no probe row can match.
    Nothing,
    /// Every non-null key lies in `min..=max`; when the build side has at most
    /// the configured number of rows, it is one of `values`; and, when its keys
    /// fit the build-side size bound, its hash is one of `members`.
    Within {
        min: Literal,
        max: Literal,
        values: Option<Vec<Literal>>,
        members: Option<Arc<KeyHashes>>,
    },
}

/// The hashes of a build side's keys, which every matching probe key's hash
/// is among. Two keys may share a hash, so a probe row may be kept that no key
/// matches; the join then drops it, as it drops every row a filter keeps.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct KeyHashes(hashbrown::HashSet<u64>);

impl KeyValues {
    /// Reads one key column of the build side.
    ///
    /// Values are listed only when the build side holds at most `max_values`
    /// rows, so the list never costs more than the build itself to make, as
    /// a hash join's IN filter is bounded by its build rows in DuckDB.
    fn of(keys: &Series, max_values: usize, max_member_bytes: usize) -> DaftResult<Self> {
        let present = match keys.nulls() {
            Some(_) => keys.filter(keys.not_null()?.bool()?)?,
            None => keys.clone(),
        };
        if present.is_empty() {
            return Ok(Self::Nothing);
        }
        let min = Literal::try_from_single_value_series(&present.min(None)?)?;
        let max = Literal::try_from_single_value_series(&present.max(None)?)?;
        let members = if present.size_bytes() <= max_member_bytes {
            let hashes = present.hash(None)?;
            Some(Arc::new(KeyHashes(hashes.as_arrow()?.values().iter().copied().collect())))
        } else {
            None
        };
        let values = if min != max && present.len() <= max_values {
            Some(distinct_values(present)?)
        } else {
            None
        };
        Ok(Self::Within {
            min,
            max,
            values,
            members,
        })
    }

    /// What this key restricts `column` to: nothing at all, or a predicate
    /// that prunes (its range, and its values when listed) and a test that
    /// filters rows (its hashes when held, else its range).
    fn restriction(&self, column: &str) -> KeyRestriction {
        let Self::Within {
            min,
            max,
            values,
            members,
        } = self
        else {
            return KeyRestriction::Nothing;
        };
        let key = resolved_col(column);
        let range = if min == max {
            key.clone().eq(lit(min.clone()))
        } else {
            key.clone()
                .gt_eq(lit(min.clone()))
                .and(key.clone().lt_eq(lit(max.clone())))
        };
        let prune = match values {
            Some(values) => range
                .clone()
                .and(key.is_in(values.iter().cloned().map(lit).collect())),
            None => range.clone(),
        };
        let rows = match members {
            Some(members) => RowTest::Members(members.clone()),
            None => RowTest::Satisfies(range),
        };
        KeyRestriction::Within {
            prune,
            rows: (column.to_string(), rows),
        }
    }
}

/// What one join key restricts a scan column to.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum KeyRestriction {
    /// No row can match: the build side holds no key.
    Nothing,
    /// Rows outside these cannot match.
    Within {
        /// Skips row groups and files whose statistics, bloom filters or
        /// partition values rule every row out.
        prune: ExprRef,
        /// Drops the rows read, of this column, that cannot match.
        rows: (String, RowTest),
    },
}

/// How a row filter tests one key column.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum RowTest {
    /// The key's hash is among the build side's.
    Members(Arc<KeyHashes>),
    /// The predicate holds, for a build side too large to hash.
    Satisfies(ExprRef),
}

impl RowTest {
    /// Whether each row of `batch` passes, null keys never passing.
    fn passes(&self, column: &str, batch: &RecordBatch) -> DaftResult<BooleanBuffer> {
        match self {
            Self::Members(members) => {
                let keys = batch.get_column(batch.schema.get_index(column)?);
                let hashes = keys.hash(None)?;
                let hashes = hashes.as_arrow()?.values();
                let mut passes =
                    BooleanBuffer::collect_bool(hashes.len(), |row| members.0.contains(&hashes[row]));
                if let Some(nulls) = keys.nulls() {
                    passes = &passes & nulls.inner();
                }
                Ok(passes)
            }
            Self::Satisfies(predicate) => {
                let predicate = BoundExpr::try_new(predicate.clone(), &batch.schema)?;
                let holds = batch.eval_expression(&predicate)?;
                let holds = holds.bool()?.as_arrow()?;
                Ok(match holds.nulls() {
                    Some(nulls) => holds.values() & nulls.inner(),
                    None => holds.values().clone(),
                })
            }
        }
    }
}

/// The distinct values of a column with no nulls.
fn distinct_values(column: Series) -> DaftResult<Vec<Literal>> {
    let key = resolved_col(column.name());
    let batch = RecordBatch::from_nonempty_columns(vec![column])?;
    let key = BoundExpr::try_new(key, &batch.schema)?;
    let distinct = batch.dedup(&[key])?;
    let column = distinct.get_column(0);
    (0..column.len())
        .map(|row| Literal::try_from_single_value_series(&column.slice(row, row + 1)?))
        .collect()
}

/// The keys a finished build side holds, one entry per join key.
#[derive(Debug)]
pub(crate) struct KeyFilter {
    keys: Vec<KeyValues>,
}

impl KeyFilter {
    /// Reads the build side's key columns, in join-key order: values are
    /// listed for at most `max_values` build rows, and hashes kept for keys of
    /// at most `max_member_bytes`.
    ///
    /// # Errors
    /// Returns an error if a key column's minimum, maximum or values cannot be
    /// taken as literals.
    pub(crate) fn from_build_keys(
        keys: &[Series],
        max_values: usize,
        max_member_bytes: usize,
    ) -> DaftResult<Self> {
        Ok(Self {
            keys: keys
                .iter()
                .map(|column| KeyValues::of(column, max_values, max_member_bytes))
                .collect::<DaftResult<_>>()?,
        })
    }

    /// What join key `key` restricts `column` to.
    fn restriction(&self, key: usize, column: &str) -> Option<KeyRestriction> {
        self.keys.get(key).map(|values| values.restriction(column))
    }
}

enum Slot {
    /// Scans waiting for this input's filter.
    Awaited(Vec<oneshot::Sender<Option<Arc<KeyFilter>>>>),
    /// The filter, kept until each scan that has yet to ask takes it.
    Published {
        filter: Option<Arc<KeyFilter>>,
        unclaimed: usize,
    },
}

struct Slots {
    by_input: HashMap<InputId, Slot>,
    /// How many scans take each input's filter.
    consumers: usize,
    /// Set once the build side has stopped: nothing more will be published.
    closed: bool,
}

/// Hands a join's key filter, once per input, from its build side to the scans
/// on its probe side.
///
/// A scan waits only while the build side runs. When the build side stops,
/// failed or finished, the bridge closes, and a scan still waiting, or asking
/// later for an input never published, is released with no filter.
pub(crate) struct KeyFilterBridge {
    slots: Mutex<Slots>,
}

impl KeyFilterBridge {
    /// A bridge whose filters are each taken by `consumers` scans.
    pub(crate) fn new(consumers: usize) -> Self {
        Self {
            slots: Mutex::new(Slots {
                by_input: HashMap::new(),
                consumers,
                closed: false,
            }),
        }
    }

    /// Publishes an input's filter; `None` when the build side offers none.
    pub(crate) fn publish(&self, input_id: InputId, filter: Option<Arc<KeyFilter>>) {
        let mut slots = self.slots.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let consumers = slots.consumers;
        let claimed = match slots.by_input.remove(&input_id) {
            Some(Slot::Awaited(waiting)) => {
                let claimed = waiting.len();
                for scan in waiting {
                    let _ = scan.send(filter.clone());
                }
                claimed
            }
            Some(Slot::Published { .. }) | None => 0,
        };
        if claimed < consumers {
            slots.by_input.insert(
                input_id,
                Slot::Published {
                    filter,
                    unclaimed: consumers - claimed,
                },
            );
        }
    }

    /// Waits for an input's filter; `None` when there is none, or the build
    /// side stopped without publishing one.
    pub(crate) async fn wait(&self, input_id: InputId) -> Option<Arc<KeyFilter>> {
        let receiver = {
            let mut slots = self.slots.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let closed = slots.closed;
            match slots.by_input.entry(input_id) {
                Entry::Occupied(mut entry) => match entry.get_mut() {
                    Slot::Published { filter, unclaimed } => {
                        let filter = filter.clone();
                        *unclaimed -= 1;
                        if *unclaimed == 0 {
                            entry.remove();
                        }
                        return filter;
                    }
                    Slot::Awaited(waiting) => {
                        let (sender, receiver) = oneshot::channel();
                        waiting.push(sender);
                        receiver
                    }
                },
                Entry::Vacant(_) if closed => return None,
                Entry::Vacant(entry) => {
                    let (sender, receiver) = oneshot::channel();
                    entry.insert(Slot::Awaited(vec![sender]));
                    receiver
                }
            }
        };
        receiver.await.ok().flatten()
    }

    /// Marks the build side stopped, releasing every scan still waiting.
    pub(crate) fn close(&self) {
        let mut slots = self.slots.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        slots.closed = true;
        slots
            .by_input
            .retain(|_, slot| matches!(slot, Slot::Published { .. }));
    }
}

/// A join's build side, publishing its keys to the scans they filter.
///
/// Dropping it closes the bridge, so no scan waits on a build side that has
/// stopped.
pub(crate) struct KeyFilterPublisher {
    bridge: Arc<KeyFilterBridge>,
    max_values: usize,
    max_member_bytes: usize,
}

impl KeyFilterPublisher {
    /// Publishes on `bridge`, listing a key's values when the build side holds
    /// at most `max_values` rows, and hashing its keys when they take at most
    /// `max_member_bytes`.
    pub(crate) fn new(
        bridge: Arc<KeyFilterBridge>,
        max_values: usize,
        max_member_bytes: usize,
    ) -> Self {
        Self {
            bridge,
            max_values,
            max_member_bytes,
        }
    }

    /// Publishes an input's filter from its finished build side's key columns,
    /// or no filter when there are none. A filter that cannot be made is not
    /// an error of the query: the scans read unfiltered and the join decides.
    pub(crate) fn publish(&self, input_id: InputId, keys: Option<DaftResult<Vec<Series>>>) {
        let filter = keys.and_then(|keys| {
            keys.and_then(|keys| {
                KeyFilter::from_build_keys(&keys, self.max_values, self.max_member_bytes)
            })
                .inspect_err(|error| log::debug!("no key filter for this join: {error}"))
                .ok()
        });
        self.bridge.publish(input_id, filter.map(Arc::new));
    }

    /// Marks the build side stopped, releasing any scan still waiting.
    pub(crate) fn close(&self) {
        self.bridge.close();
    }
}

impl Drop for KeyFilterPublisher {
    fn drop(&mut self) {
        self.close();
    }
}

/// Plans the key filters of one hash join, before its children become
/// pipelines: the publisher for its build side, and the filter each scan on
/// its probe side takes.
///
/// A join filters only when all of these hold:
/// - it drops the probe rows no build key matches: an inner or semi join;
/// - its build side is estimated within `cfg.broadcast_join_size_bytes_threshold`
///   and smaller than its probe side, so waiting for it costs little;
/// - a key compares by plain equality, is of a type [`filters_by`] accepts, and
///   reads a column the probe side carries unchanged from a Parquet scan
///   ([`scans_filtered_by`]) that the build side does not also read.
///
/// Every such scan prunes by the keys; it also filters its rows only when that
/// spares an operator's per-row state ([`ScanTarget::spares_state`]).
pub(crate) struct PlannedKeyFilter {
    /// Publishes the build side's keys; handed to the join.
    pub publisher: KeyFilterPublisher,
    /// The filter each probe-side scan takes, by the scan's source.
    pub scans: Vec<(SourceId, ScanKeyFilter)>,
}

/// The probe-side scans a join's keys filter, if any; see [`PlannedKeyFilter`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_key_filter(
    join_type: JoinType,
    null_equals_null: Option<&[bool]>,
    build_child: &LocalPhysicalPlanRef,
    probe_child: &LocalPhysicalPlan,
    build_on: &[BoundExpr],
    probe_on: &[BoundExpr],
    key_types: &[&DataType],
    cfg: &DaftExecutionConfig,
) -> DaftResult<Option<PlannedKeyFilter>> {
    if !cfg.enable_join_key_filters || !matches!(join_type, JoinType::Inner | JoinType::Semi) {
        return Ok(None);
    }
    let (StatsState::Materialized(build), StatsState::Materialized(probe)) =
        (build_child.get_stats_state(), probe_child.get_stats_state())
    else {
        return Ok(None);
    };
    let build_bytes = build.approx_stats.size_bytes;
    if build_bytes > cfg.broadcast_join_size_bytes_threshold
        || build_bytes >= probe.approx_stats.size_bytes
    {
        return Ok(None);
    }
    let build_scans = scans_read_by(build_child)?;
    let targets = probe_on
        .iter()
        .enumerate()
        .filter(|(key, _)| {
            !null_equals_null.is_some_and(|nulls_equal| nulls_equal.get(*key) == Some(&true))
                && key_types.get(*key).is_some_and(|dtype| filters_by(dtype))
        })
        .filter_map(|(key, probe_key)| Some((key, column_read_by(probe_key)?)))
        .flat_map(|(key, column)| {
            scans_filtered_by(probe_child, column)
                .into_iter()
                .map(move |target| (key, target))
        })
        .filter(|(_, target)| !build_scans.contains(&target.source))
        .collect::<Vec<_>>();
    if targets.is_empty() || build_on.len() != probe_on.len() {
        return Ok(None);
    }
    let bridge = Arc::new(KeyFilterBridge::new(targets.len()));
    Ok(Some(PlannedKeyFilter {
        publisher: KeyFilterPublisher::new(
            bridge.clone(),
            cfg.join_key_filter_max_values,
            cfg.broadcast_join_size_bytes_threshold,
        ),
        scans: targets
            .into_iter()
            .map(|(key, target)| {
                (
                    target.source,
                    ScanKeyFilter::new(bridge.clone(), key, target.column, target.spares_state),
                )
            })
            .collect(),
    }))
}

/// The input column an expression passes through unchanged, if it only reads one.
fn column_read_by(expr: &BoundExpr) -> Option<usize> {
    fn column_of(expr: &Expr) -> Option<usize> {
        match expr {
            Expr::Column(Column::Bound(BoundColumn { index, .. })) => Some(*index),
            Expr::Alias(inner, _) => column_of(inner),
            _ => None,
        }
    }
    column_of(expr.inner())
}

/// The input column holding the values of `plan`'s output column `column`,
/// found by name among `exprs`, each of which must pass a column through.
fn passed_through(
    plan: &LocalPhysicalPlan,
    input: &LocalPhysicalPlan,
    exprs: &[BoundExpr],
    column: usize,
) -> Option<usize> {
    let name = &plan.schema().fields().get(column)?.name;
    exprs
        .iter()
        .filter_map(column_read_by)
        .find(|index| input.schema().fields().get(*index).is_some_and(|field| &field.name == name))
}

/// A Parquet scan column a join key can restrict.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ScanTarget {
    pub source: SourceId,
    pub column: String,
    /// Whether the key reaches the join through a grouping, window or
    /// distinct: an operator holding state for each row it takes, which rows
    /// dropped at the scan spare. Directly below the join, the join's own probe
    /// drops them at the same cost, so only pruning pays there.
    pub spares_state: bool,
}

/// The Parquet scan columns whose values reach `plan`'s output column
/// `column` unchanged, through operators below which keeping only some values
/// of that column keeps the same rows above.
///
/// It passes through renaming projections, filters, sorts, batching, a
/// concatenation's inputs, a distinct on that column, a grouping by it, and a
/// window partitioned by it: the operators DuckDB's and DataFusion's join
/// filters also pass. It stops at a limit, a sample, a join, a computed
/// column, and a scan that limits or aggregates what it reads.
pub(crate) fn scans_filtered_by(plan: &LocalPhysicalPlan, column: usize) -> Vec<ScanTarget> {
    scans_reached(plan, column, false)
}

#[recursive::recursive]
fn scans_reached(plan: &LocalPhysicalPlan, column: usize, spares_state: bool) -> Vec<ScanTarget> {
    match plan {
        LocalPhysicalPlan::PhysicalScan(PhysicalScan {
            source_id,
            source_config,
            pushdowns,
            schema,
            ..
        }) => {
            let reads_parquet = source_config.as_deref().is_some_and(|config| {
                matches!(config, SourceConfig::File(FileFormatConfig::Parquet(_)))
            });
            match schema.fields().get(column) {
                Some(field)
                    if reads_parquet
                        && pushdowns.limit.is_none()
                        && pushdowns.aggregation.is_none() =>
                {
                    vec![ScanTarget {
                        source: *source_id,
                        column: field.name.to_string(),
                        spares_state,
                    }]
                }
                _ => Vec::new(),
            }
        }
        LocalPhysicalPlan::Project(Project { input, projection, .. }) => projection
            .get(column)
            .and_then(column_read_by)
            .map(|column| scans_reached(input, column, spares_state))
            .unwrap_or_default(),
        LocalPhysicalPlan::Filter(Filter { input, .. })
        | LocalPhysicalPlan::Sort(Sort { input, .. })
        | LocalPhysicalPlan::IntoBatches(IntoBatches { input, .. }) => {
            scans_reached(input, column, spares_state)
        }
        LocalPhysicalPlan::Concat(Concat { input, other, .. }) => {
            let mut scans = scans_reached(input, column, spares_state);
            scans.extend(scans_reached(other, column, spares_state));
            scans
        }
        LocalPhysicalPlan::Dedup(Dedup { input, columns, .. }) => {
            passed_through(plan, input, columns, column)
                .map(|column| scans_reached(input, column, true))
                .unwrap_or_default()
        }
        LocalPhysicalPlan::HashAggregate(HashAggregate { input, group_by, .. }) => {
            passed_through(plan, input, group_by, column)
                .map(|column| scans_reached(input, column, true))
                .unwrap_or_default()
        }
        LocalPhysicalPlan::WindowPartitionOnly(WindowPartitionOnly {
            input, partition_by, ..
        })
        | LocalPhysicalPlan::WindowPartitionAndOrderBy(WindowPartitionAndOrderBy {
            input,
            partition_by,
            ..
        })
        | LocalPhysicalPlan::WindowPartitionAndDynamicFrame(WindowPartitionAndDynamicFrame {
            input,
            partition_by,
            ..
        }) => passed_through(plan, input, partition_by, column)
            .map(|column| scans_reached(input, column, true))
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// Every scan `plan` reads, by source.
fn scans_read_by(plan: &LocalPhysicalPlanRef) -> DaftResult<HashSet<SourceId>> {
    let mut sources = HashSet::new();
    plan.apply(|node| {
        if let LocalPhysicalPlan::PhysicalScan(scan) = node.as_ref() {
            sources.insert(scan.source_id);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(sources)
}

/// One join key filtering one scan column.
pub(crate) struct ScanKeyFilter {
    bridge: Arc<KeyFilterBridge>,
    key: usize,
    column: String,
    filters_rows: bool,
}

impl ScanKeyFilter {
    /// Join key `key`, published on `bridge`, restricting the scan's `column`:
    /// pruning what it reads, and filtering the rows read when `filters_rows`.
    pub(crate) fn new(
        bridge: Arc<KeyFilterBridge>,
        key: usize,
        column: String,
        filters_rows: bool,
    ) -> Self {
        Self {
            bridge,
            key,
            column,
            filters_rows,
        }
    }

    /// The scan column this filter restricts.
    pub(crate) fn column(&self) -> &str {
        &self.column
    }

    /// Whether this filter drops the rows read, besides pruning; see
    /// [`ScanTarget::spares_state`].
    pub(crate) fn filters_rows(&self) -> bool {
        self.filters_rows
    }

    /// Waits for the build side, then returns what it restricts this column
    /// to for `input_id`, if anything.
    pub(crate) async fn restriction(&self, input_id: InputId) -> Option<KeyRestriction> {
        self.bridge
            .wait(input_id)
            .await
            .and_then(|filter| filter.restriction(self.key, &self.column))
    }
}

/// How many rows a row filter samples before deciding whether to keep filtering.
const SAMPLED_ROWS: usize = 64 * 1024;

/// A row filter that stops once it is seen to keep most rows.
///
/// Filtering copies the rows it keeps, so a filter keeping most of them costs
/// more than it saves; the join drops the same rows anyway. After sampling
/// [`SAMPLED_ROWS`], the filter stops if it kept more than half: the point
/// below which a restriction pays its copy in this engine (the threshold
/// filters are moved below a grouping by), and the bound DuckDB's runtime
/// filters pause at for their probabilistic forms. Shared by every task of one
/// scan and input, so it is decided once for them.
pub(crate) struct RowKeyFilter {
    tests: Vec<(String, RowTest)>,
    seen: AtomicUsize,
    kept: AtomicUsize,
    stopped: AtomicBool,
}

impl RowKeyFilter {
    /// A filter keeping the rows whose every named column passes its test.
    pub(crate) fn new(tests: Vec<(String, RowTest)>) -> Self {
        Self {
            tests,
            seen: AtomicUsize::new(0),
            kept: AtomicUsize::new(0),
            stopped: AtomicBool::new(false),
        }
    }

    /// The rows of `partition` the filter keeps, or all of them once stopped.
    ///
    /// # Errors
    /// Returns an error if a test cannot be evaluated over `partition`.
    pub(crate) fn apply(&self, partition: MicroPartition) -> DaftResult<MicroPartition> {
        if self.stopped.load(Ordering::Relaxed) || partition.is_empty() {
            return Ok(partition);
        }
        let batches = partition
            .record_batches()
            .iter()
            .map(|batch| batch.mask_filter(&self.mask(batch)?))
            .collect::<DaftResult<Vec<_>>>()?;
        let kept = MicroPartition::new_loaded(partition.schema(), Arc::new(batches), None);
        let seen = self.seen.fetch_add(partition.len(), Ordering::Relaxed) + partition.len();
        let held = self.kept.fetch_add(kept.len(), Ordering::Relaxed) + kept.len();
        if seen >= SAMPLED_ROWS && held * 2 > seen {
            self.stopped.store(true, Ordering::Relaxed);
        }
        Ok(kept)
    }

    /// Which rows of `batch` pass every test.
    fn mask(&self, batch: &RecordBatch) -> DaftResult<Series> {
        let mut passes = BooleanBuffer::new_set(batch.len());
        for (column, test) in &self.tests {
            passes = &passes & &test.passes(column, batch)?;
        }
        let mask = arrow::array::BooleanArray::new(passes, None);
        Ok(BooleanArray::from_arrow(Field::new("mask", DataType::Boolean), Arc::new(mask))?.into_series())
    }
}

#[cfg(test)]
mod tests {
    use daft_core::prelude::{Int64Array, Utf8Array};

    use super::*;

    /// Large enough that every test's keys are hashed.
    const MEMBER_BYTES: usize = 1 << 20;

    fn ints(values: &[Option<i64>]) -> Series {
        Int64Array::from_iter(Field::new("k", DataType::Int64), values.iter().copied()).into_series()
    }

    fn batch(values: &[Option<i64>]) -> RecordBatch {
        RecordBatch::from_nonempty_columns(vec![
            Int64Array::from_iter(Field::new("c", DataType::Int64), values.iter().copied())
                .into_series(),
        ])
        .unwrap()
    }

    fn partition(values: impl IntoIterator<Item = i64>) -> MicroPartition {
        let batch = batch(&values.into_iter().map(Some).collect::<Vec<_>>());
        MicroPartition::new_loaded(batch.schema.clone(), Arc::new(vec![batch]), None)
    }

    fn restriction(keys: &[Option<i64>], max_values: usize, member_bytes: usize) -> KeyRestriction {
        KeyFilter::from_build_keys(&[ints(keys)], max_values, member_bytes)
            .unwrap()
            .restriction(0, "c")
            .unwrap()
    }

    fn passing(test: &RowTest, values: &[Option<i64>]) -> Vec<bool> {
        test.passes("c", &batch(values)).unwrap().iter().collect()
    }

    #[test]
    fn keys_give_their_range_and_values() {
        let keys = ints(&[Some(7), Some(3), None, Some(7), Some(5)]);
        let KeyValues::Within {
            min, max, values, ..
        } = KeyValues::of(&keys, 10, MEMBER_BYTES).unwrap()
        else {
            panic!("keys were present");
        };
        let mut values = values.unwrap();
        values.sort_by_key(|value| value.to_string());
        assert_eq!((min, max), (Literal::Int64(3), Literal::Int64(7)));
        assert_eq!(values, vec![Literal::Int64(3), Literal::Int64(5), Literal::Int64(7)]);
    }

    #[test]
    fn values_are_listed_only_up_to_the_limit_of_build_rows() {
        let keys = ints(&[Some(1), Some(2), Some(3)]);
        assert!(matches!(
            KeyValues::of(&keys, 2, MEMBER_BYTES).unwrap(),
            KeyValues::Within { values: None, .. }
        ));
    }

    #[test]
    fn a_build_side_of_only_null_keys_matches_nothing() {
        for keys in [ints(&[None, None]), ints(&[])] {
            assert_eq!(KeyValues::of(&keys, 10, MEMBER_BYTES).unwrap(), KeyValues::Nothing);
            assert_eq!(
                KeyFilter::from_build_keys(&[keys], 10, MEMBER_BYTES)
                    .unwrap()
                    .restriction(0, "c"),
                Some(KeyRestriction::Nothing)
            );
        }
    }

    #[test]
    fn one_distinct_key_prunes_by_equality() {
        let keys = Utf8Array::from_iter("k", ["a", "a"].into_iter().map(Some)).into_series();
        let filter = KeyFilter::from_build_keys(&[keys], 10, MEMBER_BYTES).unwrap();
        let KeyRestriction::Within { prune, .. } = filter.restriction(0, "c").unwrap() else {
            panic!("keys were present");
        };
        assert_eq!(prune, resolved_col("c").eq(lit(Literal::Utf8("a".into()))));
    }

    #[test]
    fn listed_keys_prune_by_range_and_values() {
        let KeyRestriction::Within { prune, .. } = restriction(&[Some(2), Some(9)], 10, MEMBER_BYTES)
        else {
            panic!("keys were present");
        };
        let column = resolved_col("c");
        let range = column
            .clone()
            .gt_eq(lit(2i64))
            .and(column.clone().lt_eq(lit(9i64)));
        let candidates = [
            range.clone().and(column.clone().is_in(vec![lit(2i64), lit(9i64)])),
            range.and(column.is_in(vec![lit(9i64), lit(2i64)])),
        ];
        assert!(candidates.contains(&prune));
    }

    #[test]
    fn hashed_keys_pass_only_rows_holding_a_build_key() {
        let KeyRestriction::Within { rows: (column, test), .. } =
            restriction(&[Some(2), Some(9)], 10, MEMBER_BYTES)
        else {
            panic!("keys were present");
        };
        assert_eq!(column, "c");
        assert!(matches!(test, RowTest::Members(_)));
        assert_eq!(
            passing(&test, &[Some(2), Some(3), None, Some(9)]),
            vec![true, false, false, true]
        );
    }

    #[test]
    fn keys_too_large_to_hash_pass_rows_within_their_range() {
        let KeyRestriction::Within { prune, rows: (_, test) } =
            restriction(&[Some(2), Some(9), Some(4)], 2, 0)
        else {
            panic!("keys were present");
        };
        let column = resolved_col("c");
        let range = column.clone().gt_eq(lit(2i64)).and(column.lt_eq(lit(9i64)));
        assert_eq!(test, RowTest::Satisfies(range.clone()));
        assert_eq!(prune, range);
        assert_eq!(
            passing(&test, &[Some(2), Some(5), None, Some(10)]),
            vec![true, true, false, false]
        );
    }

    fn keep(keys: &[i64]) -> RowKeyFilter {
        let KeyRestriction::Within { rows, .. } = restriction(
            &keys.iter().copied().map(Some).collect::<Vec<_>>(),
            10,
            MEMBER_BYTES,
        ) else {
            panic!("keys were present");
        };
        RowKeyFilter::new(vec![rows])
    }

    #[test]
    fn a_row_filter_keeping_most_rows_stops_after_its_sample() {
        let filter = keep(&(1..SAMPLED_ROWS as i64).collect::<Vec<_>>());
        let sample = filter.apply(partition(0..SAMPLED_ROWS as i64)).unwrap();
        let after = filter.apply(partition([0, 0, 5])).unwrap();

        assert_eq!(sample.len(), SAMPLED_ROWS - 1);
        assert_eq!(after.len(), 3);
    }

    #[test]
    fn a_selective_row_filter_keeps_filtering() {
        let filter = keep(&[7]);
        let sample = filter.apply(partition(0..SAMPLED_ROWS as i64)).unwrap();
        let after = filter.apply(partition([0, 7, 5])).unwrap();

        assert_eq!(sample.len(), 1);
        assert_eq!(after.len(), 1);
    }

    #[test]
    fn floats_and_nested_types_do_not_filter() {
        assert!(filters_by(&DataType::Int32));
        assert!(filters_by(&DataType::Utf8));
        assert!(!filters_by(&DataType::Float64));
        assert!(!filters_by(&DataType::List(Box::new(DataType::Int64))));
    }

    #[tokio::test]
    async fn every_consumer_takes_a_published_filter_once() {
        let bridge = Arc::new(KeyFilterBridge::new(2));
        let filter = Arc::new(KeyFilter::from_build_keys(&[ints(&[Some(1)])], 10, MEMBER_BYTES).unwrap());
        let early = tokio::spawn({
            let bridge = bridge.clone();
            async move { bridge.wait(0).await }
        });
        tokio::task::yield_now().await;
        bridge.publish(0, Some(filter));

        assert!(early.await.unwrap().is_some());
        assert!(bridge.wait(0).await.is_some());
        assert!(bridge.slots.lock().unwrap().by_input.is_empty());
    }

    #[tokio::test]
    async fn a_closed_bridge_releases_its_waiting_scans_without_a_filter() {
        let bridge = Arc::new(KeyFilterBridge::new(1));
        let waiting = tokio::spawn({
            let bridge = bridge.clone();
            async move { bridge.wait(3).await }
        });
        tokio::task::yield_now().await;
        bridge.close();

        assert!(waiting.await.unwrap().is_none());
        assert!(bridge.wait(4).await.is_none());
    }
}
