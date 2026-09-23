use std::{collections::HashSet, sync::Arc};

use common_error::{DaftError, DaftResult};
use common_file_formats::FileFormat;
use daft_core::prelude::*;
use daft_dsl::{Expr, ExprRef};
use serde::{Deserialize, Serialize};

#[cfg(feature = "python")]
use crate::sink_info::CatalogType;
use crate::{
    LogicalPlan,
    logical_plan::{self},
    sink_info::SinkInfo,
    stats::{PlanStats, StatsState},
};

/// Refuses, before anything is written, a CSV write of a column no CSV file can hold.
///
/// A plain partition column is written as a directory name rather than into the
/// files, so only the columns that reach the files are checked.
fn ensure_csv_can_hold(schema: &Schema, partition_cols: Option<&[ExprRef]>) -> DaftResult<()> {
    let in_paths: HashSet<&str> = partition_cols
        .unwrap_or_default()
        .iter()
        .filter(|col| matches!(col.as_ref(), Expr::Column(_)))
        .map(|col| col.name())
        .collect();
    let mut written = schema
        .fields()
        .iter()
        .filter(|field| !in_paths.contains(field.name.as_ref()))
        .peekable();
    // When every column partitions the write, every column is written.
    let unholdable = if written.peek().is_some() {
        written.find(|field| field.dtype.is_nested() || field.dtype.is_python())
    } else {
        schema
            .fields()
            .iter()
            .find(|field| field.dtype.is_nested() || field.dtype.is_python())
    };
    match unholdable {
        Some(field) => Err(DaftError::NotImplemented(format!(
            "CSV cannot hold the column {:?} of type {}; select or convert it before writing",
            field.name, field.dtype
        ))),
        None => Ok(()),
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(debug_assertions, derive(Debug))]
pub struct Sink {
    pub plan_id: Option<usize>,
    pub node_id: Option<usize>,
    // Upstream node.
    pub input: Arc<LogicalPlan>,
    pub schema: SchemaRef,
    /// Information about the sink data location.
    pub sink_info: Arc<SinkInfo>,
    pub stats_state: StatsState,
}

impl Sink {
    pub(crate) fn try_new(
        input: Arc<LogicalPlan>,
        sink_info: Arc<SinkInfo>,
    ) -> logical_plan::Result<Self> {
        let schema = input.schema();

        let fields = match sink_info.as_ref() {
            SinkInfo::OutputFileInfo(output_file_info) => {
                if output_file_info.file_format == FileFormat::Csv {
                    ensure_csv_can_hold(&schema, output_file_info.partition_cols.as_deref())?;
                }
                // Each written file: where it is and how many rows it holds, then the
                // partition values it was written for.
                let mut fields = vec![
                    Field::new("path", DataType::Utf8),
                    Field::new("num_rows", DataType::Int64),
                ];
                if let Some(ref pcols) = output_file_info.partition_cols {
                    for pc in pcols {
                        let field = pc.to_field(&schema)?;
                        if fields.iter().any(|reported| reported.name == field.name) {
                            return Err(DaftError::ValueError(format!(
                                "cannot partition a write by a column named {:?}; the write reports that name for each file it writes",
                                field.name
                            ))
                            .into());
                        }
                        fields.push(field);
                    }
                }
                fields
            }
            #[cfg(feature = "python")]
            SinkInfo::CatalogInfo(catalog_info) => {
                match catalog_info.catalog {
                    CatalogType::Iceberg(_) => {
                        vec![
                            // We have to return datafile since PyIceberg Table is not picklable yet
                            Field::new("data_file", DataType::Python),
                        ]
                    }
                    CatalogType::DeltaLake(_) => vec![Field::new("add_action", DataType::Python)],
                    CatalogType::Lance(_) => vec![Field::new("fragments", DataType::Python)],
                }
            }
            #[cfg(feature = "python")]
            SinkInfo::DataSinkInfo(_) => {
                vec![Field::new("write_results", DataType::Python)]
            }
        };
        let schema = Schema::new(fields).into();
        Ok(Self {
            plan_id: None,
            node_id: None,
            input,
            schema,
            sink_info,
            stats_state: StatsState::NotMaterialized,
        })
    }

    pub fn with_plan_id(mut self, plan_id: usize) -> Self {
        self.plan_id = Some(plan_id);
        self
    }

    pub fn with_node_id(mut self, node_id: usize) -> Self {
        self.node_id = Some(node_id);
        self
    }

    pub(crate) fn with_materialized_stats(mut self) -> Self {
        // Post-write DataFrame will contain paths to files that were written.
        // TODO(desmond): Estimate output size via root directory and estimates for # of partitions given partitioning column.
        self.stats_state = StatsState::Materialized(PlanStats::empty().into());
        self
    }

    pub fn multiline_display(&self) -> Vec<String> {
        let mut res = vec![];

        match self.sink_info.as_ref() {
            SinkInfo::OutputFileInfo(output_file_info) => {
                res.push(format!("Sink: {:?}", output_file_info.file_format));
                res.extend(output_file_info.multiline_display());
            }
            #[cfg(feature = "python")]
            SinkInfo::CatalogInfo(catalog_info) => match &catalog_info.catalog {
                CatalogType::Iceberg(iceberg_info) => {
                    res.push(format!("Sink: Iceberg({})", iceberg_info.table_name));
                    res.extend(iceberg_info.multiline_display());
                }
                CatalogType::DeltaLake(deltalake_info) => {
                    res.push(format!("Sink: DeltaLake({})", deltalake_info.path));
                    res.extend(deltalake_info.multiline_display());
                }
                CatalogType::Lance(lance_info) => {
                    res.push(format!("Sink: Lance({})", lance_info.path));
                    res.extend(lance_info.multiline_display());
                }
            },
            #[cfg(feature = "python")]
            SinkInfo::DataSinkInfo(data_sink_info) => {
                res.push(format!("Sink: DataSink({})", data_sink_info.name));
            }
        }
        res.push(format!("Output schema = {}", self.schema.short_string()));
        if let StatsState::Materialized(stats) = &self.stats_state {
            res.push(format!("Stats = {}", stats));
        }
        res
    }
}
