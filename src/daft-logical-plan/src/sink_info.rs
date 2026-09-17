use std::{hash::Hash, sync::Arc};

use common_error::DaftResult;
use common_file_formats::{FileFormat, WriteMode};
use common_io_config::IOConfig;
#[cfg(feature = "python")]
use common_py_serde::{deserialize_py_object, serialize_py_object};
use daft_core::prelude::Schema;
use daft_dsl::{ExprRef, expr::bound_expr::BoundExpr};
use educe::Educe;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

#[allow(clippy::large_enum_variant)]
#[derive(PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(debug_assertions, derive(Debug))]
pub enum SinkInfo<E = ExprRef> {
    OutputFileInfo(OutputFileInfo<E>),
    #[cfg(feature = "python")]
    CatalogInfo(CatalogInfo<E>),
    #[cfg(feature = "python")]
    DataSinkInfo(DataSinkInfo),
}

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(debug_assertions, derive(Debug))]
pub struct OutputFileInfo<E = ExprRef> {
    pub root_dir: String,
    pub write_mode: WriteMode,
    pub file_format: FileFormat,
    pub format_option: Option<FormatSinkOption>,
    pub partition_cols: Option<Vec<E>>,
    pub compression: Option<String>,
    pub io_config: Option<IOConfig>,
    pub write_success_file: bool,
    pub single_file: bool,
}

#[cfg(feature = "python")]
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(debug_assertions, derive(Debug))]
pub struct CatalogInfo<E = ExprRef> {
    pub catalog: CatalogType<E>,
    pub catalog_columns: Vec<String>,
}

#[cfg(feature = "python")]
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(debug_assertions, derive(Debug))]
pub enum CatalogType<E = ExprRef> {
    Iceberg(IcebergCatalogInfo<E>),
    IcebergRowDelta(IcebergRowDeltaInfo<E>),
    DeltaLake(DeltaLakeCatalogInfo<E>),
    Lance(LanceCatalogInfo),
}

/// A write that both adds rows and removes rows named by where they are stored.
///
/// The incoming rows are tagged with what the write should do with each of them.
/// Rows to keep or add go to the data writer; rows to remove are recorded by the
/// file and position they occupy, and become delete files when `deletes` is set.
/// Without `deletes` the removals are implied by rewriting whole files instead.
#[cfg(feature = "python")]
#[derive(Educe, Clone, Serialize, Deserialize)]
#[cfg_attr(debug_assertions, derive(Debug))]
#[educe(PartialEq, Eq, Hash)]
pub struct IcebergRowDeltaInfo<E = ExprRef> {
    /// Where and how new data files are written.
    pub data: IcebergCatalogInfo<E>,
    /// How rows removed from existing files are recorded.
    pub deletes: Option<IcebergDeleteInfo>,
    /// Column holding what the merge decided for each row.
    pub action_column: String,
    /// Column holding the index of the file a row was read from.
    pub file_index_column: String,
    /// Column holding a row's position in that file.
    pub position_column: String,
}

/// Where the rows removed by a write are recorded.
#[cfg(feature = "python")]
#[derive(Educe, Clone, Serialize, Deserialize)]
#[cfg_attr(debug_assertions, derive(Debug))]
#[educe(PartialEq, Eq, Hash)]
pub struct IcebergDeleteInfo {
    /// Opens the writer for one group of removals, given the group's ordinal and
    /// the index of a file it names.
    #[serde(
        serialize_with = "serialize_py_object",
        deserialize_with = "deserialize_py_object"
    )]
    #[educe(PartialEq(ignore))]
    #[educe(Hash(ignore))]
    pub writer_factory: Arc<pyo3::Py<pyo3::PyAny>>,
    /// Path of the data file at each file index.
    pub file_paths: Vec<String>,
    /// Group each file index belongs to. Non-decreasing, so one pass over rows
    /// sorted by file index and position visits each group once.
    pub file_groups: Vec<i64>,
    /// Size a delete file is rolled at, in bytes.
    pub target_file_size: usize,
}

#[cfg(feature = "python")]
impl IcebergDeleteInfo {
    /// Group of the file at `index`.
    ///
    /// # Errors
    /// If `index` names no planned file, which means a row carried provenance
    /// from a different plan than the one being written.
    pub fn group_of(&self, index: i64) -> common_error::DaftResult<i64> {
        usize::try_from(index)
            .ok()
            .and_then(|index| self.file_groups.get(index))
            .copied()
            .ok_or_else(|| {
                common_error::DaftError::ValueError(format!(
                    "file index {index} is outside the {} planned file(s)",
                    self.file_groups.len()
                ))
            })
    }

    /// Path of the file at `index`.
    ///
    /// # Errors
    /// See [`Self::group_of`].
    pub fn path_of(&self, index: i64) -> common_error::DaftResult<&str> {
        usize::try_from(index)
            .ok()
            .and_then(|index| self.file_paths.get(index))
            .map(String::as_str)
            .ok_or_else(|| {
                common_error::DaftError::ValueError(format!(
                    "file index {index} is outside the {} planned file(s)",
                    self.file_paths.len()
                ))
            })
    }
}

#[cfg(feature = "python")]
#[derive(Educe, Clone, Serialize, Deserialize)]
#[cfg_attr(debug_assertions, derive(Debug))]
#[educe(PartialEq, Eq, Hash)]
pub struct IcebergCatalogInfo<E = ExprRef> {
    pub table_name: String,
    pub table_location: String,
    pub partition_spec_id: i64,
    pub partition_cols: Vec<E>,
    #[serde(
        serialize_with = "serialize_py_object",
        deserialize_with = "deserialize_py_object"
    )]
    #[educe(PartialEq(ignore))]
    #[educe(Hash(ignore))]
    pub iceberg_schema: Arc<pyo3::Py<pyo3::PyAny>>,
    #[serde(
        serialize_with = "serialize_py_object",
        deserialize_with = "deserialize_py_object"
    )]
    #[educe(PartialEq(ignore))]
    #[educe(Hash(ignore))]
    pub iceberg_properties: Arc<pyo3::Py<pyo3::PyAny>>,
    /// Sort order the written rows are in. Zero is unsorted.
    pub sort_order_id: i64,
    pub io_config: Option<IOConfig>,
}

#[cfg(feature = "python")]
impl<E> IcebergCatalogInfo<E> {
    pub fn multiline_display(&self) -> Vec<String> {
        let mut res = vec![];
        res.push(format!("Table Name = {}", self.table_name));
        res.push(format!("Table Location = {}", self.table_location));
        match &self.io_config {
            None => res.push("IOConfig = None".to_string()),
            Some(io_config) => res.push(format!("IOConfig = {}", io_config)),
        }
        res
    }
}

#[cfg(feature = "python")]
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(debug_assertions, derive(Debug))]
pub struct DeltaLakeCatalogInfo<E = ExprRef> {
    pub path: String,
    pub mode: String,
    pub version: i32,
    pub large_dtypes: bool,
    pub partition_cols: Option<Vec<E>>,
    pub io_config: Option<IOConfig>,
}

#[cfg(feature = "python")]
impl<E> DeltaLakeCatalogInfo<E>
where
    E: ToString,
{
    pub fn multiline_display(&self) -> Vec<String> {
        let mut res = vec![];
        res.push(format!("Table Name = {}", self.path));
        res.push(format!("Mode = {}", self.mode));
        res.push(format!("Version = {}", self.version));
        res.push(format!("Large Dtypes = {}", self.large_dtypes));
        if let Some(ref partition_cols) = self.partition_cols {
            res.push(format!(
                "Partition cols = {}",
                partition_cols.iter().map(|e| e.to_string()).join(", ")
            ));
        }
        match &self.io_config {
            None => res.push("IOConfig = None".to_string()),
            Some(io_config) => res.push(format!("IOConfig = {}", io_config)),
        }
        res
    }
}

#[cfg(feature = "python")]
#[derive(Educe, Clone, Serialize, Deserialize)]
#[educe(PartialEq, Eq, Hash)]
#[cfg_attr(debug_assertions, derive(Debug))]
pub struct LanceCatalogInfo {
    pub path: String,
    pub mode: String,
    pub io_config: Option<IOConfig>,
    #[serde(
        serialize_with = "serialize_py_object",
        deserialize_with = "deserialize_py_object"
    )]
    #[educe(PartialEq(ignore))]
    #[educe(Hash(ignore))]
    pub kwargs: Arc<pyo3::Py<pyo3::PyAny>>,
}

#[cfg(feature = "python")]
impl LanceCatalogInfo {
    pub fn multiline_display(&self) -> Vec<String> {
        let mut res = vec![];
        res.push(format!("Table Name = {}", self.path));
        res.push(format!("Mode = {}", self.mode));
        match &self.io_config {
            None => res.push("IOConfig = None".to_string()),
            Some(io_config) => res.push(format!("IOConfig = {}", io_config)),
        }
        res
    }
}

#[cfg(feature = "python")]
#[derive(Educe, Clone, Serialize, Deserialize)]
#[cfg_attr(debug_assertions, derive(Debug))]
#[educe(PartialEq, Eq, Hash)]
pub struct DataSinkInfo {
    pub name: String,
    #[serde(
        serialize_with = "serialize_py_object",
        deserialize_with = "deserialize_py_object"
    )]
    #[educe(PartialEq(ignore))]
    #[educe(Hash(ignore))]
    pub sink: Arc<pyo3::Py<pyo3::PyAny>>,
}

#[cfg(feature = "python")]
impl DataSinkInfo {
    pub fn multiline_display(&self) -> Vec<String> {
        vec![format!("DataSinkInfo = {}", self.sink)]
    }
}

impl<E> OutputFileInfo<E>
where
    E: ToString,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        root_dir: String,
        write_mode: WriteMode,
        file_format: FileFormat,
        format_option: Option<FormatSinkOption>,
        partition_cols: Option<Vec<E>>,
        compression: Option<String>,
        io_config: Option<IOConfig>,
        write_success_file: bool,
        single_file: bool,
    ) -> Self {
        Self {
            root_dir,
            write_mode,
            file_format,
            format_option,
            partition_cols,
            compression,
            io_config,
            write_success_file,
            single_file,
        }
    }

    pub fn multiline_display(&self) -> Vec<String> {
        let mut res = vec![];
        if let Some(ref partition_cols) = self.partition_cols {
            res.push(format!(
                "Partition cols = {}",
                partition_cols.iter().map(|e| e.to_string()).join(", ")
            ));
        }
        if let Some(ref compression) = self.compression {
            res.push(format!("Compression = {}", compression));
        }
        res.push(format!("Root dir = {}", self.root_dir));
        match &self.io_config {
            None => res.push("IOConfig = None".to_string()),
            Some(io_config) => res.push(format!("IOConfig = {}", io_config)),
        }
        res
    }
}

impl SinkInfo {
    pub fn bind(&self, schema: &Schema) -> DaftResult<SinkInfo<BoundExpr>> {
        match self {
            Self::OutputFileInfo(output_file_info) => Ok(SinkInfo::OutputFileInfo(
                output_file_info.clone().bind(schema)?,
            )),
            #[cfg(feature = "python")]
            Self::CatalogInfo(catalog_info) => {
                Ok(SinkInfo::CatalogInfo(catalog_info.clone().bind(schema)?))
            }
            #[cfg(feature = "python")]
            Self::DataSinkInfo(data_sink_info) => {
                Ok(SinkInfo::DataSinkInfo(data_sink_info.clone()))
            }
        }
    }
}

impl OutputFileInfo {
    pub fn bind(self, schema: &Schema) -> DaftResult<OutputFileInfo<BoundExpr>> {
        Ok(OutputFileInfo {
            root_dir: self.root_dir,
            write_mode: self.write_mode,
            file_format: self.file_format,
            format_option: self.format_option,
            partition_cols: self
                .partition_cols
                .map(|cols| BoundExpr::bind_all(&cols, schema))
                .transpose()?,
            compression: self.compression,
            io_config: self.io_config,
            write_success_file: self.write_success_file,
            single_file: self.single_file,
        })
    }
}

#[cfg(feature = "python")]
impl CatalogInfo {
    pub fn bind(self, schema: &Schema) -> DaftResult<CatalogInfo<BoundExpr>> {
        Ok(CatalogInfo {
            catalog: self.catalog.bind(schema)?,
            catalog_columns: self.catalog_columns,
        })
    }
}

#[cfg(feature = "python")]
impl CatalogType {
    pub fn bind(self, schema: &Schema) -> DaftResult<CatalogType<BoundExpr>> {
        match self {
            Self::Iceberg(iceberg_catalog_info) => {
                Ok(CatalogType::Iceberg(iceberg_catalog_info.bind(schema)?))
            }
            Self::IcebergRowDelta(row_delta_info) => {
                Ok(CatalogType::IcebergRowDelta(IcebergRowDeltaInfo {
                    data: row_delta_info.data.bind(schema)?,
                    deletes: row_delta_info.deletes,
                    action_column: row_delta_info.action_column,
                    file_index_column: row_delta_info.file_index_column,
                    position_column: row_delta_info.position_column,
                }))
            }
            Self::DeltaLake(delta_lake_catalog_info) => Ok(CatalogType::DeltaLake(
                delta_lake_catalog_info.bind(schema)?,
            )),
            Self::Lance(lance_catalog_info) => Ok(CatalogType::Lance(lance_catalog_info)),
        }
    }
}

#[cfg(feature = "python")]
impl IcebergCatalogInfo {
    pub fn bind(self, schema: &Schema) -> DaftResult<IcebergCatalogInfo<BoundExpr>> {
        Ok(IcebergCatalogInfo {
            table_name: self.table_name,
            table_location: self.table_location,
            partition_spec_id: self.partition_spec_id,
            partition_cols: BoundExpr::bind_all(&self.partition_cols, schema)?,
            iceberg_schema: self.iceberg_schema,
            iceberg_properties: self.iceberg_properties,
            sort_order_id: self.sort_order_id,
            io_config: self.io_config,
        })
    }
}

#[cfg(feature = "python")]
impl DeltaLakeCatalogInfo {
    pub fn bind(self, schema: &Schema) -> DaftResult<DeltaLakeCatalogInfo<BoundExpr>> {
        Ok(DeltaLakeCatalogInfo {
            path: self.path,
            mode: self.mode,
            version: self.version,
            large_dtypes: self.large_dtypes,
            partition_cols: self
                .partition_cols
                .map(|cols| BoundExpr::bind_all(&cols, schema))
                .transpose()?,
            io_config: self.io_config,
        })
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CsvFormatOption {
    pub delimiter: Option<u8>,
    pub quote: Option<u8>,
    pub escape: Option<u8>,
    pub header: Option<bool>,
    pub date_format: Option<String>,
    pub timestamp_format: Option<String>,
}

impl Default for CsvFormatOption {
    fn default() -> Self {
        Self {
            delimiter: Some(b','),
            quote: Some(b'"'),
            escape: Some(b'\\'),
            header: Some(true),
            date_format: None,
            timestamp_format: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JsonFormatOption {
    pub ignore_null_fields: Option<bool>,
    pub date_format: Option<String>,
    pub timestamp_format: Option<String>,
}

impl Default for JsonFormatOption {
    fn default() -> Self {
        Self {
            ignore_null_fields: Some(false),
            date_format: None,
            timestamp_format: None,
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ParquetFormatOption {
    /// Per-column compression overrides, the type matches the parquet writer builder
    /// method for setting column compression. The writer will parse the codec names.
    pub column_compression: Option<Vec<(String, String)>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FormatSinkOption {
    Csv(CsvFormatOption),
    Json(JsonFormatOption),
    Parquet(ParquetFormatOption),
}

impl FormatSinkOption {
    pub fn to_csv(self) -> CsvFormatOption {
        match self {
            Self::Csv(csv) => csv,
            _ => CsvFormatOption::default(),
        }
    }
    pub fn to_json(self) -> JsonFormatOption {
        match self {
            Self::Json(json) => json,
            _ => JsonFormatOption::default(),
        }
    }
    pub fn to_parquet(self) -> ParquetFormatOption {
        match self {
            Self::Parquet(parquet) => parquet,
            _ => ParquetFormatOption::default(),
        }
    }
}

#[cfg(feature = "python")]
#[pyo3::pyclass(from_py_object)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PyFormatSinkOption {
    pub inner: FormatSinkOption,
}

#[cfg(feature = "python")]
#[pyo3::pymethods]
impl PyFormatSinkOption {
    #[classmethod]
    pub fn csv(
        _cls: &pyo3::prelude::Bound<pyo3::types::PyType>,
        delimiter: Option<char>,
        quote: Option<char>,
        escape: Option<char>,
        header: Option<bool>,
        date_format: Option<String>,
        timestamp_format: Option<String>,
    ) -> Self {
        let to_u8 = |c: Option<char>| -> Option<u8> {
            c.and_then(|ch| if ch.is_ascii() { Some(ch as u8) } else { None })
        };
        Self {
            inner: FormatSinkOption::Csv(CsvFormatOption {
                delimiter: to_u8(delimiter),
                quote: to_u8(quote),
                escape: to_u8(escape),
                header,
                date_format,
                timestamp_format,
            }),
        }
    }

    #[classmethod]
    pub fn json(
        _cls: &pyo3::prelude::Bound<pyo3::types::PyType>,
        ignore_null_fields: Option<bool>,
        date_format: Option<String>,
        timestamp_format: Option<String>,
    ) -> Self {
        Self {
            inner: FormatSinkOption::Json(JsonFormatOption {
                ignore_null_fields,
                date_format,
                timestamp_format,
            }),
        }
    }

    #[classmethod]
    #[pyo3(signature = (column_compression=None))]
    pub fn parquet(
        _cls: &pyo3::prelude::Bound<pyo3::types::PyType>,
        column_compression: Option<Vec<(String, String)>>,
    ) -> Self {
        Self {
            inner: FormatSinkOption::Parquet(ParquetFormatOption { column_compression }),
        }
    }
}
