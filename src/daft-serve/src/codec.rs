//! Columnar encoding between engine record batches and the streaming wire
//! representation.
//!
//! Result streams are encoded message-by-message: one schema message first,
//! then one or more batch messages per emitted partition, then a final
//! metadata-only message carrying execution statistics. The schema message
//! additionally carries the engine-native schema so the receiving side
//! reconstructs exact column types (including extension types) rather than
//! re-inferring them from the interchange schema.
//!
//! In-memory partition sets shipped *with* a request use self-contained
//! streaming-IPC blobs produced and consumed by the same helpers.

use std::sync::Arc;

use arrow_flight::{FlightData, SchemaAsIpc};
use arrow_ipc::writer::{CompressionContext, DictionaryTracker, IpcDataGenerator, IpcWriteOptions};
use common_error::DaftError;
use daft_core::{prelude::SchemaRef, series::Series};
use daft_micropartition::MicroPartition;
use daft_recordbatch::RecordBatch;
use daft_schema::field::FieldRef;

use crate::{
    error::{ServeError, ServeResult},
    wire,
};

/// Encodes engine record batches into wire messages, tracking dictionary
/// state across batches of one stream.
pub struct BatchEncoder {
    arrow_schema: Arc<arrow_schema::Schema>,
    generator: IpcDataGenerator,
    tracker: DictionaryTracker,
    options: IpcWriteOptions,
    compression: CompressionContext,
}

impl BatchEncoder {
    /// Creates an encoder for a stream of batches with the given schema.
    ///
    /// # Errors
    /// Returns an error if the engine schema cannot be represented in the
    /// interchange schema.
    pub fn try_new(schema: &SchemaRef) -> ServeResult<Self> {
        let arrow_schema = Arc::new(schema.to_arrow().map_err(ServeError::Execution)?);
        Ok(Self {
            arrow_schema,
            generator: IpcDataGenerator::default(),
            tracker: DictionaryTracker::new(false),
            options: IpcWriteOptions::default(),
            compression: CompressionContext::default(),
        })
    }

    /// The stream-opening schema message, carrying the engine-native schema
    /// as attached metadata.
    ///
    /// # Errors
    /// Returns an error if the engine schema cannot be serialized.
    pub fn schema_message(&self, schema: &SchemaRef) -> ServeResult<FlightData> {
        let mut flight_schema: FlightData =
            SchemaAsIpc::new(&self.arrow_schema, &self.options).into();
        flight_schema.app_metadata = wire::encode(schema)?.into();
        Ok(flight_schema)
    }

    /// The stream-terminating message: an empty batch carrying serialized
    /// execution statistics as attached metadata. Receivers identify the
    /// trailer by its metadata and do not surface it as data.
    ///
    /// # Errors
    /// Returns an error if the statistics or the empty batch cannot be
    /// encoded.
    pub fn stats_message(&mut self, stats: &wire::QueryStatsWire) -> ServeResult<FlightData> {
        let empty = arrow_array::RecordBatch::new_empty(self.arrow_schema.clone());
        let (_, batch_data) = self
            .generator
            .encode(
                &empty,
                &mut self.tracker,
                &self.options,
                &mut self.compression,
            )
            .map_err(|e| ServeError::Execution(DaftError::ArrowRsError(e)))?;
        let mut message = FlightData::from(batch_data);
        message.app_metadata = wire::encode(stats)?.into();
        Ok(message)
    }

    /// Encodes one engine record batch into wire messages (dictionary
    /// deltas, if any, followed by the batch itself).
    ///
    /// # Errors
    /// Returns an error if columns cannot be converted to the interchange
    /// representation.
    pub fn encode(&mut self, batch: &RecordBatch) -> ServeResult<Vec<FlightData>> {
        let columns = batch
            .columns()
            .iter()
            .map(|c| c.as_materialized_series().to_arrow())
            .collect::<Result<Vec<_>, _>>()
            .map_err(ServeError::Execution)?;
        let arrow_batch = arrow_array::RecordBatch::try_new(self.arrow_schema.clone(), columns)
            .map_err(|e| ServeError::Execution(DaftError::ArrowRsError(e)))?;
        let (dictionaries, batch_data) = self
            .generator
            .encode(
                &arrow_batch,
                &mut self.tracker,
                &self.options,
                &mut self.compression,
            )
            .map_err(|e| ServeError::Execution(DaftError::ArrowRsError(e)))?;
        let mut out = Vec::with_capacity(dictionaries.len() + 1);
        out.extend(dictionaries.into_iter().map(FlightData::from));
        out.push(FlightData::from(batch_data));
        Ok(out)
    }
}

/// Converts one interchange record batch into an engine record batch using
/// the engine-native schema for exact field types.
///
/// # Errors
/// Returns an error if any column cannot be converted.
pub fn arrow_batch_to_engine_batch(
    schema: &SchemaRef,
    fields: &[FieldRef],
    batch: &arrow_array::RecordBatch,
) -> ServeResult<RecordBatch> {
    let columns = fields
        .iter()
        .zip(batch.columns())
        .map(|(field, array)| Series::from_arrow(field.clone(), array.clone()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(ServeError::Execution)?;
    RecordBatch::new_with_size(schema.clone(), columns, batch.num_rows())
        .map_err(ServeError::Execution)
}

/// Serializes one in-memory partition into a self-contained streaming-IPC
/// blob (schema message plus batches).
///
/// # Errors
/// Returns an error if the partition cannot be encoded.
pub fn micropartition_to_ipc(partition: &MicroPartition) -> ServeResult<Vec<u8>> {
    let schema = partition.schema();
    let arrow_schema = schema.to_arrow().map_err(ServeError::Execution)?;
    let mut writer = arrow_ipc::writer::StreamWriter::try_new(Vec::new(), &arrow_schema)
        .map_err(|e| ServeError::Execution(DaftError::ArrowRsError(e)))?;
    let arrow_schema = Arc::new(arrow_schema);
    for batch in partition.record_batches() {
        let columns = batch
            .columns()
            .iter()
            .map(|c| c.as_materialized_series().to_arrow())
            .collect::<Result<Vec<_>, _>>()
            .map_err(ServeError::Execution)?;
        let arrow_batch = arrow_array::RecordBatch::try_new(arrow_schema.clone(), columns)
            .map_err(|e| ServeError::Execution(DaftError::ArrowRsError(e)))?;
        writer
            .write(&arrow_batch)
            .map_err(|e| ServeError::Execution(DaftError::ArrowRsError(e)))?;
    }
    writer
        .into_inner()
        .map_err(|e| ServeError::Execution(DaftError::ArrowRsError(e)))
}

/// Reconstructs an in-memory partition from a streaming-IPC blob and its
/// engine-native schema.
///
/// # Errors
/// Returns an error if the blob does not decode or columns cannot be
/// converted to engine types.
pub fn ipc_to_micropartition(schema: &SchemaRef, ipc: &[u8]) -> ServeResult<MicroPartition> {
    let fields: Vec<FieldRef> = schema
        .fields()
        .iter()
        .map(|f| Arc::new(f.clone()))
        .collect();
    let reader = arrow_ipc::reader::StreamReader::try_new(std::io::Cursor::new(ipc), None)
        .map_err(|e| ServeError::MalformedPayload(format!("partition blob: {e}")))?;
    let mut batches = Vec::new();
    for batch in reader {
        let batch =
            batch.map_err(|e| ServeError::MalformedPayload(format!("partition blob: {e}")))?;
        batches.push(arrow_batch_to_engine_batch(schema, &fields, &batch)?);
    }
    Ok(MicroPartition::new_loaded(
        schema.clone(),
        Arc::new(batches),
        None,
    ))
}

#[cfg(test)]
mod tests {
    use daft_core::prelude::*;

    use super::*;

    fn sample_partition() -> MicroPartition {
        let a = Series::from_arrow(
            Arc::new(Field::new("a", DataType::Int64)),
            Arc::new(arrow_array::Int64Array::from(vec![1_i64, 2, 3])),
        )
        .unwrap();
        let b = Series::from_arrow(
            Arc::new(Field::new("b", DataType::Utf8)),
            Arc::new(arrow_array::LargeStringArray::from(vec![
                Some("x"),
                None,
                Some("z"),
            ])),
        )
        .unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64),
            Field::new("b", DataType::Utf8),
        ]));
        let batch = RecordBatch::new_with_size(schema.clone(), vec![a, b], 3).unwrap();
        MicroPartition::new_loaded(schema, Arc::new(vec![batch]), None)
    }

    #[test]
    fn partition_ipc_round_trips() {
        let partition = sample_partition();
        let ipc = micropartition_to_ipc(&partition).unwrap();
        let restored = ipc_to_micropartition(&partition.schema(), &ipc).unwrap();
        assert_eq!(restored.schema(), partition.schema());
        assert_eq!(restored.len(), partition.len());
        let original = partition.record_batches();
        let roundtrip = restored.record_batches();
        assert_eq!(original.len(), roundtrip.len());
        for (a, b) in original.iter().zip(roundtrip.iter()) {
            for (col_a, col_b) in a.columns().iter().zip(b.columns().iter()) {
                let arrow_a = col_a.as_materialized_series().to_arrow().unwrap();
                let arrow_b = col_b.as_materialized_series().to_arrow().unwrap();
                assert_eq!(&arrow_a, &arrow_b);
            }
            assert_eq!(a.num_rows(), b.num_rows());
        }
    }

    #[test]
    fn empty_partition_round_trips_schema_only() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64)]));
        let partition = MicroPartition::new_loaded(schema.clone(), Arc::new(vec![]), None);
        let ipc = micropartition_to_ipc(&partition).unwrap();
        let restored = ipc_to_micropartition(&schema, &ipc).unwrap();
        assert_eq!(restored.len(), 0);
        assert_eq!(restored.schema(), schema);
    }

    #[test]
    fn garbage_ipc_is_rejected() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64)]));
        let err = ipc_to_micropartition(&schema, &[0xDE, 0xAD, 0xBE, 0xEF]).unwrap_err();
        assert!(matches!(err, ServeError::MalformedPayload(_)));
    }

    #[test]
    fn batch_encoder_emits_schema_then_batches() {
        let partition = sample_partition();
        let schema = partition.schema();
        let mut encoder = BatchEncoder::try_new(&schema).unwrap();
        let schema_msg = encoder.schema_message(&schema).unwrap();
        assert!(!schema_msg.app_metadata.is_empty());
        let decoded_schema: SchemaRef = wire::decode(&schema_msg.app_metadata).unwrap();
        assert_eq!(decoded_schema, schema);

        let batches = encoder.encode(&partition.record_batches()[0]).unwrap();
        assert!(!batches.is_empty());
        assert!(batches.iter().all(|d| d.app_metadata.is_empty()));
    }

    #[test]
    fn stats_message_is_decodable_and_carries_stats() {
        let partition = sample_partition();
        let schema = partition.schema();
        let mut encoder = BatchEncoder::try_new(&schema).unwrap();
        let stats = wire::QueryStatsWire {
            query_id: "q".to_string(),
            rows: 7,
            bytes: 128,
            physical_plan_json: Some("{}".to_string()),
            stats: vec![1, 2],
        };
        let message = encoder.stats_message(&stats).unwrap();
        // The trailer is a real (empty) batch message so stream decoders
        // accept it, with the statistics attached as metadata.
        assert!(!message.data_header.is_empty());
        let decoded: wire::QueryStatsWire = wire::decode(&message.app_metadata).unwrap();
        assert_eq!(decoded.rows, 7);
        assert_eq!(decoded.query_id, "q");
    }
}
