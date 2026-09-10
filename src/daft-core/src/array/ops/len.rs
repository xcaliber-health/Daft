use arrow::{
    array::ArrayData,
    buffer::{NullBuffer, OffsetBuffer},
    datatypes::{ArrowNativeType, DataType as ArrowDataType},
};

#[cfg(feature = "python")]
use crate::prelude::PythonArray;
use crate::{
    array::{DataArray, FixedSizeListArray, ListArray, StructArray, UnionArray},
    datatypes::{DaftArrowBackedType, FileArray},
    file::DaftMediaType,
};

impl<T> DataArray<T>
where
    T: DaftArrowBackedType + 'static,
{
    pub fn size_bytes(&self) -> usize {
        let data = self.to_data();
        let nulls = data.nulls().map(|n| n.buffer().len()).unwrap_or(0);
        // A slice narrows the offsets but not the values buffer they index.
        let buffers = match data.data_type() {
            ArrowDataType::Utf8 | ArrowDataType::Binary => variable_width_bytes::<i32>(&data),
            ArrowDataType::LargeUtf8 | ArrowDataType::LargeBinary => {
                variable_width_bytes::<i64>(&data)
            }
            _ => data.buffers().iter().map(|b| b.len()).sum(),
        };
        buffers + nulls
    }
}

/// Bytes of the offsets and of the values they span, for a possibly sliced array.
fn variable_width_bytes<O: ArrowNativeType + Into<i64>>(data: &ArrayData) -> usize {
    let offsets = &data.buffers()[0].typed_data::<O>()[data.offset()..=data.offset() + data.len()];
    let values: i64 = offsets[data.len()].into() - offsets[0].into();
    std::mem::size_of_val(offsets) + usize::try_from(values).unwrap_or(0)
}

#[cfg(feature = "python")]
impl PythonArray {
    /// Estimate the size of this list by sampling and pickling its objects.
    pub fn size_bytes(&self) -> usize {
        use std::cmp::min;

        use common_py_serde::pickle_dumps;
        use pyo3::Python;
        use rand::{SeedableRng, rngs::StdRng, seq::IndexedRandom as _};

        // Sample up to 1MB or 10000 items to determine total size.
        const MAX_SAMPLE_QUANTITY: usize = 10000;
        const MAX_SAMPLE_SIZE: usize = 1024 * 1024;

        if self.is_empty() {
            return 0;
        }

        let values = self.values();

        let mut rng = StdRng::seed_from_u64(0);
        let sample_candidates =
            values.choose_multiple(&mut rng, min(values.len(), MAX_SAMPLE_QUANTITY));

        let mut sample_size_allowed = MAX_SAMPLE_SIZE;
        let mut sampled_sizes = Vec::with_capacity(sample_candidates.len());
        Python::attach(|py| {
            for c in sample_candidates {
                // Just estimate to 0 if pickle_dumps fails.
                let size = pickle_dumps(py, c).map(|v| v.len()).unwrap_or(0);
                sampled_sizes.push(size);
                sample_size_allowed = sample_size_allowed.saturating_sub(size);

                if sample_size_allowed == 0 {
                    break;
                }
            }
        });

        if sampled_sizes.len() == values.len() {
            // Sampling complete.
            // If we ended up measuring the entire list, just return the exact value.

            sampled_sizes.into_iter().sum()
        } else {
            // Otherwise, reduce to a one-item estimate and extrapolate.

            let one_item_size_estimate = if sampled_sizes.len() == 1 {
                sampled_sizes[0]
            } else {
                let sampled_len = sampled_sizes.len() as f64;

                let mean: f64 = sampled_sizes.iter().map(|&x| x as f64).sum::<f64>() / sampled_len;
                let stdev: f64 = sampled_sizes
                    .iter()
                    .map(|&x| ((x as f64) - mean).powi(2))
                    .sum::<f64>()
                    / sampled_len;

                (mean + stdev) as usize
            };

            one_item_size_estimate * values.len()
        }
    }
}

fn null_buffer_size(nulls: Option<&NullBuffer>) -> usize {
    nulls.map(|b| b.buffer().len()).unwrap_or(0)
}

fn offset_size(offsets: &OffsetBuffer<i64>) -> usize {
    // OffsetBuffer::len() returns the number of offset values (N+1 for N rows)
    offsets.len() * std::mem::size_of::<i64>()
}

impl FixedSizeListArray {
    pub fn size_bytes(&self) -> usize {
        self.flat_child.size_bytes() + null_buffer_size(self.nulls())
    }
}

impl ListArray {
    pub fn size_bytes(&self) -> usize {
        self.flat_child.size_bytes() + null_buffer_size(self.nulls()) + offset_size(self.offsets())
    }
}

impl StructArray {
    pub fn size_bytes(&self) -> usize {
        let children_size_bytes: usize = self.children.iter().map(|s| s.size_bytes()).sum();
        children_size_bytes + null_buffer_size(self.nulls())
    }
}

impl UnionArray {
    pub fn size_bytes(&self) -> usize {
        let children_size_bytes: usize = self.children.iter().map(|s| s.size_bytes()).sum();
        let offset_bytes =
            self.offsets().clone().map(|b| b.len()).unwrap_or(0) * std::mem::size_of::<i32>();
        let ids_bytes = self.ids().len() * std::mem::size_of::<i8>();
        children_size_bytes + offset_bytes + ids_bytes
    }
}

impl<T> FileArray<T>
where
    T: DaftMediaType,
{
    pub fn size_bytes(&self) -> usize {
        self.physical.size_bytes()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        datatypes::{DataType, Field, Int64Array, Utf8Array},
        series::IntoSeries,
    };

    #[test]
    fn a_sliced_string_array_reports_only_its_own_bytes() {
        let values: Vec<String> = (0..1000).map(|i| format!("value-{i:04}")).collect();
        let array = Utf8Array::from_iter("s", values.iter().map(|v| Some(v.as_str()))).into_series();
        let whole = array.size_bytes();
        let part = array.slice(10, 20).unwrap().size_bytes();
        // ten offsets plus one, and ten values of ten bytes
        assert_eq!(part, 11 * 8 + 100);
        assert!(part * 50 < whole);
    }

    #[test]
    fn a_sliced_primitive_array_reports_only_its_own_bytes() {
        let array = Int64Array::from_iter(Field::new("i", DataType::Int64), (0..1000).map(Some))
            .into_series();
        assert_eq!(array.slice(0, 10).unwrap().size_bytes(), 80);
    }
}
