use common_error::{DaftError, DaftResult};

use super::DataType;

/// Get the data type that the sum of a column of the given data type should be casted to.
pub fn try_sum_supertype(dtype: &DataType) -> DaftResult<DataType> {
    match dtype {
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => Ok(DataType::Int64),
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            Ok(DataType::UInt64)
        }
        DataType::Float32 => Ok(DataType::Float32),
        DataType::Float16 => Ok(DataType::Float16),
        DataType::Float64 => Ok(DataType::Float64),
        DataType::Decimal128(_, s) => Ok(DataType::Decimal128(38, *s)),
        other => Err(DaftError::TypeError(format!(
            "Invalid argument to sum supertype: {}",
            other
        ))),
    }
}

/// Get the data type that the product of a column of the given data type should be casted to.
pub fn try_product_supertype(dtype: &DataType) -> DaftResult<DataType> {
    match dtype {
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => Ok(DataType::Int64),
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            Ok(DataType::UInt64)
        }
        DataType::Float32 => Ok(DataType::Float32),
        DataType::Float16 => Ok(DataType::Float16),
        DataType::Float64 => Ok(DataType::Float64),
        DataType::Decimal128(_, s) => Ok(DataType::Decimal128(38, *s)),
        other => Err(DaftError::TypeError(format!(
            "Invalid argument to product supertype: {}",
            other
        ))),
    }
}

/// Get the data type that the mean of a column of the given data type should be casted to.
pub fn try_mean_aggregation_supertype(dtype: &DataType) -> DaftResult<DataType> {
    match dtype {
        d if d.is_numeric() => Ok(DataType::Float64),
        DataType::Decimal128(_, s) => {
            const P_PRIME: usize = 38;

            let s_max = std::cmp::min(P_PRIME, s + 4);

            if s_max > 38 {
                Err(DaftError::TypeError(format!(
                    "Cannot infer supertypes for mean on type: {} result scale: {s_max} exceed bounds of [0, 38]",
                    dtype
                )))
            } else if s_max > P_PRIME {
                Err(DaftError::TypeError(format!(
                    "Cannot infer supertypes for mean on type: {} result scale: {s_max} exceed precision {P_PRIME}",
                    dtype
                )))
            } else {
                Ok(DataType::Decimal128(P_PRIME, s_max))
            }
        }
        _ => Err(DaftError::TypeError(format!(
            "Mean is not supported for: {}",
            dtype
        ))),
    }
}

/// Get the data type that the stddev of a column of the given data type should be casted to.
pub fn try_stddev_aggregation_supertype(dtype: &DataType) -> DaftResult<DataType> {
    match dtype {
        d if d.is_numeric() => Ok(DataType::Float64),
        DataType::Decimal128(..) => Ok(DataType::Float64),
        DataType::Null => Ok(DataType::Float64),
        _ => Err(DaftError::TypeError(format!(
            "StdDev is not supported for: {}",
            dtype
        ))),
    }
}

/// Get the data type that the variance of a column of the given data type should be casted to.
pub fn try_variance_aggregation_supertype(dtype: &DataType) -> DaftResult<DataType> {
    match dtype {
        d if d.is_numeric() => Ok(DataType::Float64),
        DataType::Decimal128(..) => Ok(DataType::Float64),
        DataType::Null => Ok(DataType::Float64),
        _ => Err(DaftError::TypeError(format!(
            "Variance is not supported for: {}",
            dtype
        ))),
    }
}

/// Get the data type that the skew of a column of the given data type should be casted to.
pub fn try_skew_aggregation_supertype(dtype: &DataType) -> DaftResult<DataType> {
    match dtype {
        d if d.is_numeric() => Ok(DataType::Float64),
        DataType::Decimal128(..) => Ok(DataType::Float64),
        _ => Err(DaftError::TypeError(format!(
            "Skew is not supported for: {}",
            dtype
        ))),
    }
}

/// Whether percentile-like aggregations accept values of `dtype`.
#[must_use]
pub fn is_percentile_input(dtype: &DataType) -> bool {
    dtype.is_numeric() || matches!(dtype, DataType::Decimal128(..))
}

/// Get the data type that percentile-like aggregations should be casted to.
///
/// A decimal stays a decimal, settled as its mean is: an interpolated value needs
/// more places than the input keeps, so the scale widens as the mean's does. Every
/// other accepted input answers `Float64`.
pub fn try_percentile_aggregation_supertype(dtype: &DataType) -> DaftResult<DataType> {
    match dtype {
        DataType::Decimal128(..) => try_mean_aggregation_supertype(dtype),
        d if is_percentile_input(d) => Ok(DataType::Float64),
        DataType::List(inner) | DataType::FixedSizeList(inner, _) if is_percentile_input(inner) => {
            try_percentile_aggregation_supertype(inner)
        }
        other => Err(DaftError::TypeError(format!(
            "Invalid argument to percentile supertype: {other}"
        ))),
    }
}
