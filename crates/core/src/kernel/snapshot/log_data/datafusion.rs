use std::collections::HashSet;
use std::sync::Arc;

use ::datafusion::functions_aggregate::min_max::{MaxAccumulator, MinAccumulator};
use ::datafusion::physical_optimizer::pruning::PruningStatistics;
use ::datafusion::physical_plan::Accumulator;
use arrow::compute::concat_batches;
use arrow_arith::aggregate::sum;
use arrow_array::{ArrayRef, BooleanArray, Int64Array};
use arrow_schema::DataType as ArrowDataType;
use datafusion_common::scalar::ScalarValue;
use datafusion_common::stats::{ColumnStatistics, Precision, Statistics};
use datafusion_common::Column;
use delta_kernel::engine::arrow_data::ArrowEngineData;
use delta_kernel::expressions::Expression;
use delta_kernel::schema::{DataType, PrimitiveType};
use delta_kernel::{ExpressionEvaluator, ExpressionHandler};

use super::*;
use crate::kernel::arrow::extract::{extract_and_cast_opt, extract_column};
use crate::kernel::ARROW_HANDLER;

#[derive(Debug, Default, Clone)]
enum AccumulatorType {
    Min,
    Max,
    #[default]
    Unused,
}
// TODO validate this works with "wide and narrow" builds / stats

impl FileStatsAccessor<'_> {
    fn collect_count(&self, name: &str) -> Precision<usize> {
        let num_records = extract_and_cast_opt::<Int64Array>(self.stats, name);
        if let Some(num_records) = num_records {
            if let Some(null_count_mulls) = num_records.nulls() {
                if null_count_mulls.null_count() > 0 {
                    Precision::Absent
                } else {
                    sum(num_records)
                        .map(|s| Precision::Exact(s as usize))
                        .unwrap_or(Precision::Absent)
                }
            } else {
                sum(num_records)
                    .map(|s| Precision::Exact(s as usize))
                    .unwrap_or(Precision::Absent)
            }
        } else {
            Precision::Absent
        }
    }

    fn column_bounds(
        &self,
        path_step: &str,
        name: &str,
        fun_type: AccumulatorType,
    ) -> Precision<ScalarValue> {
        let mut path = name.split('.');
        let array = if let Ok(array) = extract_column(self.stats, path_step, &mut path) {
            array
        } else {
            return Precision::Absent;
        };

        if array.data_type().is_primitive() {
            let accumulator: Option<Box<dyn Accumulator>> =
                match fun_type {
                    AccumulatorType::Min => MinAccumulator::try_new(array.data_type())
                        .map_or(None, |a| Some(Box::new(a))),
                    AccumulatorType::Max => MaxAccumulator::try_new(array.data_type())
                        .map_or(None, |a| Some(Box::new(a))),
                    _ => None,
                };

            if let Some(mut accumulator) = accumulator {
                return accumulator
                    .update_batch(&[array.clone()])
                    .ok()
                    .and_then(|_| accumulator.evaluate().ok())
                    .map(Precision::Exact)
                    .unwrap_or(Precision::Absent);
            }

            return Precision::Absent;
        }

        match array.data_type() {
            ArrowDataType::Struct(fields) => {
                return fields
                    .iter()
                    .map(|f| {
                        self.column_bounds(
                            path_step,
                            &format!("{name}.{}", f.name()),
                            fun_type.clone(),
                        )
                    })
                    .map(|s| match s {
                        Precision::Exact(s) => Some(s),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>()
                    .map(|o| {
                        let arrays = o
                            .into_iter()
                            .map(|sv| sv.to_array())
                            .collect::<Result<Vec<_>, datafusion_common::DataFusionError>>()
                            .unwrap();
                        let sa = StructArray::new(fields.clone(), arrays, None);
                        Precision::Exact(ScalarValue::Struct(Arc::new(sa)))
                    })
                    .unwrap_or(Precision::Absent);
            }
            _ => Precision::Absent,
        }
    }

    fn num_records(&self) -> Precision<usize> {
        self.collect_count(COL_NUM_RECORDS)
    }

    fn total_size_files(&self) -> Precision<usize> {
        let size = self
            .sizes
            .iter()
            .flat_map(|s| s.map(|s| s as usize))
            .sum::<usize>();
        Precision::Inexact(size)
    }

    fn column_stats(&self, name: impl AsRef<str>) -> DeltaResult<ColumnStatistics> {
        let null_count_col = format!("{COL_NULL_COUNT}.{}", name.as_ref());
        let null_count = self.collect_count(&null_count_col);

        let min_value = self.column_bounds(COL_MIN_VALUES, name.as_ref(), AccumulatorType::Min);
        let min_value = match &min_value {
            Precision::Exact(value) if value.is_null() => Precision::Absent,
            // TODO this is a hack, we should not be casting here but rather when we read the checkpoint data.
            // it seems sometimes the min/max values are stored as nanoseconds and sometimes as microseconds?
            Precision::Exact(ScalarValue::TimestampNanosecond(a, b)) => Precision::Exact(
                ScalarValue::TimestampMicrosecond(a.map(|v| v / 1000), b.clone()),
            ),
            _ => min_value,
        };

        let max_value = self.column_bounds(COL_MAX_VALUES, name.as_ref(), AccumulatorType::Max);
        let max_value = match &max_value {
            Precision::Exact(value) if value.is_null() => Precision::Absent,
            Precision::Exact(ScalarValue::TimestampNanosecond(a, b)) => Precision::Exact(
                ScalarValue::TimestampMicrosecond(a.map(|v| v / 1000), b.clone()),
            ),
            _ => max_value,
        };

        Ok(ColumnStatistics {
            null_count,
            max_value,
            min_value,
            distinct_count: Precision::Absent,
        })
    }
}

trait StatsExt {
    fn add(&self, other: &Self) -> Self;
}

impl StatsExt for ColumnStatistics {
    fn add(&self, other: &Self) -> Self {
        Self {
            null_count: self.null_count.add(&other.null_count),
            max_value: self.max_value.max(&other.max_value),
            min_value: self.min_value.min(&other.min_value),
            distinct_count: self.distinct_count.add(&other.distinct_count),
        }
    }
}

impl LogDataHandler<'_> {
    fn num_records(&self) -> Precision<usize> {
        self.data
            .iter()
            .flat_map(|b| {
                FileStatsAccessor::try_new(b, self.metadata, self.schema).map(|a| a.num_records())
            })
            .reduce(|acc, num_records| acc.add(&num_records))
            .unwrap_or(Precision::Absent)
    }

    fn total_size_files(&self) -> Precision<usize> {
        self.data
            .iter()
            .flat_map(|b| {
                FileStatsAccessor::try_new(b, self.metadata, self.schema)
                    .map(|a| a.total_size_files())
            })
            .reduce(|acc, size| acc.add(&size))
            .unwrap_or(Precision::Absent)
    }

    pub(crate) fn column_stats(&self, name: impl AsRef<str>) -> Option<ColumnStatistics> {
        self.data
            .iter()
            .flat_map(|b| {
                FileStatsAccessor::try_new(b, self.metadata, self.schema)
                    .map(|a| a.column_stats(name.as_ref()))
            })
            .collect::<Result<Vec<_>, _>>()
            .ok()?
            .iter()
            .fold(None::<ColumnStatistics>, |acc, stats| match (acc, stats) {
                (None, stats) => Some(stats.clone()),
                (Some(acc), stats) => Some(acc.add(stats)),
            })
    }

    /// Get datafusion table statistics for the loaded log data
    pub fn statistics(&self) -> Option<Statistics> {
        let num_rows = self.num_records();
        let total_byte_size = self.total_size_files();
        let column_statistics = self
            .schema
            .fields()
            .map(|f| self.column_stats(f.name()))
            .collect::<Option<Vec<_>>>()?;
        Some(Statistics {
            num_rows,
            total_byte_size,
            column_statistics,
        })
    }

    fn pick_stats(&self, column: &Column, stats_field: &'static str) -> Option<ArrayRef> {
        let field = self.schema.field(&column.name)?;
        // See issue #1214. Binary type does not support natural order which is required for Datafusion to prune
        if field.data_type() == &DataType::Primitive(PrimitiveType::Binary) {
            return None;
        }
        let expression = if self.metadata.partition_columns.contains(&column.name) {
            Expression::Column(format!("add.partitionValues_parsed.{}", column.name))
        } else {
            Expression::Column(format!("add.stats_parsed.{}.{}", stats_field, column.name))
        };
        let evaluator = ARROW_HANDLER.get_evaluator(
            crate::kernel::models::fields::log_schema_ref().clone(),
            expression,
            field.data_type().clone(),
        );
        let mut results = Vec::with_capacity(self.data.len());
        for batch in self.data.iter() {
            let engine = ArrowEngineData::new(batch.clone());
            let result = evaluator.evaluate(&engine).ok()?;
            let result = result
                .as_any()
                .downcast_ref::<ArrowEngineData>()
                .ok_or(DeltaTableError::generic(
                    "failed to downcast evaluator result to ArrowEngineData.",
                ))
                .ok()?;
            results.push(result.record_batch().clone());
        }
        let batch = concat_batches(results[0].schema_ref(), &results).ok()?;
        batch.column_by_name("output").map(|c| c.clone())
    }
}

impl<'a> PruningStatistics for LogDataHandler<'a> {
    /// return the minimum values for the named column, if known.
    /// Note: the returned array must contain `num_containers()` rows
    fn min_values(&self, column: &Column) -> Option<ArrayRef> {
        self.pick_stats(column, "minValues")
    }

    /// return the maximum values for the named column, if known.
    /// Note: the returned array must contain `num_containers()` rows.
    fn max_values(&self, column: &Column) -> Option<ArrayRef> {
        self.pick_stats(column, "maxValues")
    }

    /// return the number of containers (e.g. row groups) being
    /// pruned with these statistics
    fn num_containers(&self) -> usize {
        self.data.iter().map(|f| f.num_rows()).sum()
    }

    /// return the number of null values for the named column as an
    /// `Option<UInt64Array>`.
    ///
    /// Note: the returned array must contain `num_containers()` rows.
    fn null_counts(&self, column: &Column) -> Option<ArrayRef> {
        if !self.metadata.partition_columns.contains(&column.name) {
            let counts = self.pick_stats(column, "nullCount")?;
            return arrow_cast::cast(counts.as_ref(), &ArrowDataType::UInt64).ok();
        }
        let partition_values = self.pick_stats(column, "__dummy__")?;
        let row_counts = self.row_counts(column)?;
        let row_counts = row_counts.as_any().downcast_ref::<UInt64Array>()?;
        let mut null_counts = Vec::with_capacity(partition_values.len());
        for i in 0..partition_values.len() {
            let null_count = if partition_values.is_null(i) {
                row_counts.value(i)
            } else {
                0
            };
            null_counts.push(null_count);
        }
        Some(Arc::new(UInt64Array::from(null_counts)))
    }

    /// return the number of rows for the named column in each container
    /// as an `Option<UInt64Array>`.
    ///
    /// Note: the returned array must contain `num_containers()` rows
    fn row_counts(&self, _column: &Column) -> Option<ArrayRef> {
        lazy_static::lazy_static! {
            static ref ROW_COUNTS_EVAL: Arc<dyn ExpressionEvaluator> =  ARROW_HANDLER.get_evaluator(
                crate::kernel::models::fields::log_schema_ref().clone(),
                Expression::column("add.stats_parsed.numRecords"),
                DataType::Primitive(PrimitiveType::Long),
            );
        }
        let mut results = Vec::with_capacity(self.data.len());
        for batch in self.data.iter() {
            let engine = ArrowEngineData::new(batch.clone());
            let result = ROW_COUNTS_EVAL.evaluate(&engine).ok()?;
            let result = result
                .as_any()
                .downcast_ref::<ArrowEngineData>()
                .ok_or(DeltaTableError::generic(
                    "failed to downcast evaluator result to ArrowEngineData.",
                ))
                .ok()?;
            results.push(result.record_batch().clone());
        }
        let batch = concat_batches(results[0].schema_ref(), &results).ok()?;
        arrow_cast::cast(batch.column_by_name("output")?, &ArrowDataType::UInt64).ok()
    }

    // This function is required since DataFusion 35.0, but is implemented as a no-op
    // https://github.com/apache/arrow-datafusion/blob/ec6abece2dcfa68007b87c69eefa6b0d7333f628/datafusion/core/src/datasource/physical_plan/parquet/page_filter.rs#L550
    fn contained(&self, _column: &Column, _value: &HashSet<ScalarValue>) -> Option<BooleanArray> {
        None
    }
}

#[cfg(test)]
mod tests {

    #[tokio::test]
    async fn read_delta_1_2_1_struct_stats_table() {
        let table_uri = "../test/tests/data/delta-1.2.1-only-struct-stats";
        let table_from_struct_stats = crate::open_table(table_uri).await.unwrap();
        let table_from_json_stats = crate::open_table_with_version(table_uri, 1).await.unwrap();

        let json_action = table_from_json_stats
            .snapshot()
            .unwrap()
            .snapshot
            .files()
            .find(|f| {
                f.path().ends_with(
                    "part-00000-7a509247-4f58-4453-9202-51d75dee59af-c000.snappy.parquet",
                )
            })
            .unwrap();

        let struct_action = table_from_struct_stats
            .snapshot()
            .unwrap()
            .snapshot
            .files()
            .find(|f| {
                f.path().ends_with(
                    "part-00000-7a509247-4f58-4453-9202-51d75dee59af-c000.snappy.parquet",
                )
            })
            .unwrap();

        assert_eq!(json_action.path(), struct_action.path());
        assert_eq!(
            json_action.partition_values().unwrap(),
            struct_action.partition_values().unwrap()
        );
        // assert_eq!(
        //     json_action.max_values().unwrap(),
        //     struct_action.max_values().unwrap()
        // );
        // assert_eq!(
        //     json_action.min_values().unwrap(),
        //     struct_action.min_values().unwrap()
        // );
    }

    #[tokio::test]
    async fn df_stats_delta_1_2_1_struct_stats_table() {
        let table_uri = "../test/tests/data/delta-1.2.1-only-struct-stats";
        let table_from_struct_stats = crate::open_table(table_uri).await.unwrap();

        let file_stats = table_from_struct_stats
            .snapshot()
            .unwrap()
            .snapshot
            .log_data();

        let col_stats = file_stats.statistics();
        println!("{:?}", col_stats);
    }
}
