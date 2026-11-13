//! The `DynamicScan` operator: scans the files named by its input's rows.
//!
//! The file list is data, not part of the plan, so the input runs to completion first. Each
//! input row becomes one [`ScanFile`]: its path resolved against `base_url`, its size and
//! modification time, and its file-constant column values. The files then go through the same
//! scan lowering as `ScanParquet` / `ScanJson` ([`super::scan`]).
//!
//! Deletion vectors are not applied yet. A row with a deletion vector is an error rather than a
//! scan that silently returns deleted rows; rows whose deletion-vector column is NULL scan
//! normally.

use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, AsArray as _, RecordBatch};
use datafusion::arrow::datatypes::{DataType as ArrowDataType, Int64Type};
use datafusion::common::ScalarValue;
use datafusion::logical_expr::LogicalPlan;
use delta_kernel::expressions::ColumnName;
use delta_kernel::plans::ir::nodes::{DynamicScan, ScanFile};
use delta_kernel::{FileMeta, KernelError};

use crate::expressions::datafusion_scalar_to_scalar;
use crate::{DataFusionPlanExecutor, ExecutorError};

/// Runs `input`, then lowers a scan over the files its rows name.
pub(crate) fn lower_dynamic_scan(
    executor: &DataFusionPlanExecutor,
    op: &DynamicScan,
    input: &Arc<LogicalPlan>,
) -> crate::Result<LogicalPlan> {
    let batches = super::collect(executor, input.as_ref().clone())?;
    let files = scan_files(op, &batches)?;
    super::scan::lower_scan(op.file_type, &files, &op.file_constant_columns, &op.schema)
}

/// One [`ScanFile`] per row of `batches`.
fn scan_files(op: &DynamicScan, batches: &[RecordBatch]) -> crate::Result<Vec<ScanFile>> {
    let mut files = Vec::new();
    for batch in batches {
        let (paths, _) = column(batch, &op.path_column)?;
        let sizes = long_column(batch, &op.file_size_column)?;
        let last_modified = long_column(batch, &op.last_modified_column)?;
        let deletion_vectors = op
            .dv_column
            .as_ref()
            .map(|dv_column| column(batch, dv_column))
            .transpose()?;
        let constants = op
            .file_constant_columns
            .iter()
            .map(|name| Ok(column(batch, &ColumnName::new([name.as_str()]))?.0))
            .collect::<crate::Result<Vec<_>>>()?;

        for row in 0..batch.num_rows() {
            let path = string_value(&paths, row)
                .ok_or_else(|| KernelError::generic("DynamicScan path must not be null"))?;
            if let Some((dv, ancestors)) = &deletion_vectors
                && dv.is_valid(row)
                && ancestors.iter().all(|ancestor| ancestor.is_valid(row))
            {
                return Err(ExecutorError::unsupported(format!(
                    "DynamicScan over `{path}`, which has a deletion vector"
                )));
            }
            if sizes.is_null(row) {
                return Err(KernelError::generic("DynamicScan file size must not be null").into());
            }
            let size = u64::try_from(sizes.value(row))
                .ok()
                .filter(|&size| size > 0)
                .ok_or_else(|| KernelError::generic("DynamicScan file size must be positive"))?;
            if last_modified.is_null(row) {
                return Err(KernelError::generic(
                    "DynamicScan last-modified time must not be null",
                )
                .into());
            }
            let location = op.base_url.join(&path).map_err(|err| {
                KernelError::generic(format!(
                    "DynamicScan path `{path}` does not resolve against `{}`: {err}",
                    op.base_url
                ))
            })?;
            let file_constants = constants
                .iter()
                .map(|column| {
                    let value = ScalarValue::try_from_array(column, row)?;
                    Ok(datafusion_scalar_to_scalar(&value)?)
                })
                .collect::<crate::Result<Vec<_>>>()?;
            files.push(ScanFile {
                meta: FileMeta {
                    location,
                    last_modified: last_modified.value(row),
                    size,
                },
                file_constants,
            });
        }
    }
    Ok(files)
}

/// The column at `name`'s path, and the struct columns enclosing it, outermost first.
///
/// A row is NULL at a nested path when the leaf or any enclosing struct is NULL.
fn column(batch: &RecordBatch, name: &ColumnName) -> crate::Result<(ArrayRef, Vec<ArrayRef>)> {
    let not_found = || KernelError::generic(format!("DynamicScan input has no column `{name}`"));
    let mut path = name.path().iter();
    let first = path.next().ok_or_else(not_found)?;
    let mut current = batch.column_by_name(first).cloned().ok_or_else(not_found)?;
    let mut ancestors = Vec::new();
    for field in path {
        let child = current
            .as_struct_opt()
            .and_then(|parent| parent.column_by_name(field))
            .cloned()
            .ok_or_else(not_found)?;
        ancestors.push(std::mem::replace(&mut current, child));
    }
    Ok((current, ancestors))
}

/// A `LONG` column, NULL where an enclosing struct is NULL.
fn long_column(
    batch: &RecordBatch,
    name: &ColumnName,
) -> crate::Result<datafusion::arrow::array::Int64Array> {
    let (column, ancestors) = column(batch, name)?;
    let values = column.as_primitive_opt::<Int64Type>().ok_or_else(|| {
        KernelError::generic(format!(
            "DynamicScan column `{name}` must be LONG, found {}",
            column.data_type()
        ))
    })?;
    if ancestors.is_empty() {
        return Ok(values.clone());
    }
    Ok(values
        .iter()
        .enumerate()
        .map(|(row, value)| value.filter(|_| ancestors.iter().all(|a| a.is_valid(row))))
        .collect())
}

/// The string at `row`, for any of Arrow's string types.
fn string_value(column: &ArrayRef, row: usize) -> Option<String> {
    if column.is_null(row) {
        return None;
    }
    match column.data_type() {
        ArrowDataType::Utf8 => Some(column.as_string::<i32>().value(row).to_string()),
        ArrowDataType::LargeUtf8 => Some(column.as_string::<i64>().value(row).to_string()),
        ArrowDataType::Utf8View => Some(column.as_string_view().value(row).to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::array::{Int32Array, Int64Array, StringArray};
    use datafusion::arrow::datatypes::{Field, Schema};
    use datafusion::assert_batches_sorted_eq;
    use datafusion::prelude::SessionContext;
    use delta_kernel::PlanBuilder;
    use delta_kernel::expressions::{Scalar, column_name};
    use delta_kernel::plans::ir::nodes::FileType;
    use delta_kernel::plans::{Operation, PlanExecutor as _, PlanResult};
    use delta_kernel::schema::{DataType, StructField, StructType};
    use tokio::runtime::Handle;
    use url::Url;

    use super::*;

    fn executor() -> DataFusionPlanExecutor {
        DataFusionPlanExecutor::new(SessionContext::new().task_ctx(), Handle::current())
    }

    /// Writes one parquet file with an `id` int column and returns its size.
    fn write_parquet(path: &std::path::Path, ids: Vec<i32>) -> i64 {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "id",
            ArrowDataType::Int32,
            false,
        )]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(ids))]).unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer =
            datafusion::parquet::arrow::ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        std::fs::metadata(path).unwrap().len() as i64
    }

    fn input_schema() -> Arc<StructType> {
        Arc::new(
            StructType::try_new([
                StructField::not_null("path", DataType::STRING),
                StructField::not_null("size", DataType::LONG),
                StructField::not_null("modified", DataType::LONG),
                StructField::not_null("version", DataType::LONG),
            ])
            .unwrap(),
        )
    }

    fn output_schema() -> Arc<StructType> {
        Arc::new(
            StructType::try_new([
                StructField::not_null("id", DataType::INTEGER),
                StructField::not_null("version", DataType::LONG),
            ])
            .unwrap(),
        )
    }

    fn dynamic_scan(base_url: Url) -> DynamicScan {
        DynamicScan::try_new(
            &input_schema(),
            output_schema(),
            FileType::Parquet,
            base_url,
            ["version"],
            column_name!("path"),
            column_name!("size"),
            column_name!("modified"),
            None,
        )
        .unwrap()
    }

    fn collect_result(result: PlanResult) -> Vec<RecordBatch> {
        let PlanResult::Data(data) = result else {
            panic!("expected PlanResult::Data");
        };
        data.map(|data| {
            delta_kernel::engine::arrow_data::ArrowEngineData::try_from_engine_data(data.unwrap())
                .unwrap()
                .record_batch()
                .clone()
        })
        .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn scans_the_files_named_by_its_input() {
        let tmp = tempfile::tempdir().unwrap();
        let size_a = write_parquet(&tmp.path().join("a.parquet"), vec![1, 2]);
        let size_b = write_parquet(&tmp.path().join("b.parquet"), vec![3]);
        let base_url = Url::from_directory_path(tmp.path()).unwrap();

        let plan = PlanBuilder::values(
            input_schema(),
            vec![
                vec!["a.parquet".into(), size_a.into(), 0i64.into(), 7i64.into()],
                vec!["b.parquet".into(), size_b.into(), 0i64.into(), 8i64.into()],
            ],
        )
        .unwrap()
        .dynamic_scan(dynamic_scan(base_url))
        .unwrap()
        .build()
        .unwrap();
        let batches = collect_result(executor().execute_op(Operation::QueryPlan(plan)).unwrap());
        assert_batches_sorted_eq!(
            &[
                "+----+---------+",
                "| id | version |",
                "+----+---------+",
                "| 1  | 7       |",
                "| 2  | 7       |",
                "| 3  | 8       |",
                "+----+---------+",
            ],
            &batches
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn no_input_rows_scans_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let base_url = Url::from_directory_path(tmp.path()).unwrap();
        let files = scan_files(&dynamic_scan(base_url), &[]).unwrap();
        assert!(files.is_empty());
    }

    #[test]
    fn rows_become_scan_files() {
        let base_url = Url::parse("memory://table/").unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("path", ArrowDataType::Utf8, false),
            Field::new("size", ArrowDataType::Int64, false),
            Field::new("modified", ArrowDataType::Int64, false),
            Field::new("version", ArrowDataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![
                    "part-0.parquet",
                    "sub/part-1.parquet",
                ])),
                Arc::new(Int64Array::from(vec![10, 20])),
                Arc::new(Int64Array::from(vec![100, 200])),
                Arc::new(Int64Array::from(vec![7, 8])),
            ],
        )
        .unwrap();
        let files = scan_files(&dynamic_scan(base_url), &[batch]).unwrap();
        assert_eq!(files.len(), 2);
        assert_eq!(
            files[1].meta.location.as_str(),
            "memory://table/sub/part-1.parquet"
        );
        assert_eq!(files[1].meta.size, 20);
        assert_eq!(files[1].meta.last_modified, 200);
        assert_eq!(files[1].file_constants, vec![Scalar::Long(8)]);
    }

    #[test]
    fn non_positive_size_is_rejected() {
        let base_url = Url::parse("memory://table/").unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("path", ArrowDataType::Utf8, false),
            Field::new("size", ArrowDataType::Int64, false),
            Field::new("modified", ArrowDataType::Int64, false),
            Field::new("version", ArrowDataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["part-0.parquet"])),
                Arc::new(Int64Array::from(vec![0])),
                Arc::new(Int64Array::from(vec![100])),
                Arc::new(Int64Array::from(vec![7])),
            ],
        )
        .unwrap();
        let err = scan_files(&dynamic_scan(base_url), &[batch]).unwrap_err();
        assert!(err.to_string().contains("must be positive"), "{err}");
    }
}
