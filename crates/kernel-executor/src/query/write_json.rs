//! The `WriteJson` operator: writes its inputs to one newline-delimited JSON file.
//!
//! The inputs run to completion first, in the order the node lists them, so `ordered` writes
//! hold. Every batch is checked against the declared schema, and the file goes through the
//! executor's JSON handler: kernel's encoding, an atomic create unless `overwrite` is set, and
//! `FileAlreadyExists` when the destination exists. The node's rows are the written file's
//! metadata, one row in [`FILE_META_SCHEMA`].

use std::sync::Arc;

use datafusion::arrow::array::{Int64Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::logical_expr::LogicalPlan;
use delta_kernel::engine::arrow_conversion::TryIntoArrow as _;
use delta_kernel::engine::arrow_data::ArrowEngineData;
use delta_kernel::plans::ir::nodes::{FILE_META_SCHEMA, WriteJson};
use delta_kernel::{FilteredEngineData, JsonHandler as _, KernelError, StorageHandler as _};

use crate::DataFusionPlanExecutor;

/// Writes the rows of `inputs` to `op.file_path` and lowers to a plan over the file's metadata.
pub(crate) fn lower_write_json(
    executor: &DataFusionPlanExecutor,
    op: &WriteJson,
    inputs: &[Arc<LogicalPlan>],
) -> crate::Result<LogicalPlan> {
    if inputs.is_empty() {
        return Err(KernelError::generic("WriteJson requires at least one input").into());
    }
    let schema: ArrowSchemaRef = Arc::new(
        op.schema
            .as_ref()
            .try_into_arrow()
            .map_err(|err| KernelError::schema(format!("unsupported write schema: {err}")))?,
    );

    // Run every input before writing anything, so a failing input or a schema mismatch leaves
    // the destination untouched.
    let mut batches = Vec::new();
    for input in inputs {
        for batch in super::collect(executor, input.as_ref().clone())? {
            batches.push(super::conform_batch(&batch, &schema)?);
        }
    }
    if batches.iter().all(|batch| batch.num_rows() == 0) {
        return Err(KernelError::generic("WriteJson requires at least one input row").into());
    }

    let data = batches.into_iter().map(|batch| {
        Ok(FilteredEngineData::with_all_rows_selected(Box::new(
            ArrowEngineData::new(batch),
        )))
    });
    executor
        .file_formats()
        .write_json_file(&op.file_path, Box::new(data), op.overwrite)?;
    let meta = executor.storage().head(&op.file_path)?;

    let file_meta_schema: ArrowSchemaRef = Arc::new(FILE_META_SCHEMA.as_ref().try_into_arrow()?);
    let size = i64::try_from(meta.size)
        .map_err(|_| KernelError::generic("written file size exceeds the LONG range"))?;
    let row = RecordBatch::try_new(
        Arc::clone(&file_meta_schema),
        vec![
            Arc::new(StringArray::from(vec![meta.location.to_string()])),
            Arc::new(Int64Array::from(vec![size])),
            Arc::new(Int64Array::from(vec![meta.last_modified])),
        ],
    )?;
    super::memory_plan("write_json", file_meta_schema, vec![row])
}

#[cfg(test)]
mod tests {
    use datafusion::prelude::SessionContext;
    use delta_kernel::PlanBuilder;
    use delta_kernel::expressions::Scalar;
    use delta_kernel::plans::ir::plan::Plan;
    use delta_kernel::plans::{Operation, PlanExecutor as _, PlanResult};
    use delta_kernel::schema::{DataType, SchemaRef, StructField, StructType};
    use tokio::runtime::Handle;
    use url::Url;

    use super::*;

    fn executor() -> DataFusionPlanExecutor {
        DataFusionPlanExecutor::new(SessionContext::new().task_ctx(), Handle::current())
    }

    fn schema() -> SchemaRef {
        Arc::new(
            StructType::try_new([
                StructField::not_null("id", DataType::LONG),
                StructField::nullable("name", DataType::STRING),
            ])
            .unwrap(),
        )
    }

    fn values(rows: Vec<Vec<Scalar>>) -> PlanBuilder {
        PlanBuilder::values(schema(), rows).unwrap()
    }

    fn write_plan(destination: &Url, overwrite: bool, ordered: bool) -> Plan {
        let first = values(vec![
            vec![1i64.into(), "a".into()],
            vec![2i64.into(), Scalar::null(DataType::STRING)],
        ]);
        let second = values(vec![vec![3i64.into(), "c".into()]]);
        PlanBuilder::write_json([first, second], destination.clone(), overwrite, ordered)
            .unwrap()
            .build()
            .unwrap()
    }

    fn file_meta(result: PlanResult) -> RecordBatch {
        let PlanResult::Data(data) = result else {
            panic!("expected PlanResult::Data");
        };
        let mut batches: Vec<RecordBatch> = data
            .map(|data| {
                ArrowEngineData::try_from_engine_data(data.unwrap())
                    .unwrap()
                    .record_batch()
                    .clone()
            })
            .filter(|batch| batch.num_rows() > 0)
            .collect();
        assert_eq!(batches.len(), 1);
        batches.remove(0)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn writes_inputs_in_order_and_returns_file_meta() {
        let tmp = tempfile::tempdir().unwrap();
        let destination = Url::from_file_path(tmp.path().join("00000.json")).unwrap();

        let result = executor()
            .execute_op(Operation::QueryPlan(write_plan(&destination, false, true)))
            .unwrap();
        let meta = file_meta(result);

        let written = std::fs::read_to_string(tmp.path().join("00000.json")).unwrap();
        assert_eq!(
            written,
            "{\"id\":1,\"name\":\"a\"}\n{\"id\":2}\n{\"id\":3,\"name\":\"c\"}\n"
        );
        let location = meta
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0);
        assert_eq!(location, destination.as_str());
        let size = meta
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0);
        assert_eq!(size, written.len() as i64);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn existing_destination_is_kept_without_overwrite() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("00000.json");
        std::fs::write(&path, "existing\n").unwrap();
        let destination = Url::from_file_path(&path).unwrap();

        let err = executor()
            .execute_op(Operation::QueryPlan(write_plan(&destination, false, true)))
            .err()
            .expect("an existing destination must not be replaced");
        assert!(matches!(err, KernelError::FileAlreadyExists(_)), "{err}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "existing\n");

        executor()
            .execute_op(Operation::QueryPlan(write_plan(&destination, true, false)))
            .unwrap();
        assert_ne!(std::fs::read_to_string(&path).unwrap(), "existing\n");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn no_rows_is_rejected_and_writes_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("00000.json");
        let destination = Url::from_file_path(&path).unwrap();
        let plan = PlanBuilder::write_json([values(vec![])], destination, false, true)
            .and_then(|builder| builder.build());
        // The builder may already refuse an input without rows.
        if let Ok(plan) = plan {
            let err = executor()
                .execute_op(Operation::QueryPlan(plan))
                .err()
                .expect("a write without rows must fail");
            assert!(err.to_string().contains("at least one input row"), "{err}");
        }
        assert!(!path.exists());
    }
}
