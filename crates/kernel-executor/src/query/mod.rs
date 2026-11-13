//! Translates a kernel [`Plan`] into a DataFusion `LogicalPlan` and runs it.
//!
//! [`Plan::nodes`] is topologically sorted: a node's inputs always come earlier in the slice.
//! [`to_logical_plan`] walks the nodes in order, builds one `LogicalPlan` per node from its
//! already translated inputs ([`operator`]), and returns the plan of the terminal (last) node.
//! Expressions and predicates inside operators go through [`crate::expressions`].
//!
//! Most operators lower lazily into the `LogicalPlan`. Three need data while the plan is being
//! built, so they run their inputs first and splice the result in as an in-memory table:
//!
//! - `RelationSource` reads a relation retained by a scoped executor ([`relation`]).
//! - `DynamicScan` reads the files named by its input's rows ([`operator`]).
//! - `WriteJson` writes its inputs to a file and yields the file's metadata ([`write_json`]).

use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::compute::{can_cast_types, cast};
use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::common::DataFusionError;
use datafusion::datasource::{MemTable, provider_as_source};
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::{LogicalPlan, LogicalPlanBuilder, ident};
use delta_kernel::KernelResult as DeltaResult;
use delta_kernel::plans::PlanResult;
use delta_kernel::plans::ir::plan::Plan;

use crate::{DataFusionPlanExecutor, ExecutorError};

mod declared_cast;
mod dynamic_scan;
pub mod operator;
pub mod relation;
pub mod scan;
pub mod write_json;

/// Translates `plan`, runs it on a session built from the executor's task context, and streams
/// the terminal node's rows as [`PlanResult::Data`].
pub fn execute(executor: &DataFusionPlanExecutor, plan: Plan) -> DeltaResult<PlanResult> {
    let logical = to_logical_plan(executor, &plan)?;
    let session = session(executor);
    let handle = crate::runtime::query_handle(executor.handle())?;
    let stream = crate::runtime::block_on(&handle, async move {
        session
            .execute_logical_plan(logical)
            .await?
            .execute_stream()
            .await
    })
    .map_err(ExecutorError::from)?;

    Ok(PlanResult::Data(crate::stream::into_engine_data(
        &handle, stream,
    )))
}

/// The session plans run on: the executor's task context with file scans kept whole.
fn session(executor: &DataFusionPlanExecutor) -> SessionContext {
    let ctx = executor.task_ctx();
    let mut config = ctx.session_config().clone();

    // A scan gives every file its own partition. Splitting a file further by byte range would
    // interleave its rows, and the kernel's handler contract wants them in file order.
    config.options_mut().optimizer.repartition_file_scans = false;
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_runtime_env(ctx.runtime_env())
        .with_default_features()
        .build();
    let rules = state
        .optimizers()
        .iter()
        .filter(|rule| !LEAF_PUSHDOWN_RULES.contains(&rule.name()))
        .cloned()
        .collect();
    let state = SessionStateBuilder::new_from_existing(state)
        .with_optimizer_rules(rules)
        .build();
    SessionContext::new_with_state(state)
}

/// Optimizer rules the executor's sessions leave out.
///
/// DataFusion 55's leaf-expression pushdown fails with `DuplicateUnqualifiedField` when it moves
/// a `get_field` below a projection that renames its input's columns, which every lowered scan
/// has (`scan_json.add AS add`). Kernel's metadata scan plan (`Scan::declarative_metadata_scan_plan`)
/// hits it on every table. Fixed upstream by apache/datafusion#25478 (issue #25446), which
/// first ships in DataFusion 56; restore the rules with that upgrade.
const LEAF_PUSHDOWN_RULES: &[&str] = &["extract_leaf_expressions", "push_down_leaf_projections"];

/// Runs `logical` to completion and returns all of its batches.
///
/// Used by the operators that need their input's rows while the plan is being lowered. It
/// blocks the calling thread, like [`execute`] does while it waits for the stream.
pub(crate) fn collect(
    executor: &DataFusionPlanExecutor,
    logical: LogicalPlan,
) -> crate::Result<Vec<RecordBatch>> {
    let session = session(executor);
    let handle = crate::runtime::query_handle(executor.handle())?;
    Ok(crate::runtime::block_on(&handle, async move {
        session.execute_logical_plan(logical).await?.collect().await
    })?)
}

/// A `LogicalPlan` that reads `batches`, with unqualified fields like every other operator's.
pub(crate) fn memory_plan(
    table_name: &str,
    schema: ArrowSchemaRef,
    batches: Vec<RecordBatch>,
) -> crate::Result<LogicalPlan> {
    let columns = schema
        .fields()
        .iter()
        .map(|field| ident(field.name()).alias(field.name()))
        .collect::<Vec<_>>();
    let table = MemTable::try_new(Arc::clone(&schema), vec![batches])?;
    Ok(
        LogicalPlanBuilder::scan(table_name, provider_as_source(Arc::new(table)), None)?
            .project(columns)?
            .build()?,
    )
}

/// Rebuilds `batch` with `schema`, the schema the kernel declared for it.
///
/// DataFusion may report a column as nullable, or a nested field with another nullability or
/// name, where the kernel declared it otherwise. Such a column is cast to the declared type; a
/// column whose type cannot be cast, or a NULL in a non-nullable field, is an error.
pub(crate) fn conform_batch(
    batch: &RecordBatch,
    schema: &ArrowSchemaRef,
) -> crate::Result<RecordBatch> {
    if batch.schema_ref() == schema {
        return Ok(batch.clone());
    }
    if batch.num_columns() != schema.fields().len() {
        return Err(ExecutorError::Kernel(delta_kernel::KernelError::schema(
            format!(
                "a batch with {} column(s) does not match the declared schema with {}",
                batch.num_columns(),
                schema.fields().len()
            ),
        )));
    }
    let columns = batch
        .columns()
        .iter()
        .zip(batch.schema_ref().fields())
        .zip(schema.fields())
        .map(|((column, actual), declared)| {
            if actual.name() != declared.name() {
                return Err(ExecutorError::Kernel(delta_kernel::KernelError::schema(
                    format!(
                        "column `{}` does not match the declared column `{}`",
                        actual.name(),
                        declared.name()
                    ),
                )));
            }
            if column.data_type() == declared.data_type() {
                Ok(Arc::clone(column))
            } else if can_cast_types(column.data_type(), declared.data_type()) {
                Ok(cast(column, declared.data_type())?)
            } else {
                Err(ExecutorError::Kernel(delta_kernel::KernelError::schema(
                    format!(
                        "column `{}` has type {}, not the declared {}",
                        declared.name(),
                        column.data_type(),
                        declared.data_type()
                    ),
                )))
            }
        })
        .collect::<crate::Result<Vec<_>>>()?;
    RecordBatch::try_new(Arc::clone(schema), columns)
        .map_err(|err| delta_kernel::KernelError::schema(err.to_string()).into())
}

/// Translates `plan` into the `LogicalPlan` of its terminal node.
///
/// # Errors
/// Returns an error if `plan` has no nodes, a node references an input that does not precede it,
/// or lowering any individual node fails.
pub fn to_logical_plan(
    executor: &DataFusionPlanExecutor,
    plan: &Plan,
) -> crate::Result<LogicalPlan> {
    // A node is identified by its index, and `nodes` is topologically ordered: every input index is
    // strictly less than the node's own, so a single forward pass leaves each node's inputs already
    // lowered by the time it is reached.
    let mut lowered: Vec<Arc<LogicalPlan>> = Vec::with_capacity(plan.nodes.len());

    for (node_index, node) in plan.nodes.iter().enumerate() {
        let op = &node.op;
        let available = lowered.len();
        let invalid_input = |input_index| {
            DataFusionError::Plan(format!(
                "node {node_index} ({}) references input {input_index}, but only {available} \
                 prior node(s) are available",
                operator::name(op)
            ))
        };
        let inputs: Vec<_> = node
            .inputs
            .iter()
            .map(|&input_index| {
                lowered
                    .get(input_index)
                    .map(Arc::clone)
                    .ok_or_else(|| invalid_input(input_index))
            })
            .collect::<Result<_, _>>()?;
        lowered.push(Arc::new(operator::to_logical_plan(executor, op, &inputs)?));
    }

    // The terminal node is the last one: no other node consumes it, and its rows are the plan's
    // output.
    match lowered.pop() {
        Some(terminal) => Ok(Arc::unwrap_or_clone(terminal)),
        None => Err(DataFusionError::Plan("cannot lower a plan with no nodes".to_string()).into()),
    }
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::record_batch::RecordBatch;
    use datafusion::assert_batches_eq;
    use datafusion::prelude::SessionContext;
    use delta_kernel::PlanBuilder;
    use delta_kernel::expressions::{
        Expression as KernelExpr, Predicate as KernelPredicate, Scalar, column_expr,
    };
    use delta_kernel::plans::ir::nodes::{Filter, Project, Values};
    use delta_kernel::plans::ir::plan::PlanNode;
    use delta_kernel::schema::{DataType, StructField, StructType};
    use rstest::rstest;
    use tokio::runtime::Handle;

    use super::*;

    // === Shared helpers ===

    /// An executor over a default session and the test runtime.
    fn executor() -> DataFusionPlanExecutor {
        DataFusionPlanExecutor::new(SessionContext::new().task_ctx(), Handle::current())
    }

    /// A single-field `long` schema named `a`.
    fn test_schema() -> StructType {
        StructType::try_new([StructField::nullable("a", DataType::LONG)]).unwrap()
    }

    /// A `Values` node over [`test_schema`] holding `rows` one-column rows.
    fn values_node(rows: Vec<Vec<Scalar>>) -> PlanNode {
        PlanNode::new(Values::new(test_schema(), rows), vec![])
    }

    async fn execute(plan: &Plan) -> crate::Result<Vec<RecordBatch>> {
        Ok(SessionContext::new()
            .execute_logical_plan(to_logical_plan(&executor(), plan)?)
            .await?
            .collect()
            .await?)
    }

    // === Tests ===

    #[tokio::test]
    async fn empty_plan_is_rejected() {
        let err = to_logical_plan(
            &executor(),
            &Plan {
                schema: Arc::new(test_schema()),
                nodes: vec![],
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("no nodes"), "{err}");
    }

    #[tokio::test]
    async fn terminal_node_is_the_plans_output() {
        // Two independent sources; the last node is the terminal one, so its rows are the output.
        let plan = Plan {
            schema: Arc::new(test_schema()),
            nodes: vec![
                values_node(vec![vec![1i64.into()]]),
                values_node(vec![vec![2i64.into()], vec![3i64.into()]]),
            ],
        };
        let batches = execute(&plan).await.unwrap();
        assert_batches_eq!(
            &["+---+", "| a |", "+---+", "| 2 |", "| 3 |", "+---+"],
            &batches
        );
    }

    #[tokio::test]
    async fn filter_plan_executes() {
        let plan = PlanBuilder::values(
            Arc::new(test_schema()),
            vec![vec![Scalar::null(DataType::LONG)], vec![2i64.into()]],
        )
        .unwrap()
        .filter(KernelPredicate::is_null(column_expr!("a")))
        .unwrap()
        .build()
        .unwrap();
        let batches = execute(&plan).await.unwrap();
        assert_batches_eq!(&["+---+", "| a |", "+---+", "|   |", "+---+"], &batches);
    }

    #[tokio::test]
    async fn values_project_filter_composes_through_declared_project_schema() {
        let project_schema = Arc::new(
            StructType::try_new([StructField::nullable("projected", DataType::LONG)]).unwrap(),
        );
        let project = Project {
            expr: KernelExpr::struct_from([column_expr!("a")]).into(),
            schema: project_schema.clone(),
        };
        let filter = Filter {
            predicate: KernelPredicate::is_not_null(column_expr!("projected")).into(),
        };
        let plan = Plan {
            schema: project_schema.clone(),
            nodes: vec![
                values_node(vec![vec![1i64.into()]]),
                PlanNode::new(project, vec![0]),
                PlanNode::new(filter, vec![1]),
            ],
        };

        let batches = execute(&plan).await.unwrap();
        assert_batches_eq!(
            &[
                "+-----------+",
                "| projected |",
                "+-----------+",
                "| 1         |",
                "+-----------+",
            ],
            &batches
        );
    }

    #[rstest]
    #[case::self_reference(0)]
    #[case::forward_reference(1)]
    #[case::far_out_of_range(42)]
    #[tokio::test]
    async fn invalid_input_reference_is_rejected(#[case] input_index: usize) {
        let node = PlanNode::new(Values::new(test_schema(), vec![]), vec![input_index]);
        let err = to_logical_plan(
            &executor(),
            &Plan {
                schema: Arc::new(test_schema()),
                nodes: vec![node],
            },
        )
        .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("node 0 (values)"), "{message}");
        assert!(
            message.contains(&format!("references input {input_index}")),
            "{message}"
        );
        assert!(message.contains("0 prior node(s)"), "{message}");
    }
}
