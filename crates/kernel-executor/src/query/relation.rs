//! Relations retained by a [`ScopedDataFusionPlanExecutor`] and read back by `RelationSource`.
//!
//! [`execute_and_retain`] runs a plan to completion and keeps its batches in the scope's
//! [`RelationStore`] under a fresh [`RelationId`]. A `RelationSource` node in a later plan names
//! that id, and [`lower_relation_source`] turns the retained batches into an in-memory table.
//!
//! The batches live as long as the store, which belongs to the scoped executor. Ids carry a
//! process-wide counter, so a ref from one scope is unknown to every other scope, and a
//! `RelationSource` on an executor without a scope is an error.
//!
//! [`ScopedDataFusionPlanExecutor`]: crate::ScopedDataFusionPlanExecutor

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::logical_expr::LogicalPlan;
use delta_kernel::KernelError;
use delta_kernel::engine::arrow_conversion::TryIntoArrow as _;
use delta_kernel::plans::ir::plan::Plan;
use delta_kernel::plans::{RelationId, RelationRef};
use delta_kernel::schema::SchemaRef;

use crate::{DataFusionPlanExecutor, ExecutorError};

/// Ids handed out to retained relations, shared by every scope.
static NEXT_RELATION_ID: AtomicU64 = AtomicU64::new(0);

/// One retained relation: its kernel schema and the rows, already in that schema.
#[derive(Debug)]
struct Retained {
    schema: SchemaRef,
    arrow_schema: ArrowSchemaRef,
    batches: Vec<RecordBatch>,
}

/// The relations one scoped executor retains.
#[derive(Debug, Default)]
pub struct RelationStore {
    relations: Mutex<HashMap<RelationId, Retained>>,
}

impl RelationStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of relations retained.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether no relation is retained.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<RelationId, Retained>> {
        // A panic while the lock is held leaves the map itself intact.
        self.relations
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    fn insert(
        &self,
        name: &str,
        schema: SchemaRef,
        batches: &[RecordBatch],
    ) -> crate::Result<RelationRef> {
        let arrow_schema: ArrowSchemaRef = Arc::new(schema.as_ref().try_into_arrow()?);
        let batches = batches
            .iter()
            .map(|batch| super::conform_batch(batch, &arrow_schema))
            .collect::<crate::Result<Vec<_>>>()?;
        let id = format!(
            "{name}-{}",
            NEXT_RELATION_ID.fetch_add(1, Ordering::Relaxed)
        );
        self.lock().insert(
            id.clone(),
            Retained {
                schema: Arc::clone(&schema),
                arrow_schema,
                batches,
            },
        );
        Ok(RelationRef::new(id, schema))
    }

    /// The schema and batches retained under `relation`'s id.
    fn get(&self, relation: &RelationRef) -> crate::Result<(ArrowSchemaRef, Vec<RecordBatch>)> {
        let relations = self.lock();
        let retained = relations.get(relation.id()).ok_or_else(|| {
            KernelError::generic(format!(
                "RelationSource references relation `{}`, which this executor does not retain",
                relation.id()
            ))
        })?;
        if &retained.schema != relation.schema() {
            return Err(KernelError::schema(format!(
                "RelationSource schema does not match the schema relation `{}` was retained with",
                relation.id()
            ))
            .into());
        }
        // Batch clones share their arrays.
        Ok((Arc::clone(&retained.arrow_schema), retained.batches.clone()))
    }
}

/// Runs `plan` and retains its rows in `executor`'s store.
///
/// # Errors
/// Returns an error if `executor` is not scoped, the plan fails, or its rows do not match the
/// plan's declared schema.
pub(crate) fn execute_and_retain(
    executor: &DataFusionPlanExecutor,
    name: &str,
    plan: Plan,
) -> crate::Result<RelationRef> {
    let store = store(executor)?;
    let logical = super::to_logical_plan(executor, &plan)?;
    let batches = super::collect(executor, logical)?;
    store.insert(name, Arc::clone(&plan.schema), &batches)
}

/// Source: reads a relation retained earlier in the executor's scope.
pub(crate) fn lower_relation_source(
    executor: &DataFusionPlanExecutor,
    relation: &RelationRef,
) -> crate::Result<LogicalPlan> {
    let (schema, batches) = store(executor)?.get(relation)?;
    super::memory_plan("relation_source", schema, batches)
}

fn store(executor: &DataFusionPlanExecutor) -> crate::Result<&Arc<RelationStore>> {
    executor.relations().ok_or_else(|| {
        ExecutorError::unsupported(
            "retained relations need a scoped executor, see DataFusionPlanExecutor::scoped",
        )
    })
}

#[cfg(test)]
mod tests {
    use datafusion::assert_batches_sorted_eq;
    use datafusion::prelude::SessionContext;
    use delta_kernel::PlanBuilder;
    use delta_kernel::expressions::{Predicate, Scalar, column_expr};
    use delta_kernel::plans::{Operation, PlanExecutor as _, PlanResult, ScopedPlanExecutor};
    use delta_kernel::schema::{DataType, StructField, StructType};
    use tokio::runtime::Handle;

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

    fn values_plan() -> Plan {
        PlanBuilder::values(
            schema(),
            vec![
                vec![1i64.into(), "a".into()],
                vec![2i64.into(), Scalar::null(DataType::STRING)],
                vec![3i64.into(), "c".into()],
            ],
        )
        .unwrap()
        .build()
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
    async fn retained_relation_is_readable_by_later_plans() {
        let scope = executor().scoped();
        let relation = scope.execute_and_retain("values", values_plan()).unwrap();
        assert!(relation.id().starts_with("values-"), "{}", relation.id());
        assert_eq!(relation.schema(), &schema());

        let plan = PlanBuilder::relation_source(relation.clone())
            .filter(Predicate::gt(column_expr!("id"), Scalar::from(1i64)))
            .unwrap()
            .build()
            .unwrap();
        let batches = collect_result(scope.execute_op(Operation::QueryPlan(plan)).unwrap());
        assert_batches_sorted_eq!(
            &[
                "+----+------+",
                "| id | name |",
                "+----+------+",
                "| 2  |      |",
                "| 3  | c    |",
                "+----+------+",
            ],
            &batches
        );

        // A relation can be read any number of times while the scope is alive.
        let again = PlanBuilder::relation_source(relation).build().unwrap();
        let batches = collect_result(scope.execute_op(Operation::QueryPlan(again)).unwrap());
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn retaining_a_relation_built_from_a_relation() {
        let scope = executor().scoped();
        let first = scope.execute_and_retain("first", values_plan()).unwrap();
        let derived = PlanBuilder::relation_source(first)
            .filter(Predicate::is_not_null(column_expr!("name")))
            .unwrap()
            .build()
            .unwrap();
        let second = scope.execute_and_retain("second", derived).unwrap();
        assert_eq!(scope.executor().relations().unwrap().len(), 2);

        let plan = PlanBuilder::relation_source(second).build().unwrap();
        let batches = collect_result(scope.execute_op(Operation::QueryPlan(plan)).unwrap());
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn relation_from_another_scope_is_rejected() {
        let executor = executor();
        let producer = executor.scoped();
        let relation = producer
            .execute_and_retain("values", values_plan())
            .unwrap();

        let other = executor.scoped();
        let plan = PlanBuilder::relation_source(relation).build().unwrap();
        let err = other
            .execute_op(Operation::QueryPlan(plan))
            .err()
            .expect("a ref from another scope must not resolve");
        assert!(err.to_string().contains("does not retain"), "{err}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn relation_source_needs_a_scoped_executor() {
        let scope = executor().scoped();
        let relation = scope.execute_and_retain("values", values_plan()).unwrap();
        let plan = PlanBuilder::relation_source(relation).build().unwrap();
        let err = executor()
            .execute_op(Operation::QueryPlan(plan))
            .err()
            .expect("an unscoped executor retains nothing");
        assert!(err.to_string().contains("scoped executor"), "{err}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn relation_source_with_a_different_schema_is_rejected() {
        let scope = executor().scoped();
        let relation = scope.execute_and_retain("values", values_plan()).unwrap();
        let other_schema =
            Arc::new(StructType::try_new([StructField::not_null("id", DataType::LONG)]).unwrap());
        let forged = RelationRef::new(relation.id(), other_schema);
        let plan = PlanBuilder::relation_source(forged).build().unwrap();
        let err = scope
            .execute_op(Operation::QueryPlan(plan))
            .err()
            .expect("a schema mismatch must be rejected");
        assert!(err.to_string().contains("schema does not match"), "{err}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropping_the_scope_releases_its_relations() {
        let scope = executor().scoped();
        scope.execute_and_retain("values", values_plan()).unwrap();
        let store = Arc::downgrade(scope.executor().relations().unwrap());
        assert!(store.upgrade().is_some());
        drop(scope);
        assert!(store.upgrade().is_none());
    }
}
