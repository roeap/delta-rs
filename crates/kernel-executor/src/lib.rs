//! A DataFusion implementation of the delta-kernel [`PlanExecutor`].
//!
//! Delta-kernel does no I/O or data processing itself. It builds an [`Operation`] and hands it to
//! the executor. [`DataFusionPlanExecutor`] dispatches on the operation:
//!
//! - [`Operation::IoOperation`] goes to [`io`], which runs a single request through the
//!   executor's [`DataFusionStorageHandler`] or [`DataFusionFileFormatHandler`]. Both resolve
//!   the object store for a URL through the DataFusion session's object store registry.
//! - [`Operation::QueryPlan`] goes to [`query`], which translates the kernel [`Plan`] into a
//!   DataFusion `LogicalPlan`, runs it on a session built from the executor's [`TaskContext`],
//!   and returns the batches through [`stream`] as [`PlanResult::Data`].
//!
//! [`DataFusionPlanExecutor::scoped`] returns a [`ScopedDataFusionPlanExecutor`], the
//! [`ScopedPlanExecutor`] kernel needs to retain a plan's result and read it back through
//! `RelationSource` nodes in later plans (see [`query::relation`]).
//!
//! [`DataFusionEngine`] is the `Engine` to hand to kernel: its handlers route every
//! call through the executor as well ([`engine`]).
//!
//! `execute_op` is synchronous. The executor spawns its async work on the tokio [`Handle`] it
//! was built with and waits for results over channels ([`runtime`]), so it can be called from a
//! runtime worker or from a plain thread. On a current-thread runtime, queries run on a shared
//! background runtime, so the caller can also be the runtime's own thread.

use std::sync::Arc;

use datafusion::catalog::Session;
use datafusion::execution::TaskContext;
use delta_kernel::KernelResult as DeltaResult;
use delta_kernel::plans::{
    Operation, Plan, PlanExecutor, PlanResult, RelationRef, ScopedPlanExecutor,
};
use tokio::runtime::Handle;

pub mod engine;
pub mod error;
pub mod expressions;
pub mod io;
pub mod query;
mod runtime;
pub mod stream;

pub use engine::DataFusionEngine;
pub use error::{ExecutorError, Result};
pub use io::file_formats::DataFusionFileFormatHandler;
pub use io::storage::{AsObjectStoreUrl, DataFusionStorageHandler};

use query::relation::RelationStore;

/// Executes kernel declarative plans on DataFusion.
#[derive(Clone)]
pub struct DataFusionPlanExecutor {
    ctx: Arc<TaskContext>,
    handle: Handle,
    storage: Arc<DataFusionStorageHandler>,
    file_formats: Arc<DataFusionFileFormatHandler>,
    /// The relations `RelationSource` nodes read. Only a [`ScopedDataFusionPlanExecutor`] has one.
    relations: Option<Arc<RelationStore>>,
}

impl std::fmt::Debug for DataFusionPlanExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DataFusionPlanExecutor")
            .field("session_id", &self.ctx.session_id())
            .finish_non_exhaustive()
    }
}

impl DataFusionPlanExecutor {
    /// Creates an executor from a DataFusion [`Session`], reusing its task context and the
    /// current tokio runtime handle.
    pub fn new_from_session(session: &dyn Session) -> Arc<Self> {
        Self::new(session.task_ctx(), Handle::current()).into()
    }

    /// Creates an executor from a [`TaskContext`], using the current tokio runtime handle.
    /// Useful inside physical operators where only the task context is available.
    pub fn new_from_context(ctx: Arc<TaskContext>) -> Arc<Self> {
        Self::new(ctx, Handle::current()).into()
    }

    /// Creates an executor from an explicit [`TaskContext`] and tokio runtime [`Handle`].
    ///
    /// The other constructors delegate here; call this directly to bind the executor to a
    /// specific runtime.
    pub fn new(ctx: Arc<TaskContext>, handle: Handle) -> Self {
        let storage = Arc::new(DataFusionStorageHandler::new(ctx.clone(), handle.clone()));
        let file_formats = Arc::new(DataFusionFileFormatHandler::new(
            ctx.clone(),
            handle.clone(),
        ));
        Self {
            ctx,
            handle,
            storage,
            file_formats,
            relations: None,
        }
    }

    /// Returns a [`ScopedPlanExecutor`] over the same session and runtime, with its own, empty
    /// set of retained relations.
    pub fn scoped(&self) -> ScopedDataFusionPlanExecutor {
        ScopedDataFusionPlanExecutor {
            executor: Self {
                relations: Some(Arc::new(RelationStore::new())),
                ..self.clone()
            },
        }
    }

    /// The relations this executor's plans can read, if it is scoped.
    pub(crate) fn relations(&self) -> Option<&Arc<RelationStore>> {
        self.relations.as_ref()
    }

    /// The DataFusion task context: session config and runtime environment (object store
    /// registry) plans and I/O run against.
    pub fn task_ctx(&self) -> &Arc<TaskContext> {
        &self.ctx
    }

    /// The tokio runtime handle.
    pub fn handle(&self) -> &Handle {
        &self.handle
    }

    /// The storage handler I/O operations run through.
    pub fn storage(&self) -> &Arc<DataFusionStorageHandler> {
        &self.storage
    }

    /// The JSON and Parquet handler file reads run through.
    pub fn file_formats(&self) -> &Arc<DataFusionFileFormatHandler> {
        &self.file_formats
    }
}

impl PlanExecutor for DataFusionPlanExecutor {
    fn execute_op(&self, op: Operation) -> DeltaResult<PlanResult> {
        match op {
            Operation::IoOperation(op) => io::execute(self, op),
            Operation::QueryPlan(plan) => query::execute(self, plan),
        }
    }
}

/// A [`DataFusionPlanExecutor`] that retains relations for later plans to read.
///
/// Built with [`DataFusionPlanExecutor::scoped`]. Each scope keeps its own relations: a
/// [`RelationRef`] from [`execute_and_retain`](ScopedPlanExecutor::execute_and_retain) resolves
/// only through the scope that produced it, and the rows are released when the scope is dropped.
#[derive(Debug)]
pub struct ScopedDataFusionPlanExecutor {
    executor: DataFusionPlanExecutor,
}

impl ScopedDataFusionPlanExecutor {
    /// The executor plans in this scope run on.
    pub fn executor(&self) -> &DataFusionPlanExecutor {
        &self.executor
    }
}

impl PlanExecutor for ScopedDataFusionPlanExecutor {
    fn execute_op(&self, op: Operation) -> DeltaResult<PlanResult> {
        self.executor.execute_op(op)
    }
}

impl ScopedPlanExecutor for ScopedDataFusionPlanExecutor {
    fn execute_and_retain(&self, name: &str, plan: Plan) -> DeltaResult<RelationRef> {
        Ok(query::relation::execute_and_retain(
            &self.executor,
            name,
            plan,
        )?)
    }
}
