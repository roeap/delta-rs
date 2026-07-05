//! Browser facade for delta-rs: read-only Delta table queries on `wasm32-unknown-unknown`.
//!
//! This crate wires deltalake-core's DataFusion-backed kernel engine into a browser
//! runtime: a fetch-backed [`ObjectStore`] (`FetchObjectStore`, wasm-only), the
//! [`PrimedStore`] that prefetches the `_delta_log` tail so the synchronous kernel engine
//! only ever awaits ready futures, and a small session/snapshot/query API that mangrove
//! Phase B (or any host) builds on. A minimal `wasm-bindgen` surface (`WasmDeltaTable`)
//! proves the path end-to-end; it is a proof harness, not the product API.
//!
//! Everything except the fetch store and the bindings compiles natively, so the priming
//! and query logic is tested with ordinary `cargo test` against in-memory stores.
//!
//! # v1 limits
//!
//! Read-only; no deletion vectors (fails loud); no zstd/brotli parquet pages (graceful
//! error); fully-qualified table URLs only.
//!
//! # Usage sketch
//!
//! ```ignore
//! let opened = open_table_with_store(store, &url, LogSource::List, OpenOptions::default()).await?;
//! register_snapshot(&opened.ctx, opened.snapshot.clone())?;
//! let mut chunks = query_ipc(&opened.ctx, "SELECT count(*) FROM delta").await?;
//! while let Some(chunk) = chunks.try_next().await? { /* forward Arrow IPC bytes */ }
//! ```

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
mod bindings;
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub mod fetch_store;
pub mod primed;

use std::sync::Arc;

use arrow_ipc::writer::StreamWriter;
use datafusion::execution::TaskContext;
use datafusion::execution::context::SessionContext;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::prelude::SessionConfig;
use futures::StreamExt;
use futures::stream::BoxStream;
use object_store::ObjectStore;
use url::Url;

use deltalake_core::delta_datafusion::engine::DataFusionEngine;
use deltalake_core::delta_datafusion::planner::DeltaPlanner;
use deltalake_core::delta_datafusion::{DeltaScanConfig, DeltaScanNext, DeltaSessionConfig};
use deltalake_core::kernel::{ExecutorHandle, Snapshot};
use deltalake_core::{DeltaResult, DeltaTableConfig, DeltaTableError};

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub use fetch_store::FetchObjectStore;
pub use primed::{LogSource, PrimeLimits, PrimeReport, PrimedStore};

/// Table name [`register_snapshot`] registers the scan under; SQL passed to
/// [`query_ipc`] refers to the table as `delta`.
pub const TABLE_NAME: &str = "delta";

/// Options for [`open_table_with_store`].
#[derive(Debug, Default)]
pub struct OpenOptions {
    /// Table version to load; latest when `None`.
    pub version: Option<u64>,
    /// Guardrails for [`PrimedStore::prime`].
    pub limits: PrimeLimits,
    /// Executor driving the sync kernel engine. Defaults to the target's natural choice
    /// (tokio natively, the inline executor on wasm); native tests force
    /// [`InlineExecutor`](deltalake_core::kernel::InlineExecutor) to prove the wasm
    /// execution model.
    pub executor: Option<ExecutorHandle>,
}

/// An opened table: the session wired to its primed store, plus the pinned snapshot.
#[derive(Clone)]
pub struct OpenedTable {
    // `SessionContext` has no `Debug` impl, hence the manual one below.
    /// Session whose runtime registry serves the table through the [`PrimedStore`].
    pub ctx: SessionContext,
    /// Snapshot pinned to the primed table version.
    pub snapshot: Arc<Snapshot>,
    /// The primed store, kept for re-priming after a "not primed" error.
    pub store: Arc<PrimedStore>,
    /// Telemetry from the initial prime.
    pub report: PrimeReport,
}

impl std::fmt::Debug for OpenedTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenedTable")
            .field("version", &self.snapshot.version())
            .field("store", &self.store)
            .field("report", &self.report)
            .finish_non_exhaustive()
    }
}

/// Build the facade's DataFusion session and register `store` for `table_url`.
///
/// The configuration pins the "supported plan shape" for wasm (see V3): a single
/// partition and no repartitioning, so physical plans contain no `RepartitionExec` (whose
/// stream driving spawns tasks that never run without a threaded runtime).
pub fn session(store: Arc<dyn ObjectStore>, table_url: &Url) -> DeltaResult<SessionContext> {
    let config: SessionConfig = DeltaSessionConfig::default().into();
    let config = config
        .with_target_partitions(1)
        .with_round_robin_repartition(false)
        .with_repartition_joins(false)
        .with_repartition_aggregations(false)
        .with_repartition_windows(false)
        .with_repartition_sorts(false)
        .with_repartition_file_scans(false);
    let runtime_env = RuntimeEnvBuilder::new()
        .build_arc()
        .map_err(DeltaTableError::from)?;
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(config)
        .with_runtime_env(runtime_env)
        .with_query_planner(DeltaPlanner::new())
        .build();
    let ctx = SessionContext::new_with_state(state);

    let mut base = table_url.clone();
    base.set_path("/");
    base.set_query(None);
    base.set_fragment(None);
    ctx.runtime_env().register_object_store(&base, store);
    Ok(ctx)
}

/// Open the Delta table at `table_url`, priming its log from `inner` per `source`.
///
/// `inner` is the raw store for the table's scheme/authority (the fetch store on wasm, any
/// [`ObjectStore`] natively); it is wrapped in a [`PrimedStore`], primed, and registered in
/// a fresh facade [`session`]. The snapshot is then built with the DataFusion engine — on
/// wasm entirely against the primed cache.
pub async fn open_table_with_store(
    inner: Arc<dyn ObjectStore>,
    table_url: &Url,
    source: LogSource,
    options: OpenOptions,
) -> DeltaResult<OpenedTable> {
    // Kernel engines join paths onto the table root; without the trailing slash the last
    // path segment would be replaced instead of appended.
    let mut table_url = table_url.clone();
    if !table_url.path().ends_with('/') {
        table_url.set_path(&format!("{}/", table_url.path()));
    }

    let store = Arc::new(PrimedStore::try_new(inner, &table_url)?.with_limits(options.limits));
    let report = store.prime(source).await?;

    let ctx = session(store.clone(), &table_url)?;
    let executor = options.executor.unwrap_or_else(ExecutorHandle::current);
    let engine = Arc::new(DataFusionEngine::new(ctx.task_ctx(), executor));
    let snapshot = Snapshot::try_new_with_engine(
        engine,
        table_url,
        DeltaTableConfig::default(),
        options.version,
    )
    .await?;

    Ok(OpenedTable {
        ctx,
        snapshot: Arc::new(snapshot),
        store,
        report,
    })
}

/// Register `snapshot` as the queryable table [`TABLE_NAME`] on `ctx`.
///
/// Replaces any previous registration, so re-opening at another version just re-registers.
pub fn register_snapshot(ctx: &SessionContext, snapshot: Arc<Snapshot>) -> DeltaResult<()> {
    let scan = DeltaScanNext::new(snapshot, DeltaScanConfig::default())?;
    ctx.register_table(TABLE_NAME, Arc::new(scan))
        .map_err(DeltaTableError::from)?;
    Ok(())
}

/// Execute `sql` and stream the result as Arrow IPC chunks.
///
/// The chunks form one *incremental* IPC stream: the first carries the schema message,
/// each subsequent one a record batch, the last the end-of-stream marker — concatenating
/// them yields a complete Arrow IPC stream, and consumers appending chunk-by-chunk see a
/// valid prefix at every step.
pub async fn query_ipc(
    ctx: &SessionContext,
    sql: &str,
) -> DeltaResult<BoxStream<'static, DeltaResult<Vec<u8>>>> {
    let df = ctx.sql(sql).await.map_err(DeltaTableError::from)?;
    let batches = df.execute_stream().await.map_err(DeltaTableError::from)?;
    let writer = StreamWriter::try_new(Vec::new(), batches.schema().as_ref())
        .map_err(DeltaTableError::from)?;

    struct State {
        batches: datafusion::execution::SendableRecordBatchStream,
        writer: StreamWriter<Vec<u8>>,
    }
    let stream = futures::stream::unfold(Some(State { batches, writer }), |state| async move {
        let mut state = state?;
        match state.batches.next().await {
            Some(Ok(batch)) => {
                if let Err(err) = state.writer.write(&batch) {
                    return Some((Err(DeltaTableError::from(err)), None));
                }
                let chunk = std::mem::take(state.writer.get_mut());
                Some((Ok(chunk), Some(state)))
            }
            Some(Err(err)) => Some((Err(DeltaTableError::from(err)), None)),
            None => {
                if let Err(err) = state.writer.finish() {
                    return Some((Err(DeltaTableError::from(err)), None));
                }
                let chunk = std::mem::take(state.writer.get_mut());
                (!chunk.is_empty()).then_some((Ok(chunk), None))
            }
        }
    });
    Ok(stream.boxed())
}

/// The table's Delta schema as JSON (the log's `metaData.schemaString` shape).
pub fn snapshot_schema_json(snapshot: &Snapshot) -> DeltaResult<String> {
    serde_json::to_string(snapshot.schema().as_ref())
        .map_err(|err| DeltaTableError::generic(err.to_string()))
}

/// Convenience: an ad-hoc engine over `ctx` for callers driving kernel APIs directly.
pub fn engine_for_context(ctx: Arc<TaskContext>) -> Arc<DataFusionEngine> {
    DataFusionEngine::new_from_context(ctx)
}
