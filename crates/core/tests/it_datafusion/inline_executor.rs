//! V2 keystone tests (wasm-engine D2): the native proof of the wasm execution model.
//!
//! On wasm the `DataFusionEngine` drives kernel IO with the [`InlineExecutor`], which can
//! only complete futures that are already ready — the browser facade *primes* the
//! `_delta_log` into an in-memory store before any sync kernel call. These tests replicate
//! that model natively: a fixture table is loaded into an `object_store::memory::InMemory`
//! store, the engine is forced onto the inline executor, and a snapshot build plus full
//! `scan_metadata` replay must (a) complete against the fully primed store, matching the
//! Tokio engine's result, and (b) fail loudly — not hang — when required data would block.

use std::sync::Arc;

use arrow_array::{RecordBatch, cast::AsArray};
use async_trait::async_trait;
use bytes::Bytes;
use futures::TryStreamExt;
use futures::stream::BoxStream;
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    ObjectStoreExt as _, PutMultipartOptions, PutOptions, PutPayload, PutResult,
};
use url::Url;

use deltalake_core::DeltaTableConfig;
use deltalake_core::delta_datafusion::create_session;
use deltalake_core::delta_datafusion::engine::DataFusionEngine;
use deltalake_core::kernel::engine::arrow_data::ArrowEngineData;
use deltalake_core::kernel::{InlineExecutor, Snapshot};
use deltalake_test::TestResult;

/// The checkpointed fixture: v2 checkpoint parquet without a `_last_checkpoint` hint, so a
/// snapshot build exercises listing, commit-JSON reads, the checkpoint parquet scan, *and*
/// the parquet footer read — every sync kernel handler path — under the executor under test.
const FIXTURE: &str = "with_checkpoint_no_last_checkpoint";

/// Table root within the in-memory store.
const TABLE_PREFIX: &str = "table";

fn fixture_dir() -> std::path::PathBuf {
    std::fs::canonicalize(format!(
        "{}/../test/tests/data/{FIXTURE}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("fixture directory exists")
}

fn table_root_url() -> Url {
    Url::parse(&format!("memory:///{TABLE_PREFIX}/")).unwrap()
}

/// Load every file under `dir` into an [`InMemory`] store beneath [`TABLE_PREFIX`].
async fn prime_store(dir: &std::path::Path) -> TestResult<Arc<InMemory>> {
    let store = Arc::new(InMemory::new());
    let mut pending = vec![dir.to_path_buf()];
    while let Some(current) = pending.pop() {
        for entry in std::fs::read_dir(&current)? {
            let path = entry?.path();
            if path.is_dir() {
                pending.push(path);
            } else {
                let relative = path.strip_prefix(dir)?.to_string_lossy().replace('\\', "/");
                let key = Path::from(format!("{TABLE_PREFIX}/{relative}"));
                let bytes = Bytes::from(std::fs::read(&path)?);
                store.put(&key, bytes.into()).await?;
            }
        }
    }
    Ok(store)
}

/// Collect the selected file paths from a full `scan_metadata` replay driven on `engine`.
async fn scan_metadata_paths(
    snapshot: &Snapshot,
    engine: Arc<DataFusionEngine>,
) -> TestResult<Vec<String>> {
    let scan = snapshot.scan_builder().build()?;
    let metadata: Vec<_> = scan.scan_metadata(engine).try_collect().await?;

    let mut paths = Vec::new();
    for item in metadata {
        let (data, mut selection) = item.scan_files.into_parts();
        let batch: RecordBatch = ArrowEngineData::try_from_engine_data(data)
            .map_err(deltalake_core::DeltaTableError::from)?
            .into();
        // Kernel may return a short selection vector; missing entries are selected.
        selection.resize(batch.num_rows(), true);
        let path_column = batch
            .column_by_name("path")
            .expect("scan row batch has a path column")
            .as_string::<i32>();
        for (idx, selected) in selection.into_iter().enumerate() {
            if selected {
                paths.push(path_column.value(idx).to_string());
            }
        }
    }
    paths.sort();
    Ok(paths)
}

/// A fully primed store must let the inline executor drive snapshot construction and the
/// complete `scan_metadata` replay, producing the same file list as the Tokio engine.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_inline_executor_snapshot_scan_matches_tokio() -> TestResult<()> {
    let store = prime_store(&fixture_dir()).await?;
    let session = create_session().into_inner();
    session
        .runtime_env()
        .register_object_store(&Url::parse("memory:///").unwrap(), store);

    let inline_engine: Arc<DataFusionEngine> =
        DataFusionEngine::new(session.task_ctx(), InlineExecutor).into();
    let snapshot = Snapshot::try_new_with_engine(
        inline_engine.clone(),
        table_root_url(),
        DeltaTableConfig::default(),
        None,
    )
    .await?;
    // Snapshot state: v2 checkpoint (footer-derived schema) + v3 commit replayed.
    assert_eq!(snapshot.version(), 3);

    let inline_paths = scan_metadata_paths(&snapshot, inline_engine).await?;
    assert!(!inline_paths.is_empty(), "scan selected no files");

    // Same store, same session — only the executor differs.
    let tokio_engine = DataFusionEngine::new_from_session(&session.state());
    let tokio_snapshot = Snapshot::try_new_with_engine(
        tokio_engine.clone(),
        table_root_url(),
        DeltaTableConfig::default(),
        None,
    )
    .await?;
    assert_eq!(tokio_snapshot.version(), snapshot.version());
    let tokio_paths = scan_metadata_paths(&tokio_snapshot, tokio_engine).await?;

    assert_eq!(inline_paths, tokio_paths);
    Ok(())
}

/// Wraps an [`InMemory`] store, listing all objects but never resolving reads of one path —
/// modelling an unprimed async fetch. The inline executor must surface this as a
/// "would block / not primed" error instead of hanging.
#[derive(Debug)]
struct UnprimedStore {
    inner: Arc<InMemory>,
    unprimed: Path,
}

impl std::fmt::Display for UnprimedStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UnprimedStore({})", self.unprimed)
    }
}

#[async_trait]
impl ObjectStore for UnprimedStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        if location == &self.unprimed {
            // An unprimed fetch never resolves synchronously.
            std::future::pending::<()>().await;
            unreachable!("pending future never completes");
        }
        self.inner.get_opts(location, options).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

/// Negative proof: with one commit JSON listed but unreadable-without-waiting, the inline
/// executor errors ("would block … not primed") instead of deadlocking, and the error
/// reaches the snapshot API caller.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_inline_executor_errors_on_unprimed_data() -> TestResult<()> {
    let inner = prime_store(&fixture_dir()).await?;
    let store = Arc::new(UnprimedStore {
        inner,
        unprimed: Path::from(format!(
            "{TABLE_PREFIX}/_delta_log/00000000000000000003.json"
        )),
    });
    let session = create_session().into_inner();
    session
        .runtime_env()
        .register_object_store(&Url::parse("memory:///").unwrap(), store);

    let engine: Arc<DataFusionEngine> =
        DataFusionEngine::new(session.task_ctx(), InlineExecutor).into();
    let result =
        Snapshot::try_new_with_engine(engine, table_root_url(), DeltaTableConfig::default(), None)
            .await;

    let err = result.expect_err("snapshot build must fail against unprimed data");
    assert!(
        err.to_string().contains("would block"),
        "expected a would-block/not-primed error, got: {err}"
    );
    Ok(())
}
