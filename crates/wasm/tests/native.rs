//! Native tests of the wasm facade's rlib surface.
//!
//! The wasm execution model is proven natively: fixtures are served from an
//! [`InMemory`] store standing in for the browser fetch store, priming runs against it,
//! and snapshot builds are forced onto the [`InlineExecutor`] — exactly what runs in the
//! browser, minus the network.

#![cfg(not(target_arch = "wasm32"))]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use arrow::compute::concat_batches;
use arrow_array::RecordBatch;
use arrow_ipc::reader::StreamReader;
use async_trait::async_trait;
use bytes::Bytes;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties as _};
use futures::TryStreamExt;
use futures::stream::BoxStream;
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    ObjectStoreExt as _, PutMultipartOptions, PutOptions, PutPayload, PutResult,
};
use url::Url;

use deltalake_core::delta_datafusion::engine::DataFusionEngine;
use deltalake_core::kernel::InlineExecutor;
use deltalake_wasm::{
    LogSource, OpenOptions, PrimeLimits, PrimedStore, TABLE_NAME, open_table_with_store, query_ipc,
    register_snapshot, snapshot_schema_json,
};

/// v2 checkpoint (JSON manifest + parquet sidecars in `_delta_log/_sidecars/`) with a
/// `_last_checkpoint` hint at version 8; latest table version is 9.
const FIXTURE: &str = "checkpoint-v2-table";
const FIXTURE_CHECKPOINT_VERSION: u64 = 8;
const FIXTURE_LATEST_VERSION: u64 = 9;

/// Table root within the in-memory store.
const TABLE_PREFIX: &str = "table";

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error + Send + Sync + 'static>>;

fn fixture_dir(fixture: &str) -> std::path::PathBuf {
    std::fs::canonicalize(format!(
        "{}/../test/tests/data/{fixture}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("fixture directory exists")
}

fn table_url() -> Url {
    Url::parse(&format!("memory:///{TABLE_PREFIX}/")).unwrap()
}

/// Every fixture file path relative to the table root (forward slashes).
fn fixture_files(fixture: &str) -> Vec<(String, Vec<u8>)> {
    let dir = fixture_dir(fixture);
    let mut files = Vec::new();
    let mut pending = vec![dir.clone()];
    while let Some(current) = pending.pop() {
        for entry in std::fs::read_dir(&current).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else {
                let relative = path
                    .strip_prefix(&dir)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                files.push((relative, std::fs::read(&path).unwrap()));
            }
        }
    }
    files
}

/// Load a fixture into an [`InMemory`] store beneath [`TABLE_PREFIX`].
async fn fixture_store(fixture: &str) -> Arc<InMemory> {
    let store = Arc::new(InMemory::new());
    for (relative, bytes) in fixture_files(fixture) {
        let key = Path::from(format!("{TABLE_PREFIX}/{relative}"));
        store.put(&key, Bytes::from(bytes).into()).await.unwrap();
    }
    store
}

/// A table-root-relative manifest of the fixture's `_delta_log`, minus `_last_checkpoint`
/// and any path containing `exclude`.
fn fixture_log_manifest(fixture: &str, exclude: Option<&str>) -> Vec<ObjectMeta> {
    fixture_files(fixture)
        .into_iter()
        .filter(|(relative, _)| {
            relative.starts_with("_delta_log/")
                && !relative.ends_with("_last_checkpoint")
                && exclude.is_none_or(|pattern| !relative.contains(pattern))
        })
        .map(|(relative, bytes)| ObjectMeta {
            location: Path::from(relative),
            last_modified: chrono::Utc::now(),
            size: bytes.len() as u64,
            e_tag: None,
            version: None,
        })
        .collect()
}

fn inline_options() -> OpenOptions {
    OpenOptions {
        executor: Some(InlineExecutor.into()),
        ..OpenOptions::default()
    }
}

/// Priming from a listing must cache the checkpoint tail including v2 sidecars, and
/// report the `_last_checkpoint` hint.
#[tokio::test]
async fn test_prime_list_caches_log_tail_and_sidecars() -> TestResult {
    let store = PrimedStore::try_new(fixture_store(FIXTURE).await, &table_url())?;
    let report = store.prime(LogSource::List).await?;

    assert_eq!(report.checkpoint_version, Some(FIXTURE_CHECKPOINT_VERSION));
    assert!(report.files > 0 && report.bytes > 0);

    // The primed store serves log listings from its cache: the checkpoint manifest, the
    // post-checkpoint commit, and the sidecars must all be there.
    let log_prefix = Path::from(format!("{TABLE_PREFIX}/_delta_log"));
    let cached: Vec<String> = store
        .list(Some(&log_prefix))
        .map_ok(|meta| meta.location.to_string())
        .try_collect()
        .await?;
    assert!(
        cached.iter().any(|p| p.contains("_sidecars/")),
        "sidecars missing from primed cache: {cached:?}"
    );
    assert!(
        cached
            .iter()
            .any(|p| p.ends_with("00000000000000000009.json")),
        "post-checkpoint commit missing from primed cache: {cached:?}"
    );
    assert!(
        cached
            .iter()
            .any(|p| p.contains("00000000000000000008.checkpoint.")),
        "checkpoint manifest missing from primed cache: {cached:?}"
    );
    Ok(())
}

/// Both guardrails must fail loud, naming the cap, before caching the log.
#[tokio::test]
async fn test_prime_enforces_caps() -> TestResult {
    let inner = fixture_store(FIXTURE).await;

    let store = PrimedStore::try_new(inner.clone(), &table_url())?.with_limits(PrimeLimits {
        max_files: 2,
        ..PrimeLimits::default()
    });
    let err = store.prime(LogSource::List).await.expect_err("file cap");
    assert!(err.to_string().contains("max_files"), "got: {err}");

    let store = PrimedStore::try_new(inner, &table_url())?.with_limits(PrimeLimits {
        max_bytes: 64,
        ..PrimeLimits::default()
    });
    let err = store.prime(LogSource::List).await.expect_err("byte cap");
    assert!(err.to_string().contains("max_bytes"), "got: {err}");
    Ok(())
}

/// The full open path — prime, session wiring, snapshot build over the JSON v2
/// checkpoint + parquet sidecars — completes on the inline executor and agrees with the
/// default (tokio) executor.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_open_table_inline_executor_matches_tokio() -> TestResult {
    let inner = fixture_store(FIXTURE).await;

    let inline = open_table_with_store(
        inner.clone(),
        &table_url(),
        LogSource::List,
        inline_options(),
    )
    .await?;
    assert_eq!(inline.snapshot.version(), FIXTURE_LATEST_VERSION);

    let tokio =
        open_table_with_store(inner, &table_url(), LogSource::List, OpenOptions::default()).await?;
    assert_eq!(tokio.snapshot.version(), inline.snapshot.version());

    let schema = snapshot_schema_json(&inline.snapshot)?;
    assert!(schema.contains("\"name\":\"id\""), "got schema: {schema}");
    Ok(())
}

/// A host-supplied manifest (no listing) primes to the same snapshot as a listing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_open_table_from_manifest() -> TestResult {
    let opened = open_table_with_store(
        fixture_store(FIXTURE).await,
        &table_url(),
        LogSource::Manifest(fixture_log_manifest(FIXTURE, None)),
        inline_options(),
    )
    .await?;
    assert_eq!(opened.snapshot.version(), FIXTURE_LATEST_VERSION);
    Ok(())
}

/// Opening at an explicit historical version pins the snapshot there.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_open_table_at_version() -> TestResult {
    let opened = open_table_with_store(
        fixture_store(FIXTURE).await,
        &table_url(),
        LogSource::List,
        OpenOptions {
            version: Some(FIXTURE_CHECKPOINT_VERSION),
            ..inline_options()
        },
    )
    .await?;
    assert_eq!(opened.snapshot.version(), FIXTURE_CHECKPOINT_VERSION);
    Ok(())
}

/// Decode a full IPC stream (concatenated chunks) into record batches.
fn decode_ipc(bytes: &[u8]) -> TestResult<Vec<RecordBatch>> {
    let reader = StreamReader::try_new(std::io::Cursor::new(bytes), None)?;
    Ok(reader.collect::<Result<Vec<_>, _>>()?)
}

/// `query_ipc` chunks concatenate into one valid incremental IPC stream whose contents
/// match what the same session returns through the regular DataFrame path.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_query_ipc_round_trip() -> TestResult {
    let opened = open_table_with_store(
        fixture_store(FIXTURE).await,
        &table_url(),
        LogSource::List,
        inline_options(),
    )
    .await?;
    register_snapshot(&opened.ctx, opened.snapshot.clone())?;

    let sql = format!("SELECT id, name FROM {TABLE_NAME} ORDER BY id");
    let chunks: Vec<Vec<u8>> = query_ipc(&opened.ctx, &sql).await?.try_collect().await?;
    assert!(!chunks.is_empty());

    let concatenated: Vec<u8> = chunks.concat();
    let batches = decode_ipc(&concatenated)?;
    let schema = batches[0].schema();
    let result = concat_batches(&schema, &batches)?;
    assert!(result.num_rows() > 0);

    let expected = opened.ctx.sql(&sql).await?.collect().await?;
    let expected = concat_batches(&schema, &expected)?;
    assert_eq!(result, expected);

    // Every chunk boundary is also a valid stream prefix for incremental consumers.
    let first_batch = decode_ipc(&[chunks[0].clone()].concat())?;
    assert_eq!(first_batch.len(), 1);
    Ok(())
}

/// V3: with the facade session config, representative read plans contain no partition
/// or task-spawning operators — the "supported plan shape" for driving execution on the
/// JS event loop without a threaded runtime.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_v3_plan_shape_has_no_repartition_operators() -> TestResult {
    let opened = open_table_with_store(
        fixture_store(FIXTURE).await,
        &table_url(),
        LogSource::List,
        inline_options(),
    )
    .await?;
    register_snapshot(&opened.ctx, opened.snapshot.clone())?;

    let queries = [
        format!("SELECT * FROM {TABLE_NAME} WHERE id > 3 LIMIT 5"),
        format!("SELECT name, count(*), max(id) FROM {TABLE_NAME} GROUP BY name"),
        format!("SELECT id, name FROM {TABLE_NAME} ORDER BY created_at DESC LIMIT 3"),
        format!("SELECT count(*) FROM {TABLE_NAME}"),
    ];
    for sql in &queries {
        let plan = opened.ctx.sql(sql).await?.create_physical_plan().await?;
        assert_single_partition_shape(&plan, sql);
        // The plan must also *run* to completion (natively under tokio here; the wasm
        // half of V3 is the wasm-pack smoke test driving the same shape).
        let batches: Vec<RecordBatch> =
            datafusion::physical_plan::execute_stream(plan, opened.ctx.task_ctx())?
                .try_collect()
                .await?;
        assert!(!batches.is_empty(), "no output for {sql}");
    }
    Ok(())
}

/// Operators that repartition or spawn tasks; none may appear in facade plans.
const FORBIDDEN_OPERATORS: &[&str] = &[
    "RepartitionExec",
    "CoalescePartitionsExec",
    "SortPreservingMergeExec",
];

fn assert_single_partition_shape(plan: &Arc<dyn ExecutionPlan>, sql: &str) {
    let name = plan.name();
    assert!(
        !FORBIDDEN_OPERATORS.contains(&name),
        "plan for `{sql}` contains forbidden operator {name}"
    );
    assert_eq!(
        plan.output_partitioning().partition_count(),
        1,
        "plan for `{sql}` has a multi-partition operator {name}"
    );
    for child in plan.children() {
        assert_single_partition_shape(child, sql);
    }
}

/// Wraps a store, hanging forever on reads of paths containing a marker — the native
/// stand-in for an unprimed browser fetch.
#[derive(Debug)]
struct BlockingPathsStore {
    inner: Arc<InMemory>,
    marker: &'static str,
    armed: Arc<AtomicBool>,
}

impl std::fmt::Display for BlockingPathsStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BlockingPathsStore({})", self.marker)
    }
}

#[async_trait]
impl ObjectStore for BlockingPathsStore {
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
        if self.armed.load(Ordering::Relaxed) && location.as_ref().contains(self.marker) {
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

/// A manifest that forgets the v2 sidecars primes fine — snapshot build only reads the
/// checkpoint manifest, so `open` succeeds — but the first scan replay reads the sidecars
/// and falls through to the (blocking) inner store. The inline executor must surface that
/// as a loud "would block / not primed" error, not a hang.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_unprimed_sidecar_read_fails_loud() -> TestResult {
    let store = Arc::new(BlockingPathsStore {
        inner: fixture_store(FIXTURE).await,
        marker: "_sidecars/",
        armed: Arc::new(AtomicBool::new(true)),
    });

    let opened = open_table_with_store(
        store,
        &table_url(),
        LogSource::Manifest(fixture_log_manifest(FIXTURE, Some("_sidecars/"))),
        inline_options(),
    )
    .await?;
    // Sidecars are read lazily: the snapshot builds from the checkpoint manifest alone.
    assert_eq!(opened.snapshot.version(), FIXTURE_LATEST_VERSION);

    let engine: Arc<DataFusionEngine> =
        DataFusionEngine::new(opened.ctx.task_ctx(), InlineExecutor).into();
    let scan = opened.snapshot.scan_builder().build()?;
    let err = match scan.scan_metadata(engine).try_collect::<Vec<_>>().await {
        Ok(_) => panic!("scan replay must fail against unprimed sidecars"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("would block"),
        "expected a would-block/not-primed error, got: {err}"
    );
    Ok(())
}

/// The committed zstd fixture is a valid table: natively (zstd codec present) it opens
/// and queries fine. The wasm smoke test asserts the same query errors gracefully there.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_zstd_fixture_queryable_natively() -> TestResult {
    let store = Arc::new(InMemory::new());
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/zstd-table");
    for entry in [
        "_delta_log/00000000000000000000.json",
        "part-00000-zstd.parquet",
    ] {
        let bytes = std::fs::read(dir.join(entry))?;
        let key = Path::from(format!("{TABLE_PREFIX}/{entry}"));
        store.put(&key, Bytes::from(bytes).into()).await?;
    }

    let opened =
        open_table_with_store(store, &table_url(), LogSource::List, inline_options()).await?;
    register_snapshot(&opened.ctx, opened.snapshot.clone())?;
    let batches = opened
        .ctx
        .sql(&format!("SELECT sum(value) FROM {TABLE_NAME}"))
        .await?
        .collect()
        .await?;
    let total: i64 =
        arrow::util::display::array_value_to_string(batches[0].column(0), 0)?.parse()?;
    assert_eq!(total, (0..32).sum::<i64>());
    Ok(())
}

/// Regenerate the committed `tests/data/zstd-table` fixture: a single-commit table whose
/// data file uses zstd-compressed parquet pages. The wasm build drops the zstd codec, so
/// the wasm smoke test asserts querying this table errors gracefully instead of panicking.
///
/// Run manually after fixture-affecting changes:
/// `cargo test -p deltalake-wasm --test native generate_zstd_fixture -- --ignored`
#[tokio::test]
#[ignore = "fixture generator, run manually"]
async fn generate_zstd_fixture() -> TestResult {
    use arrow_array::Int64Array;
    use parquet::arrow::ArrowWriter;
    use parquet::basic::{Compression, ZstdLevel};
    use parquet::file::properties::WriterProperties;

    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/zstd-table");
    std::fs::create_dir_all(root.join("_delta_log"))?;

    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("value", arrow::datatypes::DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from_iter_values(0..32))],
    )?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .build();
    let mut buffer = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buffer, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    let data_file = "part-00000-zstd.parquet";
    let size = buffer.len();
    std::fs::write(root.join(data_file), &buffer)?;

    let schema_string = serde_json::json!({
        "type": "struct",
        "fields": [{"name": "value", "type": "long", "nullable": true, "metadata": {}}]
    })
    .to_string();
    let commit = [
        serde_json::json!({"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}}),
        serde_json::json!({"metaData": {
            "id": "8f5f34c1-42ba-4f7a-8de4-8f4bbcc42d00",
            "format": {"provider": "parquet", "options": {}},
            "schemaString": schema_string,
            "partitionColumns": [],
            "configuration": {},
            "createdTime": 1751500000000_u64,
        }}),
        serde_json::json!({"add": {
            "path": data_file,
            "partitionValues": {},
            "size": size,
            "modificationTime": 1751500000000_u64,
            "dataChange": true,
            "stats": "{\"numRecords\":32}",
        }}),
    ]
    .map(|action| action.to_string())
    .join("\n");
    std::fs::write(
        root.join("_delta_log/00000000000000000000.json"),
        format!("{commit}\n"),
    )?;
    Ok(())
}
