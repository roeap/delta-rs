//! Wasm smoke tests: the end-to-end proof that the DataFusion query path *runs* on
//! `wasm32-unknown-unknown`, driven by the JS event loop — no tokio anywhere.
//!
//! Run with `wasm-pack test --node crates/wasm` (or `--headless --chrome`). Fixture
//! tables are embedded as bytes and served from an in-memory store, so the tests are
//! self-hosted; `FetchObjectStore` against a real HTTP endpoint is covered by the
//! (browser-only) notes in `WASM.md`.
//!
//! Regenerate `fixtures.rs` with the shell snippet in the D4 handover doc after fixture
//! changes.

#![cfg(all(target_arch = "wasm32", target_os = "unknown"))]

mod fixtures;

use std::sync::Arc;

use arrow_ipc::reader::StreamReader;
use bytes::Bytes;
use futures::TryStreamExt;
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt as _};
use url::Url;
use wasm_bindgen_test::wasm_bindgen_test;

use deltalake_wasm::{
    LogSource, OpenOptions, TABLE_NAME, open_table_with_store, query_ipc, register_snapshot,
    snapshot_schema_json,
};

const TABLE_PREFIX: &str = "table";

fn table_url() -> Url {
    Url::parse(&format!("memory:///{TABLE_PREFIX}/")).unwrap()
}

async fn fixture_store(files: &[(&'static str, &'static [u8])]) -> Arc<dyn ObjectStore> {
    let store = InMemory::new();
    for (relative, bytes) in files {
        let key = Path::from(format!("{TABLE_PREFIX}/{relative}"));
        store
            .put(&key, Bytes::from_static(bytes).into())
            .await
            .unwrap();
    }
    Arc::new(store)
}

/// Run `sql` through the IPC surface and decode the concatenated chunks.
async fn query_rows(
    ctx: &datafusion::execution::context::SessionContext,
    sql: &str,
) -> Result<Vec<arrow_array::RecordBatch>, deltalake_core::DeltaTableError> {
    let chunks: Vec<Vec<u8>> = query_ipc(ctx, sql).await?.try_collect().await?;
    let reader = StreamReader::try_new(std::io::Cursor::new(chunks.concat()), None)
        .map_err(deltalake_core::DeltaTableError::from)?;
    reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(deltalake_core::DeltaTableError::from)
}

/// End-to-end read over the v2-checkpoint fixture: prime → snapshot (inline executor)
/// → SQL through `DeltaScan` → Arrow IPC out, all on the JS event loop.
#[wasm_bindgen_test]
async fn wasm_open_and_query_v2_checkpoint_table() {
    let store = fixture_store(fixtures::CHECKPOINT_V2_TABLE).await;
    let opened =
        open_table_with_store(store, &table_url(), LogSource::List, OpenOptions::default())
            .await
            .expect("open primed table on wasm");
    assert_eq!(opened.snapshot.version(), 9);
    assert!(
        snapshot_schema_json(&opened.snapshot)
            .expect("schema json")
            .contains("\"name\":\"id\"")
    );

    register_snapshot(&opened.ctx, opened.snapshot.clone()).expect("register table");

    // count(*) and a full column read must agree — proves IPC chunks decode correctly.
    let count_batches = query_rows(&opened.ctx, &format!("SELECT count(*) FROM {TABLE_NAME}"))
        .await
        .expect("count query");
    let count = arrow_array::cast::AsArray::as_primitive::<arrow_array::types::Int64Type>(
        count_batches[0].column(0),
    )
    .value(0);
    assert!(count > 0, "expected rows in the fixture table");

    let id_batches = query_rows(
        &opened.ctx,
        &format!("SELECT id FROM {TABLE_NAME} ORDER BY id"),
    )
    .await
    .expect("id query");
    let ids: usize = id_batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(ids as i64, count);

    // Filtered read stays a subset.
    let filtered = query_rows(
        &opened.ctx,
        &format!("SELECT id FROM {TABLE_NAME} WHERE id > 2"),
    )
    .await
    .expect("filtered query");
    let filtered: usize = filtered.iter().map(|b| b.num_rows()).sum();
    assert!(filtered <= ids);
}

/// V7: `FetchObjectStore` against a real HTTP endpoint — Range GETs (206 and
/// 200-with-full-body handling), HEAD, and manifest-based priming, since plain HTTP
/// cannot list.
///
/// Gated on `WASM_SMOKE_HTTP_BASE` (compile-time env): start
/// `node tests/http-server.mjs ../test/tests/data` first, then run
/// `WASM_SMOKE_HTTP_BASE=http://127.0.0.1:8917 wasm-pack test --node`.
#[wasm_bindgen_test]
async fn wasm_fetch_store_http_table() {
    let Some(base) = option_env!("WASM_SMOKE_HTTP_BASE") else {
        return; // No server provided; the memory-store tests above still ran.
    };
    let url = Url::parse(&format!("{base}/checkpoint-v2-table/")).expect("base url");
    let store =
        Arc::new(deltalake_wasm::FetchObjectStore::try_new(url.clone()).expect("fetch store"));

    // Plain HTTP has no listing: prime from a manifest (paths + sizes come straight from
    // the embedded fixture, standing in for a catalog's commit-tail response).
    let manifest: Vec<object_store::ObjectMeta> = fixtures::CHECKPOINT_V2_TABLE
        .iter()
        .filter(|(path, _)| path.starts_with("_delta_log/") && !path.ends_with("_last_checkpoint"))
        .map(|(path, bytes)| object_store::ObjectMeta {
            location: Path::from(*path),
            last_modified: chrono::DateTime::UNIX_EPOCH,
            size: bytes.len() as u64,
            e_tag: None,
            version: None,
        })
        .collect();

    let opened = open_table_with_store(
        store,
        &url,
        LogSource::Manifest(manifest),
        OpenOptions::default(),
    )
    .await
    .expect("open table over HTTP");
    assert_eq!(opened.snapshot.version(), 9);

    register_snapshot(&opened.ctx, opened.snapshot.clone()).expect("register table");
    let batches = query_rows(
        &opened.ctx,
        &format!("SELECT id FROM {TABLE_NAME} WHERE id > 2 ORDER BY id"),
    )
    .await
    .expect("ranged parquet reads over HTTP");
    assert!(batches.iter().map(|b| b.num_rows()).sum::<usize>() > 0);
}

/// v1 limit: deletion vectors fail loud with a friendly error, not a hang or panic.
#[wasm_bindgen_test]
async fn wasm_dv_table_fails_loud() {
    let store = fixture_store(fixtures::DV_TABLE).await;
    let opened =
        open_table_with_store(store, &table_url(), LogSource::List, OpenOptions::default())
            .await
            .expect("DV table snapshot builds fine; only reading data is unsupported");
    register_snapshot(&opened.ctx, opened.snapshot.clone()).expect("register table");

    let result = query_rows(&opened.ctx, &format!("SELECT count(*) FROM {TABLE_NAME}")).await;
    let err = result.expect_err("querying a DV table must fail on wasm");
    let message = err.to_string().to_lowercase();
    assert!(
        message.contains("deletion vector"),
        "expected a deletion-vector error, got: {err}"
    );
}

/// v1 limit: zstd parquet pages (codec compiled out on wasm) error gracefully.
#[wasm_bindgen_test]
async fn wasm_zstd_table_errors_gracefully() {
    let store = fixture_store(fixtures::ZSTD_TABLE).await;
    let opened =
        open_table_with_store(store, &table_url(), LogSource::List, OpenOptions::default())
            .await
            .expect("zstd table snapshot builds fine; only page decoding is unsupported");
    register_snapshot(&opened.ctx, opened.snapshot.clone()).expect("register table");

    let result = query_rows(&opened.ctx, &format!("SELECT sum(value) FROM {TABLE_NAME}")).await;
    let err = result.expect_err("querying zstd-compressed data must fail on wasm");
    let message = err.to_string().to_lowercase();
    assert!(
        message.contains("zstd") || message.contains("compression") || message.contains("codec"),
        "expected an unsupported-codec error, got: {err}"
    );
}
