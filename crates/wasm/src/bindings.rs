//! Minimal wasm-bindgen surface — the end-to-end proof harness, not the product API
//! (mangrove Phase B builds its own surface on the rlib).
//!
//! The only synchronously blocking (inline-executor) work happens inside kernel
//! snapshot/scan-metadata calls against primed data; host in a Web Worker so those
//! bursts don't jank the main thread.

use std::sync::Arc;

use js_sys::{Function, Uint8Array};
use object_store::ObjectMeta;
use object_store::path::Path;
use url::Url;
use wasm_bindgen::prelude::*;

use crate::fetch_store::FetchObjectStore;
use crate::{
    LogSource, OpenOptions, OpenedTable, PrimeLimits, open_table_with_store, query_ipc,
    register_snapshot, snapshot_schema_json,
};

#[wasm_bindgen(start)]
fn init() {
    console_error_panic_hook::set_once();
}

fn js_err(err: impl std::fmt::Display) -> JsError {
    JsError::new(&err.to_string())
}

/// Options accepted by [`WasmDeltaTable::open`].
#[derive(serde::Deserialize, Default)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct JsOpenOptions {
    /// Table version to load; latest when omitted.
    version: Option<u64>,
    /// The catalog's latest ratified version. Required for catalog-managed
    /// (`catalogManaged`) tables; omit for filesystem/external tables.
    max_catalog_version: Option<u64>,
    /// `_delta_log` manifest (paths relative to the table root). Required for plain
    /// HTTP hosts, which cannot list.
    manifest: Option<Vec<JsManifestEntry>>,
    /// Priming guardrail overrides.
    max_files: Option<usize>,
    max_bytes: Option<u64>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct JsManifestEntry {
    path: String,
    size: u64,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct QueryStats {
    chunks: u32,
    ipc_bytes: u64,
}

/// A read-only Delta table opened over HTTP, queryable with SQL as table `delta`.
#[wasm_bindgen]
pub struct WasmDeltaTable {
    inner: OpenedTable,
}

#[wasm_bindgen]
impl WasmDeltaTable {
    /// Open the Delta table at `table_url` (fully-qualified http(s) URL).
    ///
    /// `opts`: `{ version?, manifest?: [{path, size}], maxFiles?, maxBytes? }`.
    /// Without a manifest the host must support listing, which plain HTTP does not —
    /// pass the log manifest whenever in doubt.
    pub async fn open(table_url: String, opts: JsValue) -> Result<WasmDeltaTable, JsError> {
        let opts: JsOpenOptions = if opts.is_undefined() || opts.is_null() {
            JsOpenOptions::default()
        } else {
            serde_wasm_bindgen::from_value(opts).map_err(js_err)?
        };
        let url = Url::parse(&table_url).map_err(js_err)?;
        let store = Arc::new(FetchObjectStore::try_new(url.clone()).map_err(js_err)?);

        let source = match opts.manifest {
            Some(entries) => LogSource::Manifest(
                entries
                    .into_iter()
                    .map(|entry| ObjectMeta {
                        location: Path::from(entry.path),
                        last_modified: chrono::DateTime::UNIX_EPOCH,
                        size: entry.size,
                        e_tag: None,
                        version: None,
                    })
                    .collect(),
            ),
            None => LogSource::List,
        };
        let mut limits = PrimeLimits::default();
        if let Some(max_files) = opts.max_files {
            limits.max_files = max_files;
        }
        if let Some(max_bytes) = opts.max_bytes {
            limits.max_bytes = max_bytes;
        }

        let opened = open_table_with_store(
            store,
            &url,
            source,
            OpenOptions {
                version: opts.version,
                max_catalog_version: opts.max_catalog_version,
                limits,
                executor: None,
            },
        )
        .await
        .map_err(js_err)?;
        register_snapshot(&opened.ctx, opened.snapshot.clone()).map_err(js_err)?;
        Ok(WasmDeltaTable { inner: opened })
    }

    /// Execute `sql` (the table is registered as `delta`), invoking
    /// `on_batch(Uint8Array)` with each Arrow IPC stream chunk. Concatenated chunks form
    /// one valid IPC stream.
    pub async fn query(&self, sql: String, on_batch: Function) -> Result<JsValue, JsError> {
        use futures::TryStreamExt;

        let mut chunks = query_ipc(&self.inner.ctx, &sql).await.map_err(js_err)?;
        let mut stats = QueryStats {
            chunks: 0,
            ipc_bytes: 0,
        };
        while let Some(chunk) = chunks.try_next().await.map_err(js_err)? {
            stats.chunks += 1;
            stats.ipc_bytes += chunk.len() as u64;
            let array = Uint8Array::from(chunk.as_slice());
            on_batch
                .call1(&JsValue::NULL, &array.into())
                .map_err(|err| JsError::new(&format!("on_batch callback failed: {err:?}")))?;
        }
        Ok(serde_wasm_bindgen::to_value(&stats).map_err(js_err)?)
    }

    /// The table's Delta schema as JSON.
    #[wasm_bindgen(js_name = schemaJson)]
    pub fn schema_json(&self) -> Result<String, JsError> {
        snapshot_schema_json(&self.inner.snapshot).map_err(js_err)
    }

    /// The snapshot's pinned table version.
    pub fn version(&self) -> u64 {
        self.inner.snapshot.version()
    }
}
