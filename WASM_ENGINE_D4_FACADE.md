# D4 — `deltalake-wasm` facade crate + end-to-end proof

> Part of the wasm-engine effort — read [`WASM_ENGINE.md`](./WASM_ENGINE.md) first.
>
> **Status: not started** · Depends on: **D1 + D2** (D3 optional) · Blocks: D5 and
> mangrove Phase B · Recommended model: **Fable**

## Goal

A new crate `crates/wasm` (package `deltalake-wasm`, `crate-type =
["cdylib","rlib"]`) that makes the engine *run* in a browser: a fetch-backed
`ObjectStore`, the **PrimedStore** that satisfies D2's "ready futures only"
contract, a snapshot/query API over `deltalake-core`'s existing pieces, and a
minimal wasm-bindgen surface proving end-to-end reads. Cloud-credential handling
(UC vending, SAS, workers wiring) is explicitly **mangrove Phase B**, not here —
this crate must stay mangrove-agnostic.

## Inputs from other chunks (contracts)

- D2: `ExecutorHandle` — on wasm the engine only makes progress on ready futures;
  `DataFusionEngine::new_from_session` works without tokio;
  `Snapshot::try_new_with_engine` (`crates/core/src/kernel/snapshot/mod.rs:199`)
  + `scan_metadata` complete against a primed store; DV tables fail loud.
- D1: engine handlers resolve stores via `ctx.runtime_env().object_store(url)` —
  registering our store in the session's `RuntimeEnv` is the only wiring needed.
- Provider entry: `DeltaScan::new(snapshot, config)`
  (`crates/core/src/delta_datafusion/table_provider/next/mod.rs:518`) — no
  `LogStore` needed for reads; `logstore::get_engine`'s wasm `unimplemented!()`
  (`crates/core/src/logstore/mod.rs:618`) must remain unreachable — never
  construct a `LogStore` on wasm v1.

## Crate design

Dependencies: `deltalake-core` (`default-features = false`, `features =
["datafusion"]`), `datafusion`, `object_store` (for `memory::InMemory`),
`futures`, `url`, `wasm-bindgen`, `wasm-bindgen-futures`, `js-sys`,
`serde-wasm-bindgen`, `console_error_panic_hook`, and an HTTP client:
`reqwest` (wasm = fetch backend) or `gloo-net` — pick whichever cleanly supports
**Range request headers**; verify early (part of V7). Inherit the workspace's
wasm build config (`.cargo/config.toml` `getrandom_backend="wasm_js"`, arrow-rs
parquet patch — see `WASM.md`).

### `FetchObjectStore` (`src/fetch_store.rs`)

Read-only `object_store::ObjectStore` over HTTP(S): `get`, `get_opts`
(`GetRange` → `Range` header, expect 206), `head` (HEAD or `Range: bytes=0-0`
fallback), `get_ranges`. `put*`/`delete`/`copy`/`rename` → `NotSupported`.
`list*`: implement only what priming needs — and priming prefers a **manifest**
(below) precisely because plain HTTP has no listing; if a listing endpoint
exists (S3/Azure/GCS REST), it can back `list_with_offset` later. Base-URL +
path composition must preserve presigned query strings if present.
Note: JS types are `!Send`; keep any `JsFuture` interop wrapped so the store's
futures are still usable under `object_store`'s trait bounds on wasm (single
thread — if trait bounds force `Send`, use `send_wrapper` or confine JS interop
behind channels; resolve during V7).

### `PrimedStore` (`src/primed.rs`) — the load-bearing piece

Composite `ObjectStore`: an `InMemory` cache holding everything under
`_delta_log/`, pass-through to the inner (fetch) store for data files.

```rust
pub enum LogSource {
    /// List the log via the inner store (requires a listing-capable store).
    List,
    /// Host-supplied file manifest (e.g. mangrove /delta/v1 loadTable commit tail).
    Manifest(Vec<ObjectMeta>),
}

impl PrimedStore {
    pub async fn prime(&self, table_root: &Url, source: LogSource) -> Result<PrimeReport>;
}
```

`prime()` (async — driven by the JS event loop, so free to fetch):
1. `GET _delta_log/_last_checkpoint` (tolerate 404/missing) — cache it.
2. Determine the file set: from `Manifest`, or by listing `_delta_log/` with
   offset at the checkpoint version. Include commits `*.json`, checkpoint parts
   `*.parquet`, compacted logs, and **v2 sidecars** — note
   `_delta_log/_sidecars/…` sorts *after* numeric commit names (`'_' > '9'`), so
   an offset listing captures them; a manifest must include them explicitly.
3. Fetch all files concurrently (`futures::stream::iter(..).buffered(n)`) into
   the `InMemory` store.
4. Guardrails: configurable caps on file count and total bytes (defaults: e.g.
   512 files / 256 MiB) with a clear error naming the cap; `PrimeReport` returns
   counts/bytes for the caller's telemetry.

After `prime()`, every kernel-side handler future (list from cache, JSON/parquet
`DataSourceExec` over cached bytes, footer reads) is ready ⇒ `InlineExecutor`
succeeds. A miss (e.g. table updated between prime and snapshot) surfaces as
D2's "not primed" error → caller re-primes; document this loop.

### Session + API (`src/lib.rs`, rlib surface — natively testable)

- `fn session(store: Arc<dyn ObjectStore>, table_url: &Url) -> SessionContext`:
  registers the store in the `RuntimeEnv` object-store registry under the table
  URL; config: `target_partitions = 1`, repartitioning off (V3), sensible
  `batch_size`.
- `async fn open_table(table_url, LogSource, version: Option<u64>) ->
  Result<(SessionContext, Arc<Snapshot>)>`: build `PrimedStore`, `prime()`, then
  `Snapshot::try_new_with_engine(DataFusionEngine::new_from_context(ctx), …)`.
  Wasm target only for the fetch store; the priming/snapshot logic itself must
  compile natively (V2-style tests reuse it with `InMemory` inner stores).
- `async fn query_ipc(ctx, snapshot, sql: &str) -> impl Stream<Item =
  Result<Vec<u8>>>`: register `DeltaScan::new(snapshot, config)` as a table,
  `ctx.sql(sql)` → `execute_stream` → encode each `RecordBatch` as an Arrow IPC
  stream chunk (`arrow_ipc::writer::StreamWriter` per batch, or one writer with
  per-batch flush — match what `ArrowResultStore.append(ipc)` expects: a valid
  incremental IPC stream).

### wasm-bindgen surface (`src/bindings.rs`, `cfg(target_arch = "wasm32")`)

Deliberately minimal — the proof harness, not the product API (mangrove Phase B
builds its own on the rlib):

```text
#[wasm_bindgen] WasmDeltaTable
  open(table_url: String, opts: JsValue) -> Promise<WasmDeltaTable>   // opts: version?, manifest?, caps?
  query(sql: String, on_batch: js_sys::Function) -> Promise<QueryStats> // on_batch(Uint8Array IPC chunk)
  schema_json() -> String
```

All async via `wasm_bindgen_futures::future_to_promise`; install
`console_error_panic_hook` in a `#[wasm_bindgen(start)]`. The only sync-blocking
(inline-executor) work happens inside kernel snapshot/scan-metadata calls against
primed data; recommend (in docs) hosting in a Web Worker so those bursts don't
jank the main thread.

## First actions (validations)

- **V3**: natively, build the exact session config above, plan
  `SELECT … WHERE … LIMIT …` through `DeltaScan`, and walk the physical plan
  asserting no `RepartitionExec`/spawned-task operators; then confirm on wasm
  that executing the stream via `wasm-bindgen-futures` completes. If an operator
  spawns, identify config to avoid it (this defines the "supported plan shape"
  note for Phase B).
- **V7**: Range-GET behavior of the chosen HTTP client on wasm (headers
  preserved, 206 handling, CORS notes); sidecar priming against a v2-checkpoint
  fixture table.

## Validation gates

- `cargo check -p deltalake-wasm --target wasm32-unknown-unknown` (CI gate).
- Native tests (rlib): priming (List + Manifest sources, caps, sidecars),
  `open_table` + `query_ipc` against fixture tables over `InMemory`-backed
  "fetch" stores; IPC decodes to expected rows.
- `wasm-pack test --node` (or `--headless --chrome`) smoke:
  1. embedded-bytes table (memory store) → `open` + `SELECT count(*)` + filtered
     select → IPC decodes correctly;
  2. HTTP static-server table via `FetchObjectStore` (browser test; node needs a
     fetch shim — browser preferred);
  3. DV table → loud, friendly error;
  4. zstd-compressed table → graceful "unsupported codec" error (not a panic).
- Update `WASM.md` build/run instructions.

## Risks

- Hidden tokio/spawn in DF operators at runtime (V3 front-loads this).
- `!Send` JS interop vs `object_store`/kernel `Send` bounds (see fetch-store
  note; worst case: a shim store that buffers via channels).
- CORS in browser tests — keep the wasm smoke self-hosted
  (`wasm-bindgen-test` + local server) so CI doesn't depend on external buckets.
- Table updated between prime and query → stale-but-consistent snapshot (fine;
  it's pinned to the primed version) or "not primed" error → document re-prime.

## Done criteria

All gates green; `WASM.md` updated ("runs" instead of "compiles"); status in
`WASM_ENGINE.md` updated; the crate's public rlib API documented (rustdoc) —
that API is mangrove Phase B's dependency surface.
