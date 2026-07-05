# D4 — `deltalake-wasm` facade crate + end-to-end proof

> Part of the wasm-engine effort — read [`WASM_ENGINE.md`](./WASM_ENGINE.md) first.
>
> **Status: done — see "Outcome & deviations" below (2026-07-05)** · Depends on: **D1 + D2**
> (D3 optional) · Blocks: D5 and mangrove Phase B · Recommended model: **Fable**

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

## Outcome & deviations (for D5 / Phase B)

All gates green. `crates/wasm` (package `deltalake-wasm`) exists as designed; the
wasm smoke suite (4 tests, `wasm-pack test --node`) proves the full path on
wasm32-unknown-unknown: prime → inline-executor snapshot → SQL through `DeltaScan`
→ Arrow IPC out, plus the two v1-limit error paths, plus `FetchObjectStore`
Range-GETs against a real HTTP server. Deviations and findings:

- **API deviations from the sketch in this doc** (all rustdoc'd; the rlib surface
  is Phase B's dependency):
  - `PrimedStore::try_new(inner, table_url)` holds the table root; `prime(source)`
    doesn't take it per-call. Caps live in `PrimeLimits` (`with_limits`), report in
    `PrimeReport` (files/bytes/checkpoint_version).
  - `open_table` is `open_table_with_store(inner, url, LogSource, OpenOptions) ->
    OpenedTable {ctx, snapshot, store, report}`; `OpenOptions.executor` lets native
    tests force `InlineExecutor`. The wasm-only convenience that builds a
    `FetchObjectStore` itself lives in the bindings, not the rlib.
  - `query_ipc(ctx, sql)` (table pre-registered via `register_snapshot`, name
    `TABLE_NAME = "delta"`) rather than `query_ipc(ctx, snapshot, sql)`.
  - Manifest entries are `ObjectMeta` with locations relative to the table root
    (joined internally); `_last_checkpoint` is always fetched by `prime()` and must
    not be in the manifest.
- **`_last_checkpoint` absence is cached**: a 404 during priming is remembered and
  replayed as an immediate `NotFound`, otherwise the kernel's sync read of it would
  fall through and "would-block" on wasm even for tables without checkpoints.
- **Sidecar reads are lazy** (JSON v2 checkpoint): the kernel snapshot build reads
  only the checkpoint manifest; `_sidecars/*.parquet` are first read during
  `scan_metadata`. Consequence: a manifest that omits sidecars *opens* fine and
  fails loud ("would block … not primed") at first query — covered by
  `test_unprimed_sidecar_read_fails_loud`.
- **V3 result**: with `target_partitions=1` + all repartitioning off, plans for
  select/filter/limit, aggregate, and sort queries contain no `RepartitionExec` /
  `CoalescePartitionsExec` / `SortPreservingMergeExec` and are single-partition
  end-to-end (asserted natively, executed on wasm by the smoke suite).
- **V7 result**: reqwest 0.13's fetch backend works under node ≥18 (no shim);
  Range headers arrive intact, 206 handled, and a 200-ignoring-Range server is
  handled by local slicing. `!Send` resolved with `send_wrapper` around both the
  client and each request future. Browser (`--headless --chrome`) run deferred to
  D5 CI wiring: `tests/http-server.mjs` already sends the needed CORS headers
  (`Access-Control-Allow-Origin/Headers`, `Expose-Headers: Content-Range`).
  Per-object presigned URLs are Phase B; a static query string on the base URL
  (SAS-style) is preserved on every request.
- **Two bugs found outside the crate** (the smoke suite's value):
  - kernel fork commit `81b7cb95` — `DataSkippingFilter::apply` used
    `std::time::Instant` (panics on wasm); routed through the kernel's
    `crate::time` shim. **D5 must preserve this** (third fork delta).
  - `crates/core` writer: the 5 GiB upload-part cap literal overflowed 32-bit
    `usize` at *build* time (cargo **check** passes — future wasm gates should
    build, not check).
- **zstd fixture**: `crates/wasm/tests/data/zstd-table` is generated by the
  `generate_zstd_fixture` ignored test in `tests/native.rs` and guarded natively by
  `test_zstd_fixture_queryable_natively`.
- `tests/wasm_smoke/fixtures.rs` (embedded fixture bytes) is generated; regenerate
  with the shell loop in the session transcript (or by hand) after fixture changes.

### Validation gate results

- `cargo check -p deltalake-wasm --target wasm32-unknown-unknown` — passes (D5:
  prefer `cargo build`, see overflow note above).
- Native rlib tests (`cargo test -p deltalake-wasm`) — 9 passed, 1 ignored
  (fixture generator); includes priming list/manifest/caps/sidecars, inline-vs-
  tokio parity, at-version open, IPC round-trip, V3 plan shape, would-block miss.
- `wasm-pack test --node` — 4/4: v2-checkpoint query e2e, HTTP fetch store
  (with `WASM_SMOKE_HTTP_BASE` + `tests/http-server.mjs`), DV loud error, zstd
  graceful error.
- Native regression bar: workspace builds; `-p deltalake-core --features
  datafusion` shows only the 3 pre-existing kernel-pin failures recorded in D2
  ("Pre-existing gate state"); fmt/clippy clean for the new crate on both targets.
