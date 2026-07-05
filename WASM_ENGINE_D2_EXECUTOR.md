# D2 — Executor seam + wasm compile of the DataFusion subsystem

> Part of the wasm-engine effort — read [`WASM_ENGINE.md`](./WASM_ENGINE.md) first.
>
> **Status: not started** · Depends on: — (rebase constructor changes on D1 if it
> lands first; the files otherwise barely overlap) · Blocks: D4 · Parallel with:
> D1, D3 · Recommended model: **Fable**

## Goal

Make the DataFusion scan subsystem (`delta_datafusion` + the kernel-snapshot glue)
compile and *execute* on `wasm32-unknown-unknown`: replace the tokio
`Handle`-based sync bridge in `DataFusionEngine` with a target-dependent
`ExecutorHandle`, neutralize the remaining tokio bridges on the read path, and
add the fail-loud deletion-vector guard. The execution model on wasm: **all
futures the sync engine drives must already be ready** (data primed by D4's
`PrimedStore`); a would-block future is an error, not a hang.

## First action (V1 — scope inventory)

```sh
cargo check -p deltalake-core --no-default-features --features datafusion --target wasm32-unknown-unknown
```

The `datafusion` feature has **never** been compiled for wasm (the current wasm
gate is `--no-default-features` only, see `WASM.md`). The error inventory from
this command is the authoritative scope of this chunk — the list below is the
*known* surface; expect stragglers (e.g. `IORuntime` references, `tokio::time`,
caching deps). If something fundamental fails (DataFusion itself won't build for
wasm at our pin), **stop and update `WASM_ENGINE.md`** — that invalidates
downstream assumptions. (`WASM_NOTES.md` records that arrow/parquet/DataFusion
did build for wasm during the spike, so expect breakage to be in our glue, not
upstream.)

Cfg predicate used repo-wide: `all(target_arch = "wasm32", target_os = "unknown")`.

## Known tokio surface on the read path (verified call sites)

| Site | What | wasm treatment |
|---|---|---|
| `crates/core/src/delta_datafusion/engine/mod.rs` constructors (`new_from_session` :29, `new_from_context` :36, `new` :44) | `Handle::current()` (panics without tokio) | `ExecutorHandle` (below) |
| `TracedHandle` / `BlockingStreamIterator` / `stream_future_to_iter` (engine/mod.rs, arrives with D1) | `handle.block_on` per item | genericize over `ExecutorHandle` |
| `crates/core/src/kernel/mod.rs:40-50` `spawn_blocking_in_span` (+ `spawn_blocking_with_span` :55-61) | `tokio::task::spawn_blocking` wrapping kernel sync iterators | inline-run variant (below) |
| `crates/core/src/kernel/snapshot/mod.rs:224, 315, 1101, 1131, 1260` | callers of the above (snapshot build/update, tombstones, app-ids, `read_last_checkpoint_version`) | switch to the helper |
| `crates/core/src/kernel/snapshot/stream.rs:78, 127` | `ReceiverStreamBuilder` (tokio mpsc + `JoinSet::spawn_blocking`) | wasm: `futures::stream::iter` over the sync kernel iterator |
| `crates/core/src/kernel/snapshot/scan.rs` (~:412-447) `Scan::scan_metadata` | drives kernel scan-metadata iterator via blocking spawn | same `stream::iter` treatment |
| `crates/core/src/delta_datafusion/table_provider/next/scan/mod.rs:261` | `spawn_blocking` for `FileSelection` resolution | run inline on wasm |
| `crates/core/src/delta_datafusion/table_provider/next/scan/replay.rs:149-168` | DV loading: `file.dv_info.has_vector()` check, then blocking task + `dv_stream` (`ReceiverStreamBuilder`, :81, :98) | **fail loud before the spawn** (v1: no DVs); see V4 for the field |

## Design

### `ExecutorHandle`

Location: `crates/core/src/kernel/executor.rs` (new; used by both `kernel/` glue
and `delta_datafusion/engine/`), re-exported where `TracedHandle` lives today.

```rust
#[derive(Clone, Debug)]
pub enum ExecutorHandle {
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    Tokio(TracedHandle),      // today's behavior: block_on + tracing-span re-entry
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    Inline(InlineExecutor),
}

impl ExecutorHandle {
    pub fn current() -> Self { /* native: Handle::current().into(); wasm: Inline */ }
    pub fn try_block_on<F: Future>(&self, fut: F) -> DeltaResult<F::Output> { ... }
}
```

- Shared API is `try_block_on` (wasm can fail); native maps infallibly.
- `InlineExecutor::try_block_on`: hand-rolled poll loop with a flag-setting waker
  (an `Arc<AtomicBool>`-backed `RawWaker`): poll → `Ready(v)` ⇒ return; `Pending`
  with flag set ⇒ clear flag, poll again (handles cooperative yields, e.g.
  `buffered` streams); `Pending` with flag unset ⇒
  `Err(Error::Generic("future would block on wasm — log data not primed?"))`.
  **Do not** use `futures::executor::block_on` (its parking behavior on
  wasm32-unknown-unknown is a deadlock, and we *want* would-block to be
  diagnosable). Guard against spin: cap iterations (e.g. 10_000) with the same
  error.
- **Make `InlineExecutor` compile natively too** (not cfg'd out of native builds,
  only out of `ExecutorHandle::current()`'s default) so V2's native test can force
  it: `DataFusionEngine::new(ctx, ExecutorHandle::Inline(..))`.
- `DataFusionEngine::new(ctx, executor: impl Into<ExecutorHandle>)`;
  `new_from_session`/`new_from_context` call `ExecutorHandle::current()`. Call
  sites (`next/mod.rs:645, 736`) unchanged.
- Genericize `BlockingStreamIterator`/`stream_future_to_iter` over
  `ExecutorHandle` (per-item `try_block_on`; iterator yields the error item then
  fuses).

### Inline blocking helper

Replace direct `spawn_blocking_in_span(...)` awaits with one helper (same file):

```rust
pub(crate) async fn run_blocking_in_span<F, R>(span: tracing::Span, f: F) -> DeltaResult<R>
where F: FnOnce() -> R + Send + 'static, R: Send + 'static
```

Native impl = current `spawn_blocking_in_span(span, f).await` (+ join-error
mapping); wasm impl = `Ok(span.in_scope(f))`. Update the five
`snapshot/mod.rs` call sites (they currently `.await` a `JoinHandle` — align
error handling once, here).

### Scan-metadata streaming on wasm

`ReceiverStreamBuilder` (`stream.rs`) stays native-only. On wasm,
`Scan::scan_metadata`-driving code returns
`futures::stream::iter(sync_kernel_iterator)` boxed — the kernel iterator is
lazy; each `next()` runs sync handler calls, which complete inline against primed
data. Note: each poll does synchronous work on the calling thread (JS event
loop); acceptable for previews, and D4 recommends running in a Web Worker.

### DV guard

In `replay.rs`, before the spawn: on wasm,
`if file.dv_info.has_vector() { return Err(DeltaTableError::Generic("deletion vectors are not supported on wasm (v1)")) }`
(exact error type per surrounding code). Whether `dv_stream`'s
`ReceiverStreamBuilder` field can stay merely-unused depends on **V4**: check
whether tokio `sync` types (`mpsc::channel`, `JoinSet::new`) *construct* on
wasm32-unknown-unknown at our tokio pin; if not, cfg the field and its plumbing.

## Interface contracts (out)

- `ExecutorHandle` + construction rule: *wasm engines only make progress on ready
  futures; priming is the caller's contract* (D4 consumes this).
- `DataFusionEngine::new_from_session(session)` works on wasm with no tokio
  runtime present (D4's provider path depends on it — `next/mod.rs:736`).
- `Snapshot::try_new_with_engine` + `scan_metadata` complete on wasm given a
  primed store (D4's `open_snapshot`).

## Validation gates

- V1 inventory done and resolved: the wasm `cargo check` above **passes**.
- Native suite unchanged: `cargo test -p deltalake-core --features datafusion`.
- **V2 (the keystone test, native)**: load a fixture table's files into an
  `object_store::memory::InMemory`, register it in a `RuntimeEnv`, build
  `DataFusionEngine` with a forced `InlineExecutor`, then
  `Snapshot::try_new_with_engine` + drive `scan_metadata` to completion; assert
  file list matches the tokio-engine result. Include a checkpointed fixture (so
  `DataSourceExec(ParquetSource)` + footer read run under the inline executor) and
  a **negative test**: a store missing one commit file yields the "would block /
  not primed"-class error, not a hang. This test is permanent — it is the native
  proof of the wasm execution model.
- fmt + clippy; `cargo build --workspace`.

## Risks

- Unknown breadth of V1 fallout (why this doc is sized for Fable).
- Hidden `Handle::current()`/`tokio::spawn` inside DF-feature code paths that only
  execute at runtime — V2 catches the read path; D4's V3 audits the data path.
- `Send` bounds: kernel `FileDataReadResultIterator` requires `Send + 'static`;
  wasm single-threaded futures are usually fine but JS-interop types are `!Send`
  — keep JS types out of `crates/core` entirely (they belong in D4).

## Done criteria

Wasm check green with `--features datafusion`; V2 tests merged and green
natively; DV fail-loud in place; status + any interface deviations recorded in
`WASM_ENGINE.md`.
