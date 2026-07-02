# WASM spike — decisions, learnings & next steps

Session handoff. For build instructions and the native-vs-wasm gating list, see
[`WASM.md`](./WASM.md). This file captures *why* and *what's next* so the
follow-up doesn't re-discover them.

## Where things stand

`deltalake-core` compiles for `wasm32-unknown-unknown` (native unaffected). It
does **not run** yet — `logstore::get_engine` is an `unimplemented!()` stub on
wasm. Committed across three repos on branches (not pushed):
`delta-rs` → `wasm-core-compat`, `delta-kernel-rs` → `wasm-kernel-compat`,
`arrow-rs` → `wasm-codec-58.3.0`.

## Key decisions

- **Isolated `TableProvider`, not a full core port.** The goal is a read-only
  DataFusion `TableProvider` in a new `deltalake-wasm` crate, reusing patterns
  from `crates/core/src/delta_datafusion/table_provider/next/` — not making all
  of core (writes/optimize/vacuum) wasm-clean. That subsystem is the highest-value
  isolated unblock.
- **Target `wasm32-unknown-unknown`** (browser), read-only first, **no deletion
  vectors** in v1 (fail loud on tables that have them).
- **Parquet codec fix via patch/fork, not features.** See learnings.
- **Local path/patch deps for the spike; git refs for CI later.**

## Learnings (the non-obvious bits)

- **arrow / parquet / DataFusion all DO build for wasm.** The only hard blocker
  was parquet's default `zstd`/`brotli` (C libs). Everything else is pure Rust.
- **The zstd/brotli fix must be a `[patch]`/fork of parquet**, not a feature
  flag: `datafusion-datasource-parquet` and the kernel declare `parquet` without
  `default-features = false`, and Cargo unions features graph-wide, so nothing
  downstream can turn `zstd` off.
- **The kernel's wasm-capable core is engine-agnostic and I/O-free** — the host
  supplies the `Engine`. delta-rs core is inseparable from the *arrow* engine, so
  wasm needs a real arrow-based `Engine` (the load-bearing next step), not just
  the kernel building.
- **kernel v0.25.0 relocated the `DefaultEngine`** into a separate,
  tokio-hard, native-only crate (`delta_kernel_default_engine`). Its `TaskExecutor`
  is a generic trait but only tokio impls ship — no wasm executor exists.
- **The provider reads parquet via DataFusion's own `ParquetSource`/`DataSourceExec`**,
  not the kernel's `ParquetHandler`. Kernel is used for scan metadata / file
  skipping / partition transforms (all synchronous, reusable). DVs are the only
  storage-touching kernel call in the scan path (dropped for v1).
- **The real tokio surface is the sync-iterator→`Stream` bridge**
  (`ReceiverStreamBuilder`, mpsc + `JoinSet`), not just `spawn_blocking`. On wasm,
  drive the kernel's synchronous iterators directly.
- **Validate wasm builds in the delta-rs workspace, not the kernel repo** — the
  kernel repo's own dev-deps pull mixed arrow 57/58, so the parquet patch (58.3.0)
  doesn't apply there and results mislead.
- getrandom needs two mechanisms: 0.3 via `getrandom_backend="wasm_js"` rustflag,
  0.4 via crate feature.

## Next steps

1. **Detailed planning for the wasm `Engine`** (the crux). It must implement the
   kernel `Engine` trait with: an arrow evaluation handler, a fetch-backed storage
   handler, a JSON handler, a parquet handler using DataFusion's reader, and a
   non-tokio (inline / `wasm-bindgen-futures`) executor. Decide: build it in
   `deltalake-wasm`, or add a wasm executor + wasm build to
   `delta_kernel_default_engine`.
2. **`deltalake-wasm` facade crate**: the `TableProvider` + `DeltaScanExec`
   equivalent (mostly portable arrow/DataFusion logic), DV fail-loud guard at the
   `file.dv_info.has_vector()` check (`.../next/scan/replay.rs`).
3. **Snapshot builder on wasm** from a fetch-backed `ObjectStore`, bypassing
   `get_engine`.
4. **`wasm-bindgen` harness** reading a `memory://`/`https://` table end-to-end —
   proves it runs, not just compiles.
5. **CI hygiene**: move local path/patch deps to git refs; land the parquet codec
   change and kernel object_store gating upstream (or in stable forks).
