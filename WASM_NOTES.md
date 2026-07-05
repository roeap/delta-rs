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
- **Dropping zstd/brotli is a real read limitation, not free.** parquet returns a
  graceful error ("Disabled feature at compile time: zstd") on such pages — no
  panic/corruption. Kept codecs (snappy, gzip via pure-Rust zlib-rs, lz4) cover
  Spark/snappy-default tables, but **zstd-compressed tables become unreadable**,
  and zstd is increasingly common in modern Delta writers. Proper fix (follow-up):
  wire parquet's zstd *decompression* to a pure-Rust decoder (e.g. `ruzstd`) on
  wasm — a parquet-crate change — or target `wasm32-wasip1` where `zstd-sys`
  builds. brotli is rare in practice; low impact.
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

> **Done.** The items below were planned and executed in full — see
> [`WASM_ENGINE.md`](./WASM_ENGINE.md) for the decided architecture (single
> `DataFusionEngine` for native+wasm, executor seam, primed log store, opaque
> predicate bridge), the per-chunk handover docs `WASM_ENGINE_D1..D5`, and the
> commit refs each chunk landed on. This list is kept for historical context.

1. ~~Detailed planning for the wasm `Engine`~~ — `WASM_ENGINE.md` architecture +
   decision log; D1–D3.
2. ~~`deltalake-wasm` facade crate~~ — D4 (`crates/wasm`).
3. ~~Snapshot builder on wasm~~ — D4 (`Snapshot::try_new_with_engine`, bypassing
   `get_engine`).
4. ~~`wasm-bindgen` harness~~ — D4 (`WasmDeltaTable`; `wasm-pack test --node`
   smoke suite).
5. ~~CI hygiene~~ — D5: path/patch deps now pinned git refs; obsolete kernel-fork
   delta reverted; wasm CI matrix added (`.github/workflows/wasm.yml`).
