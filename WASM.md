# WebAssembly (`wasm32-unknown-unknown`) support — status

This is an in-progress spike to make `deltalake-core` build and (eventually) run
on `wasm32-unknown-unknown` (browser / no-OS wasm), targeting a read-only
DataFusion `TableProvider` for querying Delta tables from a wasm host.

## Current status

- Delta tables are **queryable on wasm**: the `deltalake-wasm` facade crate
  (`crates/wasm`) primes the `_delta_log` into memory, builds a kernel snapshot on
  the inline executor, and runs read-only DataFusion SQL over the `next/`
  `TableProvider`, streaming Arrow IPC out. Proven by `wasm-pack test --node`
  (see "Running on wasm" below). v1 limits: read-only; no deletion vectors (loud
  error); no zstd/brotli parquet pages (graceful error); fully-qualified URLs.
- `deltalake-core` compiles for `wasm32-unknown-unknown` both with
  `--no-default-features` and with `--features datafusion`, and the **whole native
  workspace** (`cargo build --workspace`, incl. `python`) builds against the
  pinned upstream kernel.
- `logstore::get_engine` remains an `unimplemented!()` stub on wasm by design —
  the facade enters via `Snapshot::try_new_with_engine` and never constructs a
  `LogStore`.
- `nanosecond-timestamps` is **disabled in the Python crate's default features** for the
  spike (`python/Cargo.toml`). The feature is backed by kernel symbols
  (`PrimitiveType::TimestampNanos`, `Scalar::TimestampNanos`, `TableFeature::TimestampNanos`,
  `DataType::TIMESTAMP_NANOS`) that live only in the buoyant-data kernel fork, not the
  pinned upstream `delta_kernel` v0.25.0. The delta-rs gating is complete and correct; the
  feature just can't be satisfied by the pinned kernel, so leaving it on default-broke the
  workspace build (Cargo unions features graph-wide). Re-add it once the kernel dep is
  reconciled.

Build it with:

```sh
cargo build -p deltalake-core --no-default-features --lib --target wasm32-unknown-unknown
```

## Running on wasm

```sh
# CI-style gate for the facade crate (build, not just check: 32-bit usize
# overflows surface only at build time)
cargo build -p deltalake-wasm --target wasm32-unknown-unknown

# wasm smoke suite under node (embedded fixtures, no network)
cd crates/wasm && wasm-pack test --node

# optionally include the FetchObjectStore HTTP test (Range GETs against a
# local range-capable server)
node tests/http-server.mjs ../test/tests/data 8917 &
WASM_SMOKE_HTTP_BASE=http://127.0.0.1:8917 wasm-pack test --node
```

The wasm-bindgen surface (`WasmDeltaTable`: `open` / `query` / `schemaJson` /
`version`) is a proof harness; hosts build on the rlib API
(`open_table_with_store`, `register_snapshot`, `query_ipc` — see the crate's
rustdoc). Host snapshot/scan work in a Web Worker: the inline-executor bursts
are synchronous. Details and deviations: `WASM_ENGINE_D4_FACADE.md`.

## Dependency wiring (spike-only)

The build currently depends on local checkouts via path/patch in the workspace
`Cargo.toml`. These are stopgaps for the spike; CI would use git refs instead.

- `delta_kernel` → local `../delta-kernel-rs` (branch `wasm-kernel-compat`, v0.25.0).
- `delta_kernel_default_engine` → local `../delta-kernel-rs/default-engine`
  (native-only; the tokio-based default engine).
- `[patch.crates-io]` for `parquet` + the `arrow-*` family → local `../arrow-rs`
  (branch `wasm-codec-58.3.0`). The patch changes parquet's default features
  to drop the C-backed `zstd`/`brotli` codecs, which cannot build for wasm. This
  must be a patch/fork: `datafusion-datasource-parquet` and the kernel declare
  `parquet` without `default-features = false`, and Cargo unions features
  graph-wide, so no feature flag in delta-rs can turn `zstd` off.
  Since D2 the fork also target-gates `arrow-ipc`'s `zstd` dependency:
  `datafusion-common` enables `arrow-ipc/zstd` unconditionally, so on wasm the
  feature is inert (IPC zstd (de)compression returns the codec's graceful error).
- `.cargo/config.toml` sets `getrandom_backend="wasm_js"` for the wasm target
  (getrandom 0.3). getrandom 0.4 uses a crate feature instead — see
  `crates/core/Cargo.toml`.

## What builds on wasm vs. what is native-only

The wasm build drops the native async runtime and local-filesystem surface;
these are `cfg`-gated to `cfg(not(all(target_arch = "wasm32", target_os = "unknown")))`:

- The dedicated IO runtime (`DeltaIOStorageBackend`, `IORuntime`, tokio
  multi-thread) in `logstore/storage/runtime.rs`, and its threading through
  `StorageConfig` / `RawDeltaTableBuilder`.
- `tokio` native features (`rt-multi-thread`, `process`, `signal`, `fs`),
  `num_cpus`, `dirs`, and PEM-certificate client options.
- Local-filesystem path handling in `table/builder.rs` (`file://`, `~`,
  `Url::{to,from}_file_path`) — on wasm, pass fully-qualified URLs.
- The tokio-based kernel `DefaultEngine`. On wasm the host must supply its own
  `Engine`.

## Also required for the spike (not in this repo)

- `delta-kernel-rs` (`wasm-kernel-compat`): `kernel/Cargo.toml` makes the
  `object_store` cloud features (which pull `ring`/`hyper`) native-only. Also
  `default-engine/src/filesystem.rs`: `ObjectStoreStorageHandler::new` is made `pub`
  (was `pub(crate)` after the v0.25.0 relocation) so delta-rs's DataFusion engine can
  construct it directly, as it did pre-relocation.
- `arrow-rs` (`wasm-codec-58.3.0`): the parquet codec-defaults change and the
  arrow-ipc zstd target-gating above.

Native code that referenced the pre-v0.25.0 `delta_kernel::engine::default::*` paths
(`crates/core/src/delta_datafusion/engine/{file_formats,storage}.rs`) now imports the
relocated types from the `delta_kernel_default_engine` crate.

## Next steps

The wasm `Engine` + `deltalake-wasm` work is planned in detail in
[`WASM_ENGINE.md`](./WASM_ENGINE.md) (architecture, decision log, and five
executable handover documents `WASM_ENGINE_D1..D5`). Spike-era decisions and
learnings remain in [`WASM_NOTES.md`](./WASM_NOTES.md).
