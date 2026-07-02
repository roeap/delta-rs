# WebAssembly (`wasm32-unknown-unknown`) support — status

This is an in-progress spike to make `deltalake-core` build and (eventually) run
on `wasm32-unknown-unknown` (browser / no-OS wasm), targeting a read-only
DataFusion `TableProvider` for querying Delta tables from a wasm host.

## Current status

- `deltalake-core` **compiles** for `wasm32-unknown-unknown` (`--no-default-features`)
  and the native build is unaffected.
- It does **not run** on wasm yet: `logstore::get_engine` is an `unimplemented!()`
  stub on wasm. A wasm-compatible kernel `Engine` and the `deltalake-wasm` facade
  crate are the next step.

Build it with:

```sh
cargo build -p deltalake-core --no-default-features --lib --target wasm32-unknown-unknown
```

## Dependency wiring (spike-only)

The build currently depends on local checkouts via path/patch in the workspace
`Cargo.toml`. These are stopgaps for the spike; CI would use git refs instead.

- `delta_kernel` → local `../delta-kernel-rs` (branch `wasm-kernel-compat`, v0.25.0).
- `delta_kernel_default_engine` → local `../delta-kernel-rs/default-engine`
  (native-only; the tokio-based default engine).
- `[patch.crates-io]` for `parquet` + the `arrow-*` family → local `../arrow-rs`
  (branch `wasm-codec-58.3.0`). The patch only changes parquet's default features
  to drop the C-backed `zstd`/`brotli` codecs, which cannot build for wasm. This
  must be a patch/fork: `datafusion-datasource-parquet` and the kernel declare
  `parquet` without `default-features = false`, and Cargo unions features
  graph-wide, so no feature flag in delta-rs can turn `zstd` off.
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
  `object_store` cloud features (which pull `ring`/`hyper`) native-only.
- `arrow-rs` (`wasm-codec-58.3.0`): the parquet codec-defaults change above.

## Next steps

1. `deltalake-wasm` facade crate with a real wasm `Engine` (replacing the
   `get_engine` stub) and a read-only `TableProvider` adapted from
   `delta_datafusion::table_provider::next`.
2. Deletion vectors are out of scope for the first iteration (fail loud on tables
   that contain them).
3. A `wasm-bindgen` harness that reads a table over a fetch-backed object store.
