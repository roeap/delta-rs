# D1 — DataFusion-plan file-format & storage handlers (native)

> Part of the wasm-engine effort — read [`WASM_ENGINE.md`](./WASM_ENGINE.md) first.
>
> **Status: not started** · Depends on: — · Blocks: D4 (and D2 rebases its
> constructor change on this) · Parallel with: D2, D3 · Recommended model: **Opus**

## Goal

Make `DataFusionEngine`'s storage/JSON/parquet handlers pure DataFusion +
`object_store` — no `delta_kernel_default_engine` delegation — by merging the
existing unmerged branch `feat/more-df-engine` (commit `cd14f8f9`, "feat:
datafusion native kernel engine") into `wasm-core-compat`, and filling its one
read-path gap: `read_parquet_footer`. Everything in this chunk is **native** work;
D2 makes it target-portable. This is the engine the browser will run, so the
native test suite becomes the wasm engine's regression net.

## Current state (verified)

On `wasm-core-compat`, `crates/core/src/delta_datafusion/engine/`:

- `mod.rs:20-67` — `DataFusionEngine { storage, formats }`; constructors
  `new_from_session` (:29) / `new_from_context` (:36) / `new(ctx, Handle)` (:44);
  `evaluation_handler()` returns `ARROW_HANDLER` (kernel arrow evaluator — keep).
- `storage.rs` — `DataFusionStorageHandler` holds a `DashMap` registry of kernel
  `ObjectStoreStorageHandler`s from `delta_kernel_default_engine`, selected per
  store by `handle.runtime_flavor()` with a `panic!` fallback (:58-68).
  `copy_atomic`/`head` are error stubs (:118-121, :138-141) — but `head` IS
  exercised by kernel checkpoint code paths.
- `file_formats.rs` — `DataFusionFileFormatHandler` delegates to
  `DefaultParquetHandler`/`DefaultJsonHandler` (from `delta_kernel_default_engine`)
  per store, executor chosen by runtime flavor (:67-77, :95-105).
  `write_parquet_file` is `todo!()` (:154).
- The current branch added **tracing-span plumbing** the feat branch predates:
  constructor-captured `tracing::Span`s threaded through the handlers so kernel →
  engine callbacks nest under the scan span. **This must survive the merge.**

On `feat/more-df-engine` (same 4 files + `Cargo.toml`; shares merge-base
`0a09e1c6` with `wasm-core-compat`):

- `mod.rs` — adds `TracedHandle` (tokio `Handle` + span-instrumented `block_on`),
  `stream_future_to_iter` / `BlockingStreamIterator` (async `BoxStream` → sync
  `Iterator` by per-item `block_on`; documents how buffered streams give
  prefetch concurrency), and `UrlExt::is_presigned` (AWS/Azure/GCP/OSS query-param
  sniffing).
- `storage.rs` — full direct-`object_store` `StorageHandler`: `list_from`
  (`list_with_offset`, prefix/offset derivation, sorts only for `file://` since
  cloud listings are lexicographic), `read_files` (range GETs, `buffered(readahead)`,
  presigned-URL branch via `reqwest::get`), `copy_atomic` (GET + `PutMode::Create`),
  `head` — all bridged via `stream_future_to_iter`/`block_on`. Stores come from
  `ctx.runtime_env().object_store(store_url)`. Includes unit tests.
- `file_formats.rs` — the DataFusion-plan handlers:
  - `read_parquet_files` → `to_partitioned_files` → `parquet_exec(...)`:
    `ParquetSource::new(TableParquetOptions from session config)`
    `.with_parquet_file_reader_factory(CachedParquetFileReaderFactory::new(store,
    runtime_env().cache_manager.get_file_metadata_cache()))`, optional kernel
    predicate via `predicate_to_df` → `logical2physical` →
    `.with_predicate(..).with_pushdown_filters(true)`, one `FileGroup`,
    `FileScanConfigBuilder` → `DataSourceExec::from_data_source`.
  - `read_json_files` → `json_exec(...)`: `JsonSource::default()` +
    `FileScanConfigBuilder` → `DataSourceExec`.
  - `execute_iter(exec, ctx, handle)`: `execute_stream` → map `RecordBatch` →
    `ArrowEngineData` → `BlockingStreamIterator`.
  - `parse_json` → kernel `arrow_parse_json`; `write_json_file` → buffered
    `to_json_bytes` + `put_opts` under `block_on` (native commit path uses this).
  - `read_parquet_footer` and `write_parquet_file` are `todo!()`.
  - NB in `to_partitioned_files`: `PartitionedFile::new` mis-encodes paths; the
    code reassigns `object_meta.location` — preserve that fix and its comment.

Kernel trait/contract facts (local `../delta-kernel-rs`, `wasm-kernel-compat`):

- Handler traits: `StorageHandler` (`kernel/src/lib.rs:598`), `JsonHandler` (:643),
  `ParquetHandler` (:738). All sync. `read_*_files` return
  `FileDataReadResultIterator` (:217).
- **Ordering contract**: `read_json_files`/`read_parquet_files` must emit batches
  in input-file order without merging rows across file boundaries. A single
  `FileGroup` (= one partition, files read sequentially) satisfies this — do not
  split files across file groups/partitions.
- `read_parquet_footer` (`lib.rs:962`, returns `ParquetFooter { schema }`,
  :729) is called by log replay for parquet checkpoints when `_last_checkpoint`
  lacks a schema hint, and for v2 sidecars
  (`kernel/src/log_segment/mod.rs:870-871, 887`).
- Reference impl to mirror: `default-engine/src/parquet.rs:374-402` —
  `ParquetObjectReader` + `ArrowReaderMetadata::load_async` with
  `ArrowReaderOptions::new().with_skip_arrow_metadata(true)` (kernel's own
  `reader_options()` is `pub(crate)`; inline the option, don't chase visibility).

## Implementation

1. **Merge** `feat/more-df-engine` into `wasm-core-compat`
   (`git merge feat/more-df-engine` and resolve, or replay `cd14f8f9`). This is a
   reconciliation, not a cherry-pick: adopt the feat-branch handler bodies, keep
   the current branch's span-capture/instrumentation behavior (re-express it via
   `TracedHandle`'s span-instrumented `block_on` + `#[instrument]` attributes the
   feat branch already has; verify kernel-callback spans still nest by running a
   traced test).
2. **Remove `delta_kernel_default_engine` imports** from `engine/storage.rs` and
   `engine/file_formats.rs` (the merge does this); the crate remains a
   *native-only* dependency of `crates/core` solely for `logstore::get_engine`
   (`crates/core/src/logstore/mod.rs:597-613`) — do not touch that seam here.
3. **Implement `read_parquet_footer`**:

   ```rust
   fn read_parquet_footer(&self, file: &FileMeta) -> DeltaResult<ParquetFooter> {
       let store = self.ctx.runtime_env()
           .object_store(file.location.as_object_store_url())
           .map_err(Error::generic_err)?;
       let path = Path::from_url_path(file.location.path())?;
       let size = file.size;
       let handle = self.handle.clone();
       let metadata = handle.block_on(async move {
           let mut reader = ParquetObjectReader::new(store, path).with_file_size(size);
           ArrowReaderMetadata::load_async(
               &mut reader,
               ArrowReaderOptions::new().with_skip_arrow_metadata(true),
           ).await
       })?;
       Ok(ParquetFooter {
           schema: Arc::new(metadata.schema().as_ref().try_into_kernel()?),
       })
   }
   ```

   (Adjust conversion trait names to what `delta_kernel::engine::arrow_conversion`
   exposes: `TryIntoKernel`/`TryFromArrow`.) A presigned-URL branch mirroring the
   default engine is optional; on wasm presigned URLs are served by the registered
   store anyway.
4. `write_parquet_file` stays `todo!()` (read-only scope; not on any v1 path).
5. Keep `TracedHandle` as the executor type for now — **D2 replaces it with
   `ExecutorHandle`**; if D2 has already landed, wire the new type instead (the
   merge conflict will tell you).

## Interface contracts (out)

- `DataFusionEngine::new(ctx: Arc<TaskContext>, handle: impl Into<TracedHandle>)`
  — D2 changes `TracedHandle` → `ExecutorHandle`; keep the constructor shape.
- `stream_future_to_iter` / `BlockingStreamIterator` are the single sync-bridge
  primitives; all handler IO must go through them (D2 genericizes exactly these).
- Handlers resolve stores exclusively via `ctx.runtime_env().object_store(url)` —
  this is what lets D4 inject the primed/fetch stores without engine changes.
- `read_parquet_footer` returns the file's arrow-derived kernel schema with arrow
  metadata skipped (matches default-engine semantics).

## Validation gates

- `cargo test -p deltalake-core --features datafusion` — **full native suite**; this
  is the regression protection for replacing the default-engine delegation
  (log replay, checkpoint reads, commit writes via `write_json_file`, DAT tests).
- New tests:
  - footer: `read_parquet_footer` against a checkpoint parquet fixture
    (`crates/test` has table fixtures), schema equals the default-engine result;
  - **V6**: snapshot build via `Snapshot::try_new_with_engine(DataFusionEngine…)`
    of a checkpointed table whose `_last_checkpoint` has no schema hint (create the
    fixture by rewriting/stripping `_last_checkpoint` if none exists) — forces the
    footer path through kernel log replay;
  - predicate pushdown: `read_parquet_files` with a kernel predicate returns a
    subset consistent with an unpredicated read;
  - multi-file ordering: `read_json_files` over ≥3 commit files preserves input
    order (guards the single-`FileGroup` contract).
- `cargo build --workspace` (python crate included) stays green; fmt + clippy.

## Risks

- **Behavior deltas vs default-engine readers**: JSON parse options (the
  default engine sets nonstandard-JSON tolerances?), batch sizing (default engine
  uses batch/buffer 1000; DF uses session `batch_size`) — differences are
  acceptable if the suite passes, but investigate any log-replay test diff before
  "fixing" a test.
- Path encoding edge cases (spaces, `=`, unicode in partition dirs) — covered by
  existing DAT/partition tests; the `to_partitioned_files` NB exists because of
  this.
- `list_from` semantics: kernel requires strictly-greater-than-offset,
  parent-scoped, sorted listing — the feat impl derives prefix from the offset
  path; keep its tests and add one for a non-directory offset if missing.

## Done criteria

All validation gates green on `wasm-core-compat`; `delta_kernel_default_engine`
no longer imported anywhere under `crates/core/src/delta_datafusion/`; status
updated in `WASM_ENGINE.md`.
