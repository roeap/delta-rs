# WASM engine — implementation roadmap & handover index

> **Status: planned, ready to execute.** This is the index for the wasm-engine work
> ("Phase A" in `../mangrove/WASM_QUERY_PREVIEW.md`): make delta-rs *run* (not just
> compile) on `wasm32-unknown-unknown` by building a wasm-compatible delta-kernel
> `Engine` and a `deltalake-wasm` facade crate. Background: [`WASM.md`](./WASM.md)
> (build status/gating), [`WASM_NOTES.md`](./WASM_NOTES.md) (spike learnings).
>
> The work is split into five handover documents, each executable by an independent
> session. Read this file first; then read only your assigned `WASM_ENGINE_D*.md`.

## Goal

A read-only DataFusion query path over Delta tables in the browser: fetch-backed
storage, kernel snapshot + log replay, the `next/` `TableProvider`, Arrow IPC out.
v1 limits (locked): read-only; **no deletion vectors** (fail loud); **no
zstd/brotli parquet pages** (graceful error); fully-qualified URLs only.

## Architecture (decided)

**One engine for native and wasm.** We evolve the existing `DataFusionEngine`
(`crates/core/src/delta_datafusion/engine/`) instead of writing a separate wasm
engine. The `deltalake-wasm` crate contains only wasm glue (fetch store, log
priming, wasm-bindgen surface). Consequence: every native test of the DataFusion
provider exercises the same engine code the browser runs, and the DataFusion path
drops its `delta_kernel_default_engine` dependency natively.

The four kernel `Engine` handlers (kernel traits are **synchronous**; that fact
drives the whole design):

| Handler | Implementation | Doc |
|---|---|---|
| `EvaluationHandler` | kernel `ARROW_HANDLER` (`ArrowEvaluationHandler`) — already compiles on wasm today | (none — done) |
| `StorageHandler` | direct `object_store` ops from the session's `RuntimeEnv` registry, via sync-bridge | D1 |
| `JsonHandler` | DataFusion plan: `JsonSource` → `FileScanConfig` → `DataSourceExec`, driven synchronously | D1 |
| `ParquetHandler` | DataFusion plan: `ParquetSource` (+ predicate pushdown, cached reader factory) → `DataSourceExec`; `read_parquet_footer` via `ParquetObjectReader` | D1 |

**The sync/async crux and its resolution.** Kernel handler traits are sync; browser
IO is async; wasm cannot block on the JS event loop. Resolution by decomposition:

- Kernel-driven *sync* IO is confined to the `_delta_log` **metadata path** (list,
  commit JSONs, checkpoint parquet + footers). The **table data** parquet is read by
  the `next/` provider's own async `DataSourceExec`, drivable end-to-end by
  `wasm-bindgen-futures` — no blocking needed there.
- `deltalake-wasm` therefore **primes** the log: an async `prime()` prefetches the
  `_delta_log` tail (`_last_checkpoint`, checkpoint parts, sidecars, commit JSONs)
  into an in-memory store *before* any sync kernel call. After priming, every future
  the sync engine drives is immediately ready, and a trivial inline poll executor
  (`InlineExecutor`, D2) completes it without a runtime. A would-block future is a
  *bug surfaced as an error* ("data not primed"), never a deadlock.

**Executor seam.** `ExecutorHandle` (D2) replaces the tokio `Handle`/`TracedHandle`
inside the engine: native = tokio block-on with span propagation (today's
behavior); wasm = `InlineExecutor`. `DataFusionEngine::new_from_session` — the
"ad-hoc engine from the active session" pattern — is kept; only its executor
acquisition becomes target-dependent.

**Snapshot entry on wasm** bypasses `logstore::get_engine` (which stays an
unreachable `unimplemented!()` stub): the facade calls the public
`Snapshot::try_new_with_engine` (`crates/core/src/kernel/snapshot/mod.rs:199`) and
`DeltaScan::new(snapshot, config)` (`.../table_provider/next/mod.rs:518`; a
`LogStore` is only required for `insert_into`, not reads).

## Decision log

1. **EvaluationHandler: kernel arrow evaluator, not a DataFusion-expr evaluator.**
   Verified: `ARROW_HANDLER` (`crates/core/src/kernel/mod.rs:29`) already compiles
   in the passing wasm build (the workspace enables the kernel's
   `default-engine-base` feature; kernel's `reqwest` dep is
   `default-features=false`). A DataFusion-physical-expr `EvaluationHandler` would
   have to reimplement the kernel-internal transform expressions (`ParseJson`,
   `MapToStruct`, `StructPatch`, `Array`) **and** the tri-state data-skipping
   semantics — a large, correctness-critical surface where silent divergence
   corrupts file skipping — while buying zero wasm enablement. The choice stays
   swappable behind `Engine::evaluation_handler()`. See "Option-B backlog" below.
2. **DataFusion expressions still get first-class treatment via opaque predicates**
   (D3): DataFusion filter expressions the kernel model can't represent (UDFs,
   `LIKE`, `CASE`, …) are wrapped as kernel `Predicate::Opaque` ops that evaluate
   the original DF `Expr` — extending partition pruning to arbitrary DF predicates
   instead of dropping them. Conservative v1: opaque ops never contribute
   stats-based skipping (returns "don't know"), so they can never wrongly prune.
3. **File-format handlers = DataFusion execution plans.** The unmerged branch
   `feat/more-df-engine` (commit `cd14f8f9`, "feat: datafusion native kernel
   engine") already implements this; D1 merges it forward rather than rewriting.
4. **Ad-hoc engine from session confirmed** as the pattern: `TaskContext` carries
   the object-store registry, config, and caches; engine construction is cheap and
   inherits session state. Only `Handle::current()` needed replacing.
5. **Priming over blocking tricks.** Alternatives considered for sync IO on wasm —
   sync XHR in a worker; `SharedArrayBuffer` + `Atomics.wait` proxy (needs
   cross-origin isolation) — rejected for v1 as heavier and host-environment
   sensitive. Priming is deterministic, testable natively, and errors loudly.

## Work chunks

```
D1 (DF-plan handlers, native) ──┐
D2 (executor seam + wasm cfg) ──┼──► D4 (deltalake-wasm facade + e2e) ──► D5 (CI / fork hygiene)
D3 (opaque bridge, native)  ────┘
        D1 ‖ D2 ‖ D3 can run in parallel; D3 is independently landable.
```

| Doc | Scope | Depends on | Model | Status |
|---|---|---|---|---|
| [`WASM_ENGINE_D1_HANDLERS.md`](./WASM_ENGINE_D1_HANDLERS.md) | Merge `feat/more-df-engine` DF-plan handlers; implement `read_parquet_footer`; drop `delta_kernel_default_engine` from the DF path | — | **Opus** (3-way merge + behavior-parity risk vs default-engine readers) | **done — f7dadcae** |
| [`WASM_ENGINE_D2_EXECUTOR.md`](./WASM_ENGINE_D2_EXECUTOR.md) | `ExecutorHandle`/`InlineExecutor`; neutralize tokio bridges; first wasm compile of the `datafusion` feature | — (rebases on D1's constructor if D1 lands first) | **Fable** (highest uncertainty: cfg surgery, waker/poll semantics) | not started |
| [`WASM_ENGINE_D3_OPAQUE.md`](./WASM_ENGINE_D3_OPAQUE.md) | `DataFusionOpaquePredicateOp`; wire `to_kernel`/`to_datafusion` catch-alls; pruning tests | — | **Opus** (correctness-sensitive seam, well-scoped after V5 spike) | not started |
| [`WASM_ENGINE_D4_FACADE.md`](./WASM_ENGINE_D4_FACADE.md) | `deltalake-wasm` crate: fetch store, `PrimedStore`, snapshot/query API, wasm-bindgen, smoke tests | D1 + D2 | **Fable** (new crate, wasm tooling unknowns, e2e) | not started |
| [`WASM_ENGINE_D5_CI.md`](./WASM_ENGINE_D5_CI.md) | Path deps → git refs; revert obsolete fork deltas; CI matrix; fold docs | D1–D4 | **Sonnet** (mechanical) | not started |

Model recommendations assume Claude Code sessions; they track implementation
complexity/uncertainty, not doc length.

## Front-loaded validations

Each could bend the design — run them **before** deep implementation in the owning
doc. Results should be recorded in the owning doc's status section.

| # | Validation | Owner |
|---|---|---|
| V1 | `cargo check -p deltalake-core --no-default-features --features datafusion --target wasm32-unknown-unknown` — the `datafusion` feature has **never** been wasm-compiled; the error inventory defines D2's true scope | D2 (first action) |
| V2 | Native test: snapshot build + `scan_metadata` against a fully-primed `InMemory` store using a forced `InlineExecutor` — proves "all handler futures are ready after priming" for `list_with_offset`, `head`, `DataSourceExec(JsonSource)`, `DataSourceExec(ParquetSource)`, `ArrowReaderMetadata::load_async` | D2 |
| V3 | `DataSourceExec` *data* stream drivable on wasm without tokio (`target_partitions=1`; audit plans for `RepartitionExec`/spawned tasks) | D4 |
| V4 | tokio `sync`-feature types (`mpsc::channel`, `JoinSet::new`) constructible on wasm32-unknown-unknown (decides cfg-out vs never-spawn for `ReceiverStreamBuilder`/`dv_stream`) | D2 |
| V5 | Opaque-op downcast recovery for the `predicate_to_df` round-trip. **Pre-verified during planning:** the kernel's `ArrowOpaquePredicateOpAdaptor` is `pub(crate)` — recovery needs a small kernel-fork patch or the out-of-band-`Expr` fallback (both designed in D3) | D3 (first action) |
| V6 | A parquet-checkpoint fixture **without** a `_last_checkpoint` schema hint exists/can be made, so the `read_parquet_footer` path is actually exercised | D1 |
| V7 | wasm Range GETs via the fetch store against real endpoints; V2-checkpoint sidecar (`_delta_log/_sidecars/`) priming coverage | D4 |

## Option-B backlog (full DataFusion EvaluationHandler)

Not scheduled; recorded so a future unification effort doesn't re-derive scope.
Replacing `ARROW_HANDLER` with a DF-physical-expr evaluator requires:

- New `to_datafusion.rs` arms: `Expression::StructPatch`, `ParseJson` (UDF),
  `MapToStruct` (UDF implementing Delta partition-value decode rules),
  `VariadicExpressionOp::Array`, `BinaryPredicateOp::In`, `Scalar::Array/Map`,
  and defined behavior for `Unknown` (must hard-error in filter position).
- `EvaluationHandler::{new_expression_evaluator, new_predicate_evaluator, null_row,
  create_many}` over `RecordBatch` with kernel's exact null/tri-state semantics —
  the log-replay transforms (`delta-kernel-rs/kernel/src/scan/log_replay.rs:301,314`)
  and `DataSkippingFilter` depend on them precisely.
- Differential testing against `ARROW_HANDLER` on the DAT suite before switching.

## Session protocol

- Before starting a chunk, set its Status to `in progress — <date>` in the table
  above (and in the chunk doc header); on completion set `done — <commit>` and note
  any interface deviations in the chunk doc so dependent chunks see them.
- Run your chunk's front-loaded validations first; if one invalidates the design,
  **stop and update this index** rather than improvising around it.
- Branch discipline: work lands on `wasm-core-compat` (delta-rs). Kernel-fork
  changes (if any) go to `../delta-kernel-rs` branch `wasm-kernel-compat`.
- Native regression bar for every chunk:
  `cargo build --workspace` and `cargo test -p deltalake-core --features datafusion`
  stay green; plus `cargo fmt --all` / `cargo clippy --all-targets --all-features -- -D warnings`.
