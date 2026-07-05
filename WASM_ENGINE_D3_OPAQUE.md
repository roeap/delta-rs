# D3 — Opaque predicate bridge (DataFusion exprs through the kernel seam)

> Part of the wasm-engine effort — read [`WASM_ENGINE.md`](./WASM_ENGINE.md) first.
>
> **Status: done (unsigned local, branch `wasm-engine-d2-executor`)** · Depends
> on: — · Blocks: nothing (independently landable; D4 benefits) · Parallel with:
> D1, D2 · Recommended model: **Opus**
>
> **Required a kernel-fork patch** (recorded in `WASM_ENGINE.md` → "Kernel-fork
> deltas"): `ArrowOpaquePredicateOpAdaptor` made `pub` + `pub fn op()` accessor.
> See "Deviations from the plan" below for two design corrections found during
> implementation.

## Goal

Today, DataFusion filter expressions the kernel model can't represent (UDFs,
`LIKE`, `CASE`, regex, most scalar functions…) die in `to_kernel.rs`'s catch-all
and are simply not pushed into the kernel scan — no partition pruning, no
file skipping from them (results stay correct; DataFusion re-applies filters
post-scan). Kernel v0.25 provides the escape hatch: **opaque predicates** —
engine-defined ops embedded in kernel predicate trees that the kernel calls back
through well-defined traits. This chunk wraps untranslatable DataFusion `Expr`s
as opaque predicates so partition pruning works for arbitrary DF expressions,
with a conservative "never wrongly prune" contract. Native-only work; it rides
into wasm for free via D1/D2.

## Kernel facts (verified, local `../delta-kernel-rs`, branch `wasm-kernel-compat`)

All in `kernel/src/expressions/mod.rs` unless noted:

- `Expression::Opaque(OpaqueExpression)` (:401), `Predicate::Opaque(OpaquePredicate)`
  (:444), plus `Unknown(String)` variants (:410, :453 — kernel treats Unknown as
  "no skipping"; engines must NOT evaluate it). Serde intentionally fails for
  opaque (:399-400).
- `OpaquePredicateOp` trait (:158): `name()`, `eval_pred_scalar(eval_expr,
  eval_pred, exprs, inverted) -> DeltaResult<Option<bool>>` (:174) — powers
  **partition pruning**; `eval_as_data_skipping_predicate(...) -> Option<bool>`
  (:191) and `as_data_skipping_predicate(...) -> Option<Predicate>` (:210) —
  power stats-based skipping. Requires `DynPartialEq + Debug`.
  `ScalarExpressionEvaluator<'a> = dyn Fn(&Expression) -> Option<Scalar>` (:133).
- Constructors: `Predicate::opaque(op, exprs)` (:891); `OpaquePredicate { op:
  Arc<dyn OpaquePredicateOp>, exprs: Vec<Expression> }` (:298).
- **Arrow-engine extension layer** `kernel/src/engine/arrow_expression/opaque.rs`
  (compiles in our build — same gate as `ARROW_HANDLER`):
  `ArrowOpaquePredicateOp` (:47) adds columnar eval over `RecordBatch`;
  register via `Predicate::arrow_opaque(op, exprs)` (:114) which wraps the op in
  `ArrowOpaquePredicateOpAdaptor` (:172) implementing the kernel trait; the
  kernel arrow evaluator recovers it by downcast
  (`engine/arrow_expression/evaluate_expression.rs:741`; expression side :344).
  Since our `Engine::evaluation_handler()` is the kernel `ARROW_HANDLER`, opaque
  predicates embedded this way are evaluated for us wherever the kernel needs
  them — we implement the op, not the evaluator plumbing.

delta-rs seams (`crates/core/src/delta_datafusion/engine/expressions/`):

- `to_kernel.rs` — `to_delta_predicate` (:19) / `to_delta_expression` (:77);
  catch-alls to convert: unsupported `ScalarFunction` (:203-206), the general
  `_ => plan_err!` (:208), unsupported binary operators (~:299, ~:309).
- `to_datafusion.rs` — `predicate_to_df` currently rejects
  `Predicate::Opaque`/`Unknown` (:179-180). This is also the parquet-pushdown
  path (D1's `parquet_exec` calls `predicate_to_df` on the kernel predicate), so
  the round-trip matters.
- Consumer: `next/scan/plan.rs` `process_filters`/`process_predicate`
  (:524-563, :571+) calls `to_delta_predicate` (:599) to build the kernel scan
  predicate.

## First action (V5 — downcast recovery; partially pre-verified)

**Verified during planning:** `ArrowOpaquePredicateOpAdaptor` is `pub(crate)`
(`opaque.rs:172`) — delta-rs cannot name it, so downcasting `Predicate::Opaque`'s
op back to our concrete type is **not possible against the stock kernel**. Two
resolutions, in preference order:

1. **Small kernel-fork patch** (we own `wasm-kernel-compat`): make the adaptor
   type `pub` and add an accessor to the wrapped `dyn ArrowOpaquePredicateOp`
   (e.g. `pub fn op(&self) -> &dyn ArrowOpaquePredicateOp`). Recovery then is
   `pred_op.any_ref().downcast_ref::<ArrowOpaquePredicateOpAdaptor>()` →
   `.op().any_ref().downcast_ref::<DataFusionOpaquePredicateOp>()`. Record the
   patch in `WASM_ENGINE.md` status so D5 keeps it when reconciling fork deltas
   (and consider upstreaming — kernel TODO #1564 already tracks opaque serde
   gaps).
2. If patching is undesirable: skip round-trip recovery and instead keep the
   original DF `Expr` available out-of-band — `process_predicate`
   (`next/scan/plan.rs`) already holds the source filter when it builds both the
   kernel predicate and the parquet predicate, so the parquet-pushdown path can
   use the original `Expr` directly and `predicate_to_df` keeps rejecting
   `Opaque` (losing only pushdown for opaque predicates that arrive from other
   producers — acceptable v1).

Note: registering via plain `Predicate::opaque(op, …)` (`expressions/mod.rs:891`)
without the arrow adaptor is NOT an option while `ARROW_HANDLER` is the
evaluation handler — its evaluator only recognizes adaptor-wrapped ops
(`evaluate_expression.rs:741`) and errors on others. Always construct via
`Predicate::arrow_opaque` (`opaque.rs:114`).

First action remains a spike test proving the chosen path end-to-end before the
main implementation. Do not proceed on guesses.

## Design

New file `crates/core/src/delta_datafusion/engine/expressions/opaque.rs`:

```rust
#[derive(Debug)]
pub(crate) struct DataFusionOpaquePredicateOp {
    /// DF logical expr; references input columns by name.
    expr: Expr,
}
// PartialEq via Expr's PartialEq (satisfies DynPartialEq); name() = display of expr.
```

implementing `ArrowOpaquePredicateOp`:

- **Columnar** `eval_pred(exprs, batch, inverted)`: plan
  `logical2physical(&self.expr, batch.schema())`, evaluate → `BooleanArray`,
  apply `inverted` via `compute::not`. Contract: the op is only constructed for
  exprs whose column references are plain names resolvable against the batch
  the kernel hands back (the `exprs` args carry the kernel `Expression::Column`
  refs for the same columns).
- **Scalar** `eval_pred_scalar(eval_expr, _eval_pred, exprs, inverted)`: resolve
  each referenced column to a `Scalar` via the provided `ScalarExpressionEvaluator`;
  any `None` ⇒ return `Ok(None)` (unknown ⇒ no pruning). Build a 1-row
  `RecordBatch` from the scalars, run the columnar path, extract the single value
  (NULL ⇒ `None`). Any planning/eval error ⇒ `Ok(None)`, never `Err` — an opaque
  op must degrade to "don't know", not fail the scan.
- **Stats skipping** `eval_as_data_skipping_predicate` /
  `as_data_skipping_predicate`: return `None` in v1 (documented follow-up:
  map monotone/range-safe exprs onto `minValues`/`maxValues` stat columns).

Wiring:

1. `to_kernel.rs` catch-alls (:203-206, :208, and the operator fallthroughs) —
   when the node is in **predicate position**, attempt opaque wrapping instead of
   `plan_err!`, guarded by ALL of:
   - expression type is `Boolean` (check against the schema if available, else
     restrict to obviously-boolean node kinds: `Like`, `InList` with non-literals,
     `Case` with boolean output, boolean-returning `ScalarFunction`, `Not` over
     these, etc.);
   - **non-volatile** (`expr.is_volatile()` false — exclude `random()` etc.;
     volatile exprs must not prune);
   - every column reference converts to a kernel `ColumnName` (reuse the existing
     column-conversion path).
   Emit `Predicate::arrow_opaque(DataFusionOpaquePredicateOp { expr },
   referenced_columns_as_kernel_exprs)`. Keep `plan_err!` for everything else —
   expression-position nodes (projections) are NOT wrapped in v1.
2. `to_datafusion.rs:179` — replace the `Predicate::Opaque` rejection with
   downcast recovery: if it's our op, return `self.expr.clone()`; otherwise keep
   `not_impl_err!`. (`Predicate::Unknown` stays rejected.) This closes the D1
   parquet-pushdown round-trip: DF filter → kernel predicate (with opaque) →
   `parquet_exec` predicate → original DF expr.
3. `expressions/mod.rs` — export; note the invariant in the module docs:
   *opaque ops never wrongly prune: scalar-eval errors and stats hooks resolve to
   "don't know".*

## Interface contracts (out)

- `process_filters` (`next/scan/plan.rs`) now yields kernel predicates for
  previously-untranslatable filters — pushdown classification (`Exact`/`Inexact`)
  must NOT change to `Exact` for opaque-carrying predicates (partition-only
  opaque filters could theoretically be Exact, but keep them `Inexact` in v1 so
  DataFusion re-applies them — cheap and safe).
- Round-trip guarantee for D1: `predicate_to_df(to_delta_predicate(e)) == e` for
  any boolean non-volatile `e` (exact or opaque-carried).

## Validation gates

- V5 spike test (kept as a regression test).
- Unit: `eval_pred` incl. `inverted`; `eval_pred_scalar` with missing column ⇒
  `None`; volatile expr NOT wrapped; non-boolean expr NOT wrapped.
- Integration (native): a table partitioned by `part`, filter
  `substr(part, 1, 2) = 'ab'` (or a registered UDF) — assert (a) kernel scan
  metadata returns only matching partitions (pruning happened: compare file
  counts vs unfiltered), (b) query results equal the full-scan + filter baseline.
- Round-trip: `to_delta_predicate` → `predicate_to_df` recovers the original expr.
- `cargo test -p deltalake-core --features datafusion` unchanged; fmt + clippy.

## Risks

- V5 downcast reachability (mitigation above; we own the kernel fork).
- `PartialEq` semantics of `Expr` under `DynPartialEq` — two structurally equal
  ops must compare equal (kernel dedups/compares predicates); add a test.
- Accidentally wrapping exprs whose columns the kernel later can't supply to the
  scalar evaluator (non-partition columns during partition pruning) — that path
  returns `None` from the column resolve, which is safe; test it explicitly.

## Deviations from the plan (as implemented)

Two design assumptions in the sections above were corrected during
implementation. Both are load-bearing; a future reader should trust this section
over the pre-implementation "Design" text where they conflict.

1. **Partition pruning flows through `as_data_skipping_predicate`, not
   `eval_pred_scalar`.** The plan (§Design) assumed `eval_pred_scalar` "powers
   partition pruning". In kernel v0.25 the scan's partition pruning is done by
   the `DataSkippingFilter`, which calls the *indirect* hook
   `OpaquePredicateOp::as_data_skipping_predicate` (via
   `DataSkippingPredicateCreator::eval_pred_opaque`,
   `kernel/src/scan/data_skipping.rs`). An op that returns `None` there (as the
   plan proposed for v1) contributes **nothing** to pruning — the integration
   test proved 0 files pruned. `eval_pred_scalar` is only invoked for the static
   `can_statically_skip_all_files` check (`kernel/src/scan/mod.rs`) with an
   `EmptyColumnResolver`, so it can never see partition values.
   **Resolution:** `as_data_skipping_predicate` now rewrites the op over the
   exact partition-value stat columns. For each referenced column it asks the
   evaluator for `get_min_stat`/`get_max_stat`; a partition column yields the
   same `partitionValues_parsed.<col>` expression for both (min == max, exact),
   while a data column yields distinct min/max (or `None`) — so **only partition
   columns qualify** and the op never prunes on approximate stats. It re-wraps
   itself as a fresh opaque predicate over those stat expressions; the kernel
   guards the result against Remove rows and evaluates it columnarly via
   `eval_pred`. The v1 stats-skipping-for-data-columns follow-up (map
   monotone/range-safe exprs onto min/max) still stands.

2. **The op evaluates by *arg-expression*, not by name lookup; opaque wrapping
   is an explicit `process_predicate` fallback, not folded into
   `to_delta_predicate`.**
   - *Evaluation model:* because `eval_pred` runs against two different batch
     shapes (raw data with a `part` column; a stats batch with a
     `partitionValues_parsed.part` struct), the op does **not** plan
     `logical2physical(self.expr, batch.schema())` directly. It stores the DF
     `Expr` plus the ordered leaf column names, evaluates each embedded kernel
     `Expression` against the batch (`evaluate_expression`, which resolves nested
     paths), rebuilds a batch keyed by the DF column names, then runs the `Expr`.
     This is the only shape that works for both callers.
   - *Wiring placement:* the plan put opaque wrapping in `to_kernel.rs`'s
     catch-all. That is unsafe: `to_delta_predicate` is also called by
     `find_files` and by `process_predicate` *before* the schema-override
     type-mismatch guard, so auto-wrapping there let a wrong-typed predicate
     (override `Timestamp` over a `Long` column) reach the kernel and panic. The
     structural `to_delta_predicate` is therefore left unchanged (untranslatable
     ⇒ `Err`), and `try_opaque_predicate` is called **explicitly** in
     `process_predicate` (`next/scan/plan.rs`) *after* the type-mismatch guard
     and *before* the partition-refs early return. Opaque-carrying predicates are
     always classified `Inexact` (DataFusion re-applies them).
   - Consequence for the D1 round-trip: `predicate_to_df` recovers our op via
     downcast and returns the original `Expr` (kept — `test_roundtrip_opaque_predicate`).

## Done criteria — met

- Gates green: `cargo test -p deltalake-core --features datafusion` passes
  except three failures pre-existing on the base branch
  (`test_builder_rejects_unsupported_reader_protocol`,
  `test_direct_scan_rejects_unsupported_reader_protocol`,
  `test_builder_from_valid_url_local_existing_path` — unrelated to predicates);
  fmt + clippy clean for the new code.
- Invariant documented in `opaque.rs` module docs.
- V5 spike kept as regression (`v5_spike_downcast_round_trip`); partition-pruning
  integration test (`test_opaque_predicate_prunes_partitions`) proves 4→2 file
  pruning with results equal to the full-scan baseline.
- `WASM_ENGINE.md` status + kernel-fork note updated (D5 must keep the patch).
