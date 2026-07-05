# D5 — CI, dependency & fork hygiene

> Part of the wasm-engine effort — read [`WASM_ENGINE.md`](./WASM_ENGINE.md) first.
>
> **Status: not started** · Depends on: D1–D4 landed · Blocks: mangrove Phase B
> pinning · Recommended model: **Sonnet**

## Goal

Turn the spike wiring into something mangrove can pin: local path/patch deps →
git refs, prune fork deltas that D1 made obsolete, add the CI matrix, and fold
the status docs. Mangrove Phase B's stated dependency is "the `deltalake-wasm`
public API + **stable git refs** for delta-rs / delta-kernel-rs / the arrow-rs
parquet patch" (`../mangrove/WASM_QUERY_PREVIEW.md`, Phase A section).

## Work items

1. **Push the three branches** (they exist only locally today):
   `delta-rs` → `wasm-core-compat`, `../delta-kernel-rs` → `wasm-kernel-compat`,
   `../arrow-rs` → `wasm-codec-58.3.0`, to the appropriate remotes (owner's
   forks — confirm remote names with the repo owner before pushing anywhere
   non-obvious).
2. **Workspace `Cargo.toml`**: replace the path deps
   (`delta_kernel = { path = "../delta-kernel-rs/kernel" … }`,
   `delta_kernel_default_engine = { path = … }`) and the `[patch.crates-io]`
   arrow-rs family block with `git = …, rev = <pinned sha>` refs. Keep the
   explanatory comments (why the parquet patch must be a patch: feature
   unification pulls C-backed zstd/brotli otherwise — see `WASM.md`).
3. **Prune obsolete fork deltas** in `../delta-kernel-rs`:
   - commit `069115ff` made `ObjectStoreStorageHandler::new` public for the old
     delegation path; after D1 nothing in delta-rs constructs it — revert if
     truly unused (grep the workspace first).
   - Keep: the wasm32 support commit (`31f45ab7`: time shims, target-gated
     object_store cloud features, wasm deps) — still load-bearing.
   - Add anything D3 recorded as a required kernel patch (opaque downcast
     visibility — check `WASM_ENGINE_D3_OPAQUE.md` status notes).
4. **CI matrix** (GitHub Actions, mirror existing workflow style):
   - native: existing workspace build + `deltalake-core` test jobs must stay as
     they are (they now cover the DF-plan engine);
   - wasm compile gates:
     `cargo check -p deltalake-core --no-default-features --lib --target wasm32-unknown-unknown`
     and
     `cargo check -p deltalake-core --no-default-features --features datafusion --target wasm32-unknown-unknown`;
   - `cargo check -p deltalake-wasm --target wasm32-unknown-unknown`;
   - wasm smoke: `wasm-pack test --headless --chrome` (or `--node`) for
     `deltalake-wasm` per D4's tests (needs `wasm32-unknown-unknown` target +
     `wasm-pack`/`wasm-bindgen-cli` in the job).
5. **Docs folding**: update `WASM.md` (status: runs; build/run instructions incl.
   the facade; drop "does not run yet") and `WASM_NOTES.md` (mark next-steps
   complete, point at `WASM_ENGINE.md` as the record). Set all chunk statuses in
   `WASM_ENGINE.md` to done with commit refs. Note remaining known limits
   (no DVs, no zstd/brotli, `nanosecond-timestamps` disabled pending kernel-pin
   reconciliation — see `WASM.md`).
6. **Notify the mangrove side**: the pinned refs + `deltalake-wasm` rustdoc are
   Phase B's inputs; record the exact revs in `../mangrove/WASM_QUERY_PREVIEW.md`
   (Phase A → done) if working across repos, or hand the revs to the owner.

## Validation gates

- Full CI matrix green on the pushed branches.
- Fresh-clone proof: in a clean checkout (no sibling repos), `cargo build
  --workspace` and the wasm checks succeed using only git refs — this is the
  test that no path dep survived.
- `cargo build --workspace` includes the `python` crate unchanged.

## Risks

- Rev churn: pin exact SHAs, not branch names, in Cargo git deps.
- The kernel fork's resolver-v3 note (`../delta-kernel-rs/kernel/Cargo.toml`
  comments): wasm-target deps perturb `--all-features` native builds — the CI
  matrix must not add an `--all-features` job that trips this; keep feature sets
  explicit.

## Done criteria

Fresh-clone gates green; mangrove unblocked with pinned revs; docs folded;
`WASM_ENGINE.md` fully marked done.
