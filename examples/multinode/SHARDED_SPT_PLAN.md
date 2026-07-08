# Plan: Never-gather (sharded) SPT — memory-scalable variational SPT

Status: COMPLETE (end-to-end never-gather variational SPT).
Implemented in `src/core/spt_sharded.jl`; tests in `test/test_spt_sharded.jl`
(15/15 on h8, 2 workers). Delivered and validated against single-node:

- `DistributedSPTstate` container + registry (`distribute_spt_state`,
  `collect_spt_state`, `destroy!`) — round-trip bit-exact.
- Sharded primitives (`overlap`, `norm`, `scale!`, `zero!`, `add_scaled!`,
  `orthonormalize!`, `copy_sharded_state`, `similar_sharded_state`,
  `extract_root_sharded`, `combine_roots_sharded`, `precondition_sharded`).
- `build_sigma_sharded` — never-gather Tucker matvec; matches single-node
  `build_sigma!` to 1e-11.
- `spt_ci_davidson_sharded` — distributed Tucker CI solver reusing the generic
  `_tps_ci_davidson_sharded_core` via `AbstractShardedState`; matches an EXACT
  dense diagonalization of H in the fixed Tucker basis to **8.88e-15** (machine
  precision).
- `build_compressed_1st_order_state_sharded` — never-gather FOIS builder
  (each output Fock sector built wholly by its owner, reference ket blocks
  fetched); matches single-node FOIS (structure + per-root norm + cosine).
- `compute_pt1_wavefunction_sharded` — never-gather PT1/PT2 (shard-local
  rotation/`Sx`/assembly/denominators, never-gather `<X|H|0>`/`<X|F|0>` and
  reductions); PT2 energy matches single-node `compute_pt1_wavefunction` to
  1.78e-15.
- `subspace_product_tucker_sharded` — end-to-end never-gather variational SPT
  driver (`:hash` ownership so compress/`nonorth_add!`/`project_into_new_basis`
  stay worker-local); final energy matches single-node `subspace_product_tucker`
  to 10 digits (-10.1849324450) on h8.

Supporting sharded ops added: `compress_sharded`, `nonorth_add_sharded!`,
`project_into_new_basis_sharded`, `spt_expectation_sharded`,
`build_sigma_into_sharded`.

Deferred (documented, not silently missing): Tier-A stored block-sparse
Tucker-H (matrix-free only for now), and the S² spin extension (`thresh_spin`)
which `subspace_product_tucker_sharded` errors on — use `subspace_product_tucker`
/ `spt_multinode` for that.

---

Original motivation: Tucker-compressed SPT states
are **big/huge in practice**, so the existing `spt_multinode`
(`subspace_product_tucker_multinode`) — which distributes *compute* but gathers
the full FOIS/PT1/variational state on the master and runs a *local* `ci_solve`
Davidson holding `~2·max_ss_vecs·R` full vectors — is master-memory-bound and
cannot run when the state exceeds one node. Only the energy-only
`spt_pt2_multinode` / `spt_variance_multinode` are memory-safe today.

Goal: a `DistributedSPTstate` (variational vector, FOIS, PT1 split by Fock
sector and never gathered) plus a distributed Tucker CI solver, mirroring the
`DistributedTPSCIstate` + `tps_ci_davidson_sharded` architecture.

## 1. Why this is tractable (key structural facts)

The `SPTstate` is `data::OrderedDict{FockConfig, OrderedDict{TuckerConfig, Tucker}}`
— already keyed by Fock sector, exactly like `TPSCIstate`. And the heavy local
operations are Fock-sector-local or fetch-based:

- `compress` (`type_SPTstate.jl`) loops per `(fock, tconfig)` independently →
  **shard-local, zero communication**.
- `orth_dot` / `scale!` / `orth_add!` are per-block core arithmetic. Within a
  **fixed Tucker basis** (same `factors`), the SPT vector is just its
  concatenated cores, so these behave as ordinary vector ops → shardable as
  blockwise sums / local scales.
- `form_1body_operator_diagonal!(vec, Fdiag; pseudo_canon=true)` (`tucker_pt.jl`)
  rotates each block's factors and fills a per-core-entry `Fdiag` — **per-block,
  shard-local**. This is the CMF preconditioner `ci_solve` uses.
- `build_sigma!(sigma, ket, ...)` (`tucker_inner.jl`) iterates the *output*
  (bra) blocks and, per bra, loops all connected *ket* blocks. Sharded by
  destination Fock sector, each worker needs only the ket blocks connected to
  its owned bra sectors → **fetch remote ket Tucker blocks** (same "move 2" as
  `tps_ci_matvec_sharded`), keep output sharded. (`build_sigma_distributed`
  already shards the bra sectors but *ships the whole ket* and *gathers* the
  output; the sharded version fixes both.)

Crucially, inside the CI Davidson the basis is **fixed** — `ci_solve` rotates
`vec` once (pseudo-canon) then only ever `set_vector!`s cores. So every subspace
vector (V, HV, seeds, residuals, Ritz) shares the rotated `factors`, and
core-only `orth_dot`/`add` are exact.

## 2. Reuse: the Davidson core is already generic

`_tps_ci_davidson_sharded_core` (`tpsci_sharded_davidson.jl`) drives block
Davidson using only sharded primitives: `overlap`, `add_scaled!`, `norm`,
`scale!`, `copy_sharded_state`, `similar_sharded_state`, `precondition_sharded`,
`destroy!`, and an `apply_H` closure. It also contains the subtle restart
re-orthonormalization that fixed ghost eigenvalues (2026-07-07) — we must **not**
duplicate it.

Plan: introduce `abstract type AbstractShardedState{T,N,R} <: AbstractState`,
make `DistributedTPSCIstate` and `DistributedSPTstate` both subtype it, and relax
the core helpers (`_mgs_against!`, `_restart_from_ritz_basis!`,
`_tps_ci_davidson_sharded_core`) from `DistributedTPSCIstate{T,N,1}` to
`AbstractShardedState{T,N,1}`. Concrete primitives dispatch on the runtime type,
so the same core drives both. Re-run the TPSCI davidson test to prove no
regression.

## 3. New container (`src/core/spt_sharded.jl`)

```julia
mutable struct DistributedSPTstate{T,N,R} <: AbstractShardedState{T,N,R}
    id::Symbol
    clusters::Vector{MOCluster}
    p_spaces::Vector{ClusterSubspace}   # small, replicated
    q_spaces::Vector{ClusterSubspace}
    workers::Vector{Int}
    owners::OrderedDict{FockConfig{N},Int}
    lengths::OrderedDict{FockConfig{N},Int}   # coeff length = Σ_tconfig length(tuck)
    offsets::OrderedDict{FockConfig{N},Int}
    local_lengths::OrderedDict{Int,Int}
    total_length::Int
end
```

Registry `_SPT_SHARDED_STATES = Dict{Symbol,Any}()`; each worker holds a local
`SPTstate{T,N,R}` of its owned Fock sectors. Ownership mirrors TPSCI (`:balanced`
greedy weighted by coeff count, preserving existing owners on growth; `:hash`).
`distribute_spt_state` / `collect_spt_state` / `destroy!` copy the TPSCI code.

## 4. Sharded primitives (all mirror TPSCI names so the core reuses them)

- `overlap(s1,s2)::Matrix` — blockwise R×R core Gram (same-basis), reduced on master.
- `norm`, `scale!`, `zero!`, `orth_add!`/`add!`, `add_scaled!`, `copy_sharded_state`,
  `similar_sharded_state`, `orthonormalize!` (R×R symmetric orthonormalization
  → local `mult!` of cores by X).
- `extract_root_sharded` (R→R=1: copy factors, take `core[root]`),
  `combine_roots_sharded` (R=1 list → R).
- `precondition_sharded(res, Fdiag, θ)` — per core entry `res/(θ − Fdiag)` with
  the same floor guard as TPSCI.

## 5. Matvec + diagonal

- `build_sigma_sharded(v)`: per worker, allocate a zeroed local output over its
  owned bra sectors, fetch connected ket Tucker blocks (registry copy of remote
  `data[fock]`), assemble a local ket = owned ∪ fetched, call local
  `build_sigma!`, keep output sharded. Wrapped as the `:matrixfree` `apply_H`.
- `spt_diagonal_sharded!(dv)`: run `form_1body_operator_diagonal!` on each shard's
  local slice (rotates that shard's factors in place, consistent because it is
  per-block) and return a sharded R=1 `Fdiag`. Called once before the loop; the
  rotation defines the fixed basis all subspace vectors share.
- (Later, optional Tier A) a stored block-sparse Tucker H — deferred; matrix-free
  first, validated, then speed.

## 6. `spt_ci_davidson_sharded(dv, cluster_ops, clustered_ham; nroots, ...)`

Mirror `tps_ci_davidson_sharded`: rotate + build `Fdiag` (preconditioner &
basis), seed `nroots` orthonormal roots via `extract_root_sharded`, build the
matrix-free `apply_H`, call the shared `_tps_ci_davidson_sharded_core`, combine
roots. Returns `(eigvals, DistributedSPTstate)`.

## 7. Never-gather driver (`subspace_product_tucker_sharded`)

Replace each gather in `subspace_product_tucker_multinode`:
- FOIS: `build_compressed_1st_order_state_sharded` — never-gather variant of
  `build_compressed_1st_order_state_distributed` (output stays sharded; ket
  blocks of the reference fetched, not shipped whole).
- PT1: `compute_pt1_wavefunction_sharded` — do the `sigma − XF0 − Sx(E0−F0)`,
  denominators, and `orth_dot` all shard-local (no full-FOIS copies on master).
- variational solve: `spt_ci_davidson_sharded` instead of local `ci_solve`.
- `compress`/`orthonormalize!`/`nonorth_add!`/`project_into_new_basis`: sharded
  (compress & project are per-block local; add! is ownership-reconciling like
  TPSCI). S² spin extension: deferred (documented limitation, like
  `tpsci_ci_sharded`).

## 8. Validation

- Container: `distribute_spt_state`→`collect_spt_state` round-trips coefficients.
- Primitives: `overlap`/`norm`/`orthonormalize!`/matvec match single-node
  `orth_dot`/`build_sigma!` on h8 to ~1e-12.
- Solver: `spt_ci_davidson_sharded` energies/vectors match single-node `ci_solve`
  on h8 to `conv_thresh`.
- Driver: `subspace_product_tucker_sharded` matches `subspace_product_tucker`
  energies and selected space on h8.
- No regression: existing TPSCI davidson test still green after the abstract-type
  relaxation.

## 9. Staging

1. Abstract type + core relaxation (TPSCI test still green).
2. Container + primitives + matvec + diagonal + `spt_ci_davidson_sharded`,
   tested vs `ci_solve`. **← delivers the "sharded SPTstate + distributed Tucker
   CI solver" the request names.**
3. Never-gather FOIS + PT1 + `subspace_product_tucker_sharded`, tested vs
   single-node. Tier A stored Tucker-H and S² extension deferred.
