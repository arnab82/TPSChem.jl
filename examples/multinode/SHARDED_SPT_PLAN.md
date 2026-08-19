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

---

## 10. Status after the 2026-08 optimization pass

Stages 1–3 are implemented and validated (`test/test_spt_sharded.jl`, 15/15;
the end-to-end driver test is gated behind `TPSCHEM_TEST_HEAVY_MULTINODE=1` and
reproduces the single-node energy to ten digits).

### 10.1 Fixed: the matvec was latency-bound

Measured on h12 (dim 56317, 2090 Fock sectors, R=4), three workers × 2 threads:

| | before | after |
| --- | --- | --- |
| single-node `build_sigma!` | 8.9 s | 8.9 s |
| `build_sigma_sharded` | **24.1 s** | **6.1 s** |
| vs single node | 2.7× *slower* | 1.46× faster |
| parallel efficiency | 12% | 49% |
| sharded FOIS | — | 1.58× (53%) |

Each worker needs every ket Fock sector connected by `H` to one of its owned bra
sectors, and it fetched them **one `remotecall_fetch` per sector** — about 1400
round trips per worker per matvec. The payload is only ~2.4 MB per worker, so
this was latency, not bandwidth. Grouping the needed sectors by owner
(`_spt_sharded_local_copy_fock_blocks`) and prefetching the FOIS builder's
`ket_cache` the same way fixed it.

Note the existing tests compare numbers to ~1e-15 but never compare *time*, so a
path that was correct and 2.7× slower than the single-node code it exists to
replace passed cleanly for a long time. **A coarse performance assertion —
even "sharded matvec is not slower than single-node" — belongs in the test.**

### 10.2 Open: communication is O(P × |state|), not O(|state|)

This is the blocker for the target scale and batching does **not** address it.

Ownership is by *destination* Fock sector. A worker computes σ only for its own
bra sectors, but `H` couples nearly every ket sector to nearly every bra sector,
so each worker must pull essentially the **whole** ket state every matvec.
Communication therefore grows with worker count instead of shrinking with it,
and each worker's transient memory is O(|state|) regardless of P — which also
defeats the point of sharding for memory.

At h12 this is invisible (2.4 MB/worker). At the sizes this design exists for it
is fatal: a 100M-determinant FOIS at R=4 is roughly 7 GB of cores plus factors,
so every worker would pull ~7 GB per matvec *and* need room to hold it.

**Measured 2026-08-19 — "partition by contribution" does NOT help.** The
obvious alternative (assign each `(bra, ket, terms)` triple to the worker that
already owns the ket sector, accumulate partial sigma locally, reduce into the
bra owner) moves **exactly the same volume**, 1.00x at every P tested. The two
schemes are duals: H's Fock transitions come in +/- pairs, so "kets my bras
need" and "bras my kets feed" have the same structure. On the h12 FOIS the
coupling density is 6.8% with mean bra in-degree 142 of 2090 sectors, so any
1/P slice of sectors already touches essentially every sector. Do not build it.

**What the measurement says to build instead** (h12 FOIS, 3.6 MB of cores,
volume per matvec summed over all workers):

| P | destination | 2D grid | ring | peak/worker, destination | peak/worker, ring |
| --- | --- | --- | --- | --- | --- |
| 4 | 5.4 MB | 7.2 MB | 7.2 MB | 1.4 MB | 0.5 MB |
| 9 | 14.4 MB | **10.8 MB** | 16.2 MB | 1.6 MB | 0.2 MB |
| 16 | 27.0 MB | **14.4 MB** | 28.8 MB | 1.7 MB | 0.1 MB |
| 25 | 43.2 MB | **18.0 MB** | 45.1 MB | 1.7 MB | **0.1 MB** |

Two independent problems, two different fixes:

- **Volume — use a 2D (grid) partition.** Put the P workers on a q x q grid
  (P = q^2). Worker (i,j) handles contributions whose bra is in row-group i and
  whose ket is in column-group j: it pulls only column j's kets and pushes
  partial sigma only for row i's bras, both O(|state|/q) = O(|state|/sqrt(P)).
  Total traffic drops from O(P |state|) to O(sqrt(P) |state|) — confirmed above,
  the grid column grows exactly as q (1.5x, 1.33x, 1.25x). It wins from P >= 9
  and the margin widens; at P = 25 it is 2.4x less traffic.
  Per-worker transient memory also falls to O(|state|/sqrt(P)).
- **Memory — rotate the ket shards (ring / systolic).** Each worker holds one
  ket shard at a time and passes it on, so peak per-worker memory is
  O(|state|/P) — the `peak/worker, ring` column, 17x below destination
  partitioning at P = 25 and improving with P. Total volume is unchanged from
  destination partitioning, so this buys memory, not bandwidth.

Note the `peak/worker, destination` column is flat at ~1.7 MB (about half the
state) no matter how many workers are added: that is the O(|state|)-regardless-of-P
behaviour, made concrete.

For a 100M-determinant FOIS at R=4 (~7 GB) with 25 workers: destination
partitioning pulls ~3.5 GB per worker and must hold it; the 2D grid pulls and
holds ~1.4 GB; a ring holds ~0.28 GB at a time. The 2D grid is the recommended
target since it fixes both axes at once, with the ring as the fallback if
memory rather than bandwidth is binding.

Related open items: the remaining 51% overhead is serialization and load
imbalance (`:hash` balances sector *count*, not work), and the gated driver test
reports "did not converge in 6 iterations" — it passes only because the
single-node reference is equally unconverged at `atol=1e-6`, so its iteration
budget should be raised before it is trusted as an end-to-end check.
