# Plan: Across-node variational diagonalization for TPSCI

Status: IMPLEMENTED (Phases 0-3). Code in `src/core/tpsci_sharded_davidson.jl`,
wired into `tpsci_ci_multinode(...; use_sharded=true)`. Tests in
`test/test_tpsci_multinode.jl` (18 checks, sharded energies match `tps_ci_direct`
to ~2e-11). Example driver + SLURM script in `examples/multinode/`. Remaining
future work: a fully never-gather selected-CI loop (sharded selection + an
ownership-reconciling merge) so `vec_var` is never materialized on the master.

## 1. Problem

In the selected-CI variational loop, after we add the PT vector into the
variational space we must diagonalize `H` in the current `vec_var` basis. For the
large cases on our cluster the CI vector alone exceeds ~500 GB — larger than any
single node's memory. We need a solver whose vectors live **across nodes** and
are never gathered onto the master.

The current multinode entry point
[`tps_ci_davidson_distributed`](src/core/tpsci_multinode.jl) cannot do this: it
calls `get_vector(ci_vector)`, materializing the full dense vector on the master,
and wraps the dense `BlockDavidson` whose subspace is dense `dim × k` matrices.

## 2. The real bottleneck: matvec cost, not the eigensolver

Measured/observed: `tps_ci_direct` is much faster than `tps_ci_davidson`, and the
difference is **not** the eigensolver. It is how the matvec is evaluated.

| Path | H storage | Cost per Davidson matvec |
|------|-----------|--------------------------|
| `tps_ci_direct` (`tpsci_outer.jl:452`) | dense `H` built once via `build_full_H_parallel` | one BLAS `H*v` GEMM — **fast** |
| `tps_ci_davidson` (`tpsci_outer.jl:601`) | none (matrix-free) | full clustered `contract_matrix_element` contraction — **slow, repeated every iteration** |
| `tps_ci_davidson_distributed` | none (matrix-free) | same slow contraction, distributed by Fock sector |

Consequence: a naive sharded, matrix-free Davidson would be distributed but still
slow, because it re-contracts the cluster operators on every iteration.

The winning idea is the same one that makes `tps_ci_direct` fast — **build the
Hamiltonian once and reuse it** — but with H stored as distributed, block-sparse
blocks so it never sits on one node, and never as a `dim × dim` dense matrix.

## 3. Memory reality that dictates the design

Two independent size questions, at the target scale:

- **Vector** (`dim` coefficients): ~500 GB / root. Must be sharded. (Already
  solved: `DistributedTPSCIstate`.)
- **Hamiltonian**: dense `H` is `dim × dim` — utterly impossible here. But the
  *block-sparse* H (only the nonzero Fock-block pairs, each stored as a dense
  sub-block) is far smaller than `dim²`, though possibly larger than one vector.
  Whether it fits in **aggregate** cluster memory is the deciding factor.

Davidson subspace cost, independent of H: with `max_ss_vecs = k` we hold ~`k`
basis + `k` sigma vectors live = `2k × 500 GB`. Hence the chosen default
**`max_ss_vecs ≈ 4`** (≈ 8 full vectors ≈ 4 TB aggregate + diagonal + scratch),
with aggressive restart. This is a knob, but it is the binding constraint.

## 4. Proposed strategy: two tiers, chosen by an H-size estimate

### Tier A — distributed stored block-sparse H (preferred; direct-like speed)

Build H once as a distributed collection of dense Fock-pair blocks, then run the
sharded Davidson with a **block-GEMM matvec** that reuses those blocks. This
recovers `tps_ci_direct`'s per-iteration speed while keeping both H and the
vectors off any single node.

- Ownership: each output Fock-bra sector's row-blocks are owned by the worker that
  owns that sector in the `DistributedTPSCIstate` layout (reuse existing
  `owners`), so `H_block * v_ket_block` writes into a locally-owned sigma block.
- Build: per owned Fock-bra sector, for each connected Fock-ket sector, call the
  existing `build_full_H_parallel(v_bra_sector, v_ket_sector, cluster_ops,
  clustered_ham)` to get the dense sub-block; store it in a worker-local
  `Dict{(FockConfig,FockConfig), Matrix}`. Fetch remote ket config lists via the
  existing `_tpsci_sharded_get_ket_configs` machinery.
- Matvec: `sigma_bra = Σ_ket H[(bra,ket)] * v[ket]` — dense GEMM per block,
  ket blocks fetched/cached exactly as `open_matvec_sharded` already does.
- Reuse: build once before the Davidson loop; every matvec is GEMMs only.

### Tier B — sharded matrix-free Davidson (fallback; minimal memory)

If the estimated block-sparse H does not fit aggregate memory, fall back to
matrix-free: matvec via the existing
[`tps_ci_matvec_sharded`](src/core/tpsci_multinode.jl). Slower per iteration, but
only vector-scale memory. This is the safety net when even sparse H is too big.

### Selection

Before solving, estimate `nnz(H) ≈ Σ_(bra,ket connected) n_bra · n_ket` from the
Fock-sector sizes and `clustered_ham` connectivity (cheap, metadata only).
Compare `nnz(H) · sizeof(T)` against a user-supplied aggregate-memory budget
`max_mem_H`. Pick Tier A if it fits, else Tier B. Expose an override
(`h_storage = :auto | :blocks | :matrixfree`).

## 5. Components and reuse map

Already present and directly reusable (all in `tpsci_multinode.jl`):

- H·v in-space: `tps_ci_matvec_sharded` (Tier B matvec)
- inner products `⟨vᵢ|vⱼ⟩`, `⟨vᵢ|σⱼ⟩`: `overlap` (returns small k×k; only reductions cross the wire)
- `norm`, `scale!`, `add_scaled!`, `linear_combination!`
- `copy_sharded_state`, `similar_sharded_state`, `distribute_tpsci_state`, `collect_tpsci_state` (debug)
- `DistributedClusterOps` + `_materialize_cluster_ops_for_indices` (shard cluster ops too, if needed)
- Precedent for a full sharded Krylov loop: `_tpsci_sharded_minres_linsolve` / `_tpsci_sharded_cg_linsolve` (CEPA) — the sharded Davidson is the same style.

New primitives to add:

1. `compute_diagonal_sharded(ci_vector::DistributedTPSCIstate, cluster_ops, clustered_ham) -> DistributedTPSCIstate{T,N,1}`
   Worker-local `zero_trans` diagonal over owned configs, mirroring
   `compute_diagonal` (`tpsci_outer.jl:905`); `DistributedClusterOps` path via
   `_materialize_cluster_ops_for_indices`.
2. `apply_precond_sharded!(res, ritz_e, Hdiag)` — worker-local
   `res /= (ritz_e - Hdiag)`, one pass over local config blocks (mirrors
   `BlockDavidson.apply_diagonal_precond!`).
3. `rotate_sharded_vectors(V::Vector{DistributedTPSCIstate}, C::Matrix) -> Vector{DistributedTPSCIstate}`
   Build `m` Ritz vectors from `k` basis vectors via **one worker-local GEMM**
   over coefficient blocks (avoids `k·m` axpy passes over 500 GB vectors). This is
   the performance-critical primitive for the subspace collapse/restart.
4. Tier A only: `build_block_sparse_H_sharded(...)`, worker-local block store, and
   `blocked_matvec_sharded(...)`.

## 6. Sharded Davidson algorithm (Tier-agnostic driver)

Represent the subspace as `Vector{DistributedTPSCIstate{T,N,1}}` for both basis
`V` and sigma `Σ` — matches the R=1 style the whole sharded backend already uses
and avoids `R`-as-type-parameter churn across iterations.

```
tps_ci_davidson_sharded(ci_vector::DistributedTPSCIstate, cluster_ops, clustered_ham;
    nroots, conv_thresh, max_ss_vecs=4, max_iter, lindep_thresh,
    h_storage=:auto, max_mem_H, workers, blas_threads, threaded_worker)

  Hdiag  = compute_diagonal_sharded(...)                 # one full sharded vector
  Hop    = Tier A ? build_block_sparse_H_sharded(...) : matrix-free closure
  V, Σ   = [], []                                         # sharded R=1 vectors
  seed nroots orthonormal guesses (Fock-diagonal / random within layout)
  for iter in 1:max_iter
    orthonormalize new directions vs V (MGS via overlap + add_scaled!)   # sharded
    σ_new = Hop(v_new)          # Tier A: block GEMM; Tier B: tps_ci_matvec_sharded
    append to V, Σ
    build small Hss[i,j] = overlap(V[i], Σ[j])[1,1]      # k×k on master
    (F = eigen(Hss); take nroots lowest) -> ritz_e, ritz_v(k×nroots)
    Ritz vectors:  Vr = rotate_sharded_vectors(V, ritz_v)       # sharded GEMM
                   Σr = rotate_sharded_vectors(Σ, ritz_v)
    residuals r_s = Σr[s] - ritz_e[s]*Vr[s]  (add_scaled!)      # sharded
    resid[s] = norm(r_s); mark converged; if all converged -> break
    precondition r_s with (ritz_e - Hdiag) -> new directions    # apply_precond_sharded!
    if subspace size > max_ss_vecs (=4) OR lindep: restart from Vr[1:nroots]
    destroy! all transient sharded vectors each iter (explicit, no GC reliance)
  end
  return ritz_e, Vr[1:nroots]
```

Notes:
- Everything crossing the network is either a k×k reduction (`overlap`) or a
  block fetch already handled by existing code. No full vector is ever gathered.
- `destroy!` discipline matters: at 500 GB/vector, leaking one subspace vector is
  500 GB. Free aggressively (the CEPA solvers already model this).
- Restart at `max_ss_vecs=4` keeps live memory near `~8` vectors.

## 7. Integration

- Add `use_sharded::Bool` (or `solver=:sharded_davidson`) branch to
  [`tpsci_ci_multinode`](src/core/tpsci_multinode.jl), replacing the
  `orthonormalize! + tps_ci_davidson_distributed` block with:
  `dvar = distribute_tpsci_state(vec_var); e0, dvar = tps_ci_davidson_sharded(dvar, ...)`.
- Keep the PT step sharded too (already exists: `compute_pt1_wavefunction_sharded`),
  so `vec_var` is never gathered between iterations. The selected/clipped new
  configs define the next iteration's Fock layout; reuse
  `strategy=:balanced` ownership assignment for newly appearing sectors.
- The existing replicated path stays as the default for small cases.

## 8. Validation

- Correctness: on a small system where everything fits, assert
  `tps_ci_davidson_sharded` energies/vectors match `tps_ci_direct` and
  `tps_ci_davidson` to `conv_thresh`, using `collect_tpsci_state` only in the test.
- Tier A vs Tier B: assert both tiers agree on the small system.
- Distributed sanity: run with `addprocs(2)` locally, then on 2 real nodes.
- Memory: log per-worker peak (`sharded_state_summary`, `cluster_ops_summary`)
  and confirm no master-side full-vector allocation (guard against accidental
  `get_vector`/`collect_*` in the hot path).

## 9. Open questions / risks

1. **Does block-sparse H fit aggregate memory at your target sizes?** This picks
   Tier A vs Tier B and is the single most important number. Need a realistic
   `dim`, average Fock connectivity, and total cluster RAM. (I can add the
   `nnz(H)` estimator first, as a cheap standalone, so you can measure before we
   commit to Tier A.)
2. Cluster-ops memory: are `cluster_ops` small enough to replicate per node, or do
   we need `DistributedClusterOps` in this path too?
3. Seed guess quality drives iteration count (and thus how many times we pay the
   matvec). Reuse previous iteration's `vec_var` as the guess (carry Ritz vectors
   across selected-CI iterations).
4. Load balance: Fock-sector-owned block distribution can be uneven; may need the
   `:balanced` greedy assignment plus block-splitting for very large sectors.

## 10. Suggested rollout

- **Phase 0**: `nnz(H)` / memory estimator + reporting (answers Q1 with real data).
- **Phase 1**: primitives 1–3 + Tier B sharded Davidson (`max_ss_vecs=4`), validated
  against `tps_ci_direct` on a small case. Unblocks >node-memory runs immediately,
  even if slow.
- **Phase 2**: Tier A block-sparse H build + block GEMM matvec + auto-selection.
  This is where the direct-like speed comes back.
- **Phase 3**: wire `use_sharded` into `tpsci_ci_multinode`, carry guesses across
  iterations, tune restart.
