# Multinode TPSCI/CEPA Examples

These examples use one Julia distributed worker per SLURM node and many Julia
threads inside each worker. They do not require OpenMPI.

## Input file

Set `TPSCHEM_INPUT_JLD2` to a JLD2 file containing:

- `ints`
- `clusters`
- `d1`
- `init_fspace`
- `cluster_bases`
- optionally `clustered_ham`

For CEPA, also provide a pre-solved reference and energies:

- `ref_vec` or set `TPSCHEM_REF_KEY`
- `e0` or set `TPSCHEM_E0="-1.0,-0.9,..."`

## Submit

```bash
export TPSCHEM_INPUT_JLD2=/path/to/problem.jld2
export TPSCHEM_THRESH_FOI=1e-6
export TPSCHEM_MINRES_MAXITER=300
sbatch examples/multinode/run_cepa_sharded_minres_4nodes.slurm
```

For a sharded FOIS/PT1 smoke run:

```bash
export TPSCHEM_INPUT_JLD2=/path/to/problem.jld2
export TPSCHEM_THRESH_FOI=1e-8
sbatch examples/multinode/run_tpsci_sharded_pt1_4nodes.slurm
```

The CEPA driver builds the Q space once as a `DistributedTPSCIstate`, then solves
every root's amplitude equation over that same sharded Q basis — by default all
roots in one solver pass, sharing a Hamiltonian apply.

See [../CEPA_JOB_KEYWORDS.md](../CEPA_JOB_KEYWORDS.md) for what each submission
keyword does, how to choose `TPSCHEM_H_STORAGE` / `TPSCHEM_SOLVER` /
`TPSCHEM_BLOCK_ROOTS`, and the measured cost of each combination.

## TPSCI Property/RDM Post-Analysis

Use `tpsci_property_rdm_driver.jl` after you have a stored TPSCI reference
wavefunction (`ref_vec`, `ci_vector`, or the key named by `TPSCHEM_REF_KEY`).
It computes the alpha/beta 1-RDM, optional spin-flip 1-RDM, a direct
one-electron property matrix, and optionally the full spin-free 2-RDM.

```bash
export TPSCHEM_OUTPUT_JLD2=post_analysis_rdms.jld2
export TPSCHEM_REF_KEY=ref_vec
export TPSCHEM_BUILD_SPIN_FLIP_1RDM=1
export TPSCHEM_BUILD_2RDM=0        # set to 1 only when you need the full 2-RDM
sbatch examples/multinode/generic_multinode_multithread.sh \
    examples/multinode/tpsci_property_rdm_driver.jl \
    /path/to/problem_with_ref_vec.jld2
```

For a non-Hamiltonian one-electron property, store the integral matrix in the
input JLD2 and set `TPSCHEM_PROPERTY_KEY`, for example
`TPSCHEM_PROPERTY_KEY=mu_z`. If unset, the driver uses `ints.h1`.

Entry points:

- `compute_1rdm_distributed`
- `compute_1rdm_sf_distributed`
- `compute_1e_property_direct_distributed`
- `compute_2rdm_distributed`
- `add_spinfree_2rdm_ops_distributed!`
- `compute_cluster_ops_2rdm_distributed`

## SPT Variance / Sigma Norm

Use `spt_variance_driver.jl` for the SPT variance-style sigma norm
`<sigma|sigma>`. The driver reads an SPT state from `TPSCHEM_SPT_REF_KEY`, or
falls back to `v_var`, `spt_ref`, `vbst`, `ref_spt`, then finally a CMF-like
SPT reference built from `init_fspace`.

```bash
export TPSCHEM_OUTPUT_JLD2=spt_variance.jld2
export TPSCHEM_SPT_REF_KEY=v_var
export TPSCHEM_THRESH_FOI=1e-6
export TPSCHEM_NBODY=4
export TPSCHEM_OPT_REF=0
sbatch examples/multinode/generic_multinode_multithread.sh \
    examples/multinode/spt_variance_driver.jl \
    /path/to/problem_with_spt_state.jld2
```

Entry point: `compute_spt_sigma_norm_blockwise_distributed`
(`spt_variance_multinode` is an alias).

### Stored-H TPS-CEPA (faster)

`cepa_sharded_minres_driver.jl` is matrix-free: it re-contracts the Q-space
Hamiltonian on every MINRES/CG iteration. `tps_cepa_stored_h_driver.jl` instead
builds the Q-space `H_QQ` once as distributed block-sparse blocks and reuses it
via block GEMV across all roots, macro-iterations, and solver steps — same energy,
far fewer contractions (~4x faster already on a tiny 79-config Q-space).

```bash
export TPSCHEM_INPUT_JLD2=/path/to/problem_with_ref_vec.jld2
export TPSCHEM_THRESH_FOI=1e-6
export JULIA_NUM_THREADS=32       # threads for the master process
export TPSCHEM_WORKER_THREADS=96  # threads per worker-only node
                                  # the node that also runs the master gets
                                  # TPSCHEM_MASTER_NODE_WORKER_THREADS, which the
                                  # launchers default to CPUS_PER_TASK minus the
                                  # master's threads so the pair is not oversubscribed
export TPSCHEM_H_STORAGE=auto     # auto | blocks | matrixfree
export TPSCHEM_MAX_MEM_H=200      # aggregate GB budget for the stored H_Q
export TPSCHEM_SOLVER=pcg         # pcg (preconditioned CG) | minres | cg
export TPSCHEM_LINSOLVE_TOL=1e-6  # inner Krylov tol; TPSCHEM_CEPA_TOL drives the macro loop
sbatch examples/multinode/run_cepa_sharded_minres_4nodes.slurm \
    # ...or run tps_cepa_stored_h_driver.jl directly with the same env vars
```

Entry points: `do_tps_sharded_cepa(ref, cluster_ops, clustered_ham; h_storage=:blocks, ...)`
and `tps_sharded_cepa_solve(...)`. The matrix-free `do_fois_cepa_sharded` /
`tpsci_cepa_solve_sharded` names keep working unchanged — they are now thin
wrappers that call the same code path with `h_storage=:matrixfree`, so there is a
single implementation for both the matrix-free and stored-block Hamiltonians.

Where the time goes, and what the solver does about it. Distributed round trips
*were* the bottleneck — before the batching below they were roughly 91% of an
apply. They are now about 3%: a measured apply (dim_Q 32865, 2 workers) spends
2.5 ms gathering kets and 78 ms in the block GEMV, streaming stored H at
~11 GB/s. So the apply is now **memory-bandwidth bound**, and stored-H size sets
both the memory ceiling and the time per iteration. Six things matter.

- **Batched ket exchange.** The connected ket sectors a worker needs are a
  property of the stored H, so they are recorded at build time and fetched with
  one call per remote owner (as dense vectors), not one blocking round trip per
  Fock sector.
- **Fused vector algebra.** `sharded_fused_ops!` runs a whole batch of
  axpy/scale/copy/dot operations in a single fan-out, so a MINRES iteration is
  one `H*v` plus three fan-outs instead of roughly a dozen.
- **Shift folded into the apply.** The stored-block operator evaluates
  `(H_QQ - E·I)v` in one worker pass (`apply_sharded_H(op, v; eshift=...)`).
- **Warm-started macro-iterations.** For `acpf`/`aqcc`/`cisd` the shift moves only
  slightly between CEPA iterations, so each solve starts from the previous
  amplitudes (`warm_start=true`, the default) and later solves take a few Krylov
  steps. Convergence is measured against `‖b‖`, so the tolerance means the same
  thing warm or cold.
- **Threaded apply.** The per-apply GEMV is threaded across the bra sectors a
  worker owns (`threaded_worker=true`), which matters most on fat nodes: a
  single thread streaming tens of GB of stored H is the common way to leave a
  96-core node idle.
- **Preconditioned CG.** `solver=:pcg` runs Jacobi-preconditioned CG using the
  Q-space diagonal (built once, reused across every root and macro-iteration).
  It needs roughly a third of the applies of plain MINRES, and falls back to
  MINRES by itself on any root whose shifted operator is not positive definite.
  `linsolve_tol` sets the inner Krylov tolerance separately from `tol`, which
  keeps governing macro-iteration convergence.

The stored-H build itself buckets the Hamiltonian terms by which clusters they
act on, keyed by Fock transfer, so a matrix element only visits the handful of
terms that can possibly contribute instead of testing all of them; blocks are
filled in parallel across worker threads, and sector pairs no term can connect
are not stored at all.

## Across-node variational diagonalization

When the variational CI vector exceeds a single node's memory, there are two
fixed-space sharded diagonalizers:

- `tps_ci_direct_sharded`: stored block-H across workers, dense Davidson
  coefficient subspace on the master, and batched worker GEMM for `H*X`. This is
  the multinode analogue of `tps_ci_direct` and is the fast stored-H path.
- `tps_ci_davidson_sharded`: fully sharded state-vector Davidson, useful for
  matrix-free operation when stored H cannot fit.

```bash
export TPSCHEM_INPUT_JLD2=/path/to/problem_with_variational_space.jld2
export TPSCHEM_NROOTS=3
export TPSCHEM_MAX_SS_VECS=4      # keep small: each subspace vector is another full vector
export TPSCHEM_H_STORAGE=auto     # auto | blocks | matrixfree
export TPSCHEM_MAX_MEM_H=200      # GB budget for the stored block-sparse H
sbatch examples/multinode/run_tpsci_sharded_davidson_4nodes.slurm
```

Storage tiers (`TPSCHEM_H_STORAGE`):

- `blocks`     — build the Hamiltonian once as distributed, block-sparse dense
                 Fock-pair blocks and reuse them via block GEMV every iteration.
                 `tps_ci_direct_sharded` uses these blocks with dense
                 coefficient matrices and batched GEMM. Needs the block-sparse H
                 to fit aggregate memory.
- `matrixfree` — re-contract the cluster operators every matvec (via
                 `tps_ci_matvec_sharded`). Minimal memory, slower per iteration.
- `auto`       — pick `blocks` if the estimated block-sparse H fits within
                 `TPSCHEM_MAX_MEM_H` GB, else `matrixfree`.

`tps_ci_direct_sharded(...; h_storage=:auto)` is stricter: it uses `blocks` only
when the estimate fits and otherwise errors, because direct-sharded has no
matrix-free mode. Use `tps_ci_davidson_sharded(...; h_storage=:matrixfree)` when
you intentionally want the memory-only fallback.

Check the block-sparse H size before committing to a run:

```julia
rep = TPSChem.sharded_H_memory_report(dvar, clustered_ham)   # (dim, nfocks, nnz, bytes, gb)
```

The same solver is available inside the selected-CI driver via
`tpsci_ci_multinode(...; use_sharded=true, h_storage=:auto, max_mem_H=200.0)`.
Note: in `tpsci_ci_multinode` only the diagonalization is sharded; the
PT/selection steps still gather `vec_var` on the master.

## Never-gather selected TPSCI

When even the *variational* CI vector must never be gathered between selected-CI
iterations, use `tpsci_ci_sharded` (`tpsci_ci_sharded_driver.jl`). The vector
stays a `DistributedTPSCIstate` for the **entire** selected-CI loop: stored-H
direct sharded diagonalization (`tps_ci_direct_sharded`) when
`h_storage=:blocks`, sharded PT1 selection
(`compute_pt1_wavefunction_sharded`), sharded `clip!`, and an ownership-
reconciling `add!` grow the variational space in place. No full CI vector is
ever materialized on the master or on any single node. The (small) starting
reference is distributed once and never gathered again; the returned `vec_var`
stays sharded (`collect_tpsci_state` it only if it fits one node).

```bash
export TPSCHEM_INPUT_JLD2=/path/to/problem_with_ref_vec.jld2
export TPSCHEM_NROOTS=3
export TPSCHEM_THRESH_CIPSI=1e-3     # PT1 selection threshold
export TPSCHEM_THRESH_FOI=1e-5       # FOIS screening threshold
export TPSCHEM_MAX_ITER=10           # outer selected-CI iterations
export TPSCHEM_MAX_SS_VECS=4         # keep small: each subspace vector is another full vector
export TPSCHEM_H_STORAGE=auto        # auto | blocks | matrixfree
export TPSCHEM_MAX_MEM_H=200         # GB budget for the stored block-sparse H
sbatch examples/multinode/run_tpsci_ci_sharded_4nodes.slurm
```

Entry point: `tpsci_ci_sharded(ref, cluster_ops, clustered_ham; thresh_cipsi,
thresh_foi, h_storage=:auto, max_mem_H, ...)`. The storage tiers
(`TPSCHEM_H_STORAGE`) use stored block-H plus `tps_ci_direct_sharded` for the
normal fast path. Selected-CI is strict: `h_storage=auto` errors instead of
silently falling back to matrix-free when the stored-H estimate exceeds
`max_mem_H`. Set `TPSCHEM_H_STORAGE=matrixfree`, or
`TPSCHEM_ALLOW_MATRIXFREE_FALLBACK=true`, only when you intentionally accept the
slow memory-only path.

Intentional limitations (documented, not bugs): `tpsci_ci_sharded` errors on
`incremental` sigma updates and the `thresh_spin` S² space extension — use
`tpsci_ci_multinode` for those. Config-level `thresh_asci`/`thresh_var` clipping
is safe, but clipping so aggressively that an entire Fock sector leaves the
search vector while it remains in the variational state can trip the `add!`
ownership guard.

## Never-gather variational SPT

When the Tucker-compressed SPT state itself outgrows a node, run the variational
SPT loop (`subspace_product_tucker`) with `subspace_product_tucker_sharded`: the
variational vector, the FOIS, and the PT1 all stay `DistributedSPTstate`s across
the workers for the whole loop and are never assembled on the master. It uses the
distributed Tucker CI solver `spt_ci_davidson_sharded` instead of a node-local
`ci_solve`.

Entry points: `subspace_product_tucker_sharded(ref, cluster_ops, clustered_ham;
thresh_var, thresh_foi, thresh_pt, ...)`, `compute_pt1_wavefunction_sharded`,
`spt_ci_davidson_sharded`, `build_compressed_1st_order_state_sharded`. Deferred:
Tier-A stored block-sparse Tucker-H (matrix-free only for now) and the S² spin
extension (`thresh_spin`) — the driver errors on the latter; use
`subspace_product_tucker` / `spt_multinode`.

## Concrete end-to-end Cr2 examples

`run_tpsci_multinode_cr2_morokuma.jl` and `run_spt_multinode_cr2_morokuma.jl` are
self-contained scripts (in the spirit of `examples/bimetallics` /
`examples/notes/tpsci.jl`) that build the spin cluster bases and reference Fock
sectors / P-space for the Cr2 Morokuma trimer and run the never-gather TPSCI /
SPT loops. They read `ints`, `clusters`, `d1` from a CMF JLD2
(`test/data_cmf_13_cr2_morokuma.jld2` works) and shard the cluster operators.

```bash
# On a cluster (one worker per node) via the generic launcher:
sbatch generic_multinode_multithread.sh run_spt_multinode_cr2_morokuma.jl \
       ../../test/data_cmf_13_cr2_morokuma.jld2

# Locally with two workers:
julia -p 2 --project run_tpsci_multinode_cr2_morokuma.jl \
       ../../test/data_cmf_13_cr2_morokuma.jld2
```

Both accept the same env knobs as the drivers above (`TPSCHEM_CLUSTER_MAX_ROOTS`,
`TPSCHEM_NROOTS`, `TPSCHEM_THRESH*`, `TPSCHEM_MAX_ITER`, `TPSCHEM_NBODY`,
`TPSCHEM_H_STORAGE`, `TPSCHEM_BLAS_THREADS`, `TPSCHEM_OUTPUT_JLD2`, ...).
