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

The CEPA driver builds the Q space once as a `DistributedTPSCIstate`, then runs
MINRES root by root over that same sharded Q basis.

### Stored-H TPS-CEPA (faster)

`cepa_sharded_minres_driver.jl` is matrix-free: it re-contracts the Q-space
Hamiltonian on every MINRES/CG iteration. `tps_cepa_stored_h_driver.jl` instead
builds the Q-space `H_QQ` once as distributed block-sparse blocks and reuses it
via block GEMV across all roots, macro-iterations, and solver steps — same energy,
far fewer contractions (~4x faster already on a tiny 79-config Q-space).

```bash
export TPSCHEM_INPUT_JLD2=/path/to/problem_with_ref_vec.jld2
export TPSCHEM_THRESH_FOI=1e-6
export TPSCHEM_H_STORAGE=auto     # auto | blocks | matrixfree
export TPSCHEM_MAX_MEM_H=200      # GB budget for the stored block-sparse H_Q
sbatch examples/multinode/run_cepa_sharded_minres_4nodes.slurm \
    # ...or run tps_cepa_stored_h_driver.jl directly with the same env vars
```

Entry points: `do_tps_sharded_cepa(ref, cluster_ops, clustered_ham; h_storage=:blocks, ...)`
and `tps_sharded_cepa_solve(...)`. The matrix-free `do_fois_cepa_sharded` /
`tpsci_cepa_solve_sharded` names keep working unchanged — they are now thin
wrappers that call the same code path with `h_storage=:matrixfree`, so there is a
single implementation for both the matrix-free and stored-block Hamiltonians.

## Across-node variational diagonalization

When the variational CI vector exceeds a single node's memory, diagonalize it
with `tps_ci_davidson_sharded`: a Davidson solver whose subspace vectors are
themselves `DistributedTPSCIstate`s and are never gathered onto the master.

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
                 This recovers the per-iteration speed of `tps_ci_direct` without
                 ever forming a `dim × dim` dense matrix. Needs the block-sparse H
                 to fit aggregate memory.
- `matrixfree` — re-contract the cluster operators every matvec (via
                 `tps_ci_matvec_sharded`). Minimal memory, slower per iteration.
- `auto`       — pick `blocks` if the estimated block-sparse H fits within
                 `TPSCHEM_MAX_MEM_H` GB, else `matrixfree`.

Check the block-sparse H size before committing to a run:

```julia
rep = TPSChem.sharded_H_memory_report(dvar, clustered_ham)   # (dim, nfocks, nnz, bytes, gb)
```

The same solver is available inside the selected-CI driver via
`tpsci_ci_multinode(...; use_sharded=true, h_storage=:auto, max_mem_H=200.0)`.
Note: in `tpsci_ci_multinode` only the diagonalization is sharded; the
PT/selection steps still gather `vec_var` on the master. A fully never-gather
selected-CI loop is future work (needs sharded selection + an ownership-
reconciling merge).
