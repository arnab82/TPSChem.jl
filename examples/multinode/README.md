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
