#!/bin/bash
#SBATCH --job-name=spt-scaling
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=08:00:00
#SBATCH --mem=0
#SBATCH --output=spt-scaling-%j.out
#SBATCH --error=spt-scaling-%j.err

set -euo pipefail

# SPT sharding scaling benchmark.  Run spt_diagnostics_driver.jl first (it is
# metadata-only, single process, minutes) to see whether the 2D grid is expected
# to pay off for your coupling graph.
#
#   export TPSCHEM_INPUT_JLD2=/path/to/problem.jld2
#   export TPSCHEM_NROOTS=4
#   export TPSCHEM_FOIS_THRESH=1e-4
#   export TPSCHEM_SKIP_SINGLE_NODE=0   # 1 if the state will not fit the master
#   sbatch examples/multinode/spt/run_spt_scaling.slurm
#
# For a scaling curve, submit with --nodes=1,4,9,16.  The 2D grid needs a
# non-prime worker count to form a real grid; with a prime count it warns and
# falls back to the destination partition.

: "${TPSCHEM_INPUT_JLD2:?Set TPSCHEM_INPUT_JLD2 to your problem JLD2 file}"

PROJECT="${TPSCHEM_PROJECT:-$PWD}"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TPSCHEM_BLAS_THREADS="${TPSCHEM_BLAS_THREADS:-1}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

NODEFILE="${SLURM_TMPDIR:-/tmp}/tpschem_nodes_${SLURM_JOB_ID}.txt"
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$NODEFILE"
export TPSCHEM_MACHINE_FILE="$NODEFILE"

echo "Project:   $PROJECT"
echo "Nodes:"; cat "$TPSCHEM_MACHINE_FILE"
echo "JULIA_NUM_THREADS=$JULIA_NUM_THREADS  BLAS=$TPSCHEM_BLAS_THREADS"

julia --project="$PROJECT" --threads="$JULIA_NUM_THREADS" \
    "$PROJECT/examples/multinode/spt/spt_scaling_driver.jl"
