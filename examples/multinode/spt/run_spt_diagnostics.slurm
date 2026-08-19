#!/bin/bash
#SBATCH --job-name=spt-diag
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --mem=0
#SBATCH --output=spt-diag-%j.out
#SBATCH --error=spt-diag-%j.err

set -euo pipefail

# Metadata-only: no workers, one node.  Tells you whether the 2D grid matvec and
# a work-weighted FOIS assignment are worth anything for YOUR system before you
# spend node-hours on the scaling run.
#
#   export TPSCHEM_INPUT_JLD2=/path/to/problem.jld2
#   export TPSCHEM_P_LIST=4,9,16,25,36
#   sbatch examples/multinode/spt/run_spt_diagnostics.slurm

: "${TPSCHEM_INPUT_JLD2:?Set TPSCHEM_INPUT_JLD2 to your problem JLD2 file}"
PROJECT="${TPSCHEM_PROJECT:-$PWD}"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1

julia --project="$PROJECT" --threads="$JULIA_NUM_THREADS" \
    "$PROJECT/examples/multinode/spt/spt_diagnostics_driver.jl"
