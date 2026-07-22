#!/bin/bash
#SBATCH --job-name=cepa-minres-4n
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --output=cepa-minres-%j.out
#SBATCH --error=cepa-minres-%j.err

set -euo pipefail

# Example:
#   export TPSCHEM_INPUT_JLD2=/path/to/problem_with_ref_vec_and_e0.jld2
#   export TPSCHEM_THRESH_FOI=1e-6
#   export TPSCHEM_MINRES_MAXITER=300
#   sbatch examples/multinode/run_cepa_sharded_minres_4nodes.slurm

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

echo "Project: $PROJECT"
echo "Node file: $TPSCHEM_MACHINE_FILE"
cat "$TPSCHEM_MACHINE_FILE"
echo "JULIA_NUM_THREADS=$JULIA_NUM_THREADS"

julia --project="$PROJECT" --threads="$JULIA_NUM_THREADS" \
    "$PROJECT/examples/multinode/cepa_sharded_minres_driver.jl"
