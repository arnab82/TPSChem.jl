#!/bin/bash
#SBATCH --job-name=tpsci-ci-sharded-4n
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --output=tpsci-ci-sharded-%j.out
#SBATCH --error=tpsci-ci-sharded-%j.err

set -euo pipefail

# Never-gather selected TPSCI: the whole selected-CI loop (Davidson + PT1 +
# clip + variational growth) stays sharded, so vec_var is never gathered onto
# the master between iterations.
# Example:
#   export TPSCHEM_INPUT_JLD2=/path/to/problem_with_ref_vec.jld2
#   export TPSCHEM_NROOTS=3
#   export TPSCHEM_THRESH_CIPSI=1e-3      # PT1 selection threshold
#   export TPSCHEM_THRESH_FOI=1e-5        # FOIS screening threshold
#   export TPSCHEM_MAX_ITER=10            # outer selected-CI iterations
#   export TPSCHEM_MAX_SS_VECS=4          # keep small for >node-memory vectors
#   export TPSCHEM_H_STORAGE=auto         # auto | blocks | matrixfree
#   export TPSCHEM_MAX_MEM_H=200          # GB budget for the stored block-sparse H
#   sbatch examples/multinode/run_tpsci_ci_sharded_4nodes.slurm

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
    "$PROJECT/examples/multinode/tpsci_ci_sharded_driver.jl"
