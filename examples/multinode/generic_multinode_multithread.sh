#!/bin/bash
#SBATCH -J tps-multinode
#SBATCH -p general
#SBATCH --mail-type=ALL
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=95:00:00
#SBATCH --mem=0
#SBATCH -A r01859

set -euo pipefail

# Usage:
#   sbatch generic_multinode_multithread.sh input.jl data.jld2 [extra files...]
#
# The Julia input must read its data file from ARGS[1]. The launcher stages the
# input, data, and optional extra files in node-accessible scratch. The input is
# responsible for starting one Distributed worker per hostname listed in
# TPSCHEM_MACHINE_FILE.
#
# Useful overrides:
#   JULIAENV=/path/to/project
#   JULIA_DEPOT_PATH=/path/to/depot
#   JULIA_EXE=julia
#   JULIA_VERSION_ARG=+1.11       # set to an empty string for plain `julia`
#   JULIA_NUM_THREADS=32          # master threads; defaults to 32
#   TPSCHEM_WORKER_THREADS=64     # worker-only nodes; defaults to SLURM_CPUS_PER_TASK
#   TPSCHEM_MASTER_NODE_WORKER_THREADS=64   # the node that also runs the master;
#                                 # defaults to SLURM_CPUS_PER_TASK - JULIA_NUM_THREADS
#   TPSCHEM_SKIP_MASTER_NODE=true # omit the first node from the worker list
#   TPSCHEM_SAVE_SCRATCH=0        # do not copy scratch back after success
#   TPSCHEM_SCRATCH_ROOT=/path    # defaults to /N/scratch/$USER

if [ "$#" -lt 2 ]; then
    echo "Usage: sbatch generic_multinode_multithread.sh input.jl data.jld2 [extra files...]" >&2
    exit 2
fi

module reset

export PATH="$HOME/.juliaup/bin:$PATH"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$HOME/.julia}"
export JULIAENV="${JULIAENV:-$SLURM_SUBMIT_DIR}"

# Julia threads parallelize TPSChem worker kernels. Keep BLAS/OpenMP at one
# thread so each Julia thread does not start another nested thread team.
# The master primarily coordinates distributed CEPA work, so it uses fewer
# threads than the workers by default. Each worker can use the full node.
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-32}"
export TPSCHEM_WORKER_THREADS="${TPSCHEM_WORKER_THREADS:-$SLURM_CPUS_PER_TASK}"
# One node runs the master as well as a worker. Give that node's worker the
# cores the master is not using, so the pair fits in --cpus-per-task instead of
# oversubscribing it, while every worker-only node gets the full allocation.
_shared=$(( SLURM_CPUS_PER_TASK - JULIA_NUM_THREADS ))
[ "$_shared" -lt 1 ] && _shared=1
export TPSCHEM_MASTER_NODE_WORKER_THREADS="${TPSCHEM_MASTER_NODE_WORKER_THREADS:-$_shared}"
export TPSCHEM_BLAS_THREADS="${TPSCHEM_BLAS_THREADS:-1}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
INPUT_FILE="$1"
DATA_FILE="$2"
shift 2

case "$INPUT_FILE" in
    *.jl) ;;
    *)
        echo "First argument must be a Julia file ending in .jl: $INPUT_FILE" >&2
        exit 2
        ;;
esac

case "$INPUT_FILE" in
    /*) ;;
    *) INPUT_FILE="$WORKDIR/$INPUT_FILE" ;;
esac

case "$DATA_FILE" in
    /*) ;;
    *) DATA_FILE="$WORKDIR/$DATA_FILE" ;;
esac

for required_file in "$INPUT_FILE" "$DATA_FILE"; do
    if [ ! -f "$required_file" ]; then
        echo "Required file does not exist: $required_file" >&2
        exit 2
    fi
done

SCRATCH_ROOT="${TPSCHEM_SCRATCH_ROOT:-/N/scratch/$USER}"
JOB_SCRATCH="$SCRATCH_ROOT/tpschem-${SLURM_JOB_ID}"
mkdir -p "$JOB_SCRATCH"

cp "$INPUT_FILE" "$JOB_SCRATCH/"
cp "$DATA_FILE" "$JOB_SCRATCH/"
for extra_file in "$@"; do
    [ -n "$extra_file" ] || continue
    case "$extra_file" in
        /*) source_file="$extra_file" ;;
        *) source_file="$WORKDIR/$extra_file" ;;
    esac
    if [ ! -f "$source_file" ]; then
        echo "Extra file does not exist: $source_file" >&2
        exit 2
    fi
    cp "$source_file" "$JOB_SCRATCH/"
done

INPUT_BASE="$(basename "$INPUT_FILE")"
DATA_BASE="$(basename "$DATA_FILE")"
JOB_OUTPUT="$WORKDIR/${INPUT_BASE}.${SLURM_JOB_ID}.out"
SCRATCH_OUTPUT="$JOB_SCRATCH/${INPUT_BASE}.out"

export TPSCHEM_MACHINE_FILE="$JOB_SCRATCH/nodes.${SLURM_JOB_ID}.txt"
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$TPSCHEM_MACHINE_FILE"

JULIA_EXE="${JULIA_EXE:-julia}"
JULIA_VERSION_ARG="${JULIA_VERSION_ARG-+1.11}"
JULIA_RUN=("$JULIA_EXE")
if [ -n "$JULIA_VERSION_ARG" ]; then
    JULIA_RUN+=("$JULIA_VERSION_ARG")
fi

echo "Job ID:         $SLURM_JOB_ID"
echo "Project:        $JULIAENV"
echo "Depot:          $JULIA_DEPOT_PATH"
echo "Scratch:        $JOB_SCRATCH"
echo "Master threads: $JULIA_NUM_THREADS"
echo "Worker threads: $TPSCHEM_WORKER_THREADS (master's node: $TPSCHEM_MASTER_NODE_WORKER_THREADS)"
echo "BLAS threads:   $TPSCHEM_BLAS_THREADS"
echo "Nodes:"
cat "$TPSCHEM_MACHINE_FILE"

cd "$JOB_SCRATCH"
touch "$SCRATCH_OUTPUT"

sync_output() {
    while true; do
        rsync -a "$SCRATCH_OUTPUT" "$JOB_OUTPUT" || true
        sleep 60
    done
}

sync_output &
SYNC_PID=$!

cleanup() {
    kill "$SYNC_PID" 2>/dev/null || true
    wait "$SYNC_PID" 2>/dev/null || true
    rsync -a "$SCRATCH_OUTPUT" "$JOB_OUTPUT" 2>/dev/null || true
}
trap cleanup EXIT

if [ -n "${TPSCHEM_DEVELOP_PATH:-}" ]; then
    "${JULIA_RUN[@]}" --project="$JULIAENV" -e \
        'import Pkg; Pkg.develop(path=ENV["TPSCHEM_DEVELOP_PATH"]); Pkg.instantiate(); Pkg.precompile(); Pkg.status()'
else
    "${JULIA_RUN[@]}" --project="$JULIAENV" -e \
        'import Pkg; Pkg.instantiate(); Pkg.precompile(); Pkg.status()'
fi

"${JULIA_RUN[@]}" --project="$JULIAENV" --threads="$JULIA_NUM_THREADS" \
    "$INPUT_BASE" "$DATA_BASE" > "$SCRATCH_OUTPUT" 2>&1

cleanup
trap - EXIT

if [ "${TPSCHEM_SAVE_SCRATCH:-1}" = "1" ]; then
    SCRATCH_COPY="$WORKDIR/${INPUT_BASE}.${SLURM_JOB_ID}.scr"
    mkdir -p "$SCRATCH_COPY"
    cp -R "$JOB_SCRATCH/." "$SCRATCH_COPY/"
fi

cd "$WORKDIR"
rm -rf "$JOB_SCRATCH"

echo "Completed successfully. Output: $JOB_OUTPUT"
