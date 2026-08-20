#!/bin/bash
#SBATCH -J tps-cepa
#SBATCH -p general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abachhar@iu.edu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --time=95:00:00
#SBATCH --mem=0
#SBATCH -A r01859

# --exclusive: this partition (general) can co-schedule other jobs' tasks on
# the same physical node. Without it, --mem=0 ("give me all this node's
# memory") is not a safe request -- if a second job also lands here with its
# own memory grant, SLURM's cgroup accounting does not necessarily partition
# the two cleanly, and combined usage can exceed the node's real RAM with no
# warning from SLURM. That surfaces as an OS-level OOM, which on Linux often
# shows up as a bus error with no clean Julia stacktrace, not a normal OOM
# kill message. --exclusive removes the ambiguity: the whole node (every
# core, all its memory) belongs to this job alone.

set -euo pipefail

# Usage:
#   sbatch run_cepa_multinode_fe2s2.sh input_file.jl data_file.jld2 [extra files...]
#
# Keep input_file.jl and data_file.jld2 in the directory where you submit the job,
# or pass absolute paths.
# Set JULIAENV to the shared TPSChem project directory if it differs from below.

module reset

export PATH="$HOME/.juliaup/bin:$PATH"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/N/u/abachhar/BigRed200/.julia}"
export JULIAENV="${JULIAENV:-/N/u/abachhar/BigRed200/multinode-tpsci}"

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Threads. SLURM grants --cpus-per-task per node. One node runs the master as
# well as a worker, so that node's worker gets the cores the master is not using
# and every worker-only node gets the full allocation. A single number for all
# hosts is wrong either way: too low wastes cores on the worker-only nodes, too
# high oversubscribes the master's node.
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-32}"
export TPSCHEM_WORKER_THREADS="${TPSCHEM_WORKER_THREADS:-$SLURM_CPUS_PER_TASK}"
_shared=$(( SLURM_CPUS_PER_TASK - JULIA_NUM_THREADS ))
[ "$_shared" -lt 1 ] && _shared=1
export TPSCHEM_MASTER_NODE_WORKER_THREADS="${TPSCHEM_MASTER_NODE_WORKER_THREADS:-$_shared}"

export TPSCHEM_BLAS_THREADS="${TPSCHEM_BLAS_THREADS:-1}"
export JULIA_WORKER_TIMEOUT="${JULIA_WORKER_TIMEOUT:-250}"

# CEPA solver settings. :pcg is Jacobi-preconditioned CG (~3x fewer Hamiltonian
# applies than MINRES, falls back to MINRES on any root whose shifted operator is
# not positive definite). TPSCHEM_LINSOLVE_TOL is the inner Krylov tolerance;
# TPSCHEM_CEPA_TOL still drives macro-iteration convergence.
export TPSCHEM_SOLVER="${TPSCHEM_SOLVER:-pcg}"
export TPSCHEM_LINSOLVE_TOL="${TPSCHEM_LINSOLVE_TOL:-1e-6}"
export TPSCHEM_CEPA_TOL="${TPSCHEM_CEPA_TOL:-1e-8}"
# :blocks refuses up front when the stored H will not fit, instead of silently
# dropping to :matrixfree, which does not finish at these dimensions.
export TPSCHEM_H_STORAGE="${TPSCHEM_H_STORAGE:-blocks}"
# Aggregate GB budget across ALL workers, not per worker.
export TPSCHEM_MAX_MEM_H="${TPSCHEM_MAX_MEM_H:-800}"

if [ "$#" -lt 2 ]; then
    echo "Usage: sbatch run_cepa_multinode_fe2s2.sh input_file.jl data_file.jld2 [extra files...]" >&2
    exit 2
fi

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
INPUT_FILE="$1"
DATA_FILE="$2"
shift 2

case "$INPUT_FILE" in
    *.jl) ;;
    *) echo "First argument must be a Julia input file ending in .jl: $INPUT_FILE" >&2; exit 2 ;;
esac

case "$INPUT_FILE" in
    /*) ;;
    *) INPUT_FILE="$WORKDIR/$INPUT_FILE" ;;
esac

case "$DATA_FILE" in
    /*) ;;
    *) DATA_FILE="$WORKDIR/$DATA_FILE" ;;
esac

TMPDIR="/N/scratch/abachhar/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

cp "$INPUT_FILE" "$TMPDIR/"
cp "$DATA_FILE" "$TMPDIR/"
for f in "$@"; do
    [ -n "$f" ] || continue
    case "$f" in
        /*) cp "$f" "$TMPDIR/" ;;
        *) cp "$WORKDIR/$f" "$TMPDIR/" ;;
    esac
done

INPUT_BASE="$(basename "$INPUT_FILE")"
DATA_BASE="$(basename "$DATA_FILE")"
OUTFILE="${INPUT_BASE}.out"

export TPSCHEM_MACHINE_FILE="$TMPDIR/nodes.${SLURM_JOB_ID}.txt"
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$TPSCHEM_MACHINE_FILE"

cd "$TMPDIR"

echo "Project: $JULIAENV"
echo "Depot:   $JULIA_DEPOT_PATH"
echo "Scratch: $TMPDIR"
echo "Master threads: $JULIA_NUM_THREADS"
echo "Worker threads: $TPSCHEM_WORKER_THREADS (master's node: $TPSCHEM_MASTER_NODE_WORKER_THREADS)"
echo "BLAS threads:   $TPSCHEM_BLAS_THREADS"
echo "Solver:         $TPSCHEM_SOLVER (linsolve_tol=$TPSCHEM_LINSOLVE_TOL)"
echo "H storage:      $TPSCHEM_H_STORAGE (max_mem_H=$TPSCHEM_MAX_MEM_H GB aggregate)"
echo "Worker timeout: $JULIA_WORKER_TIMEOUT s"
echo "Nodes:"
cat "$TPSCHEM_MACHINE_FILE"

JULIA_EXE="${JULIA_EXE:-julia}"
JULIA_VERSION_ARG="${JULIA_VERSION_ARG-+1.11}"
JULIA_RUN=("$JULIA_EXE")
if [ -n "$JULIA_VERSION_ARG" ]; then
    JULIA_RUN+=("$JULIA_VERSION_ARG")
fi

touch "$OUTFILE"
( while true; do
      rsync -a "$OUTFILE" "$WORKDIR/${INPUT_BASE}.${SLURM_JOB_ID}.out" || true
      sleep 60
  done ) &
RSYNC_PID=$!

cleanup() {
    kill "$RSYNC_PID" 2>/dev/null || true
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
    "$INPUT_BASE" "$DATA_BASE" > "$OUTFILE" 2>&1

kill "$RSYNC_PID" 2>/dev/null || true
trap - EXIT

cp "$OUTFILE" "$WORKDIR/${INPUT_BASE}.${SLURM_JOB_ID}.out"
rm -f "$WORKDIR/${INPUT_BASE}.${SLURM_JOB_ID}.running.out" "$TPSCHEM_MACHINE_FILE"

if [ "${TPSCHEM_SAVE_SCRATCH:-1}" = "1" ]; then
    SCRATCH_COPY="$WORKDIR/${INPUT_BASE}.${SLURM_JOB_ID}.scr"
    mkdir -p "$SCRATCH_COPY"
    cp -r . "$SCRATCH_COPY/"
fi

cd "$WORKDIR"
rm -rf "$TMPDIR"
