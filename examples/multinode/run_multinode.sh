#!/bin/bash
#SBATCH -J tps-cepa
#SBATCH -p general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abachhar@iu.edu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=15:00:00
#SBATCH --mem=0
#SBATCH -A r01859

set -euo pipefail

# Usage:
#   sbatch run_multinode.sh input_file.jl data_file.jld2 [extra files...]
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
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-32}"
export TPSCHEM_WORKER_THREADS="${TPSCHEM_WORKER_THREADS:-$SLURM_CPUS_PER_TASK}"
# The node that also runs the master gets the cores the master is not using, so
# the two fit in --cpus-per-task; worker-only nodes get the full allocation.
_shared=$(( SLURM_CPUS_PER_TASK - JULIA_NUM_THREADS ))
[ "$_shared" -lt 1 ] && _shared=1
export TPSCHEM_MASTER_NODE_WORKER_THREADS="${TPSCHEM_MASTER_NODE_WORKER_THREADS:-$_shared}"
export TPSCHEM_BLAS_THREADS="${TPSCHEM_BLAS_THREADS:-1}"

if [ "$#" -lt 2 ]; then
    echo "Usage: sbatch run_multinode.sh input_file.jl data_file.jld2 [extra files...]" >&2
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
echo "Nodes:"
cat "$TPSCHEM_MACHINE_FILE"

# Precompilation forks a Julia process per package, each inheriting
# JULIA_NUM_THREADS; that is how this step gets OOM-killed on a fat node.
export JULIA_NUM_PRECOMPILE_TASKS="${JULIA_NUM_PRECOMPILE_TASKS:-4}"

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
    JULIA_NUM_THREADS=1 "${JULIA_RUN[@]}" --project="$JULIAENV" --threads=1 -e \
        'import Pkg; Pkg.develop(path=ENV["TPSCHEM_DEVELOP_PATH"]); Pkg.instantiate(); Pkg.precompile(); Pkg.status()'
else
    JULIA_NUM_THREADS=1 "${JULIA_RUN[@]}" --project="$JULIAENV" --threads=1 -e \
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
