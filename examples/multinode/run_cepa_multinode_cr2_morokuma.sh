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
#   sbatch run_cepa_multinode_cr2_morokuma.sh data_cmf_29_cr2.jld2 [extra files...]
#
# Keep run_cepa_multinode_cr2_morokuma.jl in the same directory as this script.
# Set JULIAENV to the shared TPSChem project directory if it differs from below.

module reset

export PATH="$HOME/.juliaup/bin:$PATH"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/N/u/abachhar/BigRed200/.julia}"
export JULIAENV="${JULIAENV:-/N/u/abachhar/BigRed200/multinode-tpsci}"

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TPSCHEM_BLAS_THREADS="${TPSCHEM_BLAS_THREADS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="${TPSCHEM_DRIVER:-$SCRIPT_DIR/run_cepa_multinode_cr2_morokuma.jl}"
DATA_FILE="${1:-data_cmf_29_cr2.jld2}"
shift || true

WORKDIR="$(pwd)"
TMPDIR="/N/scratch/abachhar/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

cp "$DRIVER" "$TMPDIR/"
cp "$DATA_FILE" "$TMPDIR/"
for f in "$@"; do
    [ -n "$f" ] && cp "$f" "$TMPDIR/"
done

DRIVER_BASE="$(basename "$DRIVER")"
DATA_BASE="$(basename "$DATA_FILE")"
OUTFILE="${DRIVER_BASE}.out"

export TPSCHEM_MACHINE_FILE="$TMPDIR/nodes.${SLURM_JOB_ID}.txt"
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$TPSCHEM_MACHINE_FILE"

cd "$TMPDIR"

echo "Project: $JULIAENV"
echo "Depot:   $JULIA_DEPOT_PATH"
echo "Scratch: $TMPDIR"
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
      rsync -a "$OUTFILE" "$WORKDIR/${DRIVER_BASE}.${SLURM_JOB_ID}.out" || true
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
    "$DRIVER_BASE" "$DATA_BASE" > "$OUTFILE" 2>&1

kill "$RSYNC_PID" 2>/dev/null || true
trap - EXIT

cp "$OUTFILE" "$WORKDIR/${DRIVER_BASE}.${SLURM_JOB_ID}.out"
rm -f "$WORKDIR/${DRIVER_BASE}.${SLURM_JOB_ID}.running.out" "$TPSCHEM_MACHINE_FILE"

if [ "${TPSCHEM_SAVE_SCRATCH:-1}" = "1" ]; then
    SCRATCH_COPY="$WORKDIR/${DRIVER_BASE}.${SLURM_JOB_ID}.scr"
    mkdir -p "$SCRATCH_COPY"
    cp -r . "$SCRATCH_COPY/"
fi

cd "$WORKDIR"
rm -rf "$TMPDIR"
