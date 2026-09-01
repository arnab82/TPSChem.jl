#!/bin/bash
#SBATCH -J tps-multinode
#SBATCH -p general
#SBATCH --mail-type=ALL
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=95:00:00
#SBATCH --mem=0
#SBATCH -A r01859

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: sbatch $0 input.jl data.jld2 [extra files...]" >&2
    exit 2
fi

module reset

export PATH="$HOME/.juliaup/bin:$PATH"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$HOME/.julia}"
export JULIAENV="${JULIAENV:-$SLURM_SUBMIT_DIR}"

# Master and worker threading.
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-32}"
export TPSCHEM_WORKER_THREADS="${TPSCHEM_WORKER_THREADS:-$SLURM_CPUS_PER_TASK}"

_shared=$((SLURM_CPUS_PER_TASK - JULIA_NUM_THREADS))
[ "$_shared" -lt 1 ] && _shared=1
export TPSCHEM_MASTER_NODE_WORKER_THREADS="${TPSCHEM_MASTER_NODE_WORKER_THREADS:-$_shared}"

export TPSCHEM_BLAS_THREADS="${TPSCHEM_BLAS_THREADS:-1}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ── CEPA pathway (full reference: examples/CEPA_JOB_KEYWORDS.md) ─────────────
# Only read by the CEPA drivers; harmless for any other input script. Defaults
# only -- set any of them before sbatch to override.
#
#   TPSCHEM_H_STORAGE   blocks | matrixfree | auto.
#       blocks      stored block-sparse H, one shard per worker. Milliseconds
#                   per apply. Runs a feasibility check and refuses up front
#                   rather than OOM-killing the allocation hours in.
#       matrixfree  nothing stored, H re-contracted per apply. Seconds per
#                   apply, but no memory wall.
#       auto        picks by the TPSCHEM_MAX_MEM_H budget and announces a
#                   downgrade to matrixfree if the stored H will not fit.
#   TPSCHEM_MAX_MEM_H   GB per worker that auto may spend on the stored H.
#   TPSCHEM_SOLVER      pcg | minres. pcg needs a stored H for its diagonal, so
#                   with matrixfree it announces a fall back to minres.
#   TPSCHEM_BLOCK_ROOTS true solves all roots in one pass, sharing one apply and
#                   one set of fan-outs -- measured ~3x on matrixfree, roughly a
#                   wash on blocks where the run is dominated by the H build.
#   TPSCHEM_LINSOLVE_TOL  inner Krylov tolerance. Worth loosening relative to
#                   TPSCHEM_CEPA_TOL: solving to 1e-8 while an ACPF/AQCC shift
#                   is still moving is wasted work.
#   TPSCHEM_MINRES_MAXITER  Krylov iterations per solve.
export TPSCHEM_H_STORAGE="${TPSCHEM_H_STORAGE:-blocks}"
export TPSCHEM_MAX_MEM_H="${TPSCHEM_MAX_MEM_H:-200}"
export TPSCHEM_SOLVER="${TPSCHEM_SOLVER:-pcg}"
export TPSCHEM_BLOCK_ROOTS="${TPSCHEM_BLOCK_ROOTS:-true}"
export TPSCHEM_LINSOLVE_TOL="${TPSCHEM_LINSOLVE_TOL:-1e-6}"
export TPSCHEM_MINRES_MAXITER="${TPSCHEM_MINRES_MAXITER:-300}"
export TPSCHEM_CEPA_SHIFT="${TPSCHEM_CEPA_SHIFT:-aqcc}"
export TPSCHEM_CEPA_MIT="${TPSCHEM_CEPA_MIT:-30}"

export JULIA_WORKER_TIMEOUT="${JULIA_WORKER_TIMEOUT:-250}"
export TPSCHEM_ADDPROCS_ATTEMPTS="${TPSCHEM_ADDPROCS_ATTEMPTS:-3}"
export TPSCHEM_ADDPROCS_BACKOFF="${TPSCHEM_ADDPROCS_BACKOFF:-20}"

# One distributed worker per allocated node.
export TPSCHEM_MIN_WORKERS="${TPSCHEM_MIN_WORKERS:-$SLURM_JOB_NUM_NODES}"

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
INPUT_FILE="$1"
DATA_FILE="$2"
shift 2

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

export TPSCHEM_INPUT_SCRIPT="$JOB_SCRATCH/$INPUT_BASE"

JULIA_EXE="${JULIA_EXE:-julia}"
JULIA_VERSION_ARG="${JULIA_VERSION_ARG-+1.11}"

JULIA_RUN=("$JULIA_EXE")
if [ -n "$JULIA_VERSION_ARG" ]; then
    JULIA_RUN+=("$JULIA_VERSION_ARG")
fi

echo "Job ID:                 $SLURM_JOB_ID"
echo "Project:                $JULIAENV"
echo "Depot:                  $JULIA_DEPOT_PATH"
echo "Scratch:                $JOB_SCRATCH"
echo "Master threads:         $JULIA_NUM_THREADS"
echo "Worker-only threads:    $TPSCHEM_WORKER_THREADS"
echo "Master-node worker:     $TPSCHEM_MASTER_NODE_WORKER_THREADS"
echo "Minimum workers:        $TPSCHEM_MIN_WORKERS"
echo "BLAS threads:           $TPSCHEM_BLAS_THREADS"
echo "CEPA storage:           $TPSCHEM_H_STORAGE (budget ${TPSCHEM_MAX_MEM_H} GB/worker)"
echo "CEPA solver:            $TPSCHEM_SOLVER  block_roots=$TPSCHEM_BLOCK_ROOTS"
echo "CEPA shift:             $TPSCHEM_CEPA_SHIFT  cepa_mit=$TPSCHEM_CEPA_MIT"
echo "Krylov:                 maxiter=$TPSCHEM_MINRES_MAXITER  linsolve_tol=$TPSCHEM_LINSOLVE_TOL"
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

# Create a bootstrap that starts one correctly threaded worker per node.
BOOTSTRAP="$JOB_SCRATCH/tpschem_worker_bootstrap.jl"

cat > "$BOOTSTRAP" <<'JULIA_BOOTSTRAP'
using Distributed
using Sockets
using Printf

function required_env_int(name)
    haskey(ENV, name) || error("Missing required environment variable $name")
    value = parse(Int, ENV[name])
    value > 0 || error("$name must be positive")
    return value
end

shortname(host) = String(first(split(host, '.')))

machine_file = ENV["TPSCHEM_MACHINE_FILE"]
project = ENV["JULIAENV"]
input_script = ENV["TPSCHEM_INPUT_SCRIPT"]

hosts = filter(!isempty, strip.(readlines(machine_file)))
isempty(hosts) && error("No hosts found in $machine_file")

full_threads = required_env_int("TPSCHEM_WORKER_THREADS")
shared_threads = required_env_int("TPSCHEM_MASTER_NODE_WORKER_THREADS")
minimum_workers = required_env_int("TPSCHEM_MIN_WORKERS")
attempts = required_env_int("TPSCHEM_ADDPROCS_ATTEMPTS")
backoff = required_env_int("TPSCHEM_ADDPROCS_BACKOFF")

master_host = shortname(gethostname())
joined = Int[]

println("Master host: $master_host")
println("Launching $(length(hosts)) distributed workers...")

for host in hosts
    threads = shortname(host) == master_host ? shared_threads : full_threads

    host_env = Pair{String,String}[
        "PATH" => ENV["PATH"],
        "JULIA_DEPOT_PATH" => ENV["JULIA_DEPOT_PATH"],
        "JULIA_NUM_THREADS" => string(threads),
        "OPENBLAS_NUM_THREADS" => "1",
        "OMP_NUM_THREADS" => "1",
        "MKL_NUM_THREADS" => "1",
    ]

    connected = false

    for attempt in 1:attempts
        try
            @printf(
                "  launching %-16s with %i threads, attempt %i/%i\n",
                host,
                threads,
                attempt,
                attempts,
            )
            flush(stdout)

            new_workers = addprocs(
                [host];
                exeflags=`--project=$project --threads=$threads`,
                env=host_env,
            )

            append!(joined, new_workers)
            connected = true
            break
        catch err
            @warn "Worker launch failed" host threads attempt exception=err

            if attempt < attempts
                sleep(backoff)
            end
        end
    end

    connected || @warn "Skipping host after failed launch attempts" host
end

length(joined) >= minimum_workers ||
    error("Only $(length(joined)) workers joined; minimum is $minimum_workers")

println()
println("Master pid $(myid()), workers: $(workers())")
println("Master threads: $(Threads.nthreads())")

for pid in workers()
    info = remotecall_fetch(pid) do
        (
            host = gethostname(),
            threads = Threads.nthreads(),
            env_threads = get(ENV, "JULIA_NUM_THREADS", "unset"),
            project = Base.active_project(),
        )
    end

    println("Worker $pid: $info")
end

flush(stdout)

# The input sees workers already present, so a guarded worker initializer should
# not launch a second set. ARGS contains the data filename passed to bootstrap.
include(input_script)
JULIA_BOOTSTRAP

if [ -n "${TPSCHEM_DEVELOP_PATH:-}" ]; then
    "${JULIA_RUN[@]}" --project="$JULIAENV" -e \
        'import Pkg
         Pkg.develop(path=ENV["TPSCHEM_DEVELOP_PATH"])
         Pkg.instantiate()
         Pkg.precompile()
         Pkg.status()'
else
    "${JULIA_RUN[@]}" --project="$JULIAENV" -e \
        'import Pkg
         Pkg.instantiate()
         Pkg.precompile()
         Pkg.status()'
fi

"${JULIA_RUN[@]}" \
    --project="$JULIAENV" \
    --threads="$JULIA_NUM_THREADS" \
    "$BOOTSTRAP" "$DATA_BASE" \
    > "$SCRATCH_OUTPUT" 2>&1

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