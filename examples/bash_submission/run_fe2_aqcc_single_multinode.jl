using Distributed
using LinearAlgebra
using Printf
using JLD2
using TPSChem
using TPSChem.QCBase
using TPSChem.RDM

function env_bool(name::AbstractString, default::Bool)
    value = lowercase(strip(get(ENV, name, string(default))))
    return value in ("1", "true", "yes", "y", "on")
end

function env_int(name::AbstractString, default::Integer)
    return parse(Int, get(ENV, name, string(default)))
end

function env_float(name::AbstractString, default::Real)
    return parse(Float64, get(ENV, name, string(default)))
end

function env_float_vector(name::AbstractString, default::Vector{Float64})
    haskey(ENV, name) || return default
    return parse.(Float64, split(ENV[name], ","))
end

function env_int_vector(name::AbstractString, default::Vector{Int})
    haskey(ENV, name) || return default
    return parse.(Int, split(ENV[name], ","))
end

function worker_env()
    env = Pair{String,String}[]
    for name in ("JULIA_DEPOT_PATH", "PATH", "OPENBLAS_NUM_THREADS",
                 "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        haskey(ENV, name) && push!(env, name => ENV[name])
    end
    return env
end

# Worker launch is the one step that can waste a multi-hour queue wait: under
# plain addprocs a single host that misses JULIA_WORKER_TIMEOUT throws, and the
# whole allocation dies seconds after it finally started. So retry each host,
# and if one still refuses, run on the workers that did come up. Sharding is
# `mod` over the pid list, so any worker count is valid -- fewer nodes is slower,
# which is strictly better than losing the allocation.
function addprocs_resilient(host, threads, project; attempts, backoff)
    for attempt in 1:attempts
        t0 = time()
        try
            pids = addprocs([host]; exeflags="--project=$project --threads=$threads",
                            env=worker_env())
            # Report each host the moment it joins: nothing else prints until
            # every host is up, so without this a stalling node is invisible.
            @printf("  joined %-14s pid %-6s %6.1f s%s\n", host, join(pids, ","),
                    time() - t0, attempt > 1 ? "  (attempt $attempt)" : "")
            flush(stdout)
            return pids
        catch err
            @warn "addprocs failed" host attempt attempts error=sprint(showerror, err)
            flush(stderr)
            attempt < attempts && sleep(backoff * attempt)
        end
    end
    @warn "host never joined; continuing without it" host
    flush(stderr)
    return Int[]
end

function init_multinode_workers!()
    active_project = Base.active_project()
    default_project = active_project === nothing ? pwd() : dirname(active_project)
    project = get(ENV, "JULIAENV", default_project)
    requested = 0
    if nworkers() == 1 && haskey(ENV, "TPSCHEM_MACHINE_FILE")
        hosts = filter(!isempty, strip.(readlines(ENV["TPSCHEM_MACHINE_FILE"])))
        isempty(hosts) && error("TPSCHEM_MACHINE_FILE is empty")
        if env_bool("TPSCHEM_SKIP_MASTER_NODE", false) && length(hosts) > 1
            hosts = hosts[2:end]
        end
        # Per-host thread counts. addprocs applies one exeflags to every host, so
        # spawn host by host: the node that also carries the master gets fewer
        # worker threads (TPSCHEM_MASTER_NODE_WORKER_THREADS) so the pair fits in
        # the cores SLURM granted, while worker-only nodes get the full count.
        full_threads = get(ENV, "TPSCHEM_WORKER_THREADS",
                           get(ENV, "JULIA_NUM_THREADS", string(Threads.nthreads())))
        shared_threads = get(ENV, "TPSCHEM_MASTER_NODE_WORKER_THREADS", full_threads)
        shortname(h) = String(first(split(h, '.')))
        master_host = shortname(gethostname())
        attempts = env_int("TPSCHEM_ADDPROCS_ATTEMPTS", 3)
        backoff = env_int("TPSCHEM_ADDPROCS_BACKOFF", 20)
        requested = length(hosts)
        @printf("Launching %i workers (%s s timeout, %i attempts each)...\n",
                requested, get(ENV, "JULIA_WORKER_TIMEOUT", "60"), attempts)
        flush(stdout)
        for host in hosts
            t = shortname(host) == master_host ? shared_threads : full_threads
            addprocs_resilient(host, t, project; attempts=attempts, backoff=backoff)
        end
    end

    # Load TPSChem everywhere, dropping any worker that cannot manage it instead
    # of letting one bad host abort the run. Results go into a preallocated
    # vector rather than a shared push!, so the @async tasks cannot race.
    pool = workers()
    ok = fill(false, length(pool))
    @sync for (i, pid) in enumerate(pool)
        @async begin
            try
                remotecall_fetch(Core.eval, pid, Main,
                    :(begin
                          import Pkg
                          Pkg.activate($project; io=devnull)
                          using TPSChem
                      end))
                ok[i] = true
            catch err
                @warn "worker failed to initialise; dropping" pid error=sprint(showerror, err)
                flush(stderr)
            end
        end
    end
    healthy = pool[ok]
    dead = pool[.!ok]
    isempty(dead) || rmprocs(dead; waitfor=10)

    min_workers = env_int("TPSCHEM_MIN_WORKERS", 1)
    if length(healthy) < min_workers
        error("only $(length(healthy)) worker(s) joined, need at least $min_workers " *
              "(set TPSCHEM_MIN_WORKERS to lower the floor)")
    end

    pids = TPSChem.ensure_tpsci_multinode_workers!(workers=healthy)
    if requested > 0 && length(pids) < requested
        @printf("WARNING: running degraded -- %i of %i hosts joined\n", length(pids), requested)
    end
    @printf("Master pid %i, %i workers: %s\n", myid(), length(pids), pids)
    @printf("Master active project: %s\n", Base.active_project())
    @printf("Worker active projects: %s\n",
            [remotecall_fetch(() -> Base.active_project(), pid) for pid in pids])
    @printf("Master threads: %i\n", Threads.nthreads())
    @printf("Worker thread counts: %s\n",
            [remotecall_fetch(Threads.nthreads, pid) for pid in pids])
    flush(stdout)
    return pids
end

const DATA_FILE = get(ARGS, 1, get(ENV, "TPSCHEM_INPUT_JLD2", "data_fe2_morokuma_29.jld2"))
const VERBOSE = env_int("TPSCHEM_VERBOSE", 4)
const BLAS_THREADS = env_int("TPSCHEM_BLAS_THREADS", 1)
const M = env_int("TPSCHEM_CLUSTER_MAX_ROOTS", 200)
const CLUSTER_SPIN_ROOTS = env_int_vector("TPSCHEM_CLUSTER_SPIN_ROOTS", [3, 3,3])
const INIT_FSPACE = TPSChem.FockConfig([(6, 1), (4, 4), (1, 6)])
const NROOTS = env_int("TPSCHEM_NROOTS", 6)
const THRESH_FOI_LIST = env_float_vector(
    "TPSCHEM_THRESH_FOI_LIST",
    [8e-5,4e-5],
)
const CEPA_SHIFT = get(ENV, "TPSCHEM_CEPA_SHIFT", "aqcc")
# Solve all roots in one pass, sharing one Hamiltonian apply and one set of
# fan-outs, rather than one pass per root. Measured ~3x end to end on
# h_storage=:matrixfree; roughly a wash on :blocks, where a stored apply is
# milliseconds and the run is dominated by the one-time H build.
const BLOCK_ROOTS = env_bool("TPSCHEM_BLOCK_ROOTS", true)

workers_ = init_multinode_workers!()

@printf("\n================ Fe2 morokuma sharded CEPA ================\n")
@printf("Data file:            %s\n", DATA_FILE)
@printf("Cluster max roots:    %i\n", M)
@printf("Cluster spin roots:   %s\n", CLUSTER_SPIN_ROOTS)
@printf("Reference roots:      %i\n", NROOTS)
@printf("FOI thresholds:       %s\n", THRESH_FOI_LIST)
@printf("CEPA variant:         %s\n", CEPA_SHIFT)
@printf("Roots:                %s\n",
        BLOCK_ROOTS ? "all $(NROOTS) in one solver pass" : "one pass per root")
flush(stdout)

data = JLD2.load(DATA_FILE)
ints = data["ints"]
clusters = data["clusters"]
d1 = data["d1"]

cluster_bases = TPSChem.compute_cluster_eigenbasis_spin(
    ints, clusters, d1, CLUSTER_SPIN_ROOTS, INIT_FSPACE;
    max_roots=M,
    verbose=VERBOSE,
)
clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)

@printf("\nBuilding local cluster operators \n")
flush(stdout)
cluster_ops_ref = TPSChem.compute_cluster_ops(cluster_bases, ints)
TPSChem.add_cmf_operators!(cluster_ops_ref, cluster_bases, ints, d1.a, d1.b)

ci_vector = TPSChem.TPSCIstate(clusters, INIT_FSPACE, R=NROOTS)
ci_vector = TPSChem.add_spin_focksectors(ci_vector)

eci, ref_vec, _ = TPSChem.tps_ci_direct(
    ci_vector,
    cluster_ops_ref,
    clustered_ham;
    conv_thresh=env_float("TPSCHEM_REF_TOL", 1e-8),
    verbose=VERBOSE,
)

cluster_ops_ref = nothing
GC.gc()

@printf("\nBuilding distributed cluster operators for sharded CEPA...\n")
flush(stdout)
cluster_ops = TPSChem.compute_cluster_ops_distributed(
    cluster_bases,
    ints;
    workers=workers_,
    verbose=VERBOSE,
    blas_threads=BLAS_THREADS,
)
flush(stdout)
TPSChem.add_cmf_operators_distributed!(
    cluster_ops,
    cluster_bases,
    ints,
    d1.a,
    d1.b;
    verbose=0,
    blas_threads=BLAS_THREADS,
)
@printf("Distributed cluster ops summary: %s\n", TPSChem.cluster_ops_summary(cluster_ops))
flush(stdout)

# The bases and integrals now live on the workers. Anything still referenced
# here is charged against the budget of whichever node the master shares with a
# worker, which is exactly the memory the stored H needs.
cluster_bases = nothing
ints = nothing
d1 = nothing
data = nothing
GC.gc()

# Budget per worker and scale by the workers that actually joined: a fixed
# aggregate would try to store the same H on fewer nodes after a degraded start.
per_worker_mem_H = env_float("TPSCHEM_MAX_MEM_H_PER_WORKER", 200.0)
max_mem_H = env_float("TPSCHEM_MAX_MEM_H", per_worker_mem_H * length(workers_))
@printf("Stored-H budget: %.0f GB aggregate over %i workers\n",
        max_mem_H, length(workers_))
flush(stdout)

for thresh_foi_cepa in THRESH_FOI_LIST
    @printf("\nRunning sharded CEPA (%s) thresh_foi=%.3e ...\n", CEPA_SHIFT, thresh_foi_cepa)
    GC.gc()
    try
        result = @timed TPSChem.do_tps_sharded_cepa(
            ref_vec,
            cluster_ops,
            clustered_ham;
            e0=eci,
            cepa_shift=CEPA_SHIFT,
            cepa_mit=env_int("TPSCHEM_CEPA_MIT", 30),
            thresh_foi=thresh_foi_cepa,
            nbody=env_int("TPSCHEM_NBODY", 4),
            tol=env_float("TPSCHEM_CEPA_TOL", 1e-8),
            linsolve_tol=env_float("TPSCHEM_LINSOLVE_TOL", 1e-6),
            thresh_sigma=env_float("TPSCHEM_THRESH_SIGMA", 1e-6),
            thresh_clip=env_float("TPSCHEM_THRESH_CLIP", 1e-5),
            compress=env_bool("TPSCHEM_COMPRESS_Q", false),
            solver=Symbol(get(ENV, "TPSCHEM_SOLVER", "pcg")),
            h_storage=Symbol(get(ENV, "TPSCHEM_H_STORAGE", "blocks")),
            block_roots=BLOCK_ROOTS,
            max_mem_H=max_mem_H,
            cg_maxiter=env_int("TPSCHEM_MINRES_MAXITER", 300),
            workers=workers_,
            threaded_worker=true,
            blas_threads=BLAS_THREADS,
            verbose=VERBOSE,
        )
        e_cepa, qspace = result.value
        @printf("thresh=%.0e  E(cepa) = %s   (%.1f s)\n",
                thresh_foi_cepa, string(e_cepa), result.time)
        TPSChem.destroy!(qspace)
    catch err
        # A threshold whose stored H does not fit should not end the sweep:
        # report the deficit the feasibility check printed and carry on.
        @printf("thresh=%.0e  SKIPPED: %s\n",
                thresh_foi_cepa, sprint(showerror, err))
    end
    flush(stdout)
end

TPSChem.destroy!(cluster_ops)
