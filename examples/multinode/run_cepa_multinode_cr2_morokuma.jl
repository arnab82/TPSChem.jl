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

function init_multinode_workers!()
    active_project = Base.active_project()
    default_project = active_project === nothing ? pwd() : dirname(active_project)
    project = get(ENV, "JULIAENV", default_project)
    if nworkers() == 1 && haskey(ENV, "TPSCHEM_MACHINE_FILE")
        hosts = filter(!isempty, strip.(readlines(ENV["TPSCHEM_MACHINE_FILE"])))
        isempty(hosts) && error("TPSCHEM_MACHINE_FILE is empty")
        if env_bool("TPSCHEM_SKIP_MASTER_NODE", false) && length(hosts) > 1
            hosts = hosts[2:end]
        end
        threads = get(ENV, "JULIA_NUM_THREADS", string(Threads.nthreads()))
        addprocs(hosts; exeflags="--project=$project --threads=$threads",
                 env=worker_env())
    end

    @sync for pid in workers()
        @async remotecall_fetch(Core.eval, pid, Main,
            :(begin
                  import Pkg
                  Pkg.activate($project; io=devnull)
                  using TPSChem
              end))
    end

    pids = TPSChem.ensure_tpsci_multinode_workers!(workers=workers())
    @printf("Master pid %i, %i workers: %s\n", myid(), nworkers(), pids)
    @printf("Master active project: %s\n", Base.active_project())
    @printf("Worker active projects: %s\n",
            [remotecall_fetch(() -> Base.active_project(), pid) for pid in pids])
    @printf("Master threads: %i\n", Threads.nthreads())
    @printf("Worker thread counts: %s\n",
            [remotecall_fetch(Threads.nthreads, pid) for pid in pids])
    flush(stdout)
    return pids
end

const DATA_FILE = get(ARGS, 1, get(ENV, "TPSCHEM_INPUT_JLD2", "data_cmf_29_cr2.jld2"))
const VERBOSE = env_int("TPSCHEM_VERBOSE", 1)
const BLAS_THREADS = env_int("TPSCHEM_BLAS_THREADS", 1)
const M = env_int("TPSCHEM_CLUSTER_MAX_ROOTS", 100)
const CLUSTER_SPIN_ROOTS = env_int_vector("TPSCHEM_CLUSTER_SPIN_ROOTS", [3, 3, 3])
const INIT_FSPACE = TPSChem.FockConfig([(6, 3), (4, 4), (3, 6)])
const NROOTS = env_int("TPSCHEM_NROOTS", 4)
const THRESH_FOI_LIST = env_float_vector(
    "TPSCHEM_THRESH_FOI_LIST",
    [1e-4, 5e-5, 2e-5, 1e-5, 5e-6],
)

workers_ = init_multinode_workers!()

@printf("\n================ Cr2 Morokuma sharded CEPA ================\n")
@printf("Data file:            %s\n", DATA_FILE)
@printf("Cluster max roots:    %i\n", M)
@printf("Cluster spin roots:   %s\n", CLUSTER_SPIN_ROOTS)
@printf("Reference roots:      %i\n", NROOTS)
@printf("FOI thresholds:       %s\n", THRESH_FOI_LIST)
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

@printf("\nBuilding local cluster operators for the small reference solve...\n")
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

max_mem_H_default = 200.0 * length(workers_)
max_mem_H = env_float("TPSCHEM_MAX_MEM_H", max_mem_H_default)

for thresh_foi_cepa in THRESH_FOI_LIST
    @printf("\nRunning sharded CEPA-0 thresh_foi=%.3e ...\n", thresh_foi_cepa)
    GC.gc()
    result = @timed TPSChem.do_tps_sharded_cepa(
        ref_vec,
        cluster_ops,
        clustered_ham;
        e0=eci,
        cepa_shift=get(ENV, "TPSCHEM_CEPA_SHIFT", "cepa"),
        cepa_mit=env_int("TPSCHEM_CEPA_MIT", 30),
        thresh_foi=thresh_foi_cepa,
        nbody=env_int("TPSCHEM_NBODY", 4),
        tol=env_float("TPSCHEM_CEPA_TOL", 1e-8),
        thresh_sigma=env_float("TPSCHEM_THRESH_SIGMA", 1e-6),
        thresh_clip=env_float("TPSCHEM_THRESH_CLIP", 1e-5),
        compress=env_bool("TPSCHEM_COMPRESS_Q", false),
        solver=Symbol(get(ENV, "TPSCHEM_SOLVER", "minres")),
        h_storage=Symbol(get(ENV, "TPSCHEM_H_STORAGE", "auto")),
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
    flush(stdout)
    TPSChem.destroy!(qspace)
end

TPSChem.destroy!(cluster_ops)
