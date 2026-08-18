using Distributed
using LinearAlgebra
using Printf
using JLD2
using TPSChem

const PROJECT_ROOT = dirname(dirname(@__DIR__))

function project_root()
    return get(ENV, "JULIAENV", PROJECT_ROOT)
end

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

function env_vector(name::AbstractString)
    haskey(ENV, name) || return nothing
    return parse.(Float64, split(ENV[name], ","))
end

function input_data_path()
    if haskey(ENV, "TPSCHEM_INPUT_JLD2")
        return ENV["TPSCHEM_INPUT_JLD2"]
    elseif length(ARGS) >= 1
        ENV["TPSCHEM_INPUT_JLD2"] = ARGS[1]
        return ARGS[1]
    else
        error("Set TPSCHEM_INPUT_JLD2 or pass the input JLD2 as ARGS[1]")
    end
end

function init_multinode_workers!()
    if nworkers() == 1 && haskey(ENV, "TPSCHEM_MACHINE_FILE")
        hosts = strip.(readlines(ENV["TPSCHEM_MACHINE_FILE"]))
        filter!(!isempty, hosts)
        isempty(hosts) && error("TPSCHEM_MACHINE_FILE is empty")
        if env_bool("TPSCHEM_SKIP_MASTER_NODE", false) && length(hosts) > 1
            hosts = hosts[2:end]
        end
        # The launcher scripts export TPSCHEM_WORKER_THREADS for the workers and
        # JULIA_NUM_THREADS for the master. Reading only the latter silently gives
        # every worker the master's (usually smaller) thread count.
        threads = get(ENV, "TPSCHEM_WORKER_THREADS",
                      get(ENV, "JULIA_NUM_THREADS", string(Threads.nthreads())))
        worker_env = Pair{String,String}[]
        for name in ("JULIA_DEPOT_PATH", "PATH", "OPENBLAS_NUM_THREADS",
                     "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            haskey(ENV, name) && push!(worker_env, name => ENV[name])
        end
        addprocs(hosts; exeflags="--project=$(project_root()) --threads=$(threads)",
                 env=worker_env)
    end
    @sync for pid in workers()
        pid == myid() && continue
        @async remotecall_fetch(Core.eval, pid, Main,
            :(begin
                  import Pkg
                  Pkg.activate($(project_root()); io=devnull)
                  using TPSChem
              end))
    end
    pids = TPSChem.ensure_tpsci_multinode_workers!(workers=workers())
    @printf("Master process: %i\n", myid())
    @printf("Worker processes: %s\n", pids)
    @printf("Master threads: %i\n", Threads.nthreads())
    @printf("Worker thread counts: %s\n",
            [remotecall_fetch(Threads.nthreads, pid) for pid in pids])
    flush(stdout)
    return pids
end

function load_problem_data()
    data = JLD2.load(input_data_path())
    for key in ("ints", "clusters", "d1", "init_fspace", "cluster_bases")
        haskey(data, key) || error("Input file is missing key '$key'")
    end
    return data
end

function get_clustered_ham(data)
    return haskey(data, "clustered_ham") ?
        data["clustered_ham"] :
        TPSChem.extract_ClusteredTerms(data["ints"], data["clusters"])
end

function build_distributed_cluster_ops(data, workers)
    verbose = env_int("TPSCHEM_VERBOSE", 1)
    blas_threads = env_int("TPSCHEM_BLAS_THREADS", 1)
    dops = TPSChem.compute_cluster_ops_distributed(
        data["cluster_bases"], data["ints"];
        workers=workers,
        verbose=verbose,
        blas_threads=blas_threads)
    TPSChem.add_cmf_operators_distributed!(
        dops, data["cluster_bases"], data["ints"], data["d1"].a, data["d1"].b;
        verbose=0,
        blas_threads=blas_threads)
    @printf("Distributed cluster ops summary: %s\n", TPSChem.cluster_ops_summary(dops))
    flush(stdout)
    return dops
end

function get_reference_state(data; key_default="ref_vec")
    key = get(ENV, "TPSCHEM_REF_KEY", key_default)
    if haskey(data, key)
        return data[key]
    elseif haskey(data, "ci_vector")
        return data["ci_vector"]
    end

    nroots = env_int("TPSCHEM_NROOTS", 1)
    ref_fock = TPSChem.FockConfig(data["init_fspace"])
    ref = TPSChem.TPSCIstate(data["clusters"], ref_fock, R=nroots, T=Float64)
    ref[ref_fock][TPSChem.ClusterConfig(fill(1, length(data["clusters"])))] .= 0.0
    ref[ref_fock][TPSChem.ClusterConfig(fill(1, length(data["clusters"])))][1] = 1.0
    return ref
end

function get_e0(data)
    from_env = env_vector("TPSCHEM_E0")
    from_env === nothing || return from_env
    haskey(data, "e0") && return vec(data["e0"])
    haskey(data, "E0") && return vec(data["E0"])
    return nothing
end
