# =============================================================================
# Never-gather multinode TPSCI on the Cr2 (Morokuma) trimer.
#
# Concrete companion to the env-driven `tpsci_ci_sharded_driver.jl`, in the same
# spirit as `run_cepa_multinode_cr2_morokuma.jl`. It follows the single-node
# workflow of `examples/notes/tpsci.jl` / `examples/bimetallics` — build the
# spin cluster bases, define the reference Fock sectors — but runs the selected
# CI with `tpsci_ci_sharded`, whose variational vector stays a
# `DistributedTPSCIstate` across all workers for the whole loop (never gathered).
#
# Launch (one Julia worker per node, many threads per worker):
#   sbatch run_multinode.sh run_tpsci_multinode_cr2_morokuma.jl \
#          ../../test/data_cmf_13_cr2_morokuma.jld2
# or locally with 2 workers:
#   julia -p 2 --project run_tpsci_multinode_cr2_morokuma.jl \
#          ../../test/data_cmf_13_cr2_morokuma.jld2
# =============================================================================

using Distributed
using LinearAlgebra
using Printf
using JLD2
using TPSChem
using TPSChem.QCBase
using TPSChem.RDM

function env_bool(name::AbstractString, default::Bool)
    return lowercase(strip(get(ENV, name, string(default)))) in ("1", "true", "yes", "y", "on")
end
env_int(name, default) = parse(Int, get(ENV, name, string(default)))
env_float(name, default) = parse(Float64, get(ENV, name, string(default)))
function env_int_vector(name, default::Vector{Int})
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
        env_bool("TPSCHEM_SKIP_MASTER_NODE", false) && length(hosts) > 1 && (hosts = hosts[2:end])
        # Per-host thread counts. addprocs applies one exeflags to every host, so
        # spawn host by host: the node that also carries the master gets fewer
        # worker threads (TPSCHEM_MASTER_NODE_WORKER_THREADS) so the pair fits in
        # the cores SLURM granted, while worker-only nodes get the full count.
        full_threads = get(ENV, "TPSCHEM_WORKER_THREADS",
                           get(ENV, "JULIA_NUM_THREADS", string(Threads.nthreads())))
        shared_threads = get(ENV, "TPSCHEM_MASTER_NODE_WORKER_THREADS", full_threads)
        shortname(h) = String(first(split(h, '.')))
        master_host = shortname(gethostname())
        for host in hosts
            t = shortname(host) == master_host ? shared_threads : full_threads
            addprocs([host]; exeflags="--project=$project --threads=$t",
                     env=worker_env())
        end
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
    @printf("Master threads: %i   Worker threads: %s\n", Threads.nthreads(),
            [remotecall_fetch(Threads.nthreads, pid) for pid in pids])
    flush(stdout)
    return pids
end

# --- problem / method configuration (cr2 Morokuma trimer defaults) -----------
const DATA_FILE     = get(ARGS, 1, get(ENV, "TPSCHEM_INPUT_JLD2", "data_cmf_13_cr2_morokuma.jld2"))
const VERBOSE       = env_int("TPSCHEM_VERBOSE", 1)
const BLAS_THREADS  = env_int("TPSCHEM_BLAS_THREADS", 1)
const M             = env_int("TPSCHEM_CLUSTER_MAX_ROOTS", 20)
const SPIN_ROOTS    = env_int_vector("TPSCHEM_CLUSTER_SPIN_ROOTS", [3, 3, 3])
const INIT_FSPACE   = TPSChem.FockConfig([(3, 0), (3, 3), (0, 3)])
const NROOTS        = env_int("TPSCHEM_NROOTS", 4)
const THRESH_CIPSI  = env_float("TPSCHEM_THRESH_CIPSI", 1e-3)
const THRESH_FOI    = env_float("TPSCHEM_THRESH_FOI", 1e-5)
const CONV_THRESH   = env_float("TPSCHEM_CONV_THRESH", 1e-4)
const MAX_ITER      = env_int("TPSCHEM_MAX_ITER", 10)
const NBODY         = env_int("TPSCHEM_NBODY", 4)
const H_STORAGE     = Symbol(get(ENV, "TPSCHEM_H_STORAGE", "auto"))     # auto | blocks | matrixfree

workers_ = init_multinode_workers!()

@printf("\n================ Cr2 Morokuma never-gather TPSCI ================\n")
@printf("Data file:        %s\n", DATA_FILE)
@printf("Cluster max roots:%i   spin roots: %s\n", M, SPIN_ROOTS)
@printf("Reference roots:  %i\n", NROOTS)
@printf("thresh_cipsi:     %.2e   thresh_foi: %.2e\n", THRESH_CIPSI, THRESH_FOI)
@printf("h_storage:        %s\n", H_STORAGE)
flush(stdout)

data = JLD2.load(DATA_FILE)
ints, clusters, d1 = data["ints"], data["clusters"], data["d1"]

cluster_bases = TPSChem.compute_cluster_eigenbasis_spin(
    ints, clusters, d1, SPIN_ROOTS, INIT_FSPACE; max_roots=M, verbose=VERBOSE)
clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)

# Starting reference: the CMF configuration plus its spin-coupled Fock sectors.
# tpsci_ci_sharded diagonalizes and grows this, keeping the vector sharded.
ci_vector = TPSChem.TPSCIstate(clusters, INIT_FSPACE, R=NROOTS)
ci_vector = TPSChem.add_spin_focksectors(ci_vector)
TPSChem.eye!(ci_vector)

# Shard the cluster operators too, so nothing scales with node count on the master.
@printf("\nBuilding distributed cluster operators...\n"); flush(stdout)
cluster_ops = TPSChem.compute_cluster_ops_distributed(
    cluster_bases, ints; workers=workers_, verbose=VERBOSE, blas_threads=BLAS_THREADS)
TPSChem.add_cmf_operators_distributed!(
    cluster_ops, cluster_bases, ints, d1.a, d1.b; verbose=0, blas_threads=BLAS_THREADS)
@printf("Distributed cluster ops summary: %s\n", TPSChem.cluster_ops_summary(cluster_ops))
flush(stdout)

max_mem_H = env_float("TPSCHEM_MAX_MEM_H", 200.0 * length(workers_))

e0, vec_var = TPSChem.tpsci_ci_sharded(
    ci_vector, cluster_ops, clustered_ham;
    thresh_cipsi=THRESH_CIPSI,
    thresh_foi=THRESH_FOI,
    conv_thresh=CONV_THRESH,
    max_iter=MAX_ITER,
    nbody=NBODY,
    ci_conv=env_float("TPSCHEM_CI_CONV", 1e-8),
    ci_max_ss_vecs=env_int("TPSCHEM_MAX_SS_VECS", 4),
    h_storage=H_STORAGE,
    max_mem_H=max_mem_H,
    compute_s2=env_bool("TPSCHEM_COMPUTE_S2", true),
    workers=workers_,
    threaded_worker=true,
    blas_threads=BLAS_THREADS,
    verbose=VERBOSE)

@printf("\nFinal variational energies (dim=%i):\n", length(vec_var))
for r in 1:NROOTS
    @printf("  root %3i   E = %18.10f\n", r, e0[r])
end
@printf("Solution summary: %s\n", TPSChem.sharded_state_summary(vec_var))
flush(stdout)

if haskey(ENV, "TPSCHEM_OUTPUT_JLD2")
    # Only the small energy vector is saved; the eigenvector stays sharded.
    JLD2.jldsave(ENV["TPSCHEM_OUTPUT_JLD2"]; e0=e0)
end

TPSChem.destroy!(vec_var)
TPSChem.destroy!(cluster_ops)
