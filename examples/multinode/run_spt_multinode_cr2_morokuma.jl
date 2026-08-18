# =============================================================================
# Never-gather multinode SPT (subspace_product_tucker) on the Cr2 (Morokuma) trimer.
#
# Follows the single-node SPT workflow of `examples/bimetallics/benchmark_pt2.jl`
# (spin cluster bases -> P-space definition -> SPTstate -> ci_solve reference),
# but runs the variational SPT loop with `subspace_product_tucker_sharded`: the
# variational vector, the FOIS, and the PT1 all stay `DistributedSPTstate`s
# across the workers for the whole loop and are never gathered onto the master.
# This is the memory-scalable path for when the Tucker-compressed SPT state is
# larger than a single node.
#
# Launch:
#   sbatch run_multinode.sh run_spt_multinode_cr2_morokuma.jl \
#          ../../test/data_cmf_13_cr2_morokuma.jld2
# or locally with 2 workers:
#   julia -p 2 --project run_spt_multinode_cr2_morokuma.jl \
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
        # The launcher scripts export TPSCHEM_WORKER_THREADS for the workers and
        # JULIA_NUM_THREADS for the master. Reading only the latter silently gives
        # every worker the master's (usually smaller) thread count.
        threads = get(ENV, "TPSCHEM_WORKER_THREADS",
                      get(ENV, "JULIA_NUM_THREADS", string(Threads.nthreads())))
        addprocs(hosts; exeflags="--project=$project --threads=$threads", env=worker_env())
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
const DATA_FILE    = get(ARGS, 1, get(ENV, "TPSCHEM_INPUT_JLD2", "data_cmf_13_cr2_morokuma.jld2"))
const VERBOSE      = env_int("TPSCHEM_VERBOSE", 1)
const BLAS_THREADS = env_int("TPSCHEM_BLAS_THREADS", 1)
const M            = env_int("TPSCHEM_CLUSTER_MAX_ROOTS", 20)
const SPIN_ROOTS   = env_int_vector("TPSCHEM_CLUSTER_SPIN_ROOTS", [3, 3, 3])
const INIT_FSPACE  = TPSChem.FockConfig([(3, 0), (3, 3), (0, 3)])
const NROOTS       = env_int("TPSCHEM_NROOTS", 4)
const THRESH       = env_float("TPSCHEM_THRESH", 1e-3)
const THRESH_VAR   = env_float("TPSCHEM_THRESH_VAR", THRESH)
const THRESH_FOI   = env_float("TPSCHEM_THRESH_FOI", THRESH / 50)
const THRESH_PT    = env_float("TPSCHEM_THRESH_PT", THRESH / 2)
const MAX_ITER     = env_int("TPSCHEM_MAX_ITER", 10)
const NBODY        = env_int("TPSCHEM_NBODY", 4)

workers_ = init_multinode_workers!()

@printf("\n================ Cr2 Morokuma never-gather SPT ================\n")
@printf("Data file:        %s\n", DATA_FILE)
@printf("Cluster max roots:%i   spin roots: %s\n", M, SPIN_ROOTS)
@printf("Reference roots:  %i\n", NROOTS)
@printf("thresh_var: %.2e  thresh_foi: %.2e  thresh_pt: %.2e\n",
        THRESH_VAR, THRESH_FOI, THRESH_PT)
flush(stdout)

data = JLD2.load(DATA_FILE)
ints, clusters, d1 = data["ints"], data["clusters"], data["d1"]

cluster_bases = TPSChem.compute_cluster_eigenbasis_spin(
    ints, clusters, d1, SPIN_ROOTS, INIT_FSPACE; max_roots=M, verbose=VERBOSE)
clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)

# Local cluster operators for the small P-space reference solve (single-node).
cluster_ops_ref = TPSChem.compute_cluster_ops(cluster_bases, ints)
TPSChem.add_cmf_operators!(cluster_ops_ref, cluster_bases, ints, d1.a, d1.b)

# --- P-space definition (matches examples/bimetallics/benchmark_pt2.jl) -------
p_spaces = Vector{ClusterSubspace}()

ssi = ClusterSubspace(clusters[1])
add_subspace!(ssi, (3, 0), 1:1)
add_subspace!(ssi, (2, 1), 1:1)
add_subspace!(ssi, (1, 2), 1:1)
add_subspace!(ssi, (0, 3), 1:1)
push!(p_spaces, ssi)

ssi = ClusterSubspace(clusters[2])
add_subspace!(ssi, (3, 3), 1:1)
push!(p_spaces, ssi)

ssi = ClusterSubspace(clusters[3])
add_subspace!(ssi, (3, 0), 1:1)
add_subspace!(ssi, (2, 1), 1:1)
add_subspace!(ssi, (1, 2), 1:1)
add_subspace!(ssi, (0, 3), 1:1)
push!(p_spaces, ssi)

# Reference SPT state: solve the P-space CI once, locally (it is small).
ci_vector = SPTstate(clusters, p_spaces, cluster_bases, R=NROOTS)
TPSChem.fill_p_space!(ci_vector, 6, 6)
TPSChem.eye!(ci_vector)
@printf("\nSolving P-space reference (local ci_solve)...\n"); flush(stdout)
_, vbst = TPSChem.ci_solve(ci_vector, cluster_ops_ref, clustered_ham; conv_thresh=1e-6)

# Shard the cluster operators for the never-gather SPT loop.
@printf("\nBuilding distributed cluster operators...\n"); flush(stdout)
cluster_ops = TPSChem.compute_cluster_ops_distributed(
    cluster_bases, ints; workers=workers_, verbose=VERBOSE, blas_threads=BLAS_THREADS)
TPSChem.add_cmf_operators_distributed!(
    cluster_ops, cluster_bases, ints, d1.a, d1.b; verbose=0, blas_threads=BLAS_THREADS)
@printf("Distributed cluster ops summary: %s\n", TPSChem.cluster_ops_summary(cluster_ops))
cluster_ops_ref = nothing
GC.gc()
flush(stdout)

# Never-gather variational SPT. `do_pt=true` also returns the PT1-corrected
# reference; the returned vector stays sharded.
e_var, vref = TPSChem.subspace_product_tucker_sharded(
    vbst, cluster_ops, clustered_ham;
    max_iter=MAX_ITER,
    nbody=NBODY,
    H0="Hcmf",
    thresh_var=THRESH_VAR,
    thresh_foi=THRESH_FOI,
    thresh_pt=THRESH_PT,
    ci_conv=env_float("TPSCHEM_CI_CONV", 5e-5),
    ci_max_ss_vecs=env_int("TPSCHEM_MAX_SS_VECS", 12),
    resolve_ss=env_bool("TPSCHEM_RESOLVE_SS", false),
    do_pt=env_bool("TPSCHEM_DO_PT", true),
    tol_tucker=env_float("TPSCHEM_TOL_TUCKER", 1e-5),
    workers=workers_,
    blas_threads=BLAS_THREADS,
    verbose=VERBOSE)

@printf("\nFinal variational SPT energies (dim=%i):\n", length(vref))
for r in 1:NROOTS
    @printf("  root %3i   E = %18.10f\n", r, e_var[r])
end
@printf("Solution summary: %s\n", TPSChem.sharded_spt_summary(vref))
flush(stdout)

if haskey(ENV, "TPSCHEM_OUTPUT_JLD2")
    JLD2.jldsave(ENV["TPSCHEM_OUTPUT_JLD2"]; e_var=e_var)
end

TPSChem.destroy!(vref)
TPSChem.destroy!(cluster_ops)
