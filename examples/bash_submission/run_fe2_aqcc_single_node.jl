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

const DATA_FILE = get(ARGS, 1, get(ENV, "TPSCHEM_INPUT_JLD2", "data_fe2_morokuma_29.jld2"))
const VERBOSE = env_int("TPSCHEM_VERBOSE", 1)
const BLAS_THREADS = env_int("TPSCHEM_BLAS_THREADS", 1)
const M = env_int("TPSCHEM_CLUSTER_MAX_ROOTS", 200)
const CLUSTER_SPIN_ROOTS = env_int_vector("TPSCHEM_CLUSTER_SPIN_ROOTS", [3, 3, 3])
const INIT_FSPACE = TPSChem.FockConfig([(6, 1), (4, 4), (1, 6)])
const NROOTS = env_int("TPSCHEM_NROOTS", 6)
const THRESH_FOI_LIST = env_float_vector("TPSCHEM_THRESH_FOI_LIST", [8e-5,4e-5])
const CEPA_SHIFT = get(ENV, "TPSCHEM_CEPA_SHIFT", "acpf")
const SOLVER = Symbol(get(ENV, "TPSCHEM_SOLVER", "pcg"))
# :packed -- the lower triangle only, dim_q(dim_q+1)/2 * 8 B, no dependence on
# fill. Storing H_qq is the right idea (the solve on top needs under 100 MB, seven
# R x dim_q blocks), so budget ~80% of the node = ~176 GB for the matrix. At this
# system's dim_q ~ 181000 that gives:
#   :packed  131 GB  fits, and is fill-independent
#   :sparse  179 GB at 35% fill; only smaller than :packed below 25% fill
#   :direct  263 GB  does not fit -- it stores both triangles of a symmetric matrix
# :packed also measured fastest per apply of the three (half the bytes of :direct,
# and these applies are bandwidth-bound) and quickest to build, since it writes
# straight into the packed array with no COO staging and no mirroring pass.
#
# If it turns out not to fit, fall back to TPSCHEM_BUILD_HQQ=matvec: nothing
# stored, entries recomputed per apply at O(nthreads * R * dim_q) ~ 0.5 GB, but
# roughly two orders of magnitude slower per apply. Or go multinode, where
# h_storage=:blocks shards dense per-Fock-block across the allocation.
const BUILD_HQQ = Symbol(get(ENV, "TPSCHEM_BUILD_HQQ", "packed"))
# Solve all roots in one pass, sharing one H apply, rather than one pass per
# root. Worth ~3x on :matvec and :fois, where an apply is expensive. Unavailable
# for solver=:krylov, which takes a single right-hand side.
const BLOCK_ROOTS = env_bool("TPSCHEM_BLOCK_ROOTS", true)
# Each :matvec/:fois Krylov iteration is minutes at this size, so cap the budget
# rather than letting a stalling root run to 300.
const CG_MAXITER = env_int("TPSCHEM_MINRES_MAXITER", 300)

BLAS.set_num_threads(BLAS_THREADS)

@printf("\n================ Fe2S2 single-node CEPA ================\n")
@printf("Data file:            %s\n", DATA_FILE)
@printf("Julia threads:        %i\n", Threads.nthreads())
@printf("BLAS threads:         %i\n", BLAS_THREADS)
@printf("Cluster max roots:    %i\n", M)
@printf("Cluster spin roots:   %s\n", CLUSTER_SPIN_ROOTS)
@printf("Reference roots:      %i\n", NROOTS)
@printf("FOI thresholds:       %s\n", THRESH_FOI_LIST)
@printf("CEPA variant:         %s\n", CEPA_SHIFT)
@printf("Solver:               %s (build_hqq=%s, cg_maxiter=%i)\n",
        SOLVER, BUILD_HQQ, CG_MAXITER)
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

@printf("\nBuilding cluster operators...\n")
flush(stdout)
cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints)
TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)

ci_vector = TPSChem.TPSCIstate(clusters, INIT_FSPACE, R=NROOTS)
ci_vector = TPSChem.add_spin_focksectors(ci_vector)

for thresh_foi_cepa in THRESH_FOI_LIST
    @printf("\nRunning CEPA (%s) thresh_foi=%.3e ...\n", CEPA_SHIFT, thresh_foi_cepa)
    GC.gc()
    result = @timed TPSChem.do_fois_cepa(
        ci_vector,
        cluster_ops,
        clustered_ham;
        cepa_shift=CEPA_SHIFT,
        cepa_mit=env_int("TPSCHEM_CEPA_MIT", 30),
        thresh_foi=thresh_foi_cepa,
        nbody=env_int("TPSCHEM_NBODY", 4),
        tol=env_float("TPSCHEM_CEPA_TOL", 1e-8),
        thresh_sigma=env_float("TPSCHEM_THRESH_SIGMA", 1e-6),
        thresh_clip=env_float("TPSCHEM_THRESH_CLIP", 1e-5),
        compress=env_bool("TPSCHEM_COMPRESS_Q", false),
        cg_maxiter=CG_MAXITER,
        solver=SOLVER,
        build_hqq=BUILD_HQQ,
        block_roots=BLOCK_ROOTS,
        verbose=VERBOSE,
    )
    e_cepa, pt1_vec = result.value
    @printf("thresh=%.0e  E(cepa) = %s   (%.1f s)\n",
            thresh_foi_cepa, string(e_cepa), result.time)
    flush(stdout)
end