include("common.jl")

const VERBOSE = env_int("TPSCHEM_VERBOSE", 1)
const BLAS_THREADS = env_int("TPSCHEM_BLAS_THREADS", 1)
const NROOTS = env_int("TPSCHEM_NROOTS", 1)
const OUTFILE = get(ENV, "TPSCHEM_OUTPUT_JLD2", "spt_variance_multinode.jld2")

function get_spt_reference(data)
    key = get(ENV, "TPSCHEM_SPT_REF_KEY", "")
    if !isempty(key)
        haskey(data, key) || error("TPSCHEM_SPT_REF_KEY='$key' was not found in input JLD2")
        return data[key]
    end

    for candidate in ("v_var", "spt_ref", "vbst", "ref_spt")
        haskey(data, candidate) && return data[candidate]
    end

    ref_fock = TPSChem.FockConfig(data["init_fspace"])
    return TPSChem.SPTstate(data["clusters"], ref_fock, data["cluster_bases"];
                            R=NROOTS)
end

workers_ = init_multinode_workers!()
data = load_problem_data()
clustered_ham = get_clustered_ham(data)
ref = get_spt_reference(data)

@printf("\n================ SPT variance multinode ================\n")
@printf("Input file:   %s\n", ENV["TPSCHEM_INPUT_JLD2"])
@printf("Output file:  %s\n", OUTFILE)
@printf("Reference dim:%10i\n", length(ref))
flush(stdout)

cluster_ops = build_distributed_cluster_ops(data, workers_)

sigma2 = TPSChem.compute_spt_sigma_norm_blockwise_distributed(
    ref, cluster_ops, clustered_ham;
    H0=get(ENV, "TPSCHEM_H0", "Hcmf"),
    nbody=env_int("TPSCHEM_NBODY", 4),
    thresh_foi=env_float("TPSCHEM_THRESH_FOI", 1e-6),
    max_number=nothing,
    opt_ref=env_bool("TPSCHEM_OPT_REF", false),
    ci_tol=env_float("TPSCHEM_CI_TOL", 1e-6),
    verbose=VERBOSE,
    prescreen=env_bool("TPSCHEM_PRESCREEN", false),
    workers=workers_,
    threaded_worker=true,
    blas_threads=BLAS_THREADS,
    strategy=Symbol(get(ENV, "TPSCHEM_SHARD_STRATEGY", "balanced")))

E0 = get_e0(data)
JLD2.jldopen(OUTFILE, "w") do io
    io["sigma2"] = sigma2
    if E0 !== nothing
        io["e0"] = E0
        io["variance"] = sigma2 .- E0 .^ 2
    end
end

@printf("\nSaved SPT variance data to %s\n", OUTFILE)
flush(stdout)
TPSChem.destroy!(cluster_ops)
