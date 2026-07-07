include("common.jl")

const VERBOSE = env_int("TPSCHEM_VERBOSE", 1)
const BLAS_THREADS = env_int("TPSCHEM_BLAS_THREADS", 1)
const BUILD_2RDM = env_bool("TPSCHEM_BUILD_2RDM", false)
const BUILD_SPIN_FLIP_1RDM = env_bool("TPSCHEM_BUILD_SPIN_FLIP_1RDM", true)
const SOLVE_REFERENCE = env_bool("TPSCHEM_SOLVE_REFERENCE", false)
const OUTFILE = get(ENV, "TPSCHEM_OUTPUT_JLD2", "tpsci_property_rdms_multinode.jld2")

function get_property_matrix(data)
    key = get(ENV, "TPSCHEM_PROPERTY_KEY", "")
    if !isempty(key)
        haskey(data, key) || error("TPSCHEM_PROPERTY_KEY='$key' was not found in input JLD2")
        return data[key]
    end
    return data["ints"].h1
end

workers_ = init_multinode_workers!()
data = load_problem_data()
clustered_ham = get_clustered_ham(data)

@printf("\n================ TPSCI property/RDM multinode post-analysis ================\n")
@printf("Input file:           %s\n", ENV["TPSCHEM_INPUT_JLD2"])
@printf("Output file:          %s\n", OUTFILE)
@printf("Build spin-flip 1RDM: %s\n", BUILD_SPIN_FLIP_1RDM)
@printf("Build full 2RDM:      %s\n", BUILD_2RDM)
@printf("Solve reference:      %s\n", SOLVE_REFERENCE)
flush(stdout)

psi = get_reference_state(data)

if SOLVE_REFERENCE
    @printf("\nSolving local TPSCI reference before post-analysis...\n")
    flush(stdout)
    cluster_ops_ref = TPSChem.compute_cluster_ops(data["cluster_bases"], data["ints"])
    TPSChem.add_cmf_operators!(cluster_ops_ref, data["cluster_bases"], data["ints"],
                               data["d1"].a, data["d1"].b)
    e0, psi, _ = TPSChem.tps_ci_direct(
        psi, cluster_ops_ref, clustered_ham;
        conv_thresh=env_float("TPSCHEM_REF_TOL", 1e-8),
        verbose=VERBOSE)
    @printf("Reference energies: %s\n", e0)
    cluster_ops_ref = nothing
    GC.gc()
end

cluster_ops = build_distributed_cluster_ops(data, workers_)
if BUILD_2RDM
    @printf("\nAdding distributed spin-free 2-RDM operator tables...\n")
    flush(stdout)
    TPSChem.add_spinfree_2rdm_ops_distributed!(
        cluster_ops, data["cluster_bases"];
        verbose=VERBOSE,
        blas_threads=BLAS_THREADS)
end

h_prop = get_property_matrix(data)

@printf("\nComputing alpha/beta 1-RDM blocks...\n")
flush(stdout)
gamma_aa, gamma_bb = TPSChem.compute_1rdm_distributed(
    psi, cluster_ops;
    workers=workers_,
    threaded_worker=true,
    blas_threads=BLAS_THREADS,
    verbose=VERBOSE)

@printf("\nComputing one-electron property matrix...\n")
flush(stdout)
property_matrix = TPSChem.compute_1e_property_direct_distributed(
    psi, cluster_ops, h_prop;
    workers=workers_,
    blas_threads=BLAS_THREADS,
    verbose=VERBOSE)

gamma_ab = nothing
gamma_ba = nothing
if BUILD_SPIN_FLIP_1RDM
    @printf("\nComputing spin-flip 1-RDM blocks...\n")
    flush(stdout)
    gamma_ab, gamma_ba = TPSChem.compute_1rdm_sf_distributed(
        psi, cluster_ops;
        workers=workers_,
        threaded_worker=true,
        blas_threads=BLAS_THREADS,
        verbose=VERBOSE)
end

Gamma = nothing
if BUILD_2RDM
    @printf("\nComputing full spin-free 2-RDM...\n")
    flush(stdout)
    Gamma = TPSChem.compute_2rdm_distributed(
        psi, cluster_ops;
        workers=workers_,
        threaded_worker=true,
        blas_threads=BLAS_THREADS,
        verbose=VERBOSE)
end

JLD2.jldopen(OUTFILE, "w") do io
    io["gamma_aa"] = gamma_aa
    io["gamma_bb"] = gamma_bb
    io["property_matrix"] = property_matrix
    BUILD_SPIN_FLIP_1RDM && (io["gamma_ab"] = gamma_ab; io["gamma_ba"] = gamma_ba)
    BUILD_2RDM && (io["Gamma"] = Gamma)
end

@printf("\nSaved post-analysis data to %s\n", OUTFILE)
flush(stdout)
TPSChem.destroy!(cluster_ops)
