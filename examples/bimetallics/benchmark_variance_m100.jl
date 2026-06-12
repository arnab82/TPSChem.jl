using TPSChem.QCBase
using TPSChem
using NPZ
using TPSChem.InCoreIntegrals
using TPSChem.RDM
using JLD2
using Printf
using LinearAlgebra


@load "./../../test/data_cmf_13_cr2_morokuma.jld2"

M = 100

init_fspace = FockConfig([(3, 0), (3, 3), (0, 3)])

cluster_bases = TPSChem.compute_cluster_eigenbasis_spin(
    ints, clusters, d1, [3,3,3], init_fspace, max_roots=M, verbose=1)

clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
cluster_ops   = TPSChem.compute_cluster_ops(cluster_bases, ints)
TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)

# --- P-space definition (same for all thresholds) ----------------------------
p_spaces = Vector{ClusterSubspace}()

ssi = ClusterSubspace(clusters[1])
add_subspace!(ssi, (3,0), 1:1)
add_subspace!(ssi, (2,1), 1:1)
add_subspace!(ssi, (1,2), 1:1)
add_subspace!(ssi, (0,3), 1:1)
push!(p_spaces, ssi)

ssi = ClusterSubspace(clusters[2])
add_subspace!(ssi, (3,3), 1:1)
push!(p_spaces, ssi)

ssi = ClusterSubspace(clusters[3])
add_subspace!(ssi, (3,0), 1:1)
add_subspace!(ssi, (2,1), 1:1)
add_subspace!(ssi, (1,2), 1:1)
add_subspace!(ssi, (0,3), 1:1)
push!(p_spaces, ssi)

ci_vector = SPTstate(clusters, p_spaces, cluster_bases, R=4)
TPSChem.fill_p_space!(ci_vector, 6, 6)
TPSChem.eye!(ci_vector)
_, vbst = TPSChem.ci_solve(ci_vector, cluster_ops, clustered_ham)

# --- Helper: run one variational sweep ----------------------------------------
function run_bst(v0, thresh)
    return TPSChem.subspace_product_tucker(v0, cluster_ops, clustered_ham;
        max_iter    = 20,
        nbody       = 4,
        H0          = "Hcmf",
        thresh_var  = thresh,
        thresh_spin = thresh,
        thresh_foi  = thresh / 50,
        thresh_pt   = thresh / 2,
        ci_conv     = 5e-5,
        do_pt       = false,
        tol_tucker  = 1e-5,
        resolve_ss  = true,
        verbose     = 1)
end

# --- Helper: benchmark one variance call (warm-up + measured run) -------------
function bench_sigma2(label, fn, v_var, thresh_foi)
    kwargs = (H0="Hcmf", nbody=4, thresh_foi=thresh_foi,
              max_number=nothing, opt_ref=true, ci_tol=1e-6, verbose=0)

    alloc = @allocated t = @elapsed begin
        σ2 = fn(v_var, cluster_ops, clustered_ham; kwargs...)
    end

    @printf(" %-48s  %8.2f s  %8.3f GB\n", label, t, alloc * 1e-9)
    return σ2
end

# =============================================================================
# Main benchmark loop: three variational thresholds
# =============================================================================

const THRESHOLDS = [1e-2, 8e-3, 6e-3]
const THRESH_FOI = 1e-8

results = Dict{Float64, NamedTuple}()
let
    v_cur = vbst
    for thresh in THRESHOLDS
        println()
        println("="^70)
        @printf(" Variational threshold: %.1e   (M = %d, threads = %d)\n",
                thresh, M, Threads.nthreads())
        println("="^70)

        e, v_cur = run_bst(v_cur, thresh)
        @printf(" Wavefunction length: %d\n", length(v_cur))
        flush(stdout)

        println()
        @printf(" %-48s  %8s    %8s\n", "Version", "Time", "Alloc")
        println(" " * "-"^70)

        σ2_ref  = bench_sigma2("compute_spt_sigma_norm_blockwise  (reference)",
                                TPSChem.compute_spt_sigma_norm_blockwise,
                                v_cur, THRESH_FOI)
        σ2_fast = bench_sigma2("compute_spt_sigma_norm_blockwise_alternative",
                                TPSChem.compute_spt_sigma_norm_blockwise_alternative,
                                v_cur, THRESH_FOI)

        # Numerical agreement
        println()
        max_diff = maximum(abs.(σ2_ref .- σ2_fast))
        @printf(" Max |σ²_ref - σ²_fast| = %.3e\n", max_diff)

        # Per-root variance-like quantity
        println()
        @printf(" %-6s  %-14s  %-22s  %-22s\n",
                "Root", "Energy", "Var-like (ref)", "Var-like (fast)")
        println(" " * "-"^70)
        for r in eachindex(e)
            var_ref  = σ2_ref[r]  - e[r]^2
            var_fast = σ2_fast[r] - e[r]^2
            @printf(" %5d  %14.8f  %22.10e  %22.10e\n",
                    r, e[r], var_ref, var_fast)
        end

        results[thresh] = (e=e, σ2_ref=σ2_ref, σ2_fast=σ2_fast)
    end
end
# =============================================================================
# Summary table
# =============================================================================

println()
println("="^70)
println(" Summary: variance-like across thresholds (reference)")
println("="^70)
@printf(" %-10s", "thresh")
for r in 1:length(results[THRESHOLDS[1]].e)
    @printf("  %22s", "root $r var-like")
end
println()
println(" " * "-"^70)
for thresh in THRESHOLDS
    @printf(" %-10.1e", thresh)
    r_data = results[thresh]
    for r in eachindex(r_data.e)
        @printf("  %22.10e", r_data.σ2_ref[r] - r_data.e[r]^2)
    end
    println()
end
