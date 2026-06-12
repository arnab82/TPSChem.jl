using TPSChem.QCBase
using TPSChem
using TPSChem.InCoreIntegrals
using TPSChem.RDM
using JLD2
using Printf
using LinearAlgebra

@load "./../../test/data_cmf_13_cr2_morokuma.jld2"

M = 20

init_fspace = FockConfig([(3, 0), (3, 3), (0, 3)])

cluster_bases = TPSChem.compute_cluster_eigenbasis_spin(
    ints, clusters, d1, [3,3,3], init_fspace, max_roots=M, verbose=1)

clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
cluster_ops   = TPSChem.compute_cluster_ops(cluster_bases, ints)
TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)

# --- P-space definition -------------------------------------------------------
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

# --- Run one variational sweep ------------------------------------------------
thresh = 1e-3
e_var, v_var = TPSChem.subspace_product_tucker(vbst, cluster_ops, clustered_ham;
    max_iter    = 10,
    nbody       = 4,
    H0          = "Hcmf",
    thresh_var  = thresh,
    thresh_spin = thresh / 1.2,
    thresh_foi  = thresh / 50,
    thresh_pt   = thresh / 2,
    ci_conv     = 5e-5,
    do_pt       = false,
    tol_tucker  = 1e-5,
    resolve_ss  = true,
    verbose     = 1)

@printf("\n Variational wavefunction length: %d\n", length(v_var))
@printf(" E_var = %s\n\n", string(e_var))

# =============================================================================
# Benchmark PT2 methods
# =============================================================================

const THRESH_FOI = 1e-8
const PT2_KWARGS = (H0="Hcmf", nbody=4, thresh_foi=THRESH_FOI,
                    max_number=nothing, opt_ref=false, ci_tol=1e-6, verbose=1)

function bench_pt2(label, fn, v_var)

    alloc = @allocated t = @elapsed begin
        E2 = fn(v_var, cluster_ops, clustered_ham; PT2_KWARGS...)
    end
    @printf(" %-52s  %8.2f s  %8.3f GB\n", label, t, alloc * 1e-9)
    return E2
end

println()
println("="^70)
@printf(" PT2 benchmark  (M=%d, thresh_var=%.1e, thresh_foi=%.1e, threads=%d)\n",
        M, thresh, THRESH_FOI, Threads.nthreads())
println("="^70)
@printf(" %-52s  %8s    %8s\n", "Version", "Time", "Alloc")
println(" " * "-"^70)

E2_ref  = bench_pt2("compute_pt2_energy  (reference, _pt2_job)",
                     TPSChem.compute_pt2_energy,  v_var)
E2_ref2 = bench_pt2("compute_pt2_energy2 (block-at-a-time, _pt2_job2)",
                     TPSChem.compute_pt2_energy2, v_var)
E2_fast = bench_pt2("compute_pt2_energy_blockwise (Tucker rotation)",
                     TPSChem.compute_pt2_energy_blockwise, v_var)

# Numerical agreement
println()
@printf(" Max |E2_ref  - E2_fast| = %.3e\n", maximum(abs.(E2_ref  .- E2_fast)))
@printf(" Max |E2_ref2 - E2_fast| = %.3e\n", maximum(abs.(E2_ref2 .- E2_fast)))

println()
@printf(" %-8s  %-14s  %-14s  %-14s\n", "Root", "E2_ref", "E2_ref2", "E2_fast")
println(" " * "-"^56)
for r in eachindex(E2_ref)
    @printf(" %6d  %14.8f  %14.8f  %14.8f\n", r, E2_ref[r], E2_ref2[r], E2_fast[r])
end
