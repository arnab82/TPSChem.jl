"""
rdm_serial_vs_threaded.jl

Compare serial and multi-threaded 1-RDM computations for correctness and speed.

Covers:
  - compute_1rdm      vs  compute_1rdm_threaded
  - compute_1rdm_sf   vs  compute_1rdm_sf_threaded

Run with multiple threads to see a speedup:
    julia --threads 4 rdm_serial_vs_threaded.jl

Test data: uses the he4 system from the FermiCG test suite.
Point this at a different jld2 if you have your own system ready.
"""

using TPSChem.QCBase
using TPSChem.RDM
using TPSChem
using TPSChem.InCoreIntegrals
using Printf
using LinearAlgebra
using JLD2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Path to test data relative to the package root (adjust if running from elsewhere)
DATA_FILE = joinpath(@__DIR__, "../../test/_testdata_cmf_he4.jld2")

println("="^70)
println(" RDM serial vs threaded comparison")
@printf(" Julia threads: %d\n", Threads.nthreads())
println("="^70)

# ---------------------------------------------------------------------------
# Load data and build cluster operators
# ---------------------------------------------------------------------------
println("\n--- Loading data ---")
@load DATA_FILE clusters cluster_bases init_fspace ints d1

clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
cluster_ops   = TPSChem.compute_cluster_ops(cluster_bases, ints)
TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)


# ---------------------------------------------------------------------------
# Run TPSCI to get converged eigenstates
# ---------------------------------------------------------------------------
println("\n--- TPSCI ---")
nroots   = 5
ref_fock = TPSChem.FockConfig(init_fspace)
ci_vec   = TPSChem.TPSCIstate(clusters, ref_fock, R=nroots, T=Float64)
ci_vec[ref_fock][ClusterConfig([2,1,1,1])] = [0,1,0,0,0]
ci_vec[ref_fock][ClusterConfig([1,2,1,1])] = [0,0,1,0,0]
ci_vec[ref_fock][ClusterConfig([1,1,2,1])] = [0,0,0,1,0]
ci_vec[ref_fock][ClusterConfig([1,1,1,2])] = [0,0,0,0,1]

e0, v0 = TPSChem.tpsci_ci(ci_vec, cluster_ops, clustered_ham,
                            incremental=true, ci_conv=1e-10,
                            thresh_cipsi=1e-3, thresh_foi=1e-8,
                            thresh_asci=-1, conv_thresh=1e-7,
                            ci_lindep_thresh=1e-12)

@printf("\nTPSCI energies (Hartree):\n")
for (r, e) in enumerate(e0)
    @printf("  root %d: %16.10f\n", r, e)
end

# ---------------------------------------------------------------------------
# Physical reference values
# ---------------------------------------------------------------------------
n_alpha = sum(f[1] for f in init_fspace)
n_beta  = sum(f[2] for f in init_fspace)
N_elec  = n_alpha + n_beta
norb    = sum(length(c) for c in clusters)

# ---------------------------------------------------------------------------
# 1. compute_1rdm vs compute_1rdm_threaded
# ---------------------------------------------------------------------------
println("\n", "="^70)
println(" 1-RDM: serial vs threaded")
println("="^70)

γ_aa_s, γ_bb_s = TPSChem.compute_1rdm(v0, cluster_ops)
γ_aa_t, γ_bb_t = TPSChem.compute_1rdm_threaded(v0, cluster_ops)

diff_aa = maximum(abs.(γ_aa_s .- γ_aa_t))
diff_bb = maximum(abs.(γ_bb_s .- γ_bb_t))
@printf("  Max |γ_aa_serial - γ_aa_threaded| = %.2e\n", diff_aa)
@printf("  Max |γ_bb_serial - γ_bb_threaded| = %.2e\n", diff_bb)
@printf("  Match (tol 1e-12): %s\n",
        (diff_aa < 1e-12 && diff_bb < 1e-12) ? "PASS" : "FAIL")

# Physical check: trace of 1-RDM = electron count per root
println("\n  Trace check (diagonal roots): Tr(γ_aa) + Tr(γ_bb) should = $N_elec")
@printf("  %-6s  %-10s  %-10s  %-10s\n", "Root", "Tr(γ_aa)", "Tr(γ_bb)", "Sum")
for r in 1:nroots
    tr_aa = sum(γ_aa_t[p, p, r, r] for p in 1:norb)
    tr_bb = sum(γ_bb_t[p, p, r, r] for p in 1:norb)
    @printf("  %-6d  %-10.6f  %-10.6f  %-10.6f\n", r, tr_aa, tr_bb, tr_aa+tr_bb)
end

println("\n  Timing:")
print("  serial:   "); @time TPSChem.compute_1rdm(v0, cluster_ops)
print("  threaded: "); @time TPSChem.compute_1rdm_threaded(v0, cluster_ops)

# ---------------------------------------------------------------------------
# 2. compute_1rdm_sf vs compute_1rdm_sf_threaded
# ---------------------------------------------------------------------------
println("\n", "="^70)
println(" Spin-flip 1-RDM: serial vs threaded")
println("="^70)

γ_ab_s, γ_ba_s = TPSChem.compute_1rdm_sf(v0, cluster_ops)
γ_ab_t, γ_ba_t = TPSChem.compute_1rdm_sf_threaded(v0, cluster_ops)

diff_ab = maximum(abs.(γ_ab_s .- γ_ab_t))
diff_ba = maximum(abs.(γ_ba_s .- γ_ba_t))
@printf("  Max |γ_ab_serial - γ_ab_threaded| = %.2e\n", diff_ab)
@printf("  Max |γ_ba_serial - γ_ba_threaded| = %.2e\n", diff_ba)
@printf("  Match (tol 1e-12): %s\n",
        (diff_ab < 1e-12 && diff_ba < 1e-12) ? "PASS" : "FAIL")

# Symmetry check: for real eigenstates, γ_ab[p,q,r,r] == γ_ba[q,p,r,r]
println("\n  Symmetry check per root: γ_ab[:,:,r,r] == γ_ba[:,:,r,r]'")
for r in 1:nroots
    err = maximum(abs.(γ_ab_t[:,:,r,r] .- γ_ba_t[:,:,r,r]'))
    @printf("  root %d: max |γ_ab - γ_ba'| = %.2e  %s\n",
            r, err, err < 1e-10 ? "OK" : "FAIL")
end

println("\n  Timing:")
print("  serial:   "); @time TPSChem.compute_1rdm_sf(v0, cluster_ops)
print("  threaded: "); @time TPSChem.compute_1rdm_sf_threaded(v0, cluster_ops)


