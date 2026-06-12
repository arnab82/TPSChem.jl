using QCBase
using RDM
using TPSChem
using Printf
using JLD2

@load "../test/data_cmf_13_cr2_morokuma.jld2"

ref_fock = FockConfig([(3,0),(3,3),(0,3)])
nroots   = 4
M        = 100
thresh_foi = 1e-5
W          = 90

clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
clustered_S2  = TPSChem.extract_S2(clusters)

println("\n Building Spin eigenbasis M=$M ...")
@time cb_spin = TPSChem.compute_cluster_eigenbasis_spin(ints, clusters, d1,
                [3,3,3], ref_fock, max_roots=M, verbose=0)
cluster_ops = TPSChem.compute_cluster_ops(cb_spin, ints)
TPSChem.add_cmf_operators!(cluster_ops, cb_spin, ints, d1.a, d1.b)

lbs = [sum(size(sol.vectors,2) for (_,sol) in cb.basis) for cb in cb_spin]
@printf(" Local basis sizes: %s\n", join(lbs, ", "))

# ── Peak memory monitor ────────────────────────────────────────────────────────
function with_peak_memory(f)
    peak = Ref(Base.gc_live_bytes())
    done = Ref(false)
    task = @async while !done[]
        peak[] = max(peak[], Base.gc_live_bytes())
        sleep(0.1)
    end
    result = f()
    done[] = true
    wait(task)
    return result, peak[]
end

# ══════════════════════════════════════════════════════════════════════════════
# SPT-CEPA-0  (multi-root, ci_vector reference)
# ══════════════════════════════════════════════════════════════════════════════
println("\n" * "="^W)
println(" SPT-CEPA-0  (multi-root SPTstate reference)")
nroots=4
# start by defining P/Q spaces
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

ci_vector = SPTstate(clusters, p_spaces, cb_spin, R=4) 

na = 6
nb = 6

TPSChem.fill_p_space!(ci_vector, na, nb)
TPSChem.eye!(ci_vector)
e_ci, v_bst = TPSChem.ci_solve(ci_vector, cluster_ops, clustered_ham);
@printf(" SPT CI energies: %s\n", string(e_ci))

GC.gc()
baseline_bst = Base.gc_live_bytes()
r_bst, peak_bst = with_peak_memory() do
    @timed TPSChem.do_fois_cepa(v_bst, cluster_ops, clustered_ham;
                                 max_iter     = 200,
                                 cepa_shift   = "cepa",
                                 cepa_mit     = 30,
                                 nbody        = 4,
                                 thresh_foi   = thresh_foi,
                                 tol          = 1e-8,
                                 compress_type = "matvec",
                                 prescreen    = false,
                                 verbose      = true,
                                 solver       = :minres)
end
e_bst = r_bst.value

GC.gc()
baseline_bst = Base.gc_live_bytes()
r_bst_acpf, peak_bst_acpf = with_peak_memory() do
    @timed TPSChem.do_fois_cepa(v_bst, cluster_ops, clustered_ham;
                                 max_iter     = 200,
                                 cepa_shift   = "acpf",
                                 cepa_mit     = 30,
                                 nbody        = 4,
                                 thresh_foi   = thresh_foi,
                                 tol          = 1e-8,
                                 compress_type = "matvec",
                                 prescreen    = false,
                                 verbose      = true,
                                 solver       = :minres)
end
e_bst_acpf = r_bst_acpf.value

GC.gc()
baseline_bst = Base.gc_live_bytes()
r_bst_aqcc, peak_bst_aqcc = with_peak_memory() do
    @timed TPSChem.do_fois_cepa(v_bst, cluster_ops, clustered_ham;
                                 max_iter     = 200,
                                 cepa_shift   = "aqcc",
                                 cepa_mit     = 30,
                                 nbody        = 4,
                                 thresh_foi   = thresh_foi,
                                 tol          = 1e-8,
                                 compress_type = "matvec",
                                 prescreen    = false,
                                 verbose      = true,
                                 solver       = :minres)
end
e_bst_aqcc = r_bst_aqcc.value

# ══════════════════════════════════════════════════════════════════════════════
# TPSCI-CEPA-0
# ══════
println(" TPSCI-CEPA-0  (build_hqq=:sparse)")
println("="^W)
ci_tpsci = TPSChem.TPSCIstate(clusters, ref_fock, R=nroots)
ci_tpsci = TPSChem.add_spin_focksectors(ci_tpsci)
GC.gc()
baseline_tpsci = Base.gc_live_bytes()
r_tpsci, peak_tpsci = with_peak_memory() do
    @timed TPSChem.do_fois_cepa(ci_tpsci, cluster_ops, clustered_ham;
                                 cepa_shift   = "cepa",
                                 thresh_foi   = thresh_foi,
                                 nbody        = 4,
                                 tol          = 1e-8,
                                 thresh_sigma = 1e-8,
                                 solver       = :minres,
                                 build_hqq    = :sparse,
                                 verbose      = 0)
end
e_tpsci = r_tpsci.value

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
println()
println("═"^W)
@printf(" CEPA-0 comparison — Cr2 13-orbital, M=%i, thresh_foi=%.0e\n", M, thresh_foi)
println("═"^W)
@printf(" %-20s  %10s  %12s  %12s\n", "Method", "Time (s)", "Alloc (GiB)", "Peak (GiB)")
println("─"^W)
@printf(" %-20s  %10.2f  %12.3f  %12.3f\n",
        "TPSCI-CEPA-0 :sparse",
        r_tpsci.time, r_tpsci.bytes/2^30, (peak_tpsci-baseline_tpsci)/2^30)
@printf(" %-20s  %10.2f  %12.3f  %12.3f\n",
        "SPT-CEPA-0",
        r_bst.time, r_bst.bytes/2^30, (peak_bst-baseline_bst)/2^30)
println("═"^W)

println()
@printf(" %-20s  %5s  %16s\n", "Method", "Root", "E_cepa (Ha)")
println("─"^W)
for r in 1:nroots
    @printf(" %-20s  %5i  %16.10f\n", "TPSCI-CEPA-0", r, e_tpsci[r])
end
println("─"^W)
for r in 1:nroots
    @printf(" %-20s  %5i  %16.10f\n", "SPT-CEPA-0", r, e_bst[r])
end
println("─"^W)
@printf(" Max |ΔE| SPT vs TPSCI:  %.4e Ha\n", maximum(abs.(e_bst .- e_tpsci)))
println("═"^W)