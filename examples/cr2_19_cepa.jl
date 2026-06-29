using TPSChem.QCBase
using TPSChem.RDM
using TPSChem
using Printf
using JLD2




include("compare_hqq.jl")
@load "../test/data_cmf_3d_2p_2p_3d_cr2_pantazis.jld2"

ref_fock = FockConfig([(3, 0), (0, 3), (3, 3), (3, 3), (3, 3)])
nroots   = 4
M=100
clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
clustered_S2  = TPSChem.extract_S2(clusters)

# ── Spin eigenbasis M=30 ───────────────────────────────────────────────────────
println("\n Building Spin eigenbasis M=100 ...")
@time cb_spin = TPSChem.compute_cluster_eigenbasis_spin(ints, clusters, d1,
                [3,3,3,3,3], ref_fock, max_roots=M, verbose=0)
cluster_ops = TPSChem.compute_cluster_ops(cb_spin, ints)
TPSChem.add_cmf_operators!(cluster_ops, cb_spin, ints, d1.a, d1.b)
lbs = [sum(size(sol.vectors,2) for (_,sol) in cb.basis) for cb in cb_spin]
@printf(" Local basis sizes: %s\n", join(lbs, ", "))


# ── TPSCI reference ────────────────────────────────────────────────────────────
ci_vector = TPSChem.TPSCIstate(clusters, ref_fock, R=nroots)
ci_vector = TPSChem.add_spin_focksectors(ci_vector)


# println("\n Running TPSCI ...")
# e_tpsci, v_tpsci = TPSChem.tpsci_ci(ci_vector, cluster_ops, clustered_ham,
#                                       thresh_cipsi=6e-4, max_iter=30, thresh_foi=1e-6)
# e_tpsci, v_tpsci = TPSChem.tpsci_ci(v_tpsci, cluster_ops, clustered_ham,
#                                       thresh_cipsi=4e-4, max_iter=30, thresh_foi=1e-6)
# s2_tpsci = TPSChem.compute_expectation_value_parallel(v_tpsci, cluster_ops, clustered_S2)
# @printf("\n TPSCI  TPS=%6i\n", length(v_tpsci))
# @printf(" %-5s  %14s  %6s\n", "Root", "E(var)", "<S²>")
# for r in 1:nroots
#     @printf(" %5i  %14.8f  %6.3f\n", r, e_tpsci[r], s2_tpsci[r])
# end
# ept2 = TPSChem.compute_pt2_energy(v_tpsci, cluster_ops, clustered_ham, thresh_foi=1e-8)
# @printf("\n TPSCI+PT2:\n")
# for r in 1:nroots
#     @printf(" %5i  %14.8f\n", r, e_tpsci[r]+ept2[r])
# end


W = 85
thresh_foi_cepa = 1e-4
# # ── H_qq matrix correctness check: dense vs sparse ────────────────────────────
println("\n Comparing H_qq builders (dense vs sparse) at thresh_foi=$thresh_foi_cepa ...")
compare_hqq_builders(ci_vector, cluster_ops, clustered_ham,
                               thresh_foi=thresh_foi_cepa, nbody=4)


# ── CEPA solver comparison: all three on same thresh_foi ──────────────────────
# println("\n Running CEPA-0 [solver=:krylov]  thresh_foi=$thresh_foi_cepa ...")
# GC.gc()
# r_krylov = @timed TPSChem.do_fois_cepa(v_tpsci, cluster_ops, clustered_ham,
#                                          cepa_shift="cepa", thresh_foi=thresh_foi_cepa,
#                                          nbody=4, tol=1e-8, solver=:krylov, verbose=0)
# e_krylov = r_krylov.value
println("\n Running CEPA-0 [solver=:minres, build_hqq=:direct]  thresh_foi=$thresh_foi_cepa ...")
GC.gc()
r_direct = @timed TPSChem.do_fois_cepa(ci_vector, cluster_ops, clustered_ham,
                                         cepa_shift="cepa", thresh_foi=thresh_foi_cepa,
                                         nbody=4, tol=1e-8, solver=:minres,
                                         build_hqq=:direct, verbose=0)
e_direct = r_direct.value
# compare between different sigma_thresholds for minres/direct
println("\n Running CEPA-0 [solver=:minres, build_hqq=:sparse]  thresh_foi=$thresh_foi_cepa ...")
GC.gc()
r_sparse_s = @timed TPSChem.do_fois_cepa(ci_vector, cluster_ops, clustered_ham,
                                           cepa_shift="cepa", thresh_foi=thresh_foi_cepa,
                                           nbody=4, tol=1e-8,thresh_sigma=0.0, solver=:minres,
                                           build_hqq=:sparse, verbose=0)
e_sparse_s = r_sparse_s.value
println("time taken: ", r_sparse_s.time, " seconds")
println("\n Running CEPA-0 [solver=:minres, build_hqq=:sparse]  thresh_foi=$thresh_foi_cepa ...")
GC.gc()
r_sparse_s_1e8 = @timed TPSChem.do_fois_cepa(ci_vector, cluster_ops, clustered_ham,
                                           cepa_shift="cepa", thresh_foi=thresh_foi_cepa,
                                           nbody=4, tol=1e-8,thresh_sigma=1e-8, solver=:minres,
                                           build_hqq=:sparse, verbose=0)
e_sparse_s_1e8 = r_sparse_s_1e8.value
println("time taken: ", r_sparse_s_1e8.time, " seconds")
println("\n Running CEPA-0 [solver=:minres, build_hqq=:sparse]  thresh_foi=$thresh_foi_cepa ...")
GC.gc()
r_sparse_s_1e6 = @timed TPSChem.do_fois_cepa(ci_vector, cluster_ops, clustered_ham,
                                           cepa_shift="cepa", thresh_foi=thresh_foi_cepa,
                                           nbody=4, tol=1e-8,thresh_sigma=1e-6, solver=:minres,
                                           build_hqq=:sparse, verbose=0)
e_sparse_s_1e6 = r_sparse_s_1e6.value
println("time taken: ", r_sparse_s_1e6.time, " seconds")



println("\n Running ACPF [solver=:minres, build_hqq=:sparse] ...")
@time e_acpf = TPSChem.do_fois_cepa(ci_vector, cluster_ops, clustered_ham,
                                      cepa_shift="acpf", thresh_foi=thresh_foi_cepa,
                                      nbody=4, tol=1e-8, solver=:minres,
                                      build_hqq=:sparse, verbose=0)

println("\n Running AQCC [solver=:minres, build_hqq=:sparse] ...")
@time e_aqcc = TPSChem.do_fois_cepa(ci_vector, cluster_ops, clustered_ham,
                                      cepa_shift="aqcc", thresh_foi=thresh_foi_cepa,
                                      nbody=4, tol=1e-8, solver=:minres,
                                      build_hqq=:sparse, verbose=0)
# thresh_foi_cepa = 5e-5
# println("\n Running CEPA-0 [solver=:minres, build_hqq=:sparse]  thresh_foi=$thresh_foi_cepa ...")
# GC.gc()
# r_sparse_s1 = @timed TPSChem.do_fois_cepa(ci_vector, cluster_ops, clustered_ham,
#                                            cepa_shift="cepa", thresh_foi=thresh_foi_cepa,
#                                            nbody=4, tol=1e-8, solver=:minres,
#                                            build_hqq=:sparse, verbose=0)
# println("time taken: ", r_sparse_s1.time, " seconds")
# println("\n Running ACPF [solver=:minres, build_hqq=:sparse] ...")
# @time e_acpf1 = TPSChem.do_fois_cepa(ci_vector, cluster_ops, clustered_ham,
#                                       cepa_shift="acpf", thresh_foi=thresh_foi_cepa,
#                                       nbody=4, tol=1e-8, solver=:minres,
#                                       build_hqq=:sparse, verbose=0)

# println("\n Running AQCC [solver=:minres, build_hqq=:sparse] ...")
# @time e_aqcc1 = TPSChem.do_fois_cepa(ci_vector, cluster_ops, clustered_ham,
#                                       cepa_shift="aqcc", thresh_foi=thresh_foi_cepa,
#                                       nbody=4, tol=1e-8, solver=:minres,
#                                       build_hqq=:sparse, verbose=0)
# println("\n Running CEPA-0 [solver=:minres, build_hqq=:sparse]  thresh_foi=$thresh_foi_cepa ...")
# GC.gc()
# thresh_foi_cepa = 1e-5
# r_sparse_s2 = @timed TPSChem.do_fois_cepa(ci_vector, cluster_ops, clustered_ham,
#                                         cepa_shift="cepa", thresh_foi=thresh_foi_cepa,
#                                         nbody=4, tol=1e-8, solver=:minres,
#                                         build_hqq=:sparse, verbose=0)
# println("\n Running ACPF [solver=:minres, build_hqq=:sparse] ...")
# @time e_acpf2 = TPSChem.do_fois_cepa(ci_vector, cluster_ops, clustered_ham,
#                                       cepa_shift="acpf", thresh_foi=thresh_foi_cepa,
#                                       nbody=4, tol=1e-8, solver=:minres,
#                                       build_hqq=:sparse, verbose=0)

# println("\n Running AQCC [solver=:minres, build_hqq=:sparse] ...")
# @time e_aqcc2 = TPSChem.do_fois_cepa(ci_vector, cluster_ops, clustered_ham,
#                                       cepa_shift="aqcc", thresh_foi=thresh_foi_cepa,
#                                       nbody=4, tol=1e-8, solver=:minres,
#                                       build_hqq=:sparse, verbose=0)



# thresh_foi_cepa = 5e-6
# r_sparse_s3 = @timed TPSChem.do_fois_cepa(ci_vector, cluster_ops, clustered_ham,
#                                         cepa_shift="cepa", thresh_foi=thresh_foi_cepa,
#                                         nbody=4, tol=1e-8, solver=:minres,
#                                         build_hqq=:sparse, verbose=0)
# println("\n Running ACPF [solver=:minres, build_hqq=:sparse] ...")
# @time e_acpf4 = TPSChem.do_fois_cepa(ci_vector, cluster_ops, clustered_ham,
#                                       cepa_shift="acpf", thresh_foi=thresh_foi_cepa,
#                                       nbody=4, tol=1e-8, solver=:minres,
#                                       build_hqq=:sparse, verbose=0)

# println("\n Running AQCC [solver=:minres, build_hqq=:sparse] ...")
# @time e_aqcc4 = TPSChem.do_fois_cepa(ci_vector, cluster_ops, clustered_ham,
#                                       cepa_shift="aqcc", thresh_foi=thresh_foi_cepa,
#                                       nbody=4, tol=1e-8, solver=:minres,
#                                       build_hqq=:sparse, verbose=0)                                      
# println()
# println("═"^W)
# println(" CEPA-0 solver comparison — Cr2 13-orbital, M=40, thresh_foi=$thresh_foi_cepa")
# println("═"^W)
# @printf(" %-30s  %8s  %10s  %14s\n", "Variant", "Time(s)", "Alloc(GiB)", "E_cepa[1]")
# # @printf(" %-30s  %8.2f  %10.3f  %14.8f\n",
# #         ":krylov",                r_krylov.time,   r_krylov.bytes/2^30,   e_krylov[1])
# @printf(" %-30s  %8.2f  %10.3f  %14.8f\n",
#         ":minres / :direct",      r_direct.time,   r_direct.bytes/2^30,   e_direct[1])
# @printf(" %-30s  %8.2f  %10.3f  %14.8f\n",
#         ":minres / :sparse",      r_sparse_s.time, r_sparse_s.bytes/2^30, e_sparse_s[1])
# println("─"^W)
# # @printf(" Max |ΔE| krylov vs direct: %.2e Ha\n",
# #         maximum(abs.(e_krylov .- e_direct)))
# # @printf(" Max |ΔE| krylov vs sparse: %.2e Ha\n",
# #         maximum(abs.(e_krylov .- e_sparse_s)))
# # println("═"^W)


# # ── Summary ────────────────────────────────────────────────────────────────────
# println()
# println("═"^W)
# println(" Summary — Cr2 13-orbital, spin eigenbasis M=40, cipsi=6e-4")
# println("═"^W)
# @printf(" %-12s  %5s  %14s\n", "Method", "Root", "Energy")
# for r in 1:nroots
#     @printf(" %-12s  %5i  %14.8f\n", "TPSCI",      r, e_tpsci[r])
# end
# for r in 1:nroots
#     @printf(" %-12s  %5i  %14.8f\n", "TPSCI+PT2",  r, e_tpsci[r]+ept2[r])
# end
# for (lab, ev) in [("CEPA-0", e_sparse_s), ("ACPF", e_acpf), ("AQCC", e_aqcc)]
#     for r in 1:nroots
#         @printf(" %-12s  %5i  %14.8f\n", lab, r, ev[r])
#     end
# end
