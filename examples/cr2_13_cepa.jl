using TPSChem.QCBase
using TPSChem.RDM
using TPSChem
using Printf
using JLD2


include("compare_hqq.jl")
@load "../test/data_cmf_13_cr2_morokuma.jld2"

ref_fock = FockConfig([(3,0),(3,3),(0,3)])
nroots   = 4
M        = 100
clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
clustered_S2  = TPSChem.extract_S2(clusters)

# ── Spin eigenbasis M=100 ──────────────────────────────────────────────────────
println("\n Building Spin eigenbasis M=100 ...")
@time cb_spin = TPSChem.compute_cluster_eigenbasis_spin(ints, clusters, d1,
                [3,3,3], ref_fock, max_roots=M, verbose=0)

cluster_ops = TPSChem.compute_cluster_ops(cb_spin, ints)
TPSChem.add_cmf_operators!(cluster_ops, cb_spin, ints, d1.a, d1.b)

lbs = [sum(size(sol.vectors,2) for (_,sol) in cb.basis) for cb in cb_spin]
@printf(" Local basis sizes: %s\n", join(lbs, ", "))

# ── CMF reference ──────────────────────────────────────────────────────────────
ci_vector = TPSChem.TPSCIstate(clusters, ref_fock, R=nroots)
ci_vector = TPSChem.add_spin_focksectors(ci_vector)

thresh_foi_cepa = 1e-5
W = 90

# ── Peak memory monitor ────────────────────────────────────────────────────────
# Polls gc_live_bytes() every 100 ms in a background task to capture peak heap usage.
function with_peak_memory(f)
    peak  = Ref(Base.gc_live_bytes())
    done  = Ref(false)
    task  = @async while !done[]
        peak[] = max(peak[], Base.gc_live_bytes())
        sleep(0.1)
    end
    result = f()
    done[] = true
    wait(task)
    return result, peak[]
end

# ── Run :sparse ────────────────────────────────────────────────────────────────
println("\n Running CEPA-0 [build_hqq=:sparse]  thresh_foi=$thresh_foi_cepa ...")
GC.gc()
baseline_sparse = Base.gc_live_bytes()
r_sparse, peak_sparse = with_peak_memory() do
    @timed TPSChem.do_fois_cepa(ci_vector, cluster_ops, clustered_ham,
                                 cepa_shift="cepa", thresh_foi=thresh_foi_cepa,
                                 nbody=4, tol=1e-8, thresh_sigma=1e-8,
                                 solver=:minres, build_hqq=:sparse, verbose=0)
end
e_sparse = r_sparse.value

# ── Run :matvec ────────────────────────────────────────────────────────────────
println("\n Running CEPA-0 [build_hqq=:matvec]  thresh_foi=$thresh_foi_cepa ...")
GC.gc()
baseline_matvec = Base.gc_live_bytes()
r_matvec, peak_matvec = with_peak_memory() do
    @timed TPSChem.do_fois_cepa(ci_vector, cluster_ops, clustered_ham,
                                 cepa_shift="cepa", thresh_foi=thresh_foi_cepa,
                                 nbody=4, tol=1e-8, thresh_sigma=1e-8,
                                 solver=:minres, build_hqq=:matvec, verbose=0)
end
e_matvec = r_matvec.value

# ── Summary table ──────────────────────────────────────────────────────────────
println()
println("═"^W)
@printf(" CEPA-0 build_hqq comparison — Cr2 13-orbital, M=%i, thresh_foi=%.0e\n", M, thresh_foi_cepa)
println("═"^W)
@printf(" %-16s  %10s  %12s  %12s  %12s\n",
        "build_hqq", "Time (s)", "Alloc (GiB)", "Peak (GiB)", "E_cepa[1]")
println("─"^W)
@printf(" %-16s  %10.2f  %12.3f  %12.3f  %12.8f\n",
        ":sparse",
        r_sparse.time,
        r_sparse.bytes / 2^30,
        (peak_sparse - baseline_sparse) / 2^30,
        e_sparse[1])
@printf(" %-16s  %10.2f  %12.3f  %12.3f  %12.8f\n",
        ":matvec",
        r_matvec.time,
        r_matvec.bytes / 2^30,
        (peak_matvec - baseline_matvec) / 2^30,
        e_matvec[1])
println("─"^W)
@printf(" Max |ΔE| sparse vs matvec: %.2e Ha\n", maximum(abs.(e_sparse .- e_matvec)))
println("═"^W)

# ── Per-root energies ──────────────────────────────────────────────────────────
println()
@printf(" %-16s  %5s  %14s\n", "Method", "Root", "E_cepa")
println("─"^W)
for r in 1:nroots
    @printf(" %-16s  %5i  %14.8f\n", ":sparse",  r, e_sparse[r])
end
println("─"^W)
for r in 1:nroots
    @printf(" %-16s  %5i  %14.8f\n", ":matvec",  r, e_matvec[r])
end
println("═"^W)
