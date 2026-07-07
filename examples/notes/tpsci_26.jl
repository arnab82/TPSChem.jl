using TPSChem.QCBase
using TPSChem
using NPZ
using TPSChem.InCoreIntegrals
using TPSChem.RDM
using JLD2
using Printf
using LinearAlgebra

@load "data_cmf_26.jld2" clusters init_fspace ints d1 e_cmf U dip_cmf nabla_cmf Ccmf soc_Lx_cmf soc_Ly_cmf soc_Lz_cmf K_12_cmf_26 K_13_cmf_26 K_23_cmf_26

M = 50

# CMF reference: [(5,2),(3,3),(5,2)]  →  Na=13, Nb=7, M_S=3  (FM reference)
# TPSCI reference: flip C3 to (2,5)  →  Na=10, Nb=10, M_S=0  (AFM sector)
init_fspace = FockConfig([(5, 2), (3, 3), (2, 5)])
#=
let
    cluster_bases = TPSChem.compute_cluster_eigenbasis_spin(ints, clusters, d1, [5,5,5], init_fspace, max_roots=M, verbose=1);

    # ------------------------------------------------------------------
    # Fock sectors: reference M_S=0 + CT (M_S=0) + M_S=±1 + M_S=±2
    # Reference: C1=(5,2), C2=(3,3), C3=(2,5)  Na=10, Nb=10
    #
    # M_S=0 CT sectors (inter-cluster charge transfer, Na=Nb=10):
    #   α: C1→C3      C1=(4,2), C3=(3,5)
    #   α: C2→C3      C2=(2,3), C3=(3,5)
    #   β: C3→C1      C1=(5,3), C3=(2,4)
    #   β: C2→C1      C1=(5,3), C2=(3,2)
    #   spin-exch C1↔C3  C1=(4,3), C3=(3,4)
    #
    # M_S=+1 sectors (single β→α flip, Na=11, Nb=9):
    #   C1 flip: (5,2)→(6,1)
    #   C2 flip: (3,3)→(4,2)
    #   C3 flip: (2,5)→(3,4)
    #
    # M_S=−1 sectors (single α→β flip, Na=9, Nb=11):
    #   C1 flip: (5,2)→(4,3)
    #   C2 flip: (3,3)→(2,4)
    #   C3 flip: (2,5)→(1,6)
    #
    # M_S=+2 sectors (two β→α flips, Na=12, Nb=8):
    #   C1+C3: (6,1)+(3,4),  C1+C2: (6,1)+(4,2),  C2+C3: (4,2)+(3,4)
    #
    # M_S=−2 sectors (two α→β flips, Na=8, Nb=12):
    #   C1+C3: (4,3)+(1,6),  C1+C2: (4,3)+(2,4),  C2+C3: (2,4)+(1,6)
    # ------------------------------------------------------------------
    ct_fspaces = [
        # Spin partner of AFM reference: C1 majority-β, C3 majority-α
        # Required so cluster_bases covers C1=(2,5) and C3=(5,2).
        # add_spin_focksectors is NOT used — that would inject sectors missing from cluster_bases.
        FockConfig([(2, 5), (3, 3), (5, 2)]),   # spin-flipped AFM partner
        # M_S = 0 charge-transfer sectors
        FockConfig([(4, 2), (3, 3), (3, 5)]),   # α: C1→C3
        FockConfig([(5, 2), (2, 3), (3, 5)]),   # α: C2→C3
        FockConfig([(5, 3), (3, 3), (2, 4)]),   # β: C3→C1
        FockConfig([(5, 3), (3, 2), (2, 5)]),   # β: C2→C1
        FockConfig([(4, 3), (3, 3), (3, 4)]),   # spin-exchange C1↔C3
        # M_S = +1  (β→α flip on one cluster)
        FockConfig([(6, 1), (3, 3), (2, 5)]),   # C1 flip
        FockConfig([(5, 2), (4, 2), (2, 5)]),   # C2 flip
        FockConfig([(5, 2), (3, 3), (3, 4)]),   # C3 flip
        # M_S = -1  (α→β flip on one cluster)
        FockConfig([(4, 3), (3, 3), (2, 5)]),   # C1 flip
        FockConfig([(5, 2), (2, 4), (2, 5)]),   # C2 flip
        FockConfig([(5, 2), (3, 3), (1, 6)]),   # C3 flip
        # M_S = +2  (β→α flip on two clusters)
        FockConfig([(6, 1), (3, 3), (3, 4)]),   # C1+C3
        FockConfig([(6, 1), (4, 2), (2, 5)]),   # C1+C2
        FockConfig([(5, 2), (4, 2), (3, 4)]),   # C2+C3
        # M_S = -2  (α→β flip on two clusters)
        FockConfig([(4, 3), (3, 3), (1, 6)]),   # C1+C3
        FockConfig([(4, 3), (2, 4), (2, 5)]),   # C1+C2
        FockConfig([(5, 2), (2, 4), (1, 6)]),   # C2+C3
    ]

    for fs in ct_fspaces
        cb = TPSChem.compute_cluster_eigenbasis_spin(
            ints, clusters, d1, [5,5,5], fs, max_roots=5, verbose=0)
        cluster_bases = TPSChem.merge_cluster_bases(cluster_bases, cb)
    end
end
=#
    # @save "cluster_bases_cmf_26.jld2" cluster_bases
@load "cluster_bases_cmf_26.jld2" cluster_bases
clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters);
cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints);

TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);

nroots = 10

ci_vector = TPSChem.TPSCIstate(clusters, init_fspace, R=nroots)
ci_vector = TPSChem.add_spin_focksectors(ci_vector)
# Spin-flipped AFM partner — needed to properly represent the M_S=0 AFM ground state
TPSChem.add_fockconfig!(ci_vector, FockConfig([(2, 5), (3, 3), (5, 2)]))
# M_S = 0 CT sectors
TPSChem.add_fockconfig!(ci_vector, FockConfig([(4, 2), (3, 3), (3, 5)]))
TPSChem.add_fockconfig!(ci_vector, FockConfig([(5, 2), (2, 3), (3, 5)]))
TPSChem.add_fockconfig!(ci_vector, FockConfig([(5, 3), (3, 3), (2, 4)]))
TPSChem.add_fockconfig!(ci_vector, FockConfig([(5, 3), (3, 2), (2, 5)]))
TPSChem.add_fockconfig!(ci_vector, FockConfig([(4, 3), (3, 3), (3, 4)]))
ci_vector = TPSChem.add_spin_focksectors(ci_vector)
# M_S = +1  (needed for non-zero SOC_+)
TPSChem.add_fockconfig!(ci_vector, FockConfig([(6, 1), (3, 3), (2, 5)]))
TPSChem.add_fockconfig!(ci_vector, FockConfig([(5, 2), (4, 2), (2, 5)]))
TPSChem.add_fockconfig!(ci_vector, FockConfig([(5, 2), (3, 3), (3, 4)]))
# M_S = -1  (needed for non-zero SOC_-)
TPSChem.add_fockconfig!(ci_vector, FockConfig([(4, 3), (3, 3), (2, 5)]))
TPSChem.add_fockconfig!(ci_vector, FockConfig([(5, 2), (2, 4), (2, 5)]))
TPSChem.add_fockconfig!(ci_vector, FockConfig([(5, 2), (3, 3), (1, 6)]))
# M_S = +2
TPSChem.add_fockconfig!(ci_vector, FockConfig([(6, 1), (3, 3), (3, 4)]))
TPSChem.add_fockconfig!(ci_vector, FockConfig([(6, 1), (4, 2), (2, 5)]))
TPSChem.add_fockconfig!(ci_vector, FockConfig([(5, 2), (4, 2), (3, 4)]))
# M_S = -2
TPSChem.add_fockconfig!(ci_vector, FockConfig([(4, 3), (3, 3), (1, 6)]))
TPSChem.add_fockconfig!(ci_vector, FockConfig([(4, 3), (2, 4), (2, 5)]))
TPSChem.add_fockconfig!(ci_vector, FockConfig([(5, 2), (2, 4), (1, 6)]))

fspace_0 = init_fspace

# M_S = 0 CT via TPSChem.replace
tmp_fspace = TPSChem.replace(fspace_0, (1,3), ([4,2],[3,5]))   # α: C1→C3
TPSChem.add_fockconfig!(ci_vector, tmp_fspace)
ci_vector[tmp_fspace][TPSChem.ClusterConfig([1,1,1])] = zeros(Float64, nroots)

tmp_fspace = TPSChem.replace(fspace_0, (2,3), ([2,3],[3,5]))   # α: C2→C3
TPSChem.add_fockconfig!(ci_vector, tmp_fspace)
ci_vector[tmp_fspace][TPSChem.ClusterConfig([1,1,1])] = zeros(Float64, nroots)

tmp_fspace = TPSChem.replace(fspace_0, (1,3), ([5,3],[2,4]))   # β: C3→C1
TPSChem.add_fockconfig!(ci_vector, tmp_fspace)
ci_vector[tmp_fspace][TPSChem.ClusterConfig([1,1,1])] = zeros(Float64, nroots)

tmp_fspace = TPSChem.replace(fspace_0, (1,2), ([5,3],[3,2]))   # β: C2→C1
TPSChem.add_fockconfig!(ci_vector, tmp_fspace)
ci_vector[tmp_fspace][TPSChem.ClusterConfig([1,1,1])] = zeros(Float64, nroots)

tmp_fspace = TPSChem.replace(fspace_0, (1,3), ([4,3],[3,4]))   # spin-exchange C1↔C3
TPSChem.add_fockconfig!(ci_vector, tmp_fspace)
ci_vector[tmp_fspace][TPSChem.ClusterConfig([1,1,1])] = zeros(Float64, nroots)

TPSChem.eye!(ci_vector)

eci, v = TPSChem.tps_ci_direct(ci_vector, cluster_ops, clustered_ham);

e0a, v0a = TPSChem.tpsci_ci(ci_vector, cluster_ops, clustered_ham, incremental=true,
                                thresh_cipsi = 1e-3,
                                thresh_foi   = 1e-6,
                                thresh_asci  = -1);

γ_aa, γ_bb = TPSChem.compute_1rdm(v0a, cluster_ops)

 # ------------------------------------------------------------------
# Diagnostic 1: trace of the 1-RDM per root
n_elec = sum(init_fspace[k][1] + init_fspace[k][2] for k in 1:length(clusters))
γ_total = γ_aa .+ γ_bb
@printf("\n 1-RDM trace check (should be %.1f electrons each root):\n", float(n_elec))
for r in 1:nroots
    @printf("   root %i: Tr(γ) = %.6f\n", r, tr(γ_total[:,:,r,r]))
end

# Diagnostic 2: transition dipole moments
μ_x, μ_y, μ_z = dip_cmf[1,:,:], dip_cmf[2,:,:], dip_cmf[3,:,:]
tdm_x, tdm_y, tdm_z = TPSChem.compute_transition_dipoles(γ_aa, γ_bb, μ_x, μ_y, μ_z; ref_root=1)
@printf("\n Transition dipole moments from root 1 (a.u.):\n")
@printf("   %-6s  %-12s  %-12s  %-12s\n", "Root", "<0|μ_x|n>", "<0|μ_y|n>", "<0|μ_z|n>")
for n in 2:nroots
    @printf("   %-6i  %-12.6f  %-12.6f  %-12.6f\n", n, tdm_x[n], tdm_y[n], tdm_z[n])
end

f = TPSChem.compute_oscillator_strengths(
        e0a, γ_aa, γ_bb, μ_x, μ_y, μ_z; ref_root=1)

TPSChem.print_stick_spectrum(e0a, f; units=:ev)

ω, I = TPSChem.absorption_spectrum(e0a, f; σ=0.005, lineshape=:lorentzian)

using NPZ
npzwrite("tpsci_spectrum_26.npz", Dict(
        "omega"     => ω,
        "intensity" => I,
        "energies"  => e0a,
        "fosc"      => f))

using Plots
ha2ev = 27.2114
ω_ev  = ω .* ha2ev
ΔE_ev = (e0a[2:end] .- e0a[1]) .* ha2ev
f_sticks = f[2:end]

p = plot(ω_ev, I,
        lw=2, color=:steelblue,
        xlabel="Energy (eV)", ylabel="Absorption (arb. u.)",
        title="TPSCI absorption — Cr₂O(NH₃)₈⁴⁺ (26 orb)",
        label="TPSCI (Lorentzian)", legend=:topright)

scale = maximum(I) / (maximum(f_sticks) + 1e-10)
for (e, fi) in zip(ΔE_ev, f_sticks)
        fi > 1e-4 || continue
        plot!(p, [e, e], [0.0, fi * scale], lw=1.5, color=:red, alpha=0.6, label=false)
end

savefig(p, "tpsci_spectrum_26.pdf")
display(p)

# ==============================================================================
  # Spin-flip 1-RDM and Spin-Orbit Coupling
  # ==============================================================================
  γ_ab, γ_ba = TPSChem.compute_1rdm_sf(v0a, cluster_ops)

 # SOC integrals in CMF basis — loaded from data_cmf_26.jld2
  SOC_z = zeros(ComplexF64, nroots, nroots)
  SOC_p = zeros(ComplexF64, nroots, nroots)
  SOC_m = zeros(ComplexF64, nroots, nroots)

 h_z  = im .* soc_Lz_cmf
  h_pl = (soc_Lx_cmf .+ im .* soc_Ly_cmf) ./ sqrt(2)
  h_mn = (soc_Lx_cmf .- im .* soc_Ly_cmf) ./ sqrt(2)

 for r2 in 1:nroots, r1 in 1:nroots
         Δγ = γ_aa[:, :, r1, r2] .- γ_bb[:, :, r1, r2]
         SOC_z[r1, r2] = dot(vec(h_z),  vec(Δγ))
         SOC_p[r1, r2] = dot(vec(h_pl), vec(γ_ab[:, :, r1, r2]))
         SOC_m[r1, r2] = dot(vec(h_mn), vec(γ_ba[:, :, r1, r2]))
 end

 SOC_total = SOC_z .+ SOC_p .+ SOC_m
  ha2cm = 219474.63

 @printf("\n SOC matrix |H_SOC[r1,r2]| (cm⁻¹), threshold 1 cm⁻¹:\n")
  @printf("   %-6s  %-6s  %-14s  %-14s  %-14s  %-14s\n",
          "r1", "r2", "|H_SOC|", "|SOC_z|", "|SOC_+|", "|SOC_-|")
  for r2 in 1:nroots, r1 in 1:r2
      val = abs(SOC_total[r1, r2]) * ha2cm
      val < 1.0 && continue
      @printf("   %-6i  %-6i  %-14.4f  %-14.4f  %-14.4f  %-14.4f\n",
              r1, r2,
              abs(SOC_total[r1, r2]) * ha2cm,
              abs(SOC_z[r1, r2]) * ha2cm,
              abs(SOC_p[r1, r2]) * ha2cm,
              abs(SOC_m[r1, r2]) * ha2cm)
 end

 # ==============================================================================
  # 2-RDM and Exchange Coupling
  # ==============================================================================
  println("\n Computing 2-RDM and exchange coupling constants J_IJ...")
  if any(!haskey(ops, "Ppqsr") for ops in cluster_ops)
      println(" Building spin-free 2-RDM operator tables Ppqsr...")
      TPSChem.add_spinfree_2rdm_ops!(cluster_ops, cluster_bases)
  end
  Γ = TPSChem.compute_2rdm(v0a, cluster_ops)

 # Cluster-pair exchange integrals in CMF basis — from data_cmf_26.jld2
  # K[p,q,r,s] = (pq|rs), p,r ∈ cluster A; q,s ∈ cluster B
  K_12 = K_12_cmf_26   # (n_C1, n_C2, n_C1, n_C2): C1=(10 orb), C2=(6 orb)
  K_13 = K_13_cmf_26   # (n_C1, n_C3, n_C1, n_C3): C1=(10 orb), C3=(10 orb)
  K_23 = K_23_cmf_26   # (n_C2, n_C3, n_C2, n_C3): C2=(6 orb),  C3=(10 orb)

 off = [0; cumsum([length(c.orb_list) for c in clusters])]

 nroots_diag = nroots
  J12 = zeros(nroots_diag)
  J13 = zeros(nroots_diag)
  J23 = zeros(nroots_diag)

let
    o1 = off[1]+1:off[2]
    o2 = off[2]+1:off[3]
    o3 = off[3]+1:off[4]
    norb1 = length(o1); norb2 = length(o2); norb3 = length(o3)

    for rr in 1:nroots_diag
        acc12 = 0.0; acc13 = 0.0; acc23 = 0.0
    for p in 1:norb1, r in 1:norb1, q in 1:norb2, s in 1:norb2
        acc12 += K_12[p, q, r, s] * Γ[o1[p], o2[q], o1[r], o2[s], rr, rr]
    end
    for p in 1:norb1, r in 1:norb1, q in 1:norb3, s in 1:norb3
        acc13 += K_13[p, q, r, s] * Γ[o1[p], o3[q], o1[r], o3[s], rr, rr]
    end
    for p in 1:norb2, r in 1:norb2, q in 1:norb3, s in 1:norb3
        acc23 += K_23[p, q, r, s] * Γ[o2[p], o3[q], o2[r], o3[s], rr, rr]
    end
    J12[rr] = 0.5 * acc12
    J13[rr] = 0.5 * acc13
    J23[rr] = 0.5 * acc23
end

@printf("\n Exchange coupling constants J_IJ (cm⁻¹) per root:\n")
@printf("   %-6s  %-14s  %-14s  %-14s\n", "Root", "J_12", "J_13", "J_23")
for rr in 1:nroots_diag
      @printf("   %-6i  %-14.4f  %-14.4f  %-14.4f\n",
              rr, J12[rr]*ha2cm, J13[rr]*ha2cm, J23[rr]*ha2cm)
end

e2a = TPSChem.compute_pt2_energy(v0a, cluster_ops, clustered_ham)

# ---- Save RDMs and energies for spin_correlators.jl ----
@save "tpsci_rdms_26.jld2" e0a γ_aa γ_bb Γ
println("Saved tpsci_rdms_26.jld2: e0a, γ_aa, γ_bb, Γ")
