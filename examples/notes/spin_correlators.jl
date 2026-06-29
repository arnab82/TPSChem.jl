"""
spin_correlators.jl — Spin-spin correlators and 2-RDM analysis from TPSCI wavefunctions

Loads:
  data_cmf.jld2        (or data_cmf_26.jld2)  — CMF data: clusters, ints, dip_cmf, K_IJ
  tpsci_rdms.jld2      (or tpsci_rdms_26.jld2) — TPSCI output: e0a, γ_aa, γ_bb, Γ

Computes:
  1. 2-RDM correctness checks (trace, contraction to 1-RDM, hermitian symmetry)
  2. <Sz_I> per cluster per root from the 1-RDM
  3. <N_I N_J> from inter-cluster 2-RDM diagonal
  4. Exchange coupling J from energy gap analysis (Landé interval rule)
  5. D_SS classical (orbital-center approximation) for the dominant Cr1-Cr3 pair

Usage:
  julia spin_correlators.jl                   # 13-orbital default
  julia spin_correlators.jl 26                # 26-orbital system
"""

using TPSChem.QCBase
using TPSChem
using NPZ
using TPSChem.InCoreIntegrals
using TPSChem.RDM
using JLD2
using Printf
using LinearAlgebra

# ---- Select active space size from command-line argument ----
suffix = length(ARGS) > 0 ? "_$(ARGS[1])" : ""
cmf_file = "data_cmf.jld2"

println("Loading $cmf_file ...")
@load cmf_file clusters init_fspace ints dip_cmf K_12_cmf K_13_cmf K_23_cmf d1
@load "tpsci_results.jld2"
# println("Loading $rdm_file ...")
# @load rdm_file e0a γ_aa γ_bb Γ
# or compute it
clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters);
cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints);

TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);
println("Computing 1-RDM diagonal parts")
γ_aa, γ_bb = TPSChem.compute_1rdm(v0a, cluster_ops)
display(γ_aa[:,:,1,1])
display(γ_bb[:,:,1,1])
println("Computing 1-RDM spin flip parts")
γ_ab, γ_ba = TPSChem.compute_1rdm_sf(v0a, cluster_ops)
display(γ_ab[:,:,1,1])
display(γ_ba[:,:,1,1])
println("Computing 2-RDM")
Γ = TPSChem.compute_2rdm(v0a, cluster_ops)
display(Γ[:,:,:,:,1,1])
n_orb   = size(γ_aa, 1)
nroots  = size(γ_aa, 3)
N_alpha = round(Int, tr(γ_aa[:,:,1,1]))
N_beta  = round(Int, tr(γ_bb[:,:,1,1]))
N_elec  = N_alpha + N_beta
ha2cm   = 219474.63

@printf("\n System: %d orbitals, %d electrons (%dα + %dβ), %d roots\n",
        n_orb, N_elec, N_alpha, N_beta, nroots)

# ======================================================================
# 1. 2-RDM Correctness Checks
# ======================================================================
println("\n" * "="^70)
println(" 2-RDM Correctness Checks")
println("="^70)

# ---- 1a. Trace: Σ_{p,q} Γ[p,q,p,q,r,r] = N(N-1) ----
# The spin-summed 2-RDM satisfies Tr(Γ) = N(N-1)
expected_trace = N_elec * (N_elec - 1)
println("\n 1a. Trace check:  Σ_{pq} Γ[p,q,p,q] = N(N-1) = $(expected_trace)")
for r in 1:nroots
    tr_gamma = 0.0
    for p in 1:n_orb, q in 1:n_orb
        tr_gamma += Γ[p, q, p, q, r, r]
    end
    err = abs(tr_gamma - expected_trace)
    status = err < 1e-6 ? "OK" : "FAIL"
    @printf("   Root %2d:  Tr(Γ) = %12.6f  (expected %d)  err = %.2e  [%s]\n",
            r, tr_gamma, expected_trace, err, status)
end

# ---- 1b. Contraction: (1/(N-1)) Σ_q Γ[p,q,r,q] = γ[p,r] ----
# The spin-summed 1-RDM γ = γ_aa + γ_bb; contraction of 2-RDM gives (N-1)*γ
println("\n 1b. Contraction check:  (1/(N-1)) Σ_q Γ[p,q,r,q,root,root] = γ[p,r]")
for r in 1:nroots
    γ_total = γ_aa[:,:,r,r] + γ_bb[:,:,r,r]   # (n_orb, n_orb)
    # Σ_q Γ[p,q,r,q]:
    contracted = zeros(n_orb, n_orb)
    for p in 1:n_orb, rp in 1:n_orb, q in 1:n_orb
        contracted[p, rp] += Γ[p, q, rp, q, r, r]
    end
    contracted ./= (N_elec - 1)
    max_err = maximum(abs.(contracted - γ_total))
    status = max_err < 1e-6 ? "OK" : "FAIL"
    @printf("   Root %2d:  max |γ_contracted - γ_1RDM| = %.2e  [%s]\n",
            r, max_err, status)
end

# ---- 1c. Hermitian symmetry: Γ[p,q,r,s] = Γ[q,p,s,r] ----
println("\n 1c. Hermitian symmetry check:  Γ[p,q,r,s,r,r] = Γ[q,p,s,r,r,r]")
for r in 1:nroots
    max_asym = 0.0
    for p in 1:n_orb, q in 1:n_orb, rp in 1:n_orb, s in 1:n_orb
        max_asym = max(max_asym, abs(Γ[p,q,rp,s,r,r] - Γ[q,p,s,rp,r,r]))
    end
    status = max_asym < 1e-8 ? "OK" : "FAIL"
    @printf("   Root %2d:  max |Γ[p,q,r,s] - Γ[q,p,s,r]| = %.2e  [%s]\n",
            r, max_asym, status)
end

# ======================================================================
# 2. <Sz_I> per Cluster per Root (from 1-RDM)
# ======================================================================
println("\n" * "="^70)
println(" ⟨Sz_I⟩ per Cluster (from 1-RDM diagonal)")
println("="^70)
println()

# Cluster orbital ranges (0-based global indices, 1-based in Julia)
off = [0; cumsum([length(c.orb_list) for c in clusters])]
n_clusters = length(clusters)

@printf("   %-6s", "Root")
for I in 1:n_clusters
    @printf("  Sz_C%d      ", I)
end
@printf("  Sz_total\n")
@printf("   %s\n", "-"^(6 + 13*n_clusters + 12))

for r in 1:nroots
    Sz_total = 0.0
    @printf("   %-6d", r)
    for I in 1:n_clusters
        orbs = (off[I]+1):off[I+1]
        sz_I = 0.5 * (tr(γ_aa[orbs, orbs, r, r]) - tr(γ_bb[orbs, orbs, r, r]))
        Sz_total += sz_I
        @printf("  %+10.4f  ", sz_I)
    end
    @printf("  %+10.4f\n", Sz_total)
end

# ======================================================================
# 3. <N_I N_J> from Inter-Cluster 2-RDM Diagonal
# ======================================================================
println("\n" * "="^70)
println(" ⟨N_I N_J⟩ from 2-RDM (inter-cluster number correlations)")
println("="^70)
println()
println(" For I≠J: ⟨N_I N_J⟩ = Σ_{p∈I, q∈J} Γ[p,q,p,q,r,r]")
println(" Measures charge-charge correlations between clusters.\n")

for r in 1:nroots
    @printf("   Root %d:\n", r)
    for I in 1:n_clusters, J in (I+1):n_clusters
        o_I = (off[I]+1):off[I+1]
        o_J = (off[J]+1):off[J+1]
        NiNj = 0.0
        for p in o_I, q in o_J
            NiNj += Γ[p, q, p, q, r, r]
        end
        Ni = tr(γ_aa[o_I, o_I, r, r] + γ_bb[o_I, o_I, r, r])
        Nj = tr(γ_aa[o_J, o_J, r, r] + γ_bb[o_J, o_J, r, r])
        @printf("     ⟨N_%d N_%d⟩ = %8.4f   (⟨N_%d⟩=%.4f, ⟨N_%d⟩=%.4f)\n",
                I, J, NiNj, I, Ni, J, Nj)
    end
end

# ======================================================================
# 4. Exchange Coupling J from Energy Gap Analysis (Landé Interval Rule)
# ======================================================================
println("\n" * "="^70)
println(" Exchange Coupling J from Energy Gaps (Landé Interval Rule)")
println("="^70)
println()
println(" For a Heisenberg dimer with S_1 = S_2 = s:  E(S_tot) = J*S_tot(S_tot+1)/2")
println(" The Landé rule gives J = -(E(S) - E(S-1)) / S = 2*(E(S-1) - E(S)) / (2S-1)\n")

E_cm = (e0a .- e0a[1]) .* ha2cm
@printf("   Root energies (cm⁻¹ relative to ground state):\n")
for r in 1:min(nroots, 12)
    @printf("     Root %2d:  %10.2f cm⁻¹\n", r, E_cm[r])
end
println()

# For a Cr(III)-Cr(III) dinuclear (s=3/2 each) the full spin ladder is:
# S_tot = 0, 1, 2, 3 with multiplicities 1, 3, 5, 7
# The gap between consecutive S multiplets: ΔE(S→S-1) = J*S
if nroots >= 2
    # Print raw energy gaps between first few roots (assuming spin ordering)
    println("   Raw energy gaps E(n+1) - E(n)  [may not be pure spin-ladder]:")
    for r in 1:(min(nroots,8)-1)
        @printf("     E(%d) - E(%d) = %8.2f cm⁻¹\n", r+1, r, E_cm[r+1] - E_cm[r])
    end
    println()
    println("   Note: for the Heisenberg exchange coupling, identify the spin")
    println("   quantum number of each root from ⟨Sz_total⟩ above, then apply:")
    println("   J = (E(S_tot) - E(S_tot-1)) / S_tot  (interval rule, sign convention: J<0 ferromagnetic)")
end

# ======================================================================
# 5. D_SS Classical (Orbital-Center Point-Dipole Approximation)
# ======================================================================
println("\n" * "="^70)
println(" D_SS Zero-Field Splitting — Orbital-Center Approximation")
println("="^70)
println()
println(" D_SS_ab ≈ prefactor × Σ_{p∈I, q∈J} K_pq × t_ab(R_p, R_q)")
println(" K_pq = (pq|pq) exchange integral; R_p = ⟨φ_p|r|φ_p⟩ = dip_cmf[a,p,p]")
println(" t_ab(R) = (3 R_a R_b - R²δ_ab) / R^5  (classical traceless dipole tensor)\n")

# Orbital centroids from CMF-basis dipole diagonal
orb_centers = zeros(n_orb, 3)
for a in 1:3
    for p in 1:n_orb
        orb_centers[p, a] = dip_cmf[a, p, p]
    end
end

# Classical traceless dipole-dipole tensor between orbital pairs
function orbital_dipolar_tensor(centers)
    n = size(centers, 1)
    T = zeros(3, 3, n, n)
    for p in 1:n, q in 1:n
        p == q && continue
        R  = centers[p, :] - centers[q, :]
        r2 = dot(R, R)
        r2 < 1e-8 && continue
        r5 = r2^2.5
        for a in 1:3, b in 1:3
            T[a, b, p, q] = (3 * R[a] * R[b] - (a == b ? r2 : 0.0)) / r5
        end
    end
    return T
end

T_cl = orbital_dipolar_tensor(orb_centers)

# Coupling constants
alpha_fs = 1.0 / 137.035999084
g_e      = 2.0023
S_HS     = 3.0                         # S_total = s1 + s2 = 3/2 + 3/2 = 3 for Cr(III) dimer HS state

# Compute D_SS for each cluster pair using the CMF exchange integrals (ints.h2 diagonal)
# K_IJ_cmf has the inter-cluster exchange integrals in CMF basis.
# K_IJ_cmf[p,q,r,s] = (pq|rs) from cmf.jl; K_pq = h2[p,q,p,q] = K_IJ_cmf[p,q,p,q]

# Cluster pair C1-C3 (dominant for Cr1 and Cr3, same spin s=3/2 each)
o1 = (off[1]+1):off[2]
o3 = (off[end-1]+1):off[end]   # last cluster

prefac = g_e^2 * alpha_fs^2 / 4.0 / (2 * S_HS * (2*S_HS - 1))

D_SS_13 = zeros(3, 3)
for p in o1, q in o3
    K_pq = ints.h2[p, q, p, q]
    D_SS_13 .+= K_pq .* T_cl[:, :, p, q]
end
D_SS_13 .*= prefac
D_SS_13_cm = D_SS_13 .* ha2cm

ev = eigvals(Symmetric(D_SS_13_cm))
D_val = ev[3] - ev[1]
E_val = (ev[2] - ev[1]) / 2

@printf("   C1-C3 pair (S_total=%.1f, prefac=%.4e):\n", S_HS, prefac)
@printf("   D = %8.3f cm⁻¹\n", D_val)
@printf("   E = %8.3f cm⁻¹\n", E_val)
println("   D_SS tensor (cm⁻¹):")
for a in 1:3
    @printf("     %+10.5f  %+10.5f  %+10.5f\n",
            D_SS_13_cm[a,1], D_SS_13_cm[a,2], D_SS_13_cm[a,3])
end
println()
println(" Note: the orbital-center approximation works well for well-separated clusters.")
println("       The exact result requires the full 2e spin-dipolar tensor T_ab[p,q,r,s]")
println("       and the spin-resolved 2-RDM Γ_spin = Γ_αα + Γ_ββ - Γ_αβ - Γ_βα.")
println("       Spin-resolved blocks are not available from compute_2rdm (spin-summed).")
println("       Run compute_2rdm_sf (if available) for the spin-resolved components.")
