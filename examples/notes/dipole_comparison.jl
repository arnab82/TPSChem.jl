"""
dipole_comparison.jl

Compare two ways to compute one-electron properties from a TPSCI wavefunction:

  Method A (1-RDM path):
      1. compute_1rdm  → γ_aa, γ_bb  (norb × norb × R × R)
      2. contract_1rdm_property(γ_aa, γ_bb, h_prop) → P[r1,r2]

  Method B (direct path):
      compute_1e_property_direct(v, cluster_ops, h_prop) → P[r1,r2]
      (contracts with h_prop inside the cluster loop — no full RDM stored)

"""

using TPSChem.QCBase
using TPSChem
using TPSChem.InCoreIntegrals
using TPSChem.RDM
using JLD2
using Printf
using LinearAlgebra

using TPSChem.QCBase
using TPSChem
using NPZ
using TPSChem.InCoreIntegrals
using TPSChem.RDM
using JLD2
using Printf
using LinearAlgebra

@load "data_cmf.jld2"

M = 100

init_fspace = FockConfig([(3,0), (3, 3), (0, 3)])
init_fspace1= FockConfig([(2,0), (3, 3), (0, 4)])
let
    cluster_bases = TPSChem.compute_cluster_eigenbasis_spin(ints, clusters, d1, [10,10,10], init_fspace, max_roots=M, verbose=1);

    # CT Fock sectors
    ct_fspaces = [
        FockConfig([(2,0), (3,3), (0,4)]),
        FockConfig([(2,0), (3,3), (1,3)]),
        FockConfig([(3,0), (2,3), (1,3)]),
        FockConfig([(3,1), (3,3), (0,2)]),
        FockConfig([(3,1), (3,2), (0,3)]),
        FockConfig([(2,1), (3,3), (1,2)]),
    ]

    for fs in ct_fspaces
        cb = TPSChem.compute_cluster_eigenbasis_spin(
            ints, clusters, d1, [10,10,10], fs, max_roots=5, verbose=0)
        cluster_bases = TPSChem.merge_cluster_bases(cluster_bases, cb)
    end

clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters);
cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints);

TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);

nroots=10


ci_vector = TPSChem.TPSCIstate(clusters, init_fspace, R=nroots)
ci_vector = TPSChem.add_spin_focksectors(ci_vector)

# form single excitonic states by adding CT sectors to the reference root
TPSChem.add_fockconfig!(ci_vector, FockConfig([(2,0), (3, 3), (0, 4)]))
TPSChem.add_fockconfig!(ci_vector, FockConfig([(2,0), (3, 3), (1, 3)]))
TPSChem.add_fockconfig!(ci_vector, FockConfig([(3,0), (2, 3), (1, 3)]))
TPSChem.add_fockconfig!(ci_vector, FockConfig([(3,1), (3, 3), (0, 2)]))
TPSChem.add_fockconfig!(ci_vector, FockConfig([(3,1), (3, 2), (0, 3)]))
TPSChem.add_fockconfig!(ci_vector, FockConfig([(2,1), (3, 3), (1, 2)]))
# ------------------------------------------------------------------
# Charge-transfer Fock sectors for excited electronic states.

# Single-electron CT (Na=6, Nb=6 conserved throughout):
#   α: C1(3,0)→C3(0,3)   new sectors: C1=(2,0), C3=(1,3)
#   α: C2(3,3)→C3(0,3)   new sectors: C2=(2,3), C3=(1,3)
#   β: C3(0,3)→C1(3,0)   new sectors: C3=(0,2), C1=(3,1)
#   β: C2(3,3)→C1(3,0)   new sectors: C2=(3,2), C1=(3,1)
#
# Double CT / spin-exchange C1↔C3 (−1α+1β on C1, +1α−1β on C3):
#   new sectors: C1=(2,1), C3=(1,2)  
# ------------------------------------------------------------------

fspace_0 = init_fspace

# α: C1 → C3
tmp_fspace = TPSChem.replace(fspace_0, (1,3), ([2,0],[1,3]))
TPSChem.add_fockconfig!(ci_vector, tmp_fspace)
ci_vector[tmp_fspace][TPSChem.ClusterConfig([1,1,1])] = zeros(Float64,nroots)

# α: C2 → C3
tmp_fspace = TPSChem.replace(fspace_0, (2,3), ([2,3],[1,3]))
TPSChem.add_fockconfig!(ci_vector, tmp_fspace)
ci_vector[tmp_fspace][TPSChem.ClusterConfig([1,1,1])] = zeros(Float64,nroots)

# β: C3 → C1
tmp_fspace = TPSChem.replace(fspace_0, (1,3), ([3,1],[0,2]))
TPSChem.add_fockconfig!(ci_vector, tmp_fspace)
ci_vector[tmp_fspace][TPSChem.ClusterConfig([1,1,1])] = zeros(Float64,nroots)

# β: C2 → C1
tmp_fspace = TPSChem.replace(fspace_0, (1,2), ([3,1],[3,2]))
TPSChem.add_fockconfig!(ci_vector, tmp_fspace)
ci_vector[tmp_fspace][TPSChem.ClusterConfig([1,1,1])] = zeros(Float64,nroots)

# Spin-exchange C1↔C3: -1α+1β on C1, +1α-1β on C3
tmp_fspace = TPSChem.replace(fspace_0, (1,3), ([2,1],[1,2]))
TPSChem.add_fockconfig!(ci_vector, tmp_fspace)
ci_vector[tmp_fspace][TPSChem.ClusterConfig([1,1,1])] = zeros(Float64,nroots)

# Add spin sectors for all CT FockConfigs
ci_vector = TPSChem.add_spin_focksectors(ci_vector)

TPSChem.eye!(ci_vector)

eci, v = TPSChem.tps_ci_direct(ci_vector, cluster_ops, clustered_ham);

e0a, v0a = TPSChem.tpsci_ci(ci_vector, cluster_ops, clustered_ham, incremental=true,
                            thresh_cipsi = 1e-3,
                            thresh_foi   = 1e-6,
                            thresh_asci  = -1);
γ_aa, γ_bb = TPSChem.compute_1rdm(v0a, cluster_ops)

# ------------------------------------------------------------------
# Diagnostic 1: trace of the 1-RDM per root
# Tr(γ_aa[:,:,r,r] + γ_bb[:,:,r,r]) must equal total number of electrons.
# If these are correct the code is working; if zero there is a bug.
n_elec = sum(init_fspace[k][1] + init_fspace[k][2] for k in 1:length(clusters))
γ_total = γ_aa .+ γ_bb
@printf("\n 1-RDM trace check (should be %.1f electrons each root):\n", float(n_elec))
for r in 1:nroots
    @printf("   root %i: Tr(γ) = %.6f\n", r, tr(γ_total[:,:,r,r]))
end

# Diagnostic 2: raw transition dipole values between ref root and all others.
# If these are all zero, the transitions are symmetry/spin-forbidden (not a bug).
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
npzwrite("tpsci_spectrum.npz", Dict(
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
        title="TPSCI absorption — Cr₂O(NH₃)₈⁴⁺",
        label="TPSCI (Lorentzian)", legend=:topright)
    
    scale = maximum(I) / (maximum(f_sticks) + 1e-10)
    for (e, fi) in zip(ΔE_ev, f_sticks)
        fi > 1e-4 || continue
        plot!(p, [e, e], [0.0, fi * scale], lw=1.5, color=:red, alpha=0.6, label=false)
    end
    
    savefig(p, "tpsci_spectrum.pdf")
    display(p)

# --------------------------------------------------------------------------
# Dipole integral matrices (CMF MO basis, shape 3 × norb × norb)
# --------------------------------------------------------------------------
μ_x = dip_cmf[1, :, :]   # (norb, norb)
μ_y = dip_cmf[2, :, :]
μ_z = dip_cmf[3, :, :]

nroots = size(γ_aa, 3)

# ==========================================================================
# Method A: 1-RDM contraction
# ==========================================================================
println("\n" * "="^70)
println(" Method A: 1-RDM contraction")
println("="^70)

P_x_A = TPSChem.contract_1rdm_property(γ_aa, γ_bb, μ_x)
P_y_A = TPSChem.contract_1rdm_property(γ_aa, γ_bb, μ_y)
P_z_A = TPSChem.contract_1rdm_property(γ_aa, γ_bb, μ_z)

# ==========================================================================
# Method B: direct (cluster-loop contraction, no full RDM)
# ==========================================================================
println("\n" * "="^70)
println(" Method B: direct cluster-operator contraction")
println("="^70)

P_x_B = TPSChem.compute_1e_property_direct(v0a, cluster_ops, μ_x)
P_y_B = TPSChem.compute_1e_property_direct(v0a, cluster_ops, μ_y)
P_z_B = TPSChem.compute_1e_property_direct(v0a, cluster_ops, μ_z)

# ==========================================================================
# Comparison: max |A - B|
# ==========================================================================
println("\n" * "="^70)
println(" Comparison: max |P_A[r1,r2] - P_B[r1,r2]|")
println("="^70)
err_x = maximum(abs.(P_x_A .- P_x_B))
err_y = maximum(abs.(P_y_A .- P_y_B))
err_z = maximum(abs.(P_z_A .- P_z_B))
@printf("   μ_x: %.3e   μ_y: %.3e   μ_z: %.3e\n", err_x, err_y, err_z)
tol = 1e-8
if max(err_x, err_y, err_z) < tol
    println("   OK — methods agree to within $tol a.u.")
else
    println("   WARNING — discrepancy > $tol: check orbital ordering or sign conventions")
end

# ==========================================================================
# Diagonal (expectation) values per root
# ==========================================================================
println("\n" * "="^70)
println(" Dipole expectation values <r|μ|r>  (a.u.)")
println("="^70)
@printf("   %-6s  %-14s  %-14s  %-14s  %-14s  %-14s  %-14s\n",
        "Root",
        "μ_x (1-RDM)", "μ_x (direct)",
        "μ_y (1-RDM)", "μ_y (direct)",
        "μ_z (1-RDM)", "μ_z (direct)")
for r in 1:nroots
    @printf("   %-6i  %-14.8f  %-14.8f  %-14.8f  %-14.8f  %-14.8f  %-14.8f\n",
            r,
            P_x_A[r,r], P_x_B[r,r],
            P_y_A[r,r], P_y_B[r,r],
            P_z_A[r,r], P_z_B[r,r])
end

# ==========================================================================
# Transition dipole moments from root 1 (off-diagonal |<1|μ|n>|)
# ==========================================================================
println("\n" * "="^70)
println(" Transition dipoles |<1|μ_α|n>| from root 1  (a.u.)")
println("="^70)
@printf("   %-6s  %-12s  %-12s  %-12s  %-12s  %-12s  %-12s\n",
        "n",
        "|μ_x| A", "|μ_x| B",
        "|μ_y| A", "|μ_y| B",
        "|μ_z| A", "|μ_z| B")
for n in 2:nroots
    @printf("   %-6i  %-12.6f  %-12.6f  %-12.6f  %-12.6f  %-12.6f  %-12.6f\n",
            n,
            abs(P_x_A[1,n]), abs(P_x_B[1,n]),
            abs(P_y_A[1,n]), abs(P_y_B[1,n]),
            abs(P_z_A[1,n]), abs(P_z_B[1,n]))
end

# ==========================================================================
# Oscillator strengths from both methods (should be identical)
# ==========================================================================
println("\n" * "="^70)
println(" Oscillator strengths f_0n from both methods")
println("="^70)
ha2ev = 27.2114
@printf("   %-6s  %-12s  %-12s  %-12s  %-14s\n",
        "n", "ΔE (eV)", "f (1-RDM)", "f (direct)", "|Δf|")
for n in 2:nroots
    ΔE = e0a[n] - e0a[1]
    ΔE > 0 || continue
    # f = (2/3) ΔE (|<0|μ_x|n>|² + |<0|μ_y|n>|² + |<0|μ_z|n>|²)
    f_A = (2/3) * ΔE * (abs2(P_x_A[1,n]) + abs2(P_y_A[1,n]) + abs2(P_z_A[1,n]))
    f_B = (2/3) * ΔE * (abs2(P_x_B[1,n]) + abs2(P_y_B[1,n]) + abs2(P_z_B[1,n]))
    @printf("   %-6i  %-12.4f  %-12.6f  %-12.6f  %-14.2e\n",
            n, ΔE * ha2ev, f_A, f_B, abs(f_A - f_B))
end
end

"""1-RDM trace check (should be 12.0 electrons each root):
root 1: Tr(γ) = 12.000000
root 2: Tr(γ) = 12.000000
root 3: Tr(γ) = 12.000000
root 4: Tr(γ) = 12.000000
root 5: Tr(γ) = 12.000000
root 6: Tr(γ) = 12.000000
root 7: Tr(γ) = 12.000000
root 8: Tr(γ) = 12.000000
root 9: Tr(γ) = 12.000000
root 10: Tr(γ) = 12.000000

Transition dipole moments from root 1 (a.u.):
Root    <0|μ_x|n>     <0|μ_y|n>     <0|μ_z|n>   
2       0.000000      -0.000000     -0.000000   
3       -0.000078     0.000130      0.000095    
4       0.000000      -0.000000     -0.000000   
5       0.000000      -0.000001     0.000001    
6       0.000001      -0.000001     0.000003    
7       0.013241      -0.004510     0.017038    
8       -0.012411     -0.015021     0.009471    
9       -0.001833     0.002499      -0.004456   
10      -0.000000     0.000000      0.000000    

Root    ΔE (eV)         f (osc. str.) 
------------------------------------------
2       0.009989        0.00000000    
3       0.053439        0.00000000    
4       0.109049        0.00000000    
5       2.355212        0.00000000    
6       2.371209        0.00000000    
7       2.381180        0.00002835    
8       2.383783        0.00002741    
9       2.388008        0.00000172    
10      2.413255        0.00000000    


======================================================================
Method A: 1-RDM contraction
======================================================================

======================================================================
Method B: direct cluster-operator contraction
======================================================================

======================================================================
Comparison: max |P_A[r1,r2] - P_B[r1,r2]|
======================================================================
μ_x: 3.225e-14   μ_y: 4.217e-14   μ_z: 2.982e-14
OK — methods agree to within 1.0e-8 a.u.

======================================================================
Dipole expectation values <r|μ|r>  (a.u.)
======================================================================
Root    μ_x (1-RDM)     μ_x (direct)    μ_y (1-RDM)     μ_y (direct)    μ_z (1-RDM)     μ_z (direct)  
1       -0.00040801     -0.00040801     -0.00001057     -0.00001057     -0.00001350     -0.00001350   
2       -0.00045041     -0.00045041     0.00005987      0.00005987      0.00003781      0.00003781    
3       -0.00047225     -0.00047225     0.00008417      0.00008417      0.00005830      0.00005830    
4       -0.00054253     -0.00054253     0.00019109      0.00019109      0.00014166      0.00014166    
5       -0.00332216     -0.00332216     0.00455802      0.00455802      0.00333970      0.00333970    
6       -0.04650354     -0.04650354     0.07309440      0.07309440      0.05263273      0.05263273    
7       -0.05444893     -0.05444893     0.08495113      0.08495113      0.06163305      0.06163305    
8       -0.05728552     -0.05728552     0.08909022      0.08909022      0.06462140      0.06462140    
9       0.04536345      0.04536345      -0.07088822     -0.07088822     -0.05186268     -0.05186268   
10      0.00144624      0.00144624      -0.00293712     -0.00293712     -0.00211874     -0.00211874   

======================================================================
Transition dipoles |<1|μ_α|n>| from root 1  (a.u.)
======================================================================
n       |μ_x| A       |μ_x| B       |μ_y| A       |μ_y| B       |μ_z| A       |μ_z| B     
2       0.000000      0.000000      0.000000      0.000000      0.000000      0.000000    
3       0.000078      0.000078      0.000130      0.000130      0.000095      0.000095    
4       0.000000      0.000000      0.000000      0.000000      0.000000      0.000000    
5       0.000000      0.000000      0.000001      0.000001      0.000001      0.000001    
6       0.000001      0.000001      0.000001      0.000001      0.000003      0.000003    
7       0.013241      0.013241      0.004510      0.004510      0.017038      0.017038    
8       0.012411      0.012411      0.015021      0.015021      0.009471      0.009471    
9       0.001833      0.001833      0.002499      0.002499      0.004456      0.004456    
10      0.000000      0.000000      0.000000      0.000000      0.000000      0.000000    

======================================================================
Oscillator strengths f_0n from both methods
======================================================================
n       ΔE (eV)       f (1-RDM)     f (direct)    |Δf|          
2       0.0100        0.000000      0.000000      2.26e-25      
3       0.0534        0.000000      0.000000      1.66e-22      
4       0.1090        0.000000      0.000000      3.55e-25      
5       2.3552        0.000000      0.000000      1.58e-24      
6       2.3712        0.000000      0.000000      1.86e-25      
7       2.3812        0.000028      0.000028      1.09e-18      
8       2.3838        0.000027      0.000027      5.08e-20      
9       2.3880        0.000002      0.000002      1.95e-19      
10      2.4133        0.000000      0.000000      7.27e-25      
"""