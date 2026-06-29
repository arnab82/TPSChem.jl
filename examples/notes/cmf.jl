using TPSChem
using PyCall
using TPSChem.QCBase
using TPSChem.ClusterMeanField
using NPZ
using TPSChem.InCoreIntegrals
using TPSChem.RDM
using JLD2
using Printf
using TPSChem.ActiveSpaceSolvers: FCIAnsatz
C = npzread("mo_coeffs.npy")
h0 = npzread("ints_h0.npy")
h1 = npzread("ints_h1.npy")
h2 = npzread("ints_h2.npy")
ints = InCoreInts(h0, h1, h2)

Pa = npzread("Pa.npy")
Pb = npzread("Pb.npy")
@printf(" Input energy:    %12.8f\n", compute_energy(ints, RDM1(Pa, Pb)))


init_fspace=  [(3, 0), (3, 3), (3, 0)]
clusters   =  [[1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11, 12, 13]]

clusters = [MOCluster(i, collect(clusters[i])) for i = 1:length(clusters)]
display(clusters)

rdm1 = RDM1(n_orb(ints))

ansatze=[FCIAnsatz(5,3,0), FCIAnsatz(3,3,3), FCIAnsatz(5,3,0)]
@time e_cmf, U, d1 = ClusterMeanField.cmf_oo_newton(ints, clusters, init_fspace,ansatze,rdm1, maxiter_oo = 400,
                           tol_oo=1e-8, 
                           tol_d1=1e-9, 
                           tol_ci=1e-11,
                           verbose=4, 
                           zero_intra_rots = false,
                           sequential=true)


d1=orbital_rotation(d1, U)
ints = orbital_rotation(ints, U)
Ccmf=C* U

dip_mo  = npzread("dipole_ints.npy")   # shape [3, n_act, n_act] in Julia
dip_cmf = similar(dip_mo)
for x in 1:3
    dip_cmf[x,:,:] .= U' * dip_mo[x,:,:] * U
end

nabla_mo  = npzread("nabla_ints.npy")
nabla_cmf = similar(nabla_mo)
for x in 1:3
    nabla_cmf[x,:,:] .= U' * nabla_mo[x,:,:] * U
end
soc_Lx = npzread("soc_Lx.npy")
soc_Ly = npzread("soc_Ly.npy")
soc_Lz = npzread("soc_Lz.npy")

# Rotate SOC matrices into CMF basis: h'[p,q] = U' * h[p,q] * U
# Same one-electron transformation as dipole/nabla above.
soc_Lx_cmf = U' * soc_Lx * U
soc_Ly_cmf = U' * soc_Ly * U
soc_Lz_cmf = U' * soc_Lz * U

# Re-extract cluster-pair exchange integrals from the already-rotated ints.h2.
# ints = orbital_rotation(ints, U) above already applied the full 4-index rotation,
# so we just slice out the cluster-pair blocks.
# Convention: ints.h2[p,r,q,s] = (pr|qs) for p,r in cluster A; q,s in cluster B.
c1 = clusters[1].orb_list
c2 = clusters[2].orb_list
c3 = clusters[3].orb_list
K_12_cmf = ints.h2[c1, c1, c2, c2]   # (n_C1, n_C1, n_C2, n_C2): K[p,r,q,s]=(pr|qs), p,r∈C1; q,s∈C2
K_13_cmf = ints.h2[c1, c1, c3, c3]   # (n_C1, n_C1, n_C3, n_C3): K[p,r,q,s]=(pr|qs), p,r∈C1; q,s∈C3
K_23_cmf = ints.h2[c2, c2, c3, c3]   # (n_C2, n_C2, n_C3, n_C3): K[p,r,q,s]=(pr|qs), p,r∈C2; q,s∈C3

@save "data_cmf.jld2" clusters init_fspace ints d1 e_cmf U dip_cmf nabla_cmf Ccmf soc_Lx_cmf soc_Ly_cmf soc_Lz_cmf K_12_cmf K_13_cmf K_23_cmf
