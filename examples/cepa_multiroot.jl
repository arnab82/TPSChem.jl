using LinearAlgebra
using TPSChem
using Printf
using Test
using JLD2

@load "./pt2_test/_testdata_cmf_h9.jld2"
# define clusters
ref_fock = FockConfig(init_fspace)
# Do TPS
M=100
cluster_bases = TPSChem.compute_cluster_eigenbasis_spin(ints, clusters, d1, [5,5,5], ref_fock, max_roots=M, verbose=1);
#cluster_bases = TPSChem.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=M, init_fspace=init_fspace, rdm1a=d1.a, rdm1b=d1.b, T=Float64)

clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints);

TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);

nroots=3

# TPSCI
#
nroots=3
ci_vector = TPSChem.TPSCIstate(clusters, ref_fock, R=nroots)

ci_vector = TPSChem.add_spin_focksectors(ci_vector)

display(ci_vector)
etpsci, vtpsci = TPSChem.tps_ci_direct(ci_vector, cluster_ops, clustered_ham);

@time ept1 = TPSChem.compute_pt2_energy(vtpsci, cluster_ops, clustered_ham, thresh_foi=1e-12)

# start by defining P/Q spaces
p_spaces = Vector{ClusterSubspace}()

for ci in clusters
    ssi = ClusterSubspace(clusters[ci.idx])

    num_states_in_p_space = 1
    # our clusters are near triangles, with degenerate gs, so keep two states
    add_subspace!(ssi, ref_fock[ci.idx], 1:num_states_in_p_space)
    add_subspace!(ssi, (ref_fock[ci.idx][2], ref_fock[ci.idx][1]), 1:num_states_in_p_space) # add flipped spin
    push!(p_spaces, ssi)
end

ci_vector = SPTstate(clusters, p_spaces, cluster_bases, R=nroots) 

na = 5
nb = 4
TPSChem.fill_p_space!(ci_vector, na, nb)
TPSChem.eye!(ci_vector)
ebst, vbst = TPSChem.ci_solve(ci_vector, cluster_ops, clustered_ham)

e_cepa_corr = TPSChem.do_fois_cepa(vbst, cluster_ops, clustered_ham,cepa_shift="acpf", thresh_foi=1e-5, max_iter=50, nbody=4,tol=1e-8);
e_pt2, v_pt2 =    TPSChem.do_fois_pt2(vbst, cluster_ops, clustered_ham, thresh_foi=1e-5, max_iter=50, nbody=4, tol=1e-8);
e_ci, v_ci =  TPSChem.do_fois_ci(vbst, cluster_ops, clustered_ham, thresh_foi=1e-5, max_iter=50, nbody=4, tol=1e-8);
println("cepa energy: ", e_cepa_corr)
println("CI energy: ", e_ci)
println("PT2 energy: ", e_pt2)
# display(ebst)
# display(e_cepa_corr)
# display(e_pt2)
# display(e_ci)
# for j in 1:nroots
#     @printf(" CI  : %12.6f \n",(e_ci[j]+ints.h0))
#     @printf(" PT2 : %12.6f \n",(e_pt2[j]+ints.h0))
#     @printf("CEPA : %12.6f \n",(e_cepa_corr[j]+ints.h0))
# end
# display(e_fci)
