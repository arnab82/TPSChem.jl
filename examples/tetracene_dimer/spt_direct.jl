using TPSChem.QCBase
using Printf
using TPSChem
using NPZ
using TPSChem.InCoreIntegrals
using TPSChem.RDM
using JLD2


@load  "data_cmf_TD_12.jld2"

M = 50

display(clusters)
display(init_fspace)

ref_fspace = FockConfig(init_fspace)
ecore = ints.h0

cluster_bases = TPSChem.compute_cluster_eigenbasis_spin(ints, clusters, d1, [3,3], ref_fspace, max_roots=M, verbose=1);

clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints);
# 
TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);
# @save "cmf_op_TD.jld2" clusters init_fspace ints cluster_bases cluster_ops clustered_ham
nroots = 10
ci_vector = SPTstate(clusters,TPSChem.FockConfig(init_fspace), cluster_bases, R=nroots);


# # Add the lowest energy single exciton to basis

ci_vector[TPSChem.FockConfig(init_fspace)][TPSChem.TuckerConfig((1:1,1:1))] =
    TPSChem.Tucker(tuple([zeros(Float64, 1, 1) for _ in 1:nroots]...))
TPSChem.add_single_excitons!(ci_vector,TPSChem.FockConfig(init_fspace),nroots)
TPSChem.add_double_excitons!(ci_vector,TPSChem.FockConfig(init_fspace),nroots)

#electron transfer states
fspace_0 = TPSChem.FockConfig(init_fspace)
# TPSChem.add_1electron_transfers!(ci_vector, fspace_0, 1)
TPSChem.add_spin_flip_states!(ci_vector, fspace_0,1)
display(ci_vector.data)
TPSChem.eye!(ci_vector)
display(ci_vector)
e_ci, v2 = TPSChem.ci_solve(ci_vector, cluster_ops, clustered_ham);
e_var, v_var = TPSChem.subspace_product_tucker(v2, cluster_ops, clustered_ham,
                                               max_iter    = 200,
                                               nbody       = 4,
                                               H0          = "Hcmf",
                                               thresh_var  = 1e-3,
                                               thresh_foi  = 1e-5,
                                               thresh_pt   = 1e-4,
                                               ci_conv     = 1e-5,
                                               do_pt       = true,
                                               resolve_ss  = false,
                                               tol_tucker  = 1e-4,
                                               solver      = "davidson")
@time ept2 = TPSChem.compute_pt2_energy(v_var, cluster_ops, clustered_ham, thresh_foi=1e-6,prescreen   = true,compress_twice = true)
@time ept2 = TPSChem.compute_pt2_energy2(v_var, cluster_ops, clustered_ham, thresh_foi=1e-6,prescreen   = true,compress_twice = true)
e_var, v_var = TPSChem.subspace_product_tucker(v_var, cluster_ops, clustered_ham,
                                               max_iter    = 200,
                                               nbody       = 4,
                                               H0          = "Hcmf",
                                               thresh_var  = 5e-4,
                                               thresh_foi  = 1e-6,
                                               thresh_pt   = 1e-4,
                                               ci_conv     = 1e-5,
                                               do_pt       = true,
                                               resolve_ss  = false,
                                               tol_tucker  = 1e-4,
                                               solver      = "davidson")
