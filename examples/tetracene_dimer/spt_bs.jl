using TPSChem.QCBase
using Printf
using TPSChem
using NPZ
using TPSChem.InCoreIntegrals
using TPSChem.RDM
using JLD2

@load  "data_cmf_TD_12.jld2"
M = 50
#@load "cmf_op_TD_with_ops.jld2"
display(clusters)
display(init_fspace)
ref_fspace = FockConfig(init_fspace)
ecore = ints.h0
cluster_bases = TPSChem.compute_cluster_eigenbasis_spin(ints, clusters, d1, [3,3], ref_fspace, max_roots=M, verbose=1);
 
clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints);
TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);
  
v = TPSChem.BSstate(clusters, TPSChem.FockConfig(init_fspace), cluster_bases, R=10)
TPSChem.add_single_excitons_upto_L!(v,4)
TPSChem.add_double_excitons_upto_L!(v,4)
# TPSChem.add_1electron_transfers!(v)
TPSChem.add_spin_flip_states!(v,init_fspace)
TPSChem.eye!(v)

display(v)
# e_ci, v_ci = TPSChem.ci_solve(v, cluster_ops, clustered_ham, solver="davidson");
e_ci, v_ci = TPSChem.ci_solve(v, cluster_ops, clustered_ham, solver="krylovkit", verbose=2);

v_bst = TPSChem.SPTstate(v_ci, thresh=1e-5)

display(v_bst)
TPSChem.randomize!(v_bst)
TPSChem.orthonormalize!(v_bst)
# TPSChem.eye!(v_bst)
display(v_bst)
σ = TPSChem.build_compressed_1st_order_state(v_bst, cluster_ops, clustered_ham, 
                                    nbody=4,
                                    thresh=1e-3)
σ = TPSChem.compress(σ, thresh=1e-5)
v2 = SPTstate(σ,R=10)
TPSChem.eye!(v2)
e_ci, v2 = TPSChem.ci_solve(v2, cluster_ops, clustered_ham);
e_var, v_var = TPSChem.subspace_product_tucker(v2, cluster_ops, clustered_ham,
                                               max_iter    = 20,
                                               nbody       = 4,
                                               H0          = "Hcmf",
                                               thresh_var  = 1e-2,
                                               thresh_foi  = 1e-4,
                                               thresh_pt   = 1e-3,
                                               ci_conv     = 1e-5,
                                               do_pt       = true,
                                               resolve_ss  = false,
                                               tol_tucker  = 1e-4,
                                               solver      = "davidson")
# e_ci2, v_ci2 = TPSChem.ci_solve(v_bst, cluster_ops, clustered_ham, solver="davidson");
# e_ci2, v_ci2 = TPSChem.ci_solve(v_bst, cluster_ops, clustered_ham, solver="krylovkit", verbose=2);
