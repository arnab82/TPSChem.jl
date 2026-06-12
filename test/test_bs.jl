using TPSChem
using Printf
using Test
using JLD2 

@testset "SPT vs BS" begin

    @load "_testdata_cmf_h12_64bit.jld2"
    
    clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
    cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints);
    TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);
  
    v = TPSChem.BSstate(clusters, TPSChem.FockConfig(init_fspace), cluster_bases, R=7)
    TPSChem.add_single_excitons!(v)
    TPSChem.add_1electron_transfers!(v)
    TPSChem.eye!(v)

    display(v)
    e_ci, v_ci = TPSChem.ci_solve(v, cluster_ops, clustered_ham, solver="davidson");
    #e_ci, v_ci = TPSChem.ci_solve(v, cluster_ops, clustered_ham, solver="krylovkit", verbose=2);

    v_bst = TPSChem.SPTstate(v_ci, thresh=1e-5)

    display(v_bst)
    TPSChem.randomize!(v_bst)
    TPSChem.orthonormalize!(v_bst)
    e_ci2, v_ci2 = TPSChem.ci_solve(v_bst, cluster_ops, clustered_ham, solver="davidson");
    #e_ci2, v_ci2 = TPSChem.ci_solve(v_bst, cluster_ops, clustered_ham, solver="krylovkit", verbose=2);

    for r in 1:TPSChem.nroots(v)
        @test isapprox(e_ci[r], e_ci2[r], atol=1e-8)
    end
end

@testset "SPT vs BS 2" begin
    @load "_testdata_cmf_he4.jld2"
    
    clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
    cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints);
    TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);
  
  
    v = TPSChem.BSstate(clusters, TPSChem.FockConfig(init_fspace), cluster_bases, R=5)
    TPSChem.add_single_excitons!(v)
    TPSChem.add_1electron_transfers!(v)
    TPSChem.randomize!(v)
    TPSChem.orthonormalize!(v)

    #TPSChem.eye!(v)

    display(v)
    e_ci, v_ci = TPSChem.ci_solve(v, cluster_ops, clustered_ham);
    #e_ci, v_ci = TPSChem.ci_solve(v, cluster_ops, clustered_ham, solver="krylovkit", verbose=2);

    v_bst = TPSChem.SPTstate(v_ci, thresh=1e-5)

    display(v_bst)
    #e_ci, v_ci = TPSChem.ci_solve(v, cluster_ops, clustered_ham, solver="davidson");
    TPSChem.randomize!(v_bst)
    TPSChem.orthonormalize!(v_bst)
    e_ci2, v_ci2 = TPSChem.ci_solve(v_bst, cluster_ops, clustered_ham);
    #e_ci2, v_ci2 = TPSChem.ci_solve(v_bst, cluster_ops, clustered_ham, solver="krylovkit", verbose=2);

    for r in 1:TPSChem.nroots(v)
        @test isapprox(e_ci[r], e_ci2[r], atol=1e-8)
    end
end
