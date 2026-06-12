using TPSChem
using Printf
using Test
using JLD2 

#@testset "SPTstate" begin
if false 
    @load "_testdata_cmf_h12_64bit.jld2"
    v = TPSChem.SPTstate(clusters, FockConfig(init_fspace), cluster_bases)
    
    e_ci, v_ci = TPSChem.ci_solve(v, cluster_ops, clustered_ham)
    display(e_ci)
    @test isapprox(e_ci[1], -18.31710895, atol=1e-8)

   
    v = TPSChem.SPTstate(v,R=1)
    xspace  = TPSChem.build_compressed_1st_order_state(v, cluster_ops, clustered_ham, nbody=4, thresh=1e-3)
    xspace = TPSChem.compress(xspace, thresh=1e-3)

    TPSChem.nonorth_add!(v, xspace)
    v = TPSChem.SPTstate(v,R=4)
    TPSChem.randomize!(v)
    TPSChem.orthonormalize!(v)
    
    e_ci, v = TPSChem.ci_solve(v, cluster_ops, clustered_ham)
    e_ci, v = TPSChem.ci_solve(v, cluster_ops, clustered_ham, precond=true)
    # error("nick")
    
    e_pt, v_pt = TPSChem.do_fois_pt2(v, cluster_ops, clustered_ham, thresh_foi=1e-3, max_iter=50, tol=1e-8)
    
    if true 
        e_var, v_var = TPSChem.subspace_product_tucker(v, cluster_ops, clustered_ham,
                                               max_iter    = 20,
                                               nbody       = 4,
                                               H0          = "Hcmf",
                                               thresh_var  = 1e-2,
                                               thresh_foi  = 1e-3,
                                               thresh_pt   = 1e-3,
                                               ci_conv     = 1e-5,
                                               do_pt       = true,
                                               resolve_ss  = false,
                                               tol_tucker  = 1e-4,
                                               solver      = "davidson")
    end
       
#    for i in 1:4
#        v = TPSChem.compress(v, thresh=1e-3)
#        TPSChem.orthonormalize!(v)
#        xspace  = TPSChem.build_compressed_1st_order_state(v, cluster_ops, clustered_ham, nbody=4, thresh=1e-3)
#
#        display(size(xspace))
#        xspace = TPSChem.compress(xspace, thresh=1e-3)
#        display(size(xspace))
#
#        TPSChem.zero!(xspace)
#        TPSChem.nonorth_add!(v, xspace)
#        TPSChem.orthonormalize!(v)
#        e, v = TPSChem.ci_solve(v, cluster_ops, clustered_ham)
#    end
#
#end
end

@testset "SPT" begin


    @load "_testdata_cmf_h12_64bit.jld2"
    clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
    cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints);
    TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);

    v = TPSChem.SPTstate(clusters, FockConfig(init_fspace), cluster_bases)

    e_var, v_var = TPSChem.subspace_product_tucker(v, cluster_ops, clustered_ham,
                                               max_iter    = 20,
                                               nbody       = 4,
                                               H0          = "Hcmf",
                                               thresh_var  = 1e-2,
                                               thresh_foi  = 1e-3,
                                               thresh_pt   = sqrt(1e-5),
                                               ci_conv     = 1e-5,
                                               do_pt       = true,
                                               resolve_ss  = true,
                                               tol_tucker  = 1e-4, 
                                               verbose     = 1)

    @test isapprox(e_var[1], -18.329455496968325, atol=1e-6)
    # @test isapprox(e_var[1], -18.32945552731658, atol=1e-8)
    

    e_cepa, v_cepa = TPSChem.do_fois_cepa(v, cluster_ops, clustered_ham, thresh_foi=1e-3, max_iter=50, tol=1e-8, prescreen=false)
    display(e_cepa)
    @test isapprox(e_cepa[1], -18.329789530070542, atol=1e-8)
    
    e_pt = TPSChem.compute_pt2_energy(v, cluster_ops, clustered_ham, thresh_foi=1e-3)
    
    e_pt, v_pt = TPSChem.do_fois_pt2(v, cluster_ops, clustered_ham, thresh_foi=1e-3, max_iter=50, tol=1e-8)
    display(e_pt)
    @test isapprox(e_pt[1], -18.326971095822113, atol=1e-8)

    e_ci, v_ci = TPSChem.ci_solve(v_cepa, cluster_ops, clustered_ham)
    display(e_ci)
    @test isapprox(e_ci[1], -18.329641438975646, atol=1e-8)

end
if false
end
