using TPSChem
using TPSChem.QCBase
using TPSChem.RDM
using Test
using JLD2

@testset "fois_ci" begin
    @load "_testdata_cmf_h8.jld2"

    max_roots = 4
    nroots = 2
    ref_fock = FockConfig(init_fspace)

    cluster_bases = TPSChem.compute_cluster_eigenbasis(ints, clusters;
        verbose=0,
        max_roots=max_roots,
        init_fspace=init_fspace,
        rdm1a=d1.a,
        rdm1b=d1.b)

    clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
    cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints)
    TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)

    ci_vector = TPSChem.TPSCIstate(clusters, ref_fock, R=nroots, T=Float64)
    ci_vector[ref_fock][ClusterConfig([1, 1])] = [1.0, 0.0]
    ci_vector[ref_fock][ClusterConfig([2, 1])] = [0.0, 1.0]

    e_ci, v_ci = TPSChem.do_fois_ci(ci_vector, cluster_ops, clustered_ham;
        nbody=2,
        thresh_foi=1e-4,
        tol=1e-6,
        thresh_clip=1e-8,
        threaded=false,
        compress=true,
        pt=false,
        verbose=false)

    @test length(e_ci) == nroots
    @test all(isfinite, e_ci)
    @test v_ci isa TPSCIstate
    @test length(v_ci) >= length(ci_vector)
end
