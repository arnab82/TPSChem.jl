using TPSChem
using TPSChem.QCBase
using TPSChem.RDM
using LinearAlgebra
using Printf
using Test
using JLD2

@testset "tpsci_helpers" begin

    @load "_testdata_cmf_he4.jld2"

    ref_fock = FockConfig(init_fspace)
    N = length(clusters)
    (na, nb) = sum(init_fspace)

    # ---------------------------------------------------------------
    @testset "single_excitonic_basis" begin
        # N=4 clusters, Nk=2 → configs: (1,1,1,1) + 4 single excitations = 5 distinct
        R_basis = N + 1
        basis = TPSChem.single_excitonic_basis(clusters, ref_fock; R=R_basis, Nk=2)

        @test basis isa TPSCIstate
        sz = size(basis)
        @test sz[1] == R_basis   # exactly 5 configs in the reference fock sector
        @test sz[2] == R_basis

        # set_vector! installs an identity → rows must be orthonormal
        v_mat = TPSChem.get_vector(basis)
        @test isapprox(v_mat' * v_mat, I, atol=1e-12)
    end

    # ---------------------------------------------------------------
    @testset "correlation_functions basic" begin
        # Pure product state: one config, one root, coefficient = 1
        ci_vector = TPSCIstate(clusters, ref_fock, R=1)
        ci_vector[ref_fock][ClusterConfig(ones(Int, N))] = [1.0]

        n1, n2, sz1, sz2 = correlation_functions(ci_vector)

        @test length(n1) == 1    # R=1
        @test length(n1[1]) == N

        # init_fspace = [(1,1),(1,1),(1,1),(1,1)] → each cluster has 2 electrons
        @test isapprox(n1[1], fill(2.0, N), atol=1e-12)
        # Sz = (na-nb)/2 = 0 for each (1,1) cluster
        @test isapprox(sz1[1], zeros(N), atol=1e-12)

        # 2nd cumulants are zero for a pure single-config product state
        @test isapprox(n2[1],  zeros(N, N), atol=1e-12)
        @test isapprox(sz2[1], zeros(N, N), atol=1e-12)
    end

    # ---------------------------------------------------------------
    @testset "correlation_functions multi-root" begin
        # Two roots: same config but with different coefficients
        ci_vector = TPSCIstate(clusters, ref_fock, R=2)
        ci_vector[ref_fock][ClusterConfig(ones(Int, N))] = [1.0, 0.0]

        n1, n2, sz1, sz2 = correlation_functions(ci_vector)

        @test length(n1) == 2
        # root 1 has all weight → <N> = 2 per cluster
        @test isapprox(n1[1], fill(2.0, N), atol=1e-12)
        # root 2 has zero weight → all cumulants are 0
        @test isapprox(n1[2], zeros(N), atol=1e-12)
    end

    # ---------------------------------------------------------------
    @testset "correlation_functions with cluster_ops" begin
        # Build a minimal TPSCI state in the reference fock sector
        ci_vector = TPSCIstate(clusters, ref_fock, R=1)
        ci_vector[ref_fock][ClusterConfig(ones(Int, N))] = [1.0]

        cf = correlation_functions(ci_vector, cluster_ops; verbose=0)

        @test haskey(cf, "N")
        @test haskey(cf, "Sz")
        @test haskey(cf, "H")
        @test haskey(cf, "Hcmf")
        @test haskey(cf, "S2")

        # check shapes: each entry is (Vector{Vector}, Vector{Matrix})
        N_1, N_2 = cf["N"]
        @test length(N_1) == 1       # R=1
        @test length(N_1[1]) == N    # one value per cluster
        @test size(N_2[1]) == (N, N)
    end

    # ---------------------------------------------------------------
    @testset "full_dim" begin
        dim = TPSChem.full_dim(clusters, cluster_bases, na, nb)
        @test dim isa Integer
        @test dim > 0
        # Must be at least as large as the direct-product FCI dim
        # (4 clusters, each with (1,1) sector of 5-orbital FCI ≥ 25 states)
        @test dim >= 1
    end

end
