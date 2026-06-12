using TPSChem
using TPSChem.QCBase
using TPSChem.RDM
using LinearAlgebra
using Test
using JLD2

@testset "spt_helpers" begin

    @load "_testdata_cmf_he4.jld2"

    ref_fock = FockConfig(init_fspace)
    N = length(clusters)
    (na, nb) = sum(init_fspace)

    # Helper: fresh SPTstate with p-space filled for the reference fock sector
    function make_spt(; R=1)
        v = SPTstate(clusters, ref_fock, cluster_bases; R=R)
        TPSChem.fill_p_space!(v, na, nb)
        return v
    end

    # ---------------------------------------------------------------
    @testset "add_single_excitons!" begin
        v = make_spt()

        n_before = length(v)
        TPSChem.add_single_excitons!(v, ref_fock, 1)
        n_after = length(v)

        # Each cluster that has a q-space adds one new TuckerConfig
        @test n_after > n_before

        # Every TuckerConfig in the reference fock sector must be non-empty
        @test haskey(v.data, ref_fock)
        for (tconfig, tuck) in v.data[ref_fock]
            @test length(tuck) > 0
        end
    end

    # ---------------------------------------------------------------
    @testset "add_double_excitons!" begin
        v = make_spt()

        n_before = length(v)
        TPSChem.add_double_excitons!(v, ref_fock, 1)
        n_after = length(v)

        # Each valid cluster pair (ci < cj) both with q-spaces adds a new config
        @test n_after > n_before

        @test haskey(v.data, ref_fock)
    end

    # ---------------------------------------------------------------
    @testset "add_single and double together" begin
        v = make_spt()

        n_p = length(v)
        TPSChem.add_single_excitons!(v, ref_fock, 1)
        n_single = length(v)
        TPSChem.add_double_excitons!(v, ref_fock, 1)
        n_double = length(v)

        @test n_single > n_p
        @test n_double > n_single
    end

    # ---------------------------------------------------------------
    @testset "add_1electron_transfers!" begin
        v = make_spt()

        n_fock_before = length(v.data)
        TPSChem.add_1electron_transfers!(v, ref_fock, 1)
        n_fock_after = length(v.data)

        # Should have added new fock sectors (one per ordered cluster pair × 2 spins)
        @test n_fock_after > n_fock_before

        # Every new fock sector should have the correct total electron count
        for (fock, _) in v.data
            @test TPSChem.n_elec_a(fock) + TPSChem.n_elec_b(fock) == na + nb
        end
    end

    # ---------------------------------------------------------------
    @testset "add_spin_flip_states!" begin
        # Spin flips require (0,2)/(2,0) sectors; rebuild cluster_bases with delta_elec=2
        cluster_bases_2 = TPSChem.compute_cluster_eigenbasis(
            ints, clusters; verbose=0, max_roots=5,
            init_fspace=init_fspace, delta_elec=2, rdm1a=d1.a, rdm1b=d1.b)

        v = SPTstate(clusters, ref_fock, cluster_bases_2; R=1)
        TPSChem.fill_p_space!(v, na, nb)

        n_fock_before = length(v.data)
        TPSChem.add_spin_flip_states!(v, ref_fock, 1)
        n_fock_after = length(v.data)

        @test n_fock_after > n_fock_before

        for (fock, _) in v.data
            @test TPSChem.n_elec_a(fock) + TPSChem.n_elec_b(fock) == na + nb
        end
    end

end
