using TPSChem
using TPSChem.QCBase
using TPSChem.RDM
using TPSChem.ClusterMeanField
using LinearAlgebra
using Printf
using Test

if get(ENV, "TPSCHEM_TEST_PYSCF", "0") == "1"

    using PyCall

    @testset "direct_cmf" begin

        # Minimal H4 square: 4 orbitals (sto-3g), 2 clusters of 2 orbitals, (1,1) electrons each
        atoms = [Atom(1, "H", [0.0, 0.0, 0.0]),
                 Atom(2, "H", [2.0, 0.0, 0.0]),
                 Atom(3, "H", [0.0, 2.0, 0.0]),
                 Atom(4, "H", [2.0, 2.0, 0.0])]

        mol = Molecule(0, 1, atoms, "sto-3g")

        clusters    = [MOCluster(1, [1, 2]),
                       MOCluster(2, [3, 4])]
        init_fspace = [(1, 1), (1, 1)]
        na, nb = 2, 2

        mf   = pyscf_do_scf(mol; verbose=0)
        nbas = size(mf.mo_coeff)[1]
        C    = mf.mo_coeff

        # Initial guess density
        rdm1a = zeros(nbas, nbas)
        rdm1b = zeros(nbas, nbas)
        for i in 1:na
            rdm1a[i, i] = 1.0
        end
        for i in 1:nb
            rdm1b[i, i] = 1.0
        end

        # -------------------------------------------------------
        @testset "cmf_ci_iteration (direct)" begin
            e, rdm1a_out, rdm1b_out, rdm1_dict, rdm2_dict =
                TPSChem.ClusterMeanField.cmf_ci_iteration(
                    mol, C, rdm1a, rdm1b, clusters, init_fspace;
                    verbose=0, ci_max_iter=100, ci_conv_tol=1e-8)

            @test isfinite(e)
            @test e < 0.0   # energy must be negative for bound system
            @test size(rdm1a_out) == (nbas, nbas)
            @test size(rdm1b_out) == (nbas, nbas)
            # each cluster's spin-summed 1-RDM trace = na_i + nb_i
            for ci in clusters
                na_i = init_fspace[ci.idx][1]
                nb_i = init_fspace[ci.idx][2]
                @test isapprox(tr(rdm1_dict[ci.idx]), Float64(na_i + nb_i), atol=1e-10)
            end
        end

        # -------------------------------------------------------
        @testset "cmf_ci (direct, convergence)" begin
            e, rdm1a_out, rdm1b_out, rdm1_dict, rdm2_dict =
                TPSChem.ClusterMeanField.cmf_ci(
                    mol, C, clusters, init_fspace, rdm1a, rdm1b;
                    max_iter=20, dconv=1e-8, econv=1e-10, verbose=0)

            @test isfinite(e)
            @test e < 0.0
            # density matrix must sum to correct number of electrons
            @test isapprox(tr(rdm1a_out), Float64(na), atol=1e-10)
            @test isapprox(tr(rdm1b_out), Float64(nb), atol=1e-10)
        end

        # -------------------------------------------------------
        @testset "compute_cmf_energy consistency" begin
            # run cmf_ci and verify compute_cmf_energy is consistent with the returned energy
            e_cmf, _, _, rdm1_dict, rdm2_dict =
                TPSChem.ClusterMeanField.cmf_ci(
                    mol, C, clusters, init_fspace, rdm1a, rdm1b;
                    max_iter=20, verbose=0)

            # rdm1_dict[ci.idx] is already spin-summed after the fix
            e_check = TPSChem.ClusterMeanField.compute_cmf_energy(
                mol, C, rdm1_dict, rdm2_dict, clusters; verbose=0)

            @test isfinite(e_check)
            @test isapprox(e_cmf, e_check, atol=1e-8)
        end

    end  # @testset "direct_cmf"

end  # if TPSCHEM_TEST_PYSCF
