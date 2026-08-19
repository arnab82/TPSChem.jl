using Distributed
using TPSChem
using TPSChem.QCBase
using TPSChem.RDM
using Test
using JLD2
using LinearAlgebra

# Spawn workers with the active project so the stdlib `Pkg` is reachable on them
# (bare `addprocs` under `Pkg.test` gives workers a restricted load path where
# `import Pkg` fails inside `ensure_tpsci_multinode_workers!`).
_property_multinode_added_procs = Int[]
if nprocs() == 1
    _property_multinode_added_procs = addprocs(2; exeflags="--project=$(Base.active_project())")
end

try
    @testset "multinode TPSCI properties, RDMs, and SPT variance" begin
        cd(@__DIR__) do
            @load "_testdata_cmf_he4.jld2"

            clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
            cluster_ops = TPSChem.compute_cluster_ops_2rdm(cluster_bases, ints)
            TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)

            ref_fock = FockConfig(init_fspace)
            v0 = TPSChem.TPSCIstate(clusters, ref_fock, R=2, T=Float64)
            v0[ref_fock][ClusterConfig([1, 1, 1, 1])] = [1.0, 0.0]
            v0[ref_fock][ClusterConfig([2, 1, 1, 1])] = [0.0, 1.0]

            gamma_aa, gamma_bb = TPSChem.compute_1rdm(v0, cluster_ops)
            gamma_aa_d, gamma_bb_d = TPSChem.compute_1rdm_distributed(
                v0, cluster_ops; workers=workers(), threaded_worker=false,
                blas_threads=1, verbose=0)
            @test isapprox(gamma_aa, gamma_aa_d, atol=1e-12)
            @test isapprox(gamma_bb, gamma_bb_d, atol=1e-12)

            gamma_ab, gamma_ba = TPSChem.compute_1rdm_sf(v0, cluster_ops)
            gamma_ab_d, gamma_ba_d = TPSChem.compute_1rdm_sf_distributed(
                v0, cluster_ops; workers=workers(), threaded_worker=false,
                blas_threads=1, verbose=0)
            @test isapprox(gamma_ab, gamma_ab_d, atol=1e-12)
            @test isapprox(gamma_ba, gamma_ba_d, atol=1e-12)

            P_direct = TPSChem.compute_1e_property_direct(v0, cluster_ops, ints.h1)
            P_direct_d = TPSChem.compute_1e_property_direct_distributed(
                v0, cluster_ops, ints.h1; workers=workers(), blas_threads=1,
                verbose=0)
            @test isapprox(P_direct, P_direct_d, atol=1e-12)

            Gamma = TPSChem.compute_2rdm(v0, cluster_ops)
            Gamma_d = TPSChem.compute_2rdm_distributed(
                v0, cluster_ops; workers=workers(), threaded_worker=false,
                blas_threads=1, verbose=0)
            @test isapprox(Gamma, Gamma_d, atol=1e-12)

            dops = TPSChem.compute_cluster_ops_2rdm_distributed(
                cluster_bases, ints; workers=workers(), verbose=0,
                blas_threads=1)
            TPSChem.add_cmf_operators_distributed!(
                dops, cluster_bases, ints, d1.a, d1.b; verbose=0,
                blas_threads=1)
            gamma_aa_sh, gamma_bb_sh = TPSChem.compute_1rdm_distributed(
                v0, dops; workers=workers(), threaded_worker=false,
                blas_threads=1, verbose=0)
            @test isapprox(gamma_aa, gamma_aa_sh, atol=1e-12)
            @test isapprox(gamma_bb, gamma_bb_sh, atol=1e-12)

            Gamma_sh = TPSChem.compute_2rdm_distributed(
                v0, dops; workers=workers(), threaded_worker=false,
                blas_threads=1, verbose=0)
            @test isapprox(Gamma, Gamma_sh, atol=1e-12)
            TPSChem.destroy!(dops)

            spt_ref = TPSChem.SPTstate(clusters, ref_fock, cluster_bases, R=1)
            e_spt = TPSChem.compute_expectation_value(
                spt_ref, cluster_ops, clustered_ham; nbody=2)
            e_spt_d = TPSChem.compute_expectation_value_distributed(
                spt_ref, cluster_ops, clustered_ham; nbody=2,
                workers=workers(), blas_threads=1, strategy=:hash)
            @test isapprox(e_spt, e_spt_d, atol=1e-12)

            fois = TPSChem.build_compressed_1st_order_state(
                spt_ref, cluster_ops, clustered_ham; nbody=2, thresh=1e-3)
            fois_d = TPSChem.build_compressed_1st_order_state_distributed(
                spt_ref, cluster_ops, clustered_ham; nbody=2, thresh=1e-3,
                workers=workers(), threaded_worker=false, blas_threads=1,
                strategy=:hash)
            @test length(fois) == length(fois_d)
            @test isapprox(TPSChem.nonorth_dot(fois, fois),
                           TPSChem.nonorth_dot(fois_d, fois_d),
                           rtol=1e-10, atol=1e-10)
            @test isapprox(TPSChem.nonorth_dot(fois, fois_d),
                           TPSChem.nonorth_dot(fois, fois),
                           rtol=1e-10, atol=1e-10)

            hfois = deepcopy(fois)
            TPSChem.zero!(hfois)
            TPSChem.build_sigma!(
                hfois, spt_ref, cluster_ops, clustered_ham; nbody=2, verbose=0)
            hfois_d = TPSChem.build_sigma_distributed(
                fois, spt_ref, cluster_ops, clustered_ham; nbody=2,
                verbose=0, workers=workers(), blas_threads=1,
                strategy=:hash)
            @test isapprox(TPSChem.nonorth_dot(hfois, hfois),
                           TPSChem.nonorth_dot(hfois_d, hfois_d),
                           rtol=1e-10, atol=1e-10)
            @test isapprox(TPSChem.nonorth_dot(hfois, hfois_d),
                           TPSChem.nonorth_dot(hfois, hfois),
                           rtol=1e-10, atol=1e-10)

            # The blockwise PT2 path uses a scratch-reusing one-block H0
            # contraction instead of invoking the general threaded matvec for
            # every FOIS block.  Check the specialized contraction directly on
            # representative blocks so its allocation optimization cannot
            # silently change the PT2 numerator.
            ham0 = TPSChem.extract_1body_operator(clustered_ham;
                                                  op_string="Hcmf")
            scratch = [zeros(Float64, 1000) for _ in 1:10]
            nchecked = 0
            for (fock, blocks) in fois
                for (tconfig, tuck) in blocks
                    dims = size(tuck.core[1])
                    general_tuck = TPSChem.Tucker{Float64,4,1}(
                        (zeros(Float64, dims),), tuck.factors)
                    direct_tuck = TPSChem.Tucker{Float64,4,1}(
                        (zeros(Float64, dims),), tuck.factors)
                    general = TPSChem.SPTstate(
                        spt_ref.clusters, spt_ref.p_spaces, spt_ref.q_spaces;
                        T=Float64, R=1)
                    TPSChem.add_fockconfig!(general, fock)
                    general[fock][tconfig] = general_tuck
                    TPSChem.build_sigma!(general, spt_ref, cluster_ops, ham0;
                                         nbody=1, verbose=0)
                    TPSChem._pt2_build_f0_block!(
                        direct_tuck, fock, tconfig, spt_ref, cluster_ops,
                        ham0, scratch)
                    @test isapprox(general_tuck.core[1], direct_tuck.core[1];
                                   atol=1e-12)
                    nchecked += 1
                    nchecked == 3 && break
                end
                nchecked == 3 && break
            end
            @test nchecked == 3

            e2 = TPSChem.compute_pt2_energy_blockwise(
                spt_ref, cluster_ops, clustered_ham; nbody=2, thresh_foi=1e-3,
                opt_ref=false, verbose=0)
            e2_d = TPSChem.compute_pt2_energy_blockwise_distributed(
                spt_ref, cluster_ops, clustered_ham; nbody=2, thresh_foi=1e-3,
                opt_ref=false, verbose=0, workers=workers(),
                threaded_worker=false, blas_threads=1, strategy=:hash)
            @test isapprox(e2, e2_d, rtol=1e-10, atol=1e-10)

            # The distributed PT1 wavefunction combines <X|H|0>, <X|F|0>, the
            # overlap and the Fock diagonal.  build_sigma_distributed returns
            # its blocks in worker-merge order rather than the input order, so
            # this has to be matched by (fock, tconfig) key; a positional
            # combination silently pairs unrelated blocks.
            E0_spt = TPSChem.compute_expectation_value(spt_ref, cluster_ops,
                                                       clustered_ham)
            F0_spt = TPSChem.compute_expectation_value(spt_ref, cluster_ops, ham0)
            p1_s, E2_s, ec_s = TPSChem.compute_pt1_wavefunction(
                fois, spt_ref, cluster_ops, clustered_ham, ham0, E0_spt, F0_spt;
                verbose=0)
            p1_d, E2_d, ec_d = TPSChem.compute_pt1_wavefunction_distributed(
                fois, spt_ref, cluster_ops, clustered_ham, ham0, E0_spt, F0_spt;
                verbose=0, workers=workers(), blas_threads=1, strategy=:hash)
            @test isapprox(ec_s, ec_d, rtol=1e-10, atol=1e-10)
            @test isapprox(E2_s, E2_d, rtol=1e-10, atol=1e-10)
            @test length(p1_s) == length(p1_d)
            for (fock, blocks) in p1_s
                @test haskey(p1_d, fock)
                for (tconfig, tuck) in blocks
                    @test haskey(p1_d[fock], tconfig)
                    for r in 1:length(tuck.core)
                        @test isapprox(tuck.core[r], p1_d[fock][tconfig].core[r];
                                       atol=1e-10)
                    end
                end
            end

            sigma2 = TPSChem.compute_spt_sigma_norm_blockwise(
                spt_ref, cluster_ops, clustered_ham; nbody=2, thresh_foi=1e-3,
                opt_ref=false, verbose=0)
            sigma2_d = TPSChem.compute_spt_sigma_norm_blockwise_distributed(
                spt_ref, cluster_ops, clustered_ham; nbody=2, thresh_foi=1e-3,
                opt_ref=false, workers=workers(), threaded_worker=false,
                blas_threads=1, verbose=0)
            @test isapprox(sigma2, sigma2_d, rtol=1e-10, atol=1e-10)
        end
    end
finally
    if !isempty(_property_multinode_added_procs)
        rmprocs(_property_multinode_added_procs)
    end
end
