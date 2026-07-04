using Distributed

# Ensure at least 2 distributed workers for the sharded solver. Track whether we
# added them so we can clean up and not disturb the rest of the suite.
_multinode_added_procs = Int[]
if nprocs() == 1
    _multinode_added_procs = addprocs(2)
end

using TPSChem
using TPSChem.QCBase
using TPSChem.RDM
using Test
using JLD2
using LinearAlgebra
using Printf

@testset "tpsci multinode sharded davidson" begin
    @load "_testdata_cmf_h8.jld2"

    max_roots = 4
    nroots = 3
    ref_fock = FockConfig(init_fspace)

    cluster_bases = TPSChem.compute_cluster_eigenbasis(ints, clusters;
        verbose=0, max_roots=max_roots, init_fspace=init_fspace,
        rdm1a=d1.a, rdm1b=d1.b)

    clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
    cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints)
    TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)

    # Build a multi-Fock variational space via a FOIS expansion of a small
    # reference, then define a random orthonormal guess inside that fixed space.
    ci_vector = TPSChem.TPSCIstate(clusters, ref_fock, R=nroots, T=Float64)
    ci_vector[ref_fock][ClusterConfig([1, 1])] = [1.0, 0.0, 0.0]
    ci_vector[ref_fock][ClusterConfig([2, 1])] = [0.0, 1.0, 0.0]
    ci_vector[ref_fock][ClusterConfig([1, 2])] = [0.0, 0.0, 1.0]

    _, v_ci = TPSChem.do_fois_ci(ci_vector, cluster_ops, clustered_ham;
        nbody=2, thresh_foi=1e-3, tol=1e-7, thresh_clip=1e-6,
        threaded=false, compress=true, pt=false, verbose=false)

    guess = deepcopy(v_ci)
    TPSChem.randomize!(guess)
    TPSChem.orthonormalize!(guess)

    # The variational space must have more than one Fock sector for this to be a
    # meaningful multinode test.
    @test length(guess.data) > 1
    @test length(guess) >= nroots

    # Reference: direct dense diagonalization in the same space.
    e_ref, _, _ = TPSChem.tps_ci_direct(deepcopy(guess), cluster_ops, clustered_ham;
        conv_thresh=1e-11, verbose=0)
    e_ref = sort(e_ref)

    # nnz(H) estimator is positive and consistent with the dense dimension.
    dtmp = TPSChem.distribute_tpsci_state(deepcopy(guess); workers=workers(),
                                          strategy=:balanced)
    rep = TPSChem.sharded_H_memory_report(dtmp, clustered_ham)
    @test rep.dim == length(guess)
    @test rep.nnz > 0
    @test rep.nnz <= length(guess)^2 + 1     # cannot exceed dense H element count
    TPSChem.destroy!(dtmp)

    for tier in (:matrixfree, :blocks, :auto)
        dg = TPSChem.distribute_tpsci_state(deepcopy(guess); workers=workers(),
                                            strategy=:balanced)
        e_sh, v_sh = TPSChem.tps_ci_davidson_sharded(dg, cluster_ops, clustered_ham;
            nroots=nroots, conv_thresh=1e-8, max_ss_vecs=4, max_iter=300,
            h_storage=tier, verbose=0)

        # Eigenvalues match the independent dense reference.
        @test maximum(abs.(sort(e_sh) .- e_ref)) < 1e-6

        # Returned vectors are genuine eigenvectors: Rayleigh quotient of each
        # root equals its eigenvalue. Uses only sharded operations.
        for r in 1:nroots
            x = TPSChem.extract_root_sharded(v_sh, r)
            hx = TPSChem.tps_ci_matvec_sharded(x, cluster_ops, clustered_ham;
                                               workers=workers())
            num = TPSChem.overlap(x, hx)[1, 1]
            den = TPSChem.overlap(x, x)[1, 1]
            rayleigh = num / den
            @test any(abs(rayleigh - e) < 1e-6 for e in e_ref)
            TPSChem.destroy!(x)
            TPSChem.destroy!(hx)
        end

        TPSChem.destroy!(v_sh)
        TPSChem.destroy!(dg)
    end

    # combine/extract round-trip preserves coefficients.
    dg = TPSChem.distribute_tpsci_state(deepcopy(guess); workers=workers(),
                                        strategy=:balanced)
    roots = [TPSChem.extract_root_sharded(dg, r) for r in 1:nroots]
    recombined = TPSChem.combine_roots_sharded(roots)
    local_orig = TPSChem.collect_tpsci_state(dg)
    local_round = TPSChem.collect_tpsci_state(recombined)
    @test isapprox(TPSChem.get_vector(local_orig), TPSChem.get_vector(local_round),
                   atol=1e-12)
    for r in roots
        TPSChem.destroy!(r)
    end
    TPSChem.destroy!(recombined)
    TPSChem.destroy!(dg)
end

@testset "tps-cepa sharded stored-H" begin
    @load "_testdata_cmf_h8.jld2"

    nroots = 2
    ref_fock = FockConfig(init_fspace)
    cluster_bases = TPSChem.compute_cluster_eigenbasis(ints, clusters;
        verbose=0, max_roots=4, init_fspace=init_fspace, rdm1a=d1.a, rdm1b=d1.b)
    clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
    cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints)
    TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)

    # Solve a small reference so we have a proper reference eigenvector + e0.
    ci = TPSChem.TPSCIstate(clusters, ref_fock, R=nroots, T=Float64)
    ci[ref_fock][ClusterConfig([1, 1])] = [1.0, 0.0]
    ci[ref_fock][ClusterConfig([2, 1])] = [0.0, 1.0]
    ci[ref_fock][ClusterConfig([1, 2])] = [0.0, 1.0]
    e0, ref_solved, _ = TPSChem.tps_ci_direct(deepcopy(ci), cluster_ops, clustered_ham;
        conv_thresh=1e-10, verbose=0)

    common = (cepa_shift="acpf", cepa_mit=30, nbody=2, thresh_foi=1e-4,
              tol=1e-8, solver=:minres, e0=e0, workers=workers(), verbose=0)

    # Existing matrix-free sharded CEPA is the reference.
    e_ref, q_ref = TPSChem.do_fois_cepa_sharded(deepcopy(ref_solved), cluster_ops,
                                                clustered_ham; common...)
    TPSChem.destroy!(q_ref)

    # New stored-H TPS-CEPA, both tiers, must match the matrix-free reference.
    for tier in (:blocks, :matrixfree)
        e_new, q_new = TPSChem.do_tps_sharded_cepa(deepcopy(ref_solved), cluster_ops,
            clustered_ham; h_storage=tier, common...)
        @test length(e_new) == nroots
        @test all(isfinite, e_new)
        @test maximum(abs.(e_new .- e_ref)) < 1e-7
        TPSChem.destroy!(q_new)
    end
end

if !isempty(_multinode_added_procs)
    rmprocs(_multinode_added_procs)
end
