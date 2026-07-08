using Distributed

# Ensure at least 2 distributed workers for the sharded solver. Track whether we
# added them so we can clean up and not disturb the rest of the suite. Spawn the
# workers with the current active project so the stdlib `Pkg` is on their load
# path (under `Pkg.test`, bare `addprocs` workers inherit a restricted load path
# where `import Pkg` fails inside `ensure_tpsci_multinode_workers!`).
_multinode_added_procs = Int[]
if nprocs() == 1
    _multinode_added_procs = addprocs(2; exeflags="--project=$(Base.active_project())")
end

using TPSChem
using TPSChem.QCBase
using TPSChem.RDM
using Test
using JLD2
using LinearAlgebra
using Printf
using Random

# The heavy multinode cases (iterative stored-H TPS-CEPA and the full never-gather
# sharded selected-CI loop) are the slowest part of the whole suite: on a single
# machine the distributed workers are oversubscribed and every path pays JIT
# warmup, so these run for minutes each. Gate them behind an env flag so the
# default `Pkg.test()` (and CI) stays fast — set TPSCHEM_TEST_HEAVY_MULTINODE=1 to
# run them (e.g. in a nightly job). The cheaper "sharded davidson" (both H-storage
# tiers + matvec + extract/combine) and "add! merge" testsets below always run and
# still cover the core sharded primitives.
const RUN_HEAVY_MULTINODE =
    lowercase(strip(get(ENV, "TPSCHEM_TEST_HEAVY_MULTINODE", "0"))) in
        ("1", "true", "yes", "on")

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
    Random.seed!(1234567)
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

if RUN_HEAVY_MULTINODE
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
else
    @info "Skipping heavy stored-H TPS-CEPA multinode test (set TPSCHEM_TEST_HEAVY_MULTINODE=1 to run)"
end

@testset "sharded ownership-reconciling merge (add!)" begin
    @load "_testdata_cmf_h8.jld2"
    ref_fock = FockConfig(init_fspace)
    cluster_bases = TPSChem.compute_cluster_eigenbasis(ints, clusters;
        verbose=0, max_roots=4, init_fspace=init_fspace, rdm1a=d1.a, rdm1b=d1.b)
    clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
    cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints)
    TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)

    ci = TPSChem.TPSCIstate(clusters, ref_fock, R=2, T=Float64)
    ci[ref_fock][ClusterConfig([1, 1])] = [1.0, 0.0]
    ci[ref_fock][ClusterConfig([2, 1])] = [0.0, 1.0]

    dg = TPSChem.distribute_tpsci_state(deepcopy(ci); workers=workers(),
                                        strategy=:balanced)
    sig = TPSChem.open_matvec_sharded(dg, cluster_ops, clustered_ham;
        nbody=4, thresh=1e-8, prescreen=false, verbose=0)

    # The FOIS reaches Fock sectors beyond the reference space.
    @test length(sig.owners) > length(dg.owners)
    # Sectors already present keep their owner — the precondition for add!.
    for (fock, owner) in dg.owners
        if haskey(sig.owners, fock)
            @test sig.owners[fock] == owner
        end
    end

    # Reference behavior: the same structure merge done locally.
    local_sig = TPSChem.collect_tpsci_state(sig)
    expected = deepcopy(ci)
    TPSChem.zero!(local_sig)
    TPSChem.add!(expected, local_sig)

    # Sharded structure merge (zero PT coefficients, grow the space).
    TPSChem.zero!(sig)
    TPSChem.add!(dg, sig)
    merged = TPSChem.collect_tpsci_state(dg)

    @test length(dg) == length(expected)
    for (fock, configs) in expected.data
        @test haskey(merged.data, fock)
        for (config, coeffs) in configs
            @test haskey(merged.data[fock], config)
            @test isapprox(merged[fock][config], coeffs, atol=1e-14)
        end
    end

    TPSChem.destroy!(sig)
    TPSChem.destroy!(dg)
end

if RUN_HEAVY_MULTINODE
@testset "never-gather tpsci_ci_sharded vs single-node tpsci_ci" begin
    @load "_testdata_cmf_h8.jld2"
    ref_fock = FockConfig(init_fspace)
    cluster_bases = TPSChem.compute_cluster_eigenbasis(ints, clusters;
        verbose=0, max_roots=4, init_fspace=init_fspace, rdm1a=d1.a, rdm1b=d1.b)
    clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
    cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints)
    TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)

    nroots = 2
    ci = TPSChem.TPSCIstate(clusters, ref_fock, R=nroots, T=Float64)
    ci[ref_fock][ClusterConfig([1, 1])] = [1.0, 0.0]
    ci[ref_fock][ClusterConfig([2, 1])] = [0.0, 1.0]

    common = (thresh_cipsi=1e-4, thresh_foi=1e-9, conv_thresh=1e-7,
              max_iter=8, nbody=4)

    e_ref, v_ref = TPSChem.tpsci_ci(deepcopy(ci), cluster_ops, clustered_ham;
        common..., incremental=false, davidson=false)

    e_sh, dv = TPSChem.tpsci_ci_sharded(deepcopy(ci), cluster_ops, clustered_ham;
        common..., ci_conv=1e-10, compute_s2=true, workers=workers(), verbose=0)

    # Same converged energies and the same selected variational space.
    @test maximum(abs.(e_sh .- e_ref)) < 1e-6
    @test length(dv) == length(v_ref)

    # Final sharded roots are orthonormal (checked without gathering).
    S = TPSChem.overlap(dv, dv)
    @test isapprox(S, Matrix{Float64}(LinearAlgebra.I, nroots, nroots), atol=1e-8)

    TPSChem.destroy!(dv)
end
else
    @info "Skipping heavy never-gather tpsci_ci_sharded multinode test (set TPSCHEM_TEST_HEAVY_MULTINODE=1 to run)"
end

if !isempty(_multinode_added_procs)
    rmprocs(_multinode_added_procs)
end
