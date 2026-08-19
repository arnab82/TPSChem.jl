using Distributed

_spt_added_procs = Int[]
if nprocs() == 1
    _spt_added_procs = addprocs(2; exeflags="--project=$(Base.active_project())")
end

using TPSChem
using TPSChem.QCBase
using Test
using JLD2
using LinearAlgebra
using Printf
using Random

@testset "spt sharded (DistributedSPTstate + Tucker CI)" begin
    @load "_testdata_cmf_h8.jld2"

    cluster_bases = TPSChem.compute_cluster_eigenbasis(ints, clusters;
        verbose=0, max_roots=4, init_fspace=init_fspace, rdm1a=d1.a, rdm1b=d1.b)
    clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
    cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints)
    TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)

    # Build a multi-Fock SPT variational space: FOIS-expand the CMF reference,
    # then define a random R-root guess inside that fixed Tucker space.
    v = TPSChem.SPTstate(clusters, FockConfig(init_fspace), cluster_bases)
    xspace = TPSChem.build_compressed_1st_order_state(v, cluster_ops, clustered_ham;
                                                      nbody=2, thresh=1e-3)
    xspace = TPSChem.compress(xspace, thresh=1e-3)
    TPSChem.nonorth_add!(v, xspace)

    nroots = 2
    v = TPSChem.SPTstate(v, R=nroots)
    Random.seed!(11)
    TPSChem.randomize!(v)
    TPSChem.orthonormalize!(v)

    @test length(v.data) > 1              # genuinely multi-Fock
    @test length(v) >= nroots

    # --- container round-trip -------------------------------------------------
    dv = TPSChem.distribute_spt_state(deepcopy(v); workers=workers(), strategy=:balanced)
    @test length(dv) == length(v)
    vc = TPSChem.collect_spt_state(dv)
    @test TPSChem.get_vector(vc) == TPSChem.get_vector(v)   # bit-exact copy

    # --- sharded overlap / norm vs single-node core arithmetic ----------------
    S_loc = TPSChem._spt_orth_overlap(v, v)
    S_sh = TPSChem.overlap(dv, dv)
    @test isapprox(S_sh, S_loc, atol=1e-13)
    @test isapprox(norm(TPSChem.norm(dv)), sqrt(nroots), atol=1e-12)  # orthonormal roots

    # --- never-gather matvec vs single-node build_sigma! ----------------------
    TPSChem._tpsci_sharded_cache_operator_problem!(cluster_ops, clustered_ham;
                                                   workers=workers())
    dsig = TPSChem.build_sigma_sharded(dv, cluster_ops, clustered_ham;
                                       nbody=4, workers=workers())
    sig_local = deepcopy(v)
    TPSChem.zero!(sig_local)
    TPSChem.build_sigma!(sig_local, v, cluster_ops, clustered_ham; nbody=4, verbose=0)
    sig_sh = TPSChem.collect_spt_state(dsig)
    @test isapprox(TPSChem.get_vector(sig_sh), TPSChem.get_vector(sig_local), atol=1e-11)

    # --- 2D (grid) partitioned matvec vs the same reference -------------------
    # Communication scales as sqrt(P)|state| rather than P|state|, so this is a
    # different data path (bra bases and partial sigmas move, not whole ket
    # sectors) and needs its own check that it lands on the same answer with the
    # same ownership.
    dsig2d = TPSChem.build_sigma_sharded_2d(dv, cluster_ops, clustered_ham;
                                            nbody=4, workers=workers())
    sig_2d = TPSChem.collect_spt_state(dsig2d)
    @test isapprox(TPSChem.get_vector(sig_2d), TPSChem.get_vector(sig_local), atol=1e-11)
    @test Set(keys(dsig2d.owners)) == Set(keys(dsig.owners))
    for f in keys(dsig.owners)                       # ownership must be preserved
        @test dsig2d.owners[f] == dv.owners[f]
    end
    TPSChem.destroy!(dsig2d)

    # A prime worker count cannot form a useful grid; it must fall back rather
    # than silently produce a 1 x P "grid" (which is the destination partition).
    if length(workers()) >= 2
        qr, qc = TPSChem._spt_grid_shape(length(workers()))
        @test qr * qc == length(workers())
    end

    # --- distributed Tucker CI solver vs EXACT dense H in the fixed basis -----
    # Build H densely in v's fixed Tucker basis (columns = build_sigma! of unit
    # vectors). This is an exact reference, independent of Davidson convergence.
    dim = length(v)
    vone = TPSChem.SPTstate(v, R=1)
    Hdense = zeros(dim, dim)
    for i in 1:dim
        ei = zeros(dim); ei[i] = 1.0
        TPSChem.set_vector!(vone, ei; root=1)
        sig = deepcopy(vone); TPSChem.zero!(sig)
        TPSChem.build_sigma!(sig, vone, cluster_ops, clustered_ham; nbody=4, verbose=0)
        Hdense[:, i] = TPSChem.get_vector(sig)[:, 1]
    end
    exact = sort(eigvals(Symmetric(0.5 .* (Hdense .+ Hdense'))))[1:nroots]

    # Cap the subspace so it can span the (small) test space -> clean convergence.
    dv2 = TPSChem.distribute_spt_state(deepcopy(v); workers=workers(), strategy=:balanced)
    e_sh, dvout = TPSChem.spt_ci_davidson_sharded(dv2, cluster_ops, clustered_ham;
                                                  nroots=nroots, conv_thresh=1e-11,
                                                  max_ss_vecs=dim, max_iter=200,
                                                  nbody=4, workers=workers(), verbose=0)
    solver_err = maximum(abs.(sort(e_sh) .- exact))
    @printf("    sharded SPT CI vs exact dense H: max|ΔE| = %.2e\n", solver_err)
    @test solver_err < 1e-10

    # Returned roots are orthonormal (checked without gathering).
    Sout = TPSChem.overlap(dvout, dvout)
    @test isapprox(Sout, Matrix{Float64}(I, nroots, nroots), atol=1e-7)

    # --- never-gather FOIS vs single-node build_compressed_1st_order_state ----
    # Compare as vectors in a basis-robust way: the compressed Tucker factors
    # have SVD sign/ordering freedom, so check structure + per-root norm +
    # cosine alignment (nonorth_dot handles differing factor bases).
    ref1 = TPSChem.SPTstate(v, R=1)
    fois_loc = TPSChem.build_compressed_1st_order_state(ref1, cluster_ops, clustered_ham;
                                                        nbody=4, thresh=1e-4)
    dref1 = TPSChem.distribute_spt_state(deepcopy(ref1); workers=workers(), strategy=:balanced)
    dfois = TPSChem.build_compressed_1st_order_state_sharded(dref1, cluster_ops, clustered_ham;
                                                             nbody=4, thresh=1e-4, workers=workers())
    fois_sh = TPSChem.collect_spt_state(dfois)

    # Single-node keeps empty Fock sectors (add_fockconfig! runs even when
    # compression empties the block); the sharded metadata drops zero-length
    # sectors. Compare only non-empty sectors — empties carry no coefficients
    # (the norm check below confirms identical content).
    nonempty(s) = Set(f for (f, tcs) in s.data if TPSChem._spt_block_length(tcs) > 0)
    @test nonempty(fois_sh) == nonempty(fois_loc)
    na = TPSChem.nonorth_dot(fois_loc, fois_loc)
    nb = TPSChem.nonorth_dot(fois_sh, fois_sh)
    ab = TPSChem.nonorth_dot(fois_loc, fois_sh)
    @test isapprox(na, nb, rtol=1e-8)                       # same FOIS norm
    @test isapprox(ab, sqrt.(na .* nb), rtol=1e-8)          # aligned (cosine = 1)

    # --- never-gather PT1/PT2 vs single-node compute_pt1_wavefunction ---------
    # E2/ecorr are scalars (basis-independent), so compare tightly.
    fois_pt = TPSChem.build_compressed_1st_order_state(deepcopy(v), cluster_ops,
                                                       clustered_ham; nbody=4, thresh=1e-4)
    _, e2_loc, ecorr_loc = TPSChem.compute_pt1_wavefunction(fois_pt, deepcopy(v),
                                                            cluster_ops, clustered_ham;
                                                            H0="Hcmf", verbose=0)
    dref_pt = TPSChem.distribute_spt_state(deepcopy(v); workers=workers(), strategy=:hash)
    dfois_pt = TPSChem.build_compressed_1st_order_state_sharded(dref_pt, cluster_ops,
                    clustered_ham; nbody=4, thresh=1e-4, workers=workers(),
                    strategy=:hash, existing_owners=dref_pt.owners)
    psi1_pt, e2_sh, ecorr_sh = TPSChem.compute_pt1_wavefunction_sharded(dfois_pt, dref_pt,
                    cluster_ops, clustered_ham; H0="Hcmf", workers=workers(), verbose=0)
    @printf("    sharded PT2 vs single-node: max|ΔE2| = %.2e\n",
            maximum(abs.(sort(e2_sh) .- sort(e2_loc))))
    @test isapprox(sort(ecorr_sh), sort(ecorr_loc), atol=1e-8)
    @test isapprox(sort(e2_sh), sort(e2_loc), atol=1e-8)

    # --- end-to-end never-gather driver vs single-node subspace_product_tucker
    # The full 6-iteration loop (run twice) dominates this file's runtime; its
    # components (solver, FOIS, PT1) are already tested above at machine
    # precision. Gate it behind the heavy-multinode flag to keep default CI fast.
    run_heavy = lowercase(strip(get(ENV, "TPSCHEM_TEST_HEAVY_MULTINODE", "0"))) in
        ("1", "true", "yes", "on")
    dref_out = nothing
    if run_heavy
    v0 = TPSChem.SPTstate(clusters, FockConfig(init_fspace), cluster_bases)   # R=1 CMF start
    common = (max_iter=6, nbody=4, H0="Hcmf", thresh_var=1e-3, thresh_foi=1e-4,
              thresh_pt=1e-3, ci_conv=1e-7, do_pt=true, resolve_ss=false,
              tol_tucker=1e-7)
    e_loc, _ = TPSChem.subspace_product_tucker(deepcopy(v0), cluster_ops, clustered_ham;
                                               common..., verbose=0)
    e_drv, dref_out = TPSChem.subspace_product_tucker_sharded(deepcopy(v0), cluster_ops,
                                            clustered_ham; common..., workers=workers(),
                                            verbose=0)
    @printf("    sharded SPT driver vs single-node: E=%.10f vs %.10f\n", e_drv[1], e_loc[1])
    @test isapprox(e_drv[1], e_loc[1], atol=1e-6)
    else
        @info "Skipping heavy end-to-end sharded SPT driver test (set TPSCHEM_TEST_HEAVY_MULTINODE=1 to run)"
    end

    TPSChem.destroy!(psi1_pt)
    TPSChem.destroy!(dfois_pt)
    TPSChem.destroy!(dref_pt)
    dref_out === nothing || TPSChem.destroy!(dref_out)
    TPSChem.destroy!(dfois)
    TPSChem.destroy!(dref1)
    TPSChem.destroy!(dsig)
    TPSChem.destroy!(dvout)
    TPSChem.destroy!(dv2)
    TPSChem.destroy!(dv)
end

if !isempty(_spt_added_procs)
    rmprocs(_spt_added_procs)
end
