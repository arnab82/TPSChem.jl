using Distributed

# The blocked sharded solvers need real workers to exercise the fan-out and the
# remote ket gathers; two is enough to make every path cross a process boundary.
_mrmn_added_procs = Int[]
if nprocs() == 1
    _mrmn_added_procs = addprocs(2; exeflags="--project=$(Base.active_project())")
end

using TPSChem
using TPSChem.QCBase
using TPSChem.RDM
using Test
using JLD2
using LinearAlgebra
using Printf

# Multiroot TPS-CEPA across nodes. Blocking shares one Hamiltonian apply and one
# set of fan-outs across the roots while each root keeps its own shift, Krylov
# scalars and convergence test — so every blocked primitive must reproduce the
# root-at-a-time one it replaces, and the end-to-end energies must match both the
# unblocked sharded solver and the single-node one.
@testset "tps-cepa multiroot multinode" begin
    @load "_testdata_cmf_h8.jld2"

    T = Float64
    nroots = 3
    ref_fock = FockConfig(init_fspace)
    cluster_bases = TPSChem.compute_cluster_eigenbasis(ints, clusters;
        verbose=0, max_roots=4, init_fspace=init_fspace, rdm1a=d1.a, rdm1b=d1.b)
    clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
    cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints)
    TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)

    ci = TPSChem.TPSCIstate(clusters, ref_fock, R=nroots, T=T)
    ci[ref_fock][ClusterConfig([1, 1])] = [1.0, 0.0, 0.0]
    ci[ref_fock][ClusterConfig([2, 1])] = [0.0, 1.0, 0.0]
    ci[ref_fock][ClusterConfig([1, 2])] = [0.0, 0.0, 1.0]
    e0, ref_solved, _ = TPSChem.tps_ci_direct(deepcopy(ci), cluster_ops, clustered_ham;
        conv_thresh=1e-10, verbose=0)

    wids = workers()

    # ── A sharded Q-space and an R-root vector living on it ──────────────────────
    dref = TPSChem.distribute_tpsci_state(deepcopy(ref_solved); workers=wids,
                                          strategy=:balanced)
    qspace = TPSChem.open_matvec_sharded(dref, cluster_ops, clustered_ham;
                                         nbody=2, thresh=1e-4, prescreen=false,
                                         workers=wids, verbose=0)
    TPSChem.project_out!(qspace, dref)
    sig = TPSChem.open_matvec_sharded(dref, cluster_ops, clustered_ham;
                                      nbody=2, thresh=0.0, prescreen=false,
                                      workers=wids, verbose=0)
    hblock = TPSChem.restrict_to_basis_sharded(sig, qspace)
    TPSChem.destroy!(sig)
    TPSChem.destroy!(dref)
    @test length(qspace) > 1

    # Same vector, one root at a time, for the reference computations. These have
    # to carry *qspace's* Fock ownership, not a freshly balanced one: the sharded
    # vector algebra addresses coefficients by position within each worker, so it
    # refuses to mix states whose sectors live on different processes.
    hlocal = TPSChem.collect_tpsci_state(hblock)
    hroots = []
    for i in 1:nroots
        tmp = TPSChem.distribute_tpsci_state(TPSChem.extract_chosen_root(hlocal, i);
                                             workers=wids, strategy=:balanced)
        push!(hroots, TPSChem.restrict_to_basis_sharded(tmp, qspace))
        TPSChem.destroy!(tmp)
    end

    eshifts = T[e0[i] - 0.05*i for i in 1:nroots]   # a distinct shift per root

    # ── 1. Fused vector algebra: R roots in one batch vs one root per batch ──────
    scal = T[0.5, -1.5, 2.0]
    blk = TPSChem.sharded_block_fused_ops!(
        [hblock], [TPSChem.svb_nrm2sq(T, 1), TPSChem.svb_scale(T, 1, scal),
                   TPSChem.svb_nrm2sq(T, 1)])
    for i in 1:nroots
        one_ = TPSChem.sharded_fused_ops!(
            [hroots[i]], [TPSChem.sv_nrm2sq(T, 1), TPSChem.sv_scale(T, 1, scal[i]),
                          TPSChem.sv_nrm2sq(T, 1)])
        @test isapprox(blk[1, i], one_[1], rtol=1e-12)
        @test isapprox(blk[2, i], one_[2], rtol=1e-12)
    end
    # Undo the scaling so later comparisons see the original vector.
    TPSChem.sharded_block_fused_ops!([hblock], [TPSChem.svb_scale(T, 1, T[1/s for s in scal])])
    for i in 1:nroots
        TPSChem.sharded_fused_ops!([hroots[i]], [TPSChem.sv_scale(T, 1, 1/scal[i])])
    end

    # ── 2. Block-H apply, both storage tiers ────────────────────────────────────
    for tier in (:blocks, :matrixfree)
        op = tier == :blocks ?
            TPSChem.build_block_h_sharded(qspace, cluster_ops, clustered_ham;
                                          workers=wids, verbose=0) :
            TPSChem.MatrixFreeShardedH(cluster_ops, clustered_ham, wids, true, 1)

        ablock = TPSChem.collect_tpsci_state(
            TPSChem.apply_sharded_H_block(op, hblock, eshifts))
        for i in 1:nroots
            a1 = TPSChem.collect_tpsci_state(
                TPSChem.apply_sharded_H(op, hroots[i]; eshift=eshifts[i]))
            err = 0.0
            for (fock, cfgs) in a1.data, (cfg, c1) in cfgs
                err = max(err, abs(c1[1] - ablock.data[fock][cfg][i]))
            end
            @test err < 1e-12
        end

        # ── 3. Blocked linear solvers vs the root-at-a-time ones ────────────────
        xb, ib = TPSChem.tps_sharded_cepa_minres_block(hblock, op, eshifts;
                                                       tol=1e-10, maxiter=400)
        xbl = TPSChem.collect_tpsci_state(xb)
        for i in 1:nroots
            xs, _ = TPSChem.tps_sharded_cepa_minres_linsolve(hroots[i], op;
                                                             eshift=eshifts[i],
                                                             tol=1e-10, maxiter=400)
            xsl = TPSChem.collect_tpsci_state(xs)
            err = 0.0
            for (fock, cfgs) in xsl.data, (cfg, c1) in cfgs
                err = max(err, abs(c1[1] - xbl.data[fock][cfg][i]))
            end
            @test err < 1e-8
            TPSChem.destroy!(xs)
        end
        TPSChem.destroy!(xb)

        if tier == :blocks
            # PCG needs a stored H for its diagonal. Shift well below the spectrum
            # so the shifted operator really is positive definite.
            Hdiag = TPSChem.compute_diagonal_sharded(op, qspace)
            dmin = TPSChem.sharded_state_min(Hdiag)
            pd = T[dmin - 1.0 - 0.1*i for i in 1:nroots]

            zb = TPSChem.collect_tpsci_state(
                TPSChem.precondition_block_sharded(hblock, Hdiag, pd))
            dl = TPSChem.collect_tpsci_state(Hdiag)
            err = 0.0
            for (fock, cfgs) in hlocal.data, (cfg, hc) in cfgs
                for i in 1:nroots
                    err = max(err, abs(zb.data[fock][cfg][i] -
                                       hc[i] / (dl.data[fock][cfg][1] - pd[i])))
                end
            end
            @test err < 1e-12

            xp, ip = TPSChem.tps_sharded_cepa_pcg_block(hblock, op, Hdiag, pd;
                                                        tol=1e-10, maxiter=400)
            @test !any(ip.indefinite)
            xpl = TPSChem.collect_tpsci_state(xp)
            for i in 1:nroots
                xs, _ = TPSChem.tps_sharded_cepa_pcg_linsolve(hroots[i], op, Hdiag;
                                                              eshift=pd[i], tol=1e-10,
                                                              maxiter=400)
                xsl = TPSChem.collect_tpsci_state(xs)
                err = 0.0
                for (fock, cfgs) in xsl.data, (cfg, c1) in cfgs
                    err = max(err, abs(c1[1] - xpl.data[fock][cfg][i]))
                end
                @test err < 1e-8
                TPSChem.destroy!(xs)
            end
            TPSChem.destroy!(xp)
            TPSChem.destroy!(Hdiag)
            TPSChem.destroy!(op)
        end
    end

    for h in hroots
        TPSChem.destroy!(h)
    end
    TPSChem.destroy!(hblock)
    TPSChem.destroy!(qspace)

    # ── 4. End to end: blocked vs unblocked vs single node ──────────────────────
    common = (cepa_mit=30, nbody=2, thresh_foi=1e-4, tol=1e-9, thresh_sigma=0.0,
              e0=e0, workers=wids, verbose=0)
    e_single_node, _ = TPSChem.do_fois_cepa(deepcopy(ref_solved), cluster_ops,
                                            clustered_ham; cepa_shift="cepa",
                                            nbody=2, thresh_foi=1e-4, tol=1e-9,
                                            thresh_sigma=0.0, solver=:minres,
                                            build_hqq=:sparse, verbose=0)
    for (tier, slv) in ((:blocks, :minres), (:blocks, :pcg), (:matrixfree, :minres))
        e_one, q_one = TPSChem.do_tps_sharded_cepa(deepcopy(ref_solved), cluster_ops,
            clustered_ham; cepa_shift="cepa", h_storage=tier, solver=slv,
            multiroot=false, common...)
        TPSChem.destroy!(q_one)
        e_blk, q_blk = TPSChem.do_tps_sharded_cepa(deepcopy(ref_solved), cluster_ops,
            clustered_ham; cepa_shift="cepa", h_storage=tier, solver=slv,
            multiroot=true, common...)
        TPSChem.destroy!(q_blk)
        @test length(e_blk) == nroots
        @test all(isfinite, e_blk)
        @test maximum(abs.(e_blk .- e_one)) < 1e-8
        @test maximum(abs.(e_blk .- e_single_node)) < 1e-7
    end

    # acpf drives the shift iteratively, which is what exercises the warm start.
    e_acpf_one, q1 = TPSChem.do_tps_sharded_cepa(deepcopy(ref_solved), cluster_ops,
        clustered_ham; cepa_shift="acpf", h_storage=:blocks, solver=:minres,
        multiroot=false, common...)
    TPSChem.destroy!(q1)
    e_acpf_blk, q2 = TPSChem.do_tps_sharded_cepa(deepcopy(ref_solved), cluster_ops,
        clustered_ham; cepa_shift="acpf", h_storage=:blocks, solver=:minres,
        multiroot=true, common...)
    TPSChem.destroy!(q2)
    @test maximum(abs.(e_acpf_blk .- e_acpf_one)) < 1e-8
end

isempty(_mrmn_added_procs) || rmprocs(_mrmn_added_procs)
