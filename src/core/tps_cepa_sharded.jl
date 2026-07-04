"""
Stored-Hamiltonian TPS-CEPA across nodes.

The existing `tpsci_cepa_solve_sharded`/`do_fois_cepa_sharded` are matrix-free: the
shifted Q-space operator (H_QQ - eshift*I) is re-contracted from the cluster
operators on every MINRES/CG iteration. Because CEPA runs many linear-solver
iterations, over multiple macro-iterations, for multiple roots, all against the
SAME fixed Q-space, it pays to build H_QQ once as distributed block-sparse blocks
and reuse it via block GEMV every iteration.

These `tps_sharded_cepa_*` routines do exactly that. They are separate from the
matrix-free `tpsci_*` versions (which are left untouched); the Hamiltonian
applicator is pluggable (`MatrixFreeShardedH` or `ShardedBlockH`), so `h_storage`
selects re-contract vs stored-block per run. The Q-space H is never a dense
`dim_q x dim_q` matrix and is never gathered on the master.
"""

# Shifted Q-space apply used by the CEPA linear solvers: (H_QQ - eshift*I) v.
# `op` applies H_QQ (matrix-free or block-stored); see apply_sharded_H.
function tps_sharded_apply_shifted_hq(op, v::DistributedTPSCIstate{T,N,1}, eshift) where {T,N}
    out = apply_sharded_H(op, v)
    add_scaled!(out, -eshift, v)
    return out
end

"""
    tps_sharded_cepa_cg_linsolve(b, op; eshift, tol, maxiter, verbose)

Conjugate-gradient solve of `(H_QQ - eshift*I) x = b` over the sharded Q-space,
applying the Hamiltonian through the pluggable operator `op`.
"""
function tps_sharded_cepa_cg_linsolve(b::DistributedTPSCIstate{T,N,1}, op;
                                      eshift,
                                      tol=1e-5,
                                      maxiter=300,
                                      verbose=0) where {T,N}
    x = similar_sharded_state(b)
    r = copy_sharded_state(b)
    p = copy_sharded_state(r)
    rr = overlap(r, r)[1, 1]
    r0 = sqrt(abs(rr))
    threshold = tol * max(r0, one(T))
    converged = r0 <= threshold
    iters = 0

    for it in 1:maxiter
        converged && break
        iters = it
        Ap = tps_sharded_apply_shifted_hq(op, p, eshift)
        pAp = overlap(p, Ap)[1, 1]
        abs(pAp) > eps(T) ||
            error("TPS-CEPA sharded CG encountered a near-zero denominator")
        alpha = rr / pAp
        add_scaled!(x, alpha, p)
        add_scaled!(r, -alpha, Ap)
        destroy!(Ap)

        rr_new = overlap(r, r)[1, 1]
        resid = sqrt(abs(rr_new))
        if verbose > 1
            @printf("   TPS-CEPA CG iter %4i residual %12.4e\n", it, resid)
        end
        if resid <= threshold
            rr = rr_new
            converged = true
            break
        end
        beta = rr_new / rr
        linear_combination!(p, beta, r, one(T))
        rr = rr_new
    end

    final_resid = sqrt(abs(rr))
    destroy!(r)
    destroy!(p)
    return x, (iters=iters, residual=final_resid, converged=converged)
end

"""
    tps_sharded_cepa_minres_linsolve(b, op; eshift, tol, abstol, maxiter, verbose)

MINRES solve of the symmetric (indefinite) system `(H_QQ - eshift*I) x = b` over
the sharded Q-space, applying the Hamiltonian through the pluggable operator `op`.
"""
function tps_sharded_cepa_minres_linsolve(b::DistributedTPSCIstate{T,N,1}, op;
                                          eshift,
                                          tol=1e-5,
                                          abstol=zero(T),
                                          maxiter=300,
                                          verbose=0) where {T,N}
    x = similar_sharded_state(b)
    v_prev = similar_sharded_state(b)
    v_curr = copy_sharded_state(b)
    v_next = similar_sharded_state(b)
    w_prev = similar_sharded_state(b)
    w_curr = similar_sharded_state(b)
    w_next = similar_sharded_state(b)

    resnorm = norm(v_curr)[1]
    threshold = max(tol * resnorm, abstol)
    if resnorm <= threshold
        for tmp in (v_prev, v_curr, v_next, w_prev, w_curr, w_next)
            destroy!(tmp)
        end
        return x, (iters=0, residual=resnorm, converged=true)
    end

    scale!(v_curr, inv(resnorm))
    H = zeros(T, 4)
    rhs = [resnorm, zero(T)]
    c_prev = one(T)
    s_prev = zero(T)
    c_curr = one(T)
    s_curr = zero(T)
    converged = false
    iters = 0

    for it in 1:maxiter
        iters = it

        destroy!(v_next)
        v_next = tps_sharded_apply_shifted_hq(op, v_curr, eshift)
        it > 1 && add_scaled!(v_next, -H[2], v_prev)

        proj = overlap(v_curr, v_next)[1, 1]
        H[3] = real(proj)
        add_scaled!(v_next, -proj, v_curr)

        H[4] = norm(v_next)[1]
        H[4] > eps(T) && scale!(v_next, inv(H[4]))

        if it > 2
            H[1] = s_prev * H[2]
            H[2] = c_prev * H[2]
        end

        if it > 1
            tmp = -s_curr * H[2] + c_curr * H[3]
            H[2] = c_curr * H[2] + s_curr * H[3]
            H[3] = tmp
        end

        c, s, H[3] = _tpsci_givens(H[3], H[4])
        rhs[2] = -s * rhs[1]
        rhs[1] = c * rhs[1]

        destroy!(w_next)
        w_next = copy_sharded_state(v_curr)
        it > 1 && add_scaled!(w_next, -H[2], w_curr)
        it > 2 && add_scaled!(w_next, -H[1], w_prev)
        abs(H[3]) > eps(T) ||
            error("TPS-CEPA sharded MINRES encountered a near-zero Hessenberg pivot")
        scale!(w_next, inv(H[3]))

        add_scaled!(x, rhs[1], w_next)

        v_prev, v_curr, v_next = v_curr, v_next, v_prev
        w_prev, w_curr, w_next = w_curr, w_next, w_prev
        c_prev, s_prev, c_curr, s_curr = c_curr, s_curr, c, s
        rhs[1] = rhs[2]
        H[2] = H[4]

        resnorm = abs(rhs[2])
        if verbose > 1
            @printf("   TPS-CEPA MINRES iter %4i residual %12.4e\n", it, resnorm)
        end
        if resnorm <= threshold
            converged = true
            break
        end
    end

    for tmp in (v_prev, v_curr, v_next, w_prev, w_curr, w_next)
        destroy!(tmp)
    end
    return x, (iters=iters, residual=resnorm, converged=converged)
end

# Build the reusable Q-space Hamiltonian operator (once) for a CEPA solve.
function _tps_sharded_cepa_build_hq_op(cepa_vector::DistributedTPSCIstate,
                                       cluster_ops, clustered_ham;
                                       h_storage::Symbol, max_mem_H, workers,
                                       threaded_worker, blas_threads, verbose)
    tier = h_storage
    if tier == :auto
        rep = sharded_H_memory_report(cepa_vector, clustered_ham)
        tier = rep.gb <= max_mem_H ? :blocks : :matrixfree
        verbose >= 0 &&
            @printf(" TPS-CEPA H storage :auto -> :%s  (block-sparse H_Q ~ %.3f GB, budget %.1f GB)\n",
                    tier, rep.gb, max_mem_H)
    end
    if tier == :blocks
        op = build_block_h_sharded(cepa_vector, cluster_ops, clustered_ham;
                                   workers=workers, blas_threads=blas_threads,
                                   verbose=verbose)
        return op, true
    elseif tier == :matrixfree
        op = MatrixFreeShardedH(cluster_ops, clustered_ham, workers,
                                threaded_worker, Int(blas_threads))
        return op, false
    else
        error("Unknown h_storage: $h_storage (use :auto, :blocks, or :matrixfree)")
    end
end

"""
    tps_sharded_cepa_solve(ref_vector, e0, cepa_vector, cluster_ops, clustered_ham,
                           cepa_shift="cepa", cepa_mit=50; h_storage=:auto, ...)

Solve the TPS-CEPA equations with the Q-space amplitudes stored as a sharded
`DistributedTPSCIstate`. The Q-space Hamiltonian is built once (`h_storage=:blocks`
or chosen by `:auto`) and reused across all roots, macro-iterations, and
MINRES/CG steps. Nothing is gathered on the master.
"""
function tps_sharded_cepa_solve(ref_vector::TPSCIstate{T,N,R}, e0::Vector,
                                cepa_vector::DistributedTPSCIstate{T,N,Rq},
                                cluster_ops, clustered_ham,
                                cepa_shift="cepa",
                                cepa_mit=50;
                                tol=1e-5,
                                cg_maxiter=300,
                                nbody=4,
                                thresh_sigma=1e-8,
                                solver=:minres,
                                h_storage::Symbol=:auto,
                                max_mem_H=50.0,
                                workers=cepa_vector.workers,
                                threaded_worker=true,
                                blas_threads=1,
                                verbose=0) where {T,N,R,Rq}
    worker_ids = ensure_tpsci_multinode_workers!(workers=workers)
    worker_ids == cepa_vector.workers ||
        error("tps_sharded_cepa_solve requires the CEPA vector on the requested workers")
    n_clusters = length(ref_vector.clusters)

    @printf(" TPS-CEPA (stored-H) solver: dim_q=%i, R=%i, shift=%s\n",
            length(cepa_vector), R, cepa_shift)
    flush(stdout)

    # Build the Q-space Hamiltonian operator ONCE and reuse it everywhere below.
    hq_op, is_block = _tps_sharded_cepa_build_hq_op(cepa_vector, cluster_ops,
                                                    clustered_ham;
                                                    h_storage=h_storage,
                                                    max_mem_H=max_mem_H,
                                                    workers=worker_ids,
                                                    threaded_worker=threaded_worker,
                                                    blas_threads=blas_threads,
                                                    verbose=verbose)

    Ec = zeros(T, R)
    try
        for i in 1:R
            @printf(" Compute sharded coupling vector h for root %i\n", i)
            ref_i = extract_chosen_root(ref_vector, i)
            dref_i = distribute_tpsci_state(ref_i; workers=worker_ids,
                                            strategy=:balanced,
                                            blas_threads=blas_threads)
            # Coupling vector h = <Q|H|ref> stays matrix-free (rectangular ref->Q,
            # computed once per root; not the iterated within-Q operator).
            sig_i = open_matvec_sharded(dref_i, cluster_ops, clustered_ham;
                                        nbody=nbody,
                                        thresh=thresh_sigma,
                                        prescreen=false,
                                        workers=worker_ids,
                                        threaded_worker=threaded_worker,
                                        blas_threads=blas_threads)
            h_i = restrict_to_basis_sharded(sig_i, cepa_vector)
            destroy!(sig_i)
            destroy!(dref_i)

            Ec_i = zero(T)
            Ec_prev_i = T(Inf)

            for it in 1:cepa_mit
                shift = zero(T)
                if cepa_shift == "cepa"
                    shift = zero(T)
                elseif cepa_shift == "acpf"
                    shift = Ec_i * 2.0 / n_clusters
                elseif cepa_shift == "aqcc"
                    shift = (1.0 - (n_clusters - 3.0) * (n_clusters - 2.0) /
                             (n_clusters * (n_clusters - 1.0))) * Ec_i
                elseif cepa_shift == "cisd"
                    shift = Ec_i
                else
                    error("Unknown cepa_shift: $cepa_shift")
                end

                @printf(" CEPA Iter %3i  Root %i  Shift = %12.8f\n", it, i, shift)
                rhs = copy_sharded_state(h_i)
                scale!(rhs, -one(T))
                if solver == :minres
                    Cd_i, history = tps_sharded_cepa_minres_linsolve(
                        rhs, hq_op;
                        eshift=e0[i] + shift, tol=tol, maxiter=cg_maxiter,
                        verbose=verbose)
                elseif solver == :cg || solver == :krylov
                    Cd_i, history = tps_sharded_cepa_cg_linsolve(
                        rhs, hq_op;
                        eshift=e0[i] + shift, tol=tol, maxiter=cg_maxiter,
                        verbose=verbose)
                else
                    error("Unknown TPS-CEPA solver: $solver")
                end
                Ec_i = overlap(Cd_i, h_i)[1, 1]
                if verbose > 0
                    @printf(" Iter %3i  Root %i  nops=%4i  res=%8.2e  E_corr = %16.12f\n",
                            it, i, history.iters, history.residual, Ec_i)
                end
                destroy!(Cd_i)
                destroy!(rhs)

                cepa_shift == "cepa" && break
                abs(Ec_i - Ec_prev_i) < tol && break
                Ec_prev_i = Ec_i
            end

            Ec[i] = Ec_i
            destroy!(h_i)
        end
    finally
        is_block && destroy!(hq_op)
    end
    return Ec, e0 .+ Ec
end

"""
    do_tps_sharded_cepa(ref, cluster_ops, clustered_ham; h_storage=:auto, ...)

Build the TPS-CEPA FOIS/Q-space as a sharded `DistributedTPSCIstate`, then solve
CEPA with the Q-space Hamiltonian built once and reused (`h_storage=:blocks`, or
`:auto` chosen by memory budget). The returned second object is the sharded Q
vector; call `destroy!` on it when finished.
"""
function do_tps_sharded_cepa(ref::TPSCIstate{T,N,R}, cluster_ops, clustered_ham;
                             cepa_shift="cepa",
                             cepa_mit=30,
                             nbody=4,
                             thresh_foi=1e-6,
                             thresh_clip=1e-5,
                             tol=1e-8,
                             thresh_sigma=1e-8,
                             compress=false,
                             reference_solver=:sharded_davidson,
                             solver=:minres,
                             h_storage::Symbol=:auto,
                             max_mem_H=50.0,
                             e0=nothing,
                             ci_conv=1e-8,
                             ci_max_iter=100,
                             ci_max_ss_vecs=4,
                             ci_lindep_thresh=1e-12,
                             cg_maxiter=300,
                             workers=Distributed.workers(),
                             threaded_worker=true,
                             blas_threads=1,
                             verbose=1) where {T,N,R}
    worker_ids = ensure_tpsci_multinode_workers!(workers=workers)
    @printf("\n-------------------------------------------------------\n")
    @printf(" Do TPS-CEPA (stored-H, across nodes)\n")
    @printf("   thresh_foi              = %-8.1e\n", thresh_foi)
    @printf("   nbody                   = %-i\n", nbody)
    @printf("   Length of Reference     = %-i\n", length(ref))
    @printf("   Calculation type        = %s\n", cepa_shift)
    @printf("   H storage               = %s\n", h_storage)
    @printf("   Workers                 = %s\n", worker_ids)
    @printf("\n-------------------------------------------------------\n")
    flush(stdout)

    ref_vec = deepcopy(ref)
    if e0 === nothing
        @printf(" Solve zeroth-order problem. Dimension = %10i\n", length(ref_vec))
        if reference_solver == :direct
            cluster_ops isa DistributedClusterOps &&
                error("reference_solver=:direct needs a local Vector{ClusterOps}; use :sharded_davidson with DistributedClusterOps, or pass e0")
            @time e0, ref_vec = tps_ci_direct(ref_vec, cluster_ops, clustered_ham,
                                              conv_thresh=tol)
        elseif reference_solver == :distributed_davidson
            cluster_ops isa DistributedClusterOps &&
                error("reference_solver=:distributed_davidson needs a local Vector{ClusterOps}; use :sharded_davidson, or pass e0")
            orthonormalize!(ref_vec)
            @time e0, ref_vec = tps_ci_davidson_distributed(
                ref_vec, cluster_ops, clustered_ham;
                conv_thresh=tol, max_iter=ci_max_iter, max_ss_vecs=ci_max_ss_vecs,
                lindep_thresh=ci_lindep_thresh, workers=worker_ids,
                blas_threads=blas_threads)
        elseif reference_solver == :sharded_davidson
            orthonormalize!(ref_vec)
            dref0 = distribute_tpsci_state(ref_vec; workers=worker_ids,
                                           strategy=:balanced, blas_threads=blas_threads)
            e0, dref0_out = tps_ci_davidson_sharded(dref0, cluster_ops, clustered_ham;
                nroots=R, conv_thresh=ci_conv, max_iter=ci_max_iter,
                max_ss_vecs=ci_max_ss_vecs, lindep_thresh=ci_lindep_thresh,
                h_storage=h_storage, max_mem_H=max_mem_H, workers=worker_ids,
                threaded_worker=threaded_worker, blas_threads=blas_threads,
                verbose=verbose)
            ref_vec = collect_tpsci_state(dref0_out)
            e0 = Vector{T}(e0)
            destroy!(dref0_out)
            destroy!(dref0)
        else
            error("Unknown reference_solver: $reference_solver")
        end
    else
        e0 = Vector{T}(e0)
    end

    println()
    println(" Compute sharded FOIS. Reference space dim = ", length(ref_vec))
    dref = distribute_tpsci_state(ref_vec; workers=worker_ids,
                                  strategy=:balanced, blas_threads=blas_threads)
    pt1_vec = open_matvec_sharded(dref, cluster_ops, clustered_ham;
                                  nbody=nbody, thresh=thresh_foi, prescreen=false,
                                  workers=worker_ids, threaded_worker=threaded_worker,
                                  blas_threads=blas_threads)
    project_out!(pt1_vec, dref)

    if compress
        dim1 = length(pt1_vec)
        clip!(pt1_vec, thresh=thresh_clip)
        dim2 = length(pt1_vec)
        @printf(" %-50s%10i -> %-10i (thresh = %8.1e)\n",
                "FOIS Compressed from: ", dim1, dim2, thresh_clip)
    end

    S10 = overlap(pt1_vec, dref)
    for i in 1:R
        @printf(" %-50s%10.6f\n", "Overlap between <1|0>: ", S10[i, i])
    end
    destroy!(dref)

    println()
    println(" Do TPS-CEPA (stored-H): shared FOIS dim = ", length(pt1_vec))
    @time Ec, e_cepa = tps_sharded_cepa_solve(
        ref_vec, e0, pt1_vec, cluster_ops, clustered_ham, cepa_shift, cepa_mit;
        tol=tol, cg_maxiter=cg_maxiter, nbody=nbody, thresh_sigma=thresh_sigma,
        solver=solver, h_storage=h_storage, max_mem_H=max_mem_H,
        workers=worker_ids, threaded_worker=threaded_worker,
        blas_threads=blas_threads, verbose=verbose)

    for i in 1:R
        @printf(" E(cepa) root %i  corr= %12.8f  total= %12.8f\n",
                i, Ec[i], e_cepa[i])
    end

    return e_cepa, pt1_vec
end
