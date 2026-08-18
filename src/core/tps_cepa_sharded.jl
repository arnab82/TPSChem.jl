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
# `op` applies H_QQ (matrix-free or block-stored); see apply_sharded_H. The
# stored-block operator folds the diagonal shift into the same worker pass, so
# this is one fan-out there and two for the re-contracting matrix-free operator.
# `id`, when given, names the destination so the solvers can recycle a fixed set
# of Krylov buffers instead of allocating (and registering) a new sharded state
# on every iteration.
function tps_sharded_apply_shifted_hq(op::ShardedBlockH, v::DistributedTPSCIstate{T,N,1},
                                      eshift; id=nothing) where {T,N}
    return apply_sharded_H(op, v; eshift=eshift, id=id)
end

function tps_sharded_apply_shifted_hq(op, v::DistributedTPSCIstate{T,N,1},
                                      eshift; id=nothing) where {T,N}
    out = apply_sharded_H(op, v; id=id)
    add_scaled!(out, -eshift, v)
    return out
end

# Apply into the recycled buffer `buf`. Operators that cannot honor the requested
# destination id hand back their own state, in which case the stale buffer is
# released here rather than leaking a sharded state per iteration.
function _tps_sharded_apply_into!(op, v::DistributedTPSCIstate{T,N,1}, eshift,
                                  buf::DistributedTPSCIstate{T,N,1}) where {T,N}
    out = tps_sharded_apply_shifted_hq(op, v, eshift; id=buf.id)
    out.id == buf.id || destroy!(buf)
    return out
end

"""
    tps_sharded_cepa_cg_linsolve(b, op; eshift, tol, maxiter, verbose)

Conjugate-gradient solve of `(H_QQ - eshift*I) x = b` over the sharded Q-space,
applying the Hamiltonian through the pluggable operator `op`.

The per-iteration vector algebra is issued as fused batches (`sharded_fused_ops!`),
so an iteration costs one Hamiltonian apply plus three fan-outs rather than one
fan-out per axpy/dot.
"""
function tps_sharded_cepa_cg_linsolve(b::DistributedTPSCIstate{T,N,1}, op;
                                      eshift,
                                      tol=1e-5,
                                      maxiter=300,
                                      verbose=0) where {T,N}
    x = similar_sharded_state(b)
    r = copy_sharded_state(b)
    p = copy_sharded_state(r)
    Ap = similar_sharded_state(b)
    # Index order used by every fused batch below.
    vecs = [x, r, p, Ap]
    IX, IR, IP, IAP = 1, 2, 3, 4

    rr = sharded_fused_ops!(vecs, [sv_nrm2sq(T, IR)])[1]
    r0 = sqrt(abs(rr))
    threshold = tol * max(r0, one(T))
    converged = r0 <= threshold
    iters = 0

    for it in 1:maxiter
        converged && break
        iters = it
        vecs[IAP] = _tps_sharded_apply_into!(op, p, eshift, vecs[IAP])
        pAp = sharded_fused_ops!(vecs, [sv_dot(T, IP, IAP)])[1]
        abs(pAp) > eps(T) ||
            error("TPS-CEPA sharded CG encountered a near-zero denominator")
        alpha = rr / pAp
        rr_new = sharded_fused_ops!(vecs, [sv_axpy(T, IX, alpha, IP),
                                           sv_axpy(T, IR, -alpha, IAP),
                                           sv_nrm2sq(T, IR)])[1]

        resid = sqrt(abs(rr_new))
        if verbose > 2
            @printf("   TPS-CEPA CG iter %4i residual %12.4e\n", it, resid)
        end
        if resid <= threshold
            rr = rr_new
            converged = true
            break
        end
        beta = rr_new / rr
        # p .= beta*p + r
        sharded_fused_ops!(vecs, [sv_axpby(T, IP, beta, one(T), IR)])
        rr = rr_new
    end

    final_resid = sqrt(abs(rr))
    destroy!(r)
    destroy!(p)
    destroy!(vecs[IAP])
    return x, (iters=iters, residual=final_resid, converged=converged)
end

"""
    tps_sharded_cepa_minres_linsolve(b, op; eshift, tol, abstol, maxiter, verbose, x0)

MINRES solve of the symmetric (indefinite) system `(H_QQ - eshift*I) x = b` over
the sharded Q-space, applying the Hamiltonian through the pluggable operator `op`.

`x0`, when given, is the initial guess **and the destination**: the solve runs on
the residual system and accumulates the correction into `x0` in place, which is
then returned. Reusing it as the solution vector is what keeps warm starting free
of an extra Q-sized vector. Successive CEPA macro-iterations differ only by a
small change of shift, so starting from the previous amplitudes cuts the later
solves to a handful of iterations.

Each iteration costs one Hamiltonian apply plus three fused fan-outs; the Krylov
buffers are allocated once and rotated by name, so nothing is copied or
reallocated per iteration.
"""
function tps_sharded_cepa_minres_linsolve(b::DistributedTPSCIstate{T,N,1}, op;
                                          eshift,
                                          tol=1e-5,
                                          abstol=zero(T),
                                          maxiter=300,
                                          verbose=0,
                                          x0=nothing) where {T,N}
    # With a guess, the guess IS the solution vector: the Krylov correction is
    # accumulated into it, so warm starting costs no extra Q-sized vector.
    x = x0 === nothing ? similar_sharded_state(b) : x0
    v_prev = similar_sharded_state(b)
    v_curr = copy_sharded_state(b)
    v_next = similar_sharded_state(b)
    w_prev = similar_sharded_state(b)
    w_curr = similar_sharded_state(b)
    w_next = similar_sharded_state(b)

    # Converge on the residual relative to `b`, not to the starting residual:
    # with a warm start the latter is already small, and scaling the tolerance by
    # it would demand a far tighter solve than the cold start ever did.
    bnorm = sqrt(abs(sharded_fused_ops!([v_curr], [sv_nrm2sq(T, 1)])[1]))
    resnorm = bnorm
    if x0 !== nothing
        # v_curr <- b - (H - eshift)x0, and solve for the correction.
        ax0 = tps_sharded_apply_shifted_hq(op, x0, eshift)
        r2 = sharded_fused_ops!([v_curr, ax0], [sv_axpy(T, 1, -one(T), 2),
                                                sv_nrm2sq(T, 1)])[1]
        destroy!(ax0)
        resnorm = sqrt(abs(r2))
    end

    threshold = max(tol * bnorm, abstol)
    if resnorm <= threshold
        for tmp in (v_prev, v_curr, v_next, w_prev, w_curr, w_next)
            destroy!(tmp)
        end
        return x, (iters=0, residual=resnorm, converged=true)
    end

    sharded_fused_ops!([v_curr], [sv_scale(T, 1, inv(resnorm))])
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

        # One Hamiltonian apply, writing into the recycled `v_next` buffer.
        v_next = _tps_sharded_apply_into!(op, v_curr, eshift, v_next)

        # Fan-out 1: finish the Lanczos vector against v_prev and take <v_curr|v_next>.
        vecs = [v_next, v_prev, v_curr, w_next, w_curr, w_prev, x]
        ops = ShardedVecOp{T}[]
        it > 1 && push!(ops, sv_axpy(T, 1, -H[2], 2))
        push!(ops, sv_dot(T, 1, 3))
        proj = sharded_fused_ops!(vecs, ops)[1]
        H[3] = real(proj)

        # Fan-out 2: orthogonalize against v_curr and measure the new vector.
        H[4] = sqrt(abs(sharded_fused_ops!(vecs, [sv_axpy(T, 1, -proj, 3),
                                                  sv_nrm2sq(T, 1)])[1]))

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
        abs(H[3]) > eps(T) ||
            error("TPS-CEPA sharded MINRES encountered a near-zero Hessenberg pivot")

        # Fan-out 3: normalize v_next, build w_next from the three-term recurrence
        # and take the solution step.
        ops = ShardedVecOp{T}[]
        H[4] > eps(T) && push!(ops, sv_scale(T, 1, inv(H[4])))
        push!(ops, sv_copy(T, 4, 3))                       # w_next .= v_curr
        it > 1 && push!(ops, sv_axpy(T, 4, -H[2], 5))      # -= H2 * w_curr
        it > 2 && push!(ops, sv_axpy(T, 4, -H[1], 6))      # -= H1 * w_prev
        push!(ops, sv_scale(T, 4, inv(H[3])))
        push!(ops, sv_axpy(T, 7, rhs[1], 4))               # x += rhs1 * w_next
        sharded_fused_ops!(vecs, ops)

        v_prev, v_curr, v_next = v_curr, v_next, v_prev
        w_prev, w_curr, w_next = w_curr, w_next, w_prev
        c_prev, s_prev, c_curr, s_curr = c_curr, s_curr, c, s
        rhs[1] = rhs[2]
        H[2] = H[4]

        resnorm = abs(rhs[2])
        if verbose > 2
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


# Smallest entry of a sharded scalar field, reduced on the master. Used to test
# whether the shifted operator is positive definite before trusting CG.
function _tpsci_sharded_local_min(state_id::Symbol)
    st = _tpsci_sharded_get_state(state_id)
    m = Inf
    for (_, cfgs) in st.data
        for (_, c) in cfgs
            m = min(m, Float64(c[1]))
        end
    end
    return m
end

function sharded_state_min(s::DistributedTPSCIstate)
    vals = Dict{Int,Float64}()
    @sync for pid in s.workers
        @async vals[pid] = Distributed.remotecall_fetch(_tpsci_sharded_local_min,
                                                        pid, s.id)
    end
    return isempty(vals) ? Inf : minimum(values(vals))
end

"""
    tps_sharded_cepa_pcg_linsolve(b, op, Hdiag; eshift, tol, maxiter, verbose, x0)

Jacobi-preconditioned conjugate gradient for `(H_QQ - eshift*I) x = b`.

`Hdiag` is the Q-space Hamiltonian diagonal (root independent, so it is built
once per solve and reused for every root and macro-iteration). The
preconditioner is `M = diag(H_QQ) - eshift`, applied through the existing
sharded `precondition_sharded` kernel, which already floors near-zero
denominators.

CG assumes the shifted operator is positive definite; the caller is expected to
have checked that with `sharded_state_min` and fallen back to MINRES otherwise.
"""
function tps_sharded_cepa_pcg_linsolve(b::DistributedTPSCIstate{T,N,1}, op,
                                       Hdiag::DistributedTPSCIstate{T,N,1};
                                       eshift,
                                       tol=1e-5,
                                       maxiter=300,
                                       verbose=0,
                                       x0=nothing) where {T,N}
    x = x0 === nothing ? similar_sharded_state(b) : x0
    r = copy_sharded_state(b)
    Ap = similar_sharded_state(b)

    # Measure convergence against ||b||, not against the starting residual: a
    # warm start makes the latter small, which would silently tighten `tol` on
    # every macro-iteration and cancel the benefit of warm starting.
    bnorm = sqrt(abs(sharded_fused_ops!([r], [sv_nrm2sq(T, 1)])[1]))
    threshold = tol * max(bnorm, one(T))

    # Warm start: solve for the correction, r = b - A*x0.
    if x0 !== nothing
        Ax = _tps_sharded_apply_into!(op, x, eshift, Ap)
        sharded_fused_ops!([r, Ax], [sv_axpy(T, 1, -one(T), 2)])
        Ap = Ax
    end

    # z = r / (diag - eshift). precondition_sharded returns r/(eshift - diag),
    # so the fused batch below flips the sign as it stages the vectors.
    z = precondition_sharded(r, Hdiag, eshift)
    p = copy_sharded_state(z)
    vecs = [x, r, p, Ap, z]
    IX, IR, IP, IAP, IZ = 1, 2, 3, 4, 5
    sharded_fused_ops!(vecs, [sv_scale(T, IZ, -one(T)),
                              sv_scale(T, IP, -one(T))])

    rr = sharded_fused_ops!(vecs, [sv_nrm2sq(T, IR)])[1]
    resid = sqrt(abs(rr))
    rz = sharded_fused_ops!(vecs, [sv_dot(T, IR, IZ)])[1]
    converged = resid <= threshold
    pAp0 = zero(T)
    iters = 0

    for it in 1:maxiter
        converged && break
        iters = it
        vecs[IAP] = _tps_sharded_apply_into!(op, vecs[IP], eshift, vecs[IAP])
        pAp = sharded_fused_ops!(vecs, [sv_dot(T, IP, IAP)])[1]

        # A genuinely indefinite operator gives pAp <= 0 and CG cannot proceed.
        pAp > zero(T) ||
            error("TPS-CEPA sharded PCG: shifted operator is not positive " *
                  "definite (pAp = $pAp). Use solver=:minres for this root.")
        it == 1 && (pAp0 = pAp)
        # Near the solution p -> 0, so pAp underflows on its own. That is
        # convergence stalling at the floating-point floor, not a breakdown:
        # test it relative to the first iteration rather than against eps.
        if pAp <= eps(T) * pAp0
            verbose > 1 && @printf("   TPS-CEPA PCG stalled at fp precision (iter %i)\n", it)
            break
        end

        alpha = rz / pAp
        rr = sharded_fused_ops!(vecs, [sv_axpy(T, IX, alpha, IP),
                                       sv_axpy(T, IR, -alpha, IAP),
                                       sv_nrm2sq(T, IR)])[1]
        resid = sqrt(abs(rr))
        if verbose > 2
            @printf("   TPS-CEPA PCG iter %4i residual %12.4e\n", it, resid)
        end
        if resid <= threshold
            converged = true
            break
        end

        vecs[IZ] = precondition_sharded(vecs[IR], Hdiag, eshift; id=vecs[IZ].id)
        rz_new = sharded_fused_ops!(vecs, [sv_scale(T, IZ, -one(T)),
                                           sv_dot(T, IR, IZ)])[1]
        beta = rz_new / rz
        # p .= z + beta*p
        sharded_fused_ops!(vecs, [sv_axpby(T, IP, beta, one(T), IZ)])
        rz = rz_new
    end

    destroy!(vecs[IR])
    destroy!(vecs[IP])
    destroy!(vecs[IAP])
    destroy!(vecs[IZ])
    return vecs[IX], (iters=iters, residual=resid, converged=converged)
end

# Build the reusable Q-space Hamiltonian operator (once) for a CEPA solve.
function _tps_sharded_cepa_build_hq_op(cepa_vector::DistributedTPSCIstate,
                                       cluster_ops, clustered_ham;
                                       h_storage::Symbol, max_mem_H, workers,
                                       threaded_worker, blas_threads, verbose)
    tier = h_storage
    if tier == :auto
        tier, rep, _ = _sharded_H_auto_tier(cepa_vector, clustered_ham, workers,
                                            max_mem_H; verbose=verbose,
                                            label="TPS-CEPA H_Q")
        # verbose=0 silences the feasibility table, but a downgrade to matrix-free
        # changes the cost of the run by orders of magnitude, so it is always
        # announced. Silence should hide detail, never a decision like this one.
        if tier == :matrixfree && verbose <= 0
            @printf(" TPS-CEPA H_Q does not fit (%.2f GB aggregate); falling back to :matrixfree\n",
                    rep.gb)
            flush(stdout)
        end
    elseif tier == :blocks
        # Explicit :blocks still gets a feasibility check — silently OOM-killing a
        # multi-node job hours in is far worse than refusing up front.
        rep = sharded_H_memory_report(cepa_vector, clustered_ham)
        fit = sharded_H_fit_report(rep, workers)
        verbose > 0 && print_sharded_H_fit(rep, fit; label="TPS-CEPA H_Q")
        fit.fits ||
            error("h_storage=:blocks needs $(round(rep.max_worker_gb; digits=2)) GB on " *
                  "worker $(rep.max_worker_pid), but worker $(fit.worst_pid) is short by " *
                  "$(round(fit.worst_deficit_gb; digits=2)) GB. Add nodes, put one worker " *
                  "per node, or use h_storage=:matrixfree.")
    end
    if tier == :blocks
        op = build_block_h_sharded(cepa_vector, cluster_ops, clustered_ham;
                                   workers=workers, blas_threads=blas_threads,
                                   threaded_worker=threaded_worker,
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
                                linsolve_tol=nothing,
                                warm_start=true,
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

    if verbose > 0
        @printf(" TPS-CEPA (stored-H) solver: dim_q=%i, R=%i, shift=%s\n",
                length(cepa_vector), R, cepa_shift)
        flush(stdout)
    end

    # Build the Q-space Hamiltonian operator ONCE and reuse it everywhere below.
    hq_op, is_block = _tps_sharded_cepa_build_hq_op(cepa_vector, cluster_ops,
                                                    clustered_ham;
                                                    h_storage=h_storage,
                                                    max_mem_H=max_mem_H,
                                                    workers=worker_ids,
                                                    threaded_worker=threaded_worker,
                                                    blas_threads=blas_threads,
                                                    verbose=verbose)

    # The Q-space diagonal is root independent, so build it once here and reuse
    # it for every root and macro-iteration. Only the shift changes.
    Hdiag = nothing
    use_pcg = solver === :pcg
    if use_pcg
        if is_block
            Hdiag = compute_diagonal_sharded(hq_op, cepa_vector)
        else
            @printf(" solver=:pcg needs a stored H for the diagonal; using :minres\n")
            use_pcg = false
        end
    end

    # Inner Krylov tolerance defaults to the outer one, but is worth loosening:
    # solving to 1e-8 while the shift is still moving between macro-iterations is
    # wasted work.
    ltol = linsolve_tol === nothing ? tol : linsolve_tol

    Ec = zeros(T, R)
    try
        for i in 1:R
            verbose > 1 && @printf(" Compute sharded coupling vector h for root %i\n", i)
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
                                        blas_threads=blas_threads,
                                        verbose=(verbose > 1 ? 1 : 0))
            h_i = restrict_to_basis_sharded(sig_i, cepa_vector)
            destroy!(sig_i)
            destroy!(dref_i)

            Ec_i = zero(T)
            Ec_prev_i = T(Inf)
            # The right-hand side -h is the same for every macro-iteration; only
            # the shift changes. Build it once instead of copying and rescaling
            # the whole Q vector each time round.
            rhs = copy_sharded_state(h_i)
            sharded_fused_ops!([rhs], [sv_scale(T, 1, -one(T))])
            Cd_i = nothing

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

                verbose > 1 && @printf(" CEPA Iter %3i  Root %i  Shift = %12.8f\n", it, i, shift)
                eshift_i = e0[i] + shift
                # CG only converges for a positive-definite shifted operator. The
                # cheapest sufficient check is the diagonal: if any Q-space state
                # sits below the shift, fall back to MINRES for this solve.
                pcg_ok = use_pcg
                if pcg_ok
                    dmin = sharded_state_min(Hdiag) - Float64(eshift_i)
                    if dmin <= 0
                        verbose > 0 && @printf("   shifted diagonal is not positive (min %.3e); using MINRES\n", dmin)
                        pcg_ok = false
                    end
                end
                if pcg_ok
                    Cd_new, history = tps_sharded_cepa_pcg_linsolve(
                        rhs, hq_op, Hdiag;
                        eshift=eshift_i, tol=ltol, maxiter=cg_maxiter,
                        verbose=verbose, x0=warm_start ? Cd_i : nothing)
                elseif solver == :minres || solver == :pcg
                    # Successive macro-iterations move the shift by only a small
                    # amount, so the previous amplitudes are an excellent guess
                    # and later solves converge in a few Krylov steps.
                    Cd_new, history = tps_sharded_cepa_minres_linsolve(
                        rhs, hq_op;
                        eshift=eshift_i, tol=ltol, maxiter=cg_maxiter,
                        verbose=verbose, x0=warm_start ? Cd_i : nothing)
                elseif solver == :cg || solver == :krylov
                    Cd_new, history = tps_sharded_cepa_cg_linsolve(
                        rhs, hq_op;
                        eshift=eshift_i, tol=ltol, maxiter=cg_maxiter,
                        verbose=verbose)
                else
                    error("Unknown TPS-CEPA solver: $solver")
                end
                # A warm-started MINRES accumulates into the guess and hands the
                # same vector back; only free the old amplitudes when the solver
                # actually allocated a new one.
                Cd_i === nothing || Cd_new.id === Cd_i.id || destroy!(Cd_i)
                Cd_i = Cd_new
                Ec_i = sharded_fused_ops!([Cd_i, h_i], [sv_dot(T, 1, 2)])[1]
                if verbose > 0
                    @printf(" Iter %3i  Root %i  nops=%4i  res=%8.2e  E_corr = %16.12f\n",
                            it, i, history.iters, history.residual, Ec_i)
                end

                cepa_shift == "cepa" && break
                abs(Ec_i - Ec_prev_i) < tol && break
                Ec_prev_i = Ec_i
            end

            Ec[i] = Ec_i
            Cd_i === nothing || destroy!(Cd_i)
            destroy!(rhs)
            destroy!(h_i)
        end
    finally
        Hdiag === nothing || destroy!(Hdiag)
        is_block && destroy!(hq_op)
    end
    return Ec, e0 .+ Ec
end

"""
    do_tps_sharded_cepa(ref, cluster_ops, clustered_ham; h_storage=:auto, ...)

Verbosity: `verbose=0` is silent apart from warnings that change behaviour
(a storage-tier fallback, a solver fallback); `1` (default) prints the run
banner, the H-storage feasibility decision, space dimensions, one line per
macro-iteration and the final energies; `2` adds per-root and per-shift detail
plus timings; `3` adds every Krylov iteration's residual.

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
                             linsolve_tol=nothing,
                             warm_start=true,
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
    if verbose > 0
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
    end

    ref_vec = deepcopy(ref)
    if e0 === nothing
        verbose > 0 && @printf(" Solve zeroth-order problem. Dimension = %10i\n", length(ref_vec))
        if reference_solver == :direct
            cluster_ops isa DistributedClusterOps &&
                error("reference_solver=:direct needs a local Vector{ClusterOps}; use :sharded_davidson with DistributedClusterOps, or pass e0")
            e0, ref_vec = tps_ci_direct(ref_vec, cluster_ops, clustered_ham,
                                              conv_thresh=tol)
        elseif reference_solver == :distributed_davidson
            cluster_ops isa DistributedClusterOps &&
                error("reference_solver=:distributed_davidson needs a local Vector{ClusterOps}; use :sharded_davidson, or pass e0")
            orthonormalize!(ref_vec)
            e0, ref_vec = tps_ci_davidson_distributed(
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

    if verbose > 0
        println()
        println(" Compute sharded FOIS. Reference space dim = ", length(ref_vec))
    end
    dref = distribute_tpsci_state(ref_vec; workers=worker_ids,
                                  strategy=:balanced, blas_threads=blas_threads)
    pt1_vec = open_matvec_sharded(dref, cluster_ops, clustered_ham;
                                  nbody=nbody, thresh=thresh_foi, prescreen=false,
                                  workers=worker_ids, threaded_worker=threaded_worker,
                                  blas_threads=blas_threads,
                                  verbose=(verbose > 0 ? 1 : 0))
    project_out!(pt1_vec, dref)

    if compress
        dim1 = length(pt1_vec)
        clip!(pt1_vec, thresh=thresh_clip)
        dim2 = length(pt1_vec)
        verbose > 0 && @printf(" %-50s%10i -> %-10i (thresh = %8.1e)\n",
                "FOIS Compressed from: ", dim1, dim2, thresh_clip)
    end

    S10 = overlap(pt1_vec, dref)
    if verbose > 0
        for i in 1:R
            @printf(" %-50s%10.6f\n", "Overlap between <1|0>: ", S10[i, i])
        end
    end
    destroy!(dref)

    if verbose > 0
        println()
        println(" Do TPS-CEPA (stored-H): shared FOIS dim = ", length(pt1_vec))
    end
    _t_cepa = time()
    Ec, e_cepa = tps_sharded_cepa_solve(
        ref_vec, e0, pt1_vec, cluster_ops, clustered_ham, cepa_shift, cepa_mit;
        tol=tol, cg_maxiter=cg_maxiter, nbody=nbody, thresh_sigma=thresh_sigma,
        solver=solver, linsolve_tol=linsolve_tol,
        warm_start=warm_start, h_storage=h_storage,
        max_mem_H=max_mem_H,
        workers=worker_ids, threaded_worker=threaded_worker,
        blas_threads=blas_threads, verbose=verbose)
    verbose > 1 && @printf(" TPS-CEPA solve: %.2f s\n", time() - _t_cepa)

    if verbose > 0
        for i in 1:R
            @printf(" E(cepa) root %i  corr= %12.8f  total= %12.8f\n",
                    i, Ec[i], e_cepa[i])
        end
    end

    return e_cepa, pt1_vec
end
