"""
Multinode helpers for SPT and SPT-PT2.

These entry points keep the existing single-node SPT methods untouched. They
parallelize the naturally large Fock-sector job loops over Julia distributed
workers and keep threaded work inside each worker. The returned SPT states are
gathered on the master because the current Tucker CI solver is local.
"""

function _spt_empty_state_like(v::SPTstate{T,N,R}) where {T,N,R}
    return SPTstate(v.clusters, v.p_spaces, v.q_spaces, T=T, R=R)
end

function _spt_valid_fock(fock::FockConfig{N}, clusters) where {N}
    all(f[1] >= 0 for f in fock) || return false
    all(f[2] >= 0 for f in fock) || return false
    all(f[1] <= length(clusters[fi]) for (fi, f) in enumerate(fock)) || return false
    all(f[2] <= length(clusters[fi]) for (fi, f) in enumerate(fock)) || return false
    return true
end

function _spt_has_cluster_space(fock::FockConfig, clusters, cluster_ops)
    for c in clusters
        haskey(cluster_ops[c.idx], "H") || return false
        haskey(cluster_ops[c.idx]["H"], (fock[c.idx], fock[c.idx])) || return false
    end
    return true
end

function _spt_has_cluster_space(fock::FockConfig, clusters,
                                cluster_ops::DistributedClusterOps)
    for c in clusters
        haskey(cluster_ops.h_sectors, c.idx) || return false
        fock[c.idx] in cluster_ops.h_sectors[c.idx] || return false
    end
    return true
end

function _spt_make_fock_jobs(psi::SPTstate{T,N,R}, cluster_ops, clustered_ham;
                             require_cluster_space::Bool) where {T,N,R}
    jobs = Dict{FockConfig{N},Vector{Tuple}}()
    clusters = psi.clusters
    for (fock_ket, configs_ket) in psi.data
        for (ftrans, terms) in clustered_ham
            fock_bra = ftrans + fock_ket
            _spt_valid_fock(fock_bra, clusters) || continue
            if require_cluster_space
                _spt_has_cluster_space(fock_bra, clusters, cluster_ops) || continue
            end

            job_input = (terms, fock_ket, configs_ket)
            if haskey(jobs, fock_bra)
                push!(jobs[fock_bra], job_input)
            else
                jobs[fock_bra] = [job_input]
            end
        end
    end

    jobs_vec = Tuple{FockConfig{N},Vector{Tuple}}[]
    for (fock_bra, job) in jobs
        push!(jobs_vec, (fock_bra, job))
    end
    return jobs_vec
end

function _spt_job_weight(job_item)
    _, jobs = job_item
    weight = 0
    for (terms, _, configs_ket) in jobs
        weight += max(length(terms), 1) * max(length(configs_ket), 1)
    end
    return max(weight, 1)
end

function _spt_least_loaded_pid(loads, pids)
    best = pids[1]
    best_load = loads[best]
    for pid in pids
        if loads[pid] < best_load
            best = pid
            best_load = loads[pid]
        end
    end
    return best
end

function _spt_job_chunks(jobs_vec::Vector{Tuple{FockConfig{N},Vector{Tuple}}},
                         pids; strategy::Symbol=:balanced) where {N}
    chunks = Dict(pid => Tuple{FockConfig{N},Vector{Tuple}}[] for pid in pids)
    loads = Dict(pid => 0 for pid in pids)
    for job_item in jobs_vec
        fock, _ = job_item
        pid = if strategy == :balanced
            _spt_least_loaded_pid(loads, pids)
        elseif strategy == :hash
            pids[Int(mod(hash(fock), UInt(length(pids)))) + 1]
        else
            error("Unknown SPT multinode sharding strategy: $strategy")
        end
        push!(chunks[pid], job_item)
        loads[pid] += _spt_job_weight(job_item)
    end
    return chunks
end

function _spt_needed_cluster_indices_from_jobs(jobs_chunk)
    needed = Set{Int}()
    for (_, jobs) in jobs_chunk
        for (terms, _, _) in jobs
            for term in terms
                for c in term.clusters
                    push!(needed, c.idx)
                end
            end
        end
    end
    return sort(collect(needed))
end

function _spt_needed_cluster_indices_for_sigma(sigma_chunk::SPTstate{T,N,R},
                                               ket::SPTstate{T,N,R},
                                               clustered_ham) where {T,N,R}
    needed = Set{Int}()
    for (fock_bra, _) in sigma_chunk
        for (fock_ket, _) in ket
            fock_trans = fock_bra - fock_ket
            haskey(clustered_ham, fock_trans) || continue
            for term in clustered_ham[fock_trans]
                for c in term.clusters
                    push!(needed, c.idx)
                end
            end
        end
    end
    return sort(collect(needed))
end

_spt_materialize_cluster_ops(cluster_ops, needed) = cluster_ops

function _spt_materialize_cluster_ops(cluster_ops::DistributedClusterOps, needed)
    return _materialize_cluster_ops_for_indices(cluster_ops, needed)
end

function _spt_merge_unique_fock_blocks!(dest::SPTstate, src::SPTstate)
    for (fock, configs) in src.data
        haskey(dest, fock) && error("Duplicate SPT Fock sector returned by multinode job: $fock")
        dest[fock] = configs
    end
    return dest
end

function _spt_slice_state_by_focks(state::SPTstate{T,N,R}, focks;
                                   zero_cores::Bool=false) where {T,N,R}
    out = _spt_empty_state_like(state)
    for fock in focks
        haskey(state, fock) || continue
        out[fock] = deepcopy(state[fock])
    end
    zero_cores && zero!(out)
    return out
end

function _spt_fock_chunks_by_length(state::SPTstate{T,N,R}, pids;
                                    strategy::Symbol=:balanced) where {T,N,R}
    chunks = Dict(pid => FockConfig{N}[] for pid in pids)
    loads = Dict(pid => 0 for pid in pids)
    for (fock, tconfigs) in state.data
        weight = sum(length(tuck) for (_, tuck) in tconfigs; init=0)
        pid = if strategy == :balanced
            _spt_least_loaded_pid(loads, pids)
        elseif strategy == :hash
            pids[Int(mod(hash(fock), UInt(length(pids)))) + 1]
        else
            error("Unknown SPT multinode sharding strategy: $strategy")
        end
        push!(chunks[pid], fock)
        loads[pid] += max(weight, 1)
    end
    return chunks
end

function _spt_fois_worker_chunk(psi::SPTstate{T,N,R}, jobs_chunk, cluster_ops,
                                nbody, thresh, max_number, prescreen,
                                compress_twice, threaded_worker,
                                blas_threads) where {T,N,R}
    blas_threads === nothing || BLAS.set_num_threads(blas_threads)
    local_ops = _spt_materialize_cluster_ops(
        cluster_ops, _spt_needed_cluster_indices_from_jobs(jobs_chunk))

    nt = threaded_worker ? Threads.maxthreadid() : 1
    partial = [_spt_empty_state_like(psi) for _ in 1:nt]
    scr_v = [[zeros(T, 1000) for _ in 1:N] for _ in 1:nt]

    if threaded_worker
        Threads.@threads :static for idx in eachindex(jobs_chunk)
            tid = Threads.threadid()
            fock_bra, jobs = jobs_chunk[idx]
            _build_compressed_1st_order_state_job(
                fock_bra, jobs, partial[tid], local_ops, nbody, thresh,
                max_number, prescreen, compress_twice, scr_v[tid])
        end
    else
        for (fock_bra, jobs) in jobs_chunk
            _build_compressed_1st_order_state_job(
                fock_bra, jobs, partial[1], local_ops, nbody, thresh,
                max_number, prescreen, compress_twice, scr_v[1])
        end
    end

    out = _spt_empty_state_like(psi)
    for part in partial
        _spt_merge_unique_fock_blocks!(out, part)
    end
    return out
end

"""
    build_compressed_1st_order_state_distributed(psi, cluster_ops, clustered_ham; ...)

Distributed SPT FOIS builder. Fock-sector jobs are split across Julia workers,
with each worker using local threads. The result is gathered as a normal
`SPTstate` for the existing local Tucker compression/CI workflow.
"""
function build_compressed_1st_order_state_distributed(psi::SPTstate{T,N,R},
                                                      cluster_ops,
                                                      clustered_ham;
                                                      thresh=1e-7,
                                                      max_number=nothing,
                                                      nbody=4,
                                                      prescreen=false,
                                                      compress_iter=false,
                                                      compress_twice=true,
                                                      workers=Distributed.workers(),
                                                      threaded_worker=true,
                                                      blas_threads=1,
                                                      strategy::Symbol=:balanced) where {T,N,R}
    pids = ensure_tpsci_multinode_workers!(workers=workers)
    jobs_vec = _spt_make_fock_jobs(psi, cluster_ops, clustered_ham,
                                   require_cluster_space=true)
    chunks = _spt_job_chunks(jobs_vec, pids; strategy=strategy)

    println(" In build_compressed_1st_order_state_distributed")
    @printf(" %-50s%10i\n", "Number of tasks: ", length(jobs_vec))
    @printf(" %-50s%10i\n", "Number of workers: ", length(pids))
    @printf(" %-50s%10s\n", "Threads per worker enabled: ", string(threaded_worker))
    flush(stdout)

    worker_states = Dict{Int,SPTstate{T,N,R}}()
    t = @elapsed begin
        @sync for pid in pids
            @async begin
                worker_states[pid] = Distributed.remotecall_fetch(
                    _spt_fois_worker_chunk, pid, psi, chunks[pid], cluster_ops,
                    nbody, thresh, max_number, prescreen, compress_twice,
                    threaded_worker, blas_threads)
            end
        end
    end
    @printf(" %-50s%10.6f seconds\n", "Compute distributed SPT FOIS: ", t)

    sigma = _spt_empty_state_like(psi)
    for pid in pids
        _spt_merge_unique_fock_blocks!(sigma, worker_states[pid])
    end

    if compress_iter
        @printf(" Compressing final sigma vector:\n")
        sigma = compress_iteratively(sigma, thresh)
    end
    return sigma
end

function _spt_build_sigma_worker_chunk(sigma_chunk::SPTstate{T,N,R},
                                       ket::SPTstate{T,N,R}, cluster_ops,
                                       clustered_ham, nbody, cache, verbose,
                                       zero_output, blas_threads) where {T,N,R}
    blas_threads === nothing || BLAS.set_num_threads(blas_threads)
    local_ops = _spt_materialize_cluster_ops(
        cluster_ops,
        _spt_needed_cluster_indices_for_sigma(sigma_chunk, ket, clustered_ham))
    sig = deepcopy(sigma_chunk)
    zero_output && zero!(sig)
    build_sigma!(sig, ket, local_ops, clustered_ham;
                 nbody=nbody, cache=cache, verbose=verbose)
    return sig
end

"""
    build_sigma_distributed(sigma_basis, ket, cluster_ops, clustered_ham; ...)

Apply a clustered operator into an existing SPT basis by sharding destination
Fock sectors across workers. The returned value is a gathered `SPTstate` with
the same basis as `sigma_basis`.
"""
function build_sigma_distributed(sigma_basis::SPTstate{T,N,R},
                                 ket::SPTstate{T,N,R}, cluster_ops,
                                 clustered_ham;
                                 nbody=4,
                                 cache=false,
                                 verbose=1,
                                 zero_output=true,
                                 workers=Distributed.workers(),
                                 blas_threads=1,
                                 strategy::Symbol=:balanced) where {T,N,R}
    pids = ensure_tpsci_multinode_workers!(workers=workers)
    chunks = _spt_fock_chunks_by_length(sigma_basis, pids; strategy=strategy)
    worker_states = Dict{Int,SPTstate{T,N,R}}()

    @sync for pid in pids
        @async begin
            sigma_chunk = _spt_slice_state_by_focks(sigma_basis, chunks[pid])
            worker_states[pid] = Distributed.remotecall_fetch(
                _spt_build_sigma_worker_chunk, pid, sigma_chunk, ket,
                cluster_ops, clustered_ham, nbody, cache, verbose,
                zero_output, blas_threads)
        end
    end

    out = _spt_empty_state_like(sigma_basis)
    for pid in pids
        _spt_merge_unique_fock_blocks!(out, worker_states[pid])
    end
    return out
end

"""
    compute_expectation_value_distributed(vector, cluster_ops, clustered_op; ...)

Compute `<vector|clustered_op|vector>` using a distributed SPT matvec over the
existing SPT basis.
"""
function compute_expectation_value_distributed(vector::SPTstate{T,N,R},
                                               cluster_ops, clustered_op;
                                               nbody=4,
                                               workers=Distributed.workers(),
                                               blas_threads=1,
                                               strategy::Symbol=:balanced) where {T,N,R}
    tmp = deepcopy(vector)
    zero!(tmp)
    tmp = build_sigma_distributed(tmp, vector, cluster_ops, clustered_op;
                                  nbody=nbody, cache=false, verbose=0,
                                  zero_output=false, workers=workers,
                                  blas_threads=blas_threads,
                                  strategy=strategy)
    return orth_dot(tmp, vector)
end

function _spt_collect_ops_for_local_step(cluster_ops)
    return cluster_ops
end

function _spt_collect_ops_for_local_step(cluster_ops::DistributedClusterOps)
    @warn "Collecting DistributedClusterOps on the master for a local SPT step"
    return collect_cluster_ops(cluster_ops)
end

"""
    compute_pt1_wavefunction_distributed(sigma, psi0, cluster_ops, clustered_ham; ...)

SPT-PT1/PT2 first-order wavefunction builder using distributed restricted SPT
matvecs for `<X|H|0>` and `<X|H0|0>`.
"""
function compute_pt1_wavefunction_distributed(sigma_in::SPTstate{T,N,R},
                                              psi0::SPTstate{T,N,R},
                                              cluster_ops, clustered_ham,
                                              clustered_ham_0, E0, F0;
                                              H0="Hcmf",
                                              nbody=4,
                                              verbose=1,
                                              workers=Distributed.workers(),
                                              threaded_worker=true,
                                              blas_threads=1,
                                              strategy::Symbol=:balanced) where {T,N,R}
    verbose < 2 || @printf(" copy sigma...\n")
    sigma = deepcopy(sigma_in)
    zero!(sigma)

    local_ops = _spt_collect_ops_for_local_step(cluster_ops)
    Fdiag = SPTstate(sigma, R=1)
    form_1body_operator_diagonal!(sigma, Fdiag, local_ops;
                                  H0=H0, pseudo_canon=true)

    Sx = project_into_new_basis(psi0, sigma)

    verbose < 1 || @printf(" %-50s%10i\n", "Length of input      FOIS: ", length(sigma))
    time = @elapsed sigma = build_sigma_distributed(
        sigma, psi0, cluster_ops, clustered_ham;
        nbody=nbody, cache=false, verbose=verbose, zero_output=true,
        workers=workers, blas_threads=blas_threads, strategy=strategy)
    verbose < 1 || @printf(" %-50s%10.6f seconds\n", "Compute <X|H|0>: ", time)

    XF0 = deepcopy(sigma)
    zero!(XF0)
    time = @elapsed XF0 = build_sigma_distributed(
        XF0, psi0, cluster_ops, clustered_ham_0;
        nbody=1, cache=false, verbose=verbose, zero_output=false,
        workers=workers, blas_threads=blas_threads, strategy=strategy)
    verbose < 1 || @printf(" %-50s%10.6f seconds\n", "Compute <X|F|0>: ", time)

    sigma = sigma - XF0 - scale(Sx, E0 .- F0)

    psi1 = deepcopy(sigma)
    Fv = get_vector(Fdiag, 1)
    for r in 1:R
        psi1r = get_vector(psi1, r)
        denom = F0[r] .- Fv .+ 1e-12
        psi1r ./= denom
        set_vector!(psi1, psi1r[:, 1], root=r)
    end

    ecorr = orth_dot(sigma, psi1)
    E2 = zeros(T, R)
    for r in 1:R
        E2[r] = E0[r] + ecorr[r]
    end
    return psi1, E2, ecorr
end

function compute_pt1_wavefunction_distributed(sigma_in::SPTstate{T,N,R},
                                              psi0::SPTstate{T,N,R},
                                              cluster_ops, clustered_ham;
                                              H0="Hcmf",
                                              nbody=4,
                                              verbose=1,
                                              workers=Distributed.workers(),
                                              threaded_worker=true,
                                              blas_threads=1,
                                              strategy::Symbol=:balanced) where {T,N,R}
    verbose < 1 || println()
    verbose < 1 || println(" |...................................SPT-PT2 (multinode)...............................")

    clustered_ham_0 = extract_1body_operator(clustered_ham, op_string=H0)
    E0 = compute_expectation_value_distributed(
        psi0, cluster_ops, clustered_ham;
        nbody=nbody, workers=workers, blas_threads=blas_threads,
        strategy=strategy)
    F0 = compute_expectation_value_distributed(
        psi0, cluster_ops, clustered_ham_0;
        nbody=1, workers=workers, blas_threads=blas_threads,
        strategy=strategy)

    if verbose > 1
        @printf(" %5s %12s %12s\n", "Root", "<0|H|0>", "<0|F|0>")
        for r in 1:R
            @printf(" %5s %12.8f %12.8f\n", r, E0[r], F0[r])
        end
    end

    psi1, E2, ecorr = compute_pt1_wavefunction_distributed(
        sigma_in, psi0, cluster_ops, clustered_ham, clustered_ham_0, E0, F0;
        H0=H0, nbody=nbody, verbose=verbose, workers=workers,
        threaded_worker=threaded_worker, blas_threads=blas_threads,
        strategy=strategy)

    verbose < 1 || for r in 1:R
        @printf(" State %3i: %-35s%14.8f\n", r, "E(PT2) corr: ", ecorr[r])
    end
    verbose < 1 || @printf(" %5s %12s %12s\n", "Root", "E(0)", "E(2)")
    verbose < 1 || for r in 1:R
        @printf(" %5s %12.8f %12.8f\n", r, E0[r], E2[r])
    end
    verbose < 1 || println(" ......................................................................................|")
    return psi1, E2, ecorr
end

function _spt_pt2_worker_chunk(jobs_chunk, kernel::Symbol,
                               ref_vec::SPTstate{T,N,R}, cluster_ops,
                               clustered_ham, clustered_ham_0, nbody,
                               verbose, thresh, max_number, E0, F0,
                               prescreen, compress_twice, H0,
                               threaded_worker, blas_threads) where {T,N,R}
    blas_threads === nothing || BLAS.set_num_threads(blas_threads)
    local_ops = _spt_materialize_cluster_ops(
        cluster_ops, _spt_needed_cluster_indices_from_jobs(jobs_chunk))

    nt = threaded_worker ? Threads.maxthreadid() : 1
    e2_thread = [zeros(T, R) for _ in 1:nt]

    function run_job!(tid, fock_sig, job)
        if kernel == :standard
            e2_thread[tid] .+= _pt2_job(
                fock_sig, job, ref_vec, local_ops, clustered_ham,
                clustered_ham_0, nbody, verbose, thresh, max_number, E0, F0,
                prescreen, compress_twice)
        elseif kernel == :job2
            e2_thread[tid] .+= _pt2_job2(
                fock_sig, job, ref_vec, local_ops, clustered_ham,
                clustered_ham_0, nbody, verbose, thresh, max_number, E0, F0,
                prescreen, compress_twice)
        elseif kernel == :blockwise
            e2_thread[tid] .+= _pt2_job_blockwise(
                fock_sig, job, ref_vec, local_ops, clustered_ham,
                clustered_ham_0, nbody, thresh, max_number, E0, F0,
                prescreen, H0)
        else
            error("Unknown SPT-PT2 multinode kernel: $kernel")
        end
    end

    elapsed = @elapsed begin
        if threaded_worker
            Threads.@threads :static for idx in eachindex(jobs_chunk)
                fock_sig, job = jobs_chunk[idx]
                run_job!(Threads.threadid(), fock_sig, job)
            end
        else
            for (fock_sig, job) in jobs_chunk
                run_job!(1, fock_sig, job)
            end
        end
    end
    return (ecorr=sum(e2_thread), seconds=elapsed, njobs=length(jobs_chunk))
end

"""
    compute_pt2_energy_distributed(ref, cluster_ops, clustered_ham; kernel=:standard, ...)

Distributed SPT-PT2 energy. The PT2 Fock-sector jobs are split across Julia
workers and reduced on the master. Use `kernel=:blockwise` for the lower-memory
blockwise PT2 kernel.
"""
function compute_pt2_energy_distributed(ref::SPTstate{T,N,R}, cluster_ops,
                                        clustered_ham;
                                        H0="Hcmf",
                                        nbody=4,
                                        thresh_foi=1e-6,
                                        max_number=nothing,
                                        opt_ref=true,
                                        ci_tol=1e-6,
                                        verbose=1,
                                        prescreen=false,
                                        compress_twice=false,
                                        kernel::Symbol=:standard,
                                        workers=Distributed.workers(),
                                        threaded_worker=true,
                                        blas_threads=1,
                                        strategy::Symbol=:balanced) where {T,N,R}
    println()
    println(" |...................................SPT-PT2 (multinode)...............................")
    verbose < 1 || println(" H0          : ", H0)
    verbose < 1 || println(" nbody       : ", nbody)
    verbose < 1 || println(" thresh_foi  : ", thresh_foi)
    verbose < 1 || println(" max_number  : ", max_number)
    verbose < 1 || println(" opt_ref     : ", opt_ref)
    verbose < 1 || println(" ci_tol      : ", ci_tol)
    verbose < 1 || println(" kernel      : ", kernel)
    verbose < 1 || println(" verbose     : ", verbose)
    verbose < 1 || @printf("\n")
    verbose < 1 || @printf(" %-50s%10i\n", "Length of Reference: ", length(ref))

    pids = ensure_tpsci_multinode_workers!(workers=workers)
    ref_vec = deepcopy(ref)

    E0 = zeros(T, R)
    if opt_ref
        @printf(" %-50s\n", "Solve zeroth-order problem: ")
        local_ops = _spt_collect_ops_for_local_step(cluster_ops)
        time = @elapsed E0, ref_vec = ci_solve(ref_vec, local_ops,
                                               clustered_ham,
                                               conv_thresh=ci_tol)
        @printf(" %-50s%10.6f seconds\n", "Diagonalization time: ", time)
    else
        @printf(" %-50s", "Compute zeroth-order energy: ")
        flush(stdout)
        @time E0 = compute_expectation_value_distributed(
            ref_vec, cluster_ops, clustered_ham; nbody=nbody, workers=pids,
            blas_threads=blas_threads, strategy=strategy)
    end

    clustered_ham_0 = extract_1body_operator(clustered_ham, op_string=H0)
    @printf(" %-50s", "Compute <0|H0|0>: ")
    @time F0 = compute_expectation_value_distributed(
        ref_vec, cluster_ops, clustered_ham_0; nbody=1, workers=pids,
        blas_threads=blas_threads, strategy=strategy)

    if verbose > 0
        @printf(" %5s %12s %12s\n", "Root", "<0|H|0>", "<0|F|0>")
        for r in 1:R
            @printf(" %5s %12.8f %12.8f\n", r, E0[r], F0[r])
        end
    end

    jobs_vec = _spt_make_fock_jobs(ref_vec, cluster_ops, clustered_ham,
                                   require_cluster_space=false)
    chunks = _spt_job_chunks(jobs_vec, pids; strategy=strategy)

    println(" Number of jobs:    ", length(jobs_vec))
    println(" Number of workers: ", length(pids))
    println(" Threads per worker enabled: ", threaded_worker)
    flush(stdout)

    results = Dict{Int,Any}()
    t_total = @elapsed begin
        @sync for pid in pids
            @async begin
                results[pid] = Distributed.remotecall_fetch(
                    _spt_pt2_worker_chunk, pid, chunks[pid], kernel,
                    ref_vec, cluster_ops, clustered_ham, clustered_ham_0,
                    nbody, verbose, thresh_foi, max_number, E0, F0, prescreen,
                    compress_twice, H0, threaded_worker, blas_threads)
            end
        end
    end

    ecorr = zeros(T, R)
    for pid in pids
        ecorr .+= results[pid].ecorr
    end

    @printf(" %-48s%10.1f s\n", "Wall time spent computing E2: ", t_total)
    if verbose > 0
        for pid in pids
            @printf(" Worker %5i: %8i jobs  %10.1f s\n",
                    pid, results[pid].njobs, results[pid].seconds)
        end
    end

    E2 = zeros(T, R)
    for r in 1:R
        E2[r] = E0[r] + ecorr[r]
        @printf(" State %3i: %-35s%14.8f\n", r, "E(PT2) corr: ", ecorr[r])
    end

    @printf(" %5s %12s %12s\n", "Root", "E(0)", "E(2)")
    for r in 1:R
        @printf(" %5s %12.8f %12.8f\n", r, E0[r], E2[r])
    end
    println(" ......................................................................................|")
    return E2
end

function compute_pt2_energy_blockwise_distributed(ref::SPTstate, cluster_ops,
                                                  clustered_ham; kwargs...)
    return compute_pt2_energy_distributed(ref, cluster_ops, clustered_ham;
                                          kernel=:blockwise, kwargs...)
end

const spt_pt2_multinode = compute_pt2_energy_distributed

"""
    subspace_product_tucker_multinode(input_vec, cluster_ops, clustered_ham; ...)

Multinode SPT driver. FOIS construction, PT1 restricted matvecs, expectation
values, and SPT-PT2 work use distributed workers; the compressed SPT basis is
gathered for the existing local Tucker CI solve.
"""
function subspace_product_tucker_multinode(input_vec::SPTstate{T,N,R},
                                           cluster_ops, clustered_ham;
                                           max_iter=20,
                                           nbody=4,
                                           H0="Hcmf",
                                           thresh_var=1e-4,
                                           thresh_foi=1e-6,
                                           thresh_pt=1e-5,
                                           thresh_spin=nothing,
                                           ci_conv=1e-5,
                                           ci_max_iter=50,
                                           ci_max_ss_vecs=12,
                                           ci_lindep_thresh=1e-10,
                                           resolve_ss=false,
                                           do_pt=true,
                                           tol_tucker=1e-6,
                                           solver="davidson",
                                           verbose=1,
                                           workers=Distributed.workers(),
                                           threaded_worker=true,
                                           blas_threads=1,
                                           strategy::Symbol=:balanced) where {T,N,R}
    pids = ensure_tpsci_multinode_workers!(workers=workers)
    local_ops = _spt_collect_ops_for_local_step(cluster_ops)

    e_last = 0.0
    e0 = 0.0
    e_var = 0.0
    e_pt2 = 0.0
    ref_vec = deepcopy(input_vec)
    var_vec_old = deepcopy(input_vec)
    clustered_S2 = extract_S2(input_vec.clusters)

    e_projected_list = []
    e_variational_list = []
    dim_projected_list = []
    dim_variational_list = []
    converged = false

    to = TimerOutput()
    println(" max_iter         : ", max_iter)
    println(" nbody            : ", nbody)
    println(" H0               : ", H0)
    println(" thresh_var       : ", thresh_var)
    println(" thresh_foi       : ", thresh_foi)
    println(" thresh_pt        : ", thresh_pt)
    println(" thresh_spin      : ", thresh_spin)
    println(" ci_conv          : ", ci_conv)
    println(" ci_max_iter      : ", ci_max_iter)
    println(" ci_max_ss_vecs   : ", ci_max_ss_vecs)
    println(" ci_lindep_thresh : ", ci_lindep_thresh)
    println(" resolve_ss       : ", resolve_ss)
    println(" do_pt            : ", do_pt)
    println(" tol_tucker       : ", tol_tucker)
    println(" workers          : ", pids)
    println(" threaded_worker  : ", threaded_worker)

    for iter in 1:max_iter
        println()
        println()
        println()
        println(" ===================================================================")
        @printf("     SPT Multinode Iteration: %4i epsilon: %12.8f\n", iter, thresh_var)
        println(" ===================================================================")

        dim1 = length(ref_vec)
        ref_vec = compress(ref_vec, thresh=thresh_var)
        orthonormalize!(ref_vec)
        dim2 = length(ref_vec)
        @printf(" %-50s%10i -> %-10i (%-11s = %8.1e)\n",
                "Ref state compressed from: ", dim1, dim2, "thresh_var", thresh_var)

        if resolve_ss
            @printf(" %-50s\n", "Get eigenstate for compressed reference space: ")
            flush(stdout)
            @timeit to "CI small" e0, ref_vec = ci_solve(
                ref_vec, local_ops, clustered_ham;
                conv_thresh=ci_conv, max_iter=ci_max_iter,
                max_ss_vecs=ci_max_ss_vecs, nbody=nbody,
                lindep_thresh=ci_lindep_thresh, solver=solver)
            push!(e_projected_list, e0)
            push!(dim_projected_list, length(ref_vec))
        else
            @printf(" %-50s", "Compute zeroth-order energy: ")
            flush(stdout)
            @time e0 = compute_expectation_value_distributed(
                ref_vec, cluster_ops, clustered_ham; nbody=nbody,
                workers=pids, blas_threads=blas_threads, strategy=strategy)
            push!(e_projected_list, e0)
            push!(dim_projected_list, length(ref_vec))
        end

        iter == 1 && (e_last = e0)

        @printf(" %-50s", "Compute <S^2>: ")
        @time @timeit to "S2" s2 = compute_expectation_value_distributed(
            ref_vec, cluster_ops, clustered_S2; nbody=nbody, workers=pids,
            blas_threads=blas_threads, strategy=strategy)

        @printf(" %5s %12s %12s\n", "Root", "Energy", "S2")
        for r in 1:R
            @printf(" %5s %12.8f %12.8f\n", r, e0[r], abs(s2[r]))
        end
        flush(stdout)

        println()
        time = @elapsed alloc = @allocated @timeit to "FOIS" pt1_vec =
            build_compressed_1st_order_state_distributed(
                ref_vec, cluster_ops, clustered_ham; nbody=nbody,
                thresh=thresh_foi, workers=pids,
                threaded_worker=threaded_worker, blas_threads=blas_threads,
                strategy=strategy)
        verbose < 1 || @printf(" %-50s%10.6f seconds %10.2e Gb\n",
                                "Compute Compressed FOIS: ", time, alloc / 1e9)

        if do_pt
            println()
            @printf(" %-50s%10i\n",
                    "Compute PT1 wavefunction. Reference space dim: ",
                    length(ref_vec))
            time = @elapsed @timeit to "PT1" pt1_vec, e_pt2 =
                compute_pt1_wavefunction_distributed(
                    pt1_vec, ref_vec, cluster_ops, clustered_ham;
                    H0=H0, nbody=nbody, verbose=verbose, workers=pids,
                    threaded_worker=threaded_worker,
                    blas_threads=blas_threads, strategy=strategy)
            @printf(" %-50s%10.6f seconds\n",
                    "Total time spent building PT1: ", time)
        end

        dim1 = length(pt1_vec)
        length(pt1_vec) == 0 && error("zero length FOIS?")
        @timeit to "compress" pt1_vec = compress(pt1_vec, thresh=thresh_foi)
        dim2 = length(pt1_vec)
        @printf(" %-50s%10i -> %-10i (%-11s = %8.1e)\n",
                "FOIS compressed from: ", dim1, dim2, "thresh_foi", thresh_foi)

        var_vec = deepcopy(ref_vec)

        println()
        time = @elapsed begin
            dim1 = length(var_vec)
            @timeit to "add" nonorth_add!(var_vec, pt1_vec)
            @timeit to "compress" var_vec = compress(var_vec, thresh=thresh_pt)
            dim2 = length(var_vec)
            @printf(" %-50s%10i -> %-10i (%-11s = %8.1e)\n",
                    "Variational space increased from: ", dim1, dim2,
                    "thresh_pt", thresh_pt)
            orthonormalize!(var_vec)
        end
        @printf(" %-50s%10.6f seconds\n", "Add new space to variational space: ", time)

        @timeit to "project" var_vec = project_into_new_basis(var_vec_old, var_vec)
        orthonormalize!(var_vec)

        @timeit to "s2 extension" if thresh_spin != nothing
            @printf("\n Perform S^2 Spin Extension.\n")
            dim1 = length(var_vec)
            @printf(" Computing S^2 residual vector:\n")
            r = compute_spin_residual(var_vec, local_ops, thresh=thresh_foi)
            @printf(" Compressing S^2 residual vector:\n")
            r = compress_iteratively(r, thresh_spin)
            if length(r) > 1.0
                var_vec = nonorth_add(var_vec, r)
            end
            dim2 = length(var_vec)
            @printf(" %-50s%10i -> %-10i (%-11s = %8.1e)\n",
                    "Variational space increased from: ", dim1, dim2,
                    "thresh_spin", thresh_spin)
            @timeit to "project" var_vec = project_into_new_basis(var_vec_old, var_vec)
            orthonormalize!(var_vec)
        end

        @timeit to "CI big" e_var, var_vec = ci_solve(
            var_vec, local_ops, clustered_ham;
            conv_thresh=ci_conv, max_iter=ci_max_iter,
            max_ss_vecs=ci_max_ss_vecs, lindep_thresh=ci_lindep_thresh,
            solver=solver)

        var_vec_old = deepcopy(var_vec)
        push!(e_variational_list, e_var)
        push!(dim_variational_list, length(var_vec))

        @printf(" %-20s", "E(Reference): ")
        [@printf("%12.8f ", e0[r]) for r in 1:R]
        println()

        ref_vec = var_vec

        if do_pt
            @printf(" %-20s", "E(PT2): ")
            [@printf("%12.8f ", e_pt2[r]) for r in 1:R]
            println()
        end
        @printf(" %-20s", "E(SPT): ")
        [@printf("%12.8f ", e_var[r]) for r in 1:R]
        println()
        println("")

        if maximum(abs.(e_last - e_var)) < tol_tucker
            converged = true
            break
        end
        e_last = e_var
    end

    if converged
        @printf("*Converged %-20s", "E(Ref): ")
        [@printf("%12.8f ", e0[r]) for r in 1:R]
        println("")
        @printf("*Converged %-20s", "E(SPT): ")
        [@printf("%12.8f ", e_var[r]) for r in 1:R]
    else
        @printf(" Not converged %-20s", "E(Ref): ")
        [@printf("%12.8f ", e0[r]) for r in 1:R]
        println("")
        @printf(" Not converged %-20s", "E(SPT): ")
        [@printf("%12.8f ", e_var[r]) for r in 1:R]
    end
    println()
    println()

    println(" Energies per SPT iteration:")
    println("   Projected Energies: ")
    for (i, ei) in enumerate(e_projected_list)
        @printf("   Iter: %3i  ", i)
        [@printf(" %12.8f", ei[r]) for r in 1:R]
        @printf(" Dim: %9i\n", dim_projected_list[i])
    end
    println()
    println("   Variational Energies: ")
    for (i, ei) in enumerate(e_variational_list)
        @printf("   Iter: %3i  ", i)
        [@printf(" %12.8f", ei[r]) for r in 1:R]
        @printf(" Dim: %9i\n", dim_variational_list[i])
    end
    show(to)
    println()
    @printf(" ==================================================================|\n")
    return e_var, ref_vec
end

const spt_multinode = subspace_product_tucker_multinode
