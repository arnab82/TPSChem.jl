"""
Multinode wrapper for SPT variance-style sigma-norm calculations.

The single-node `compute_spt_sigma_norm_blockwise` routine is left untouched.
This file distributes the same Fock-sector jobs over Julia workers and reduces
the per-root sigma2 vector on the master.
"""

function _spt_sigma_norm_worker_chunk(jobs_chunk,
                                      ref_vec::SPTstate{T,N,R},
                                      cluster_ops,
                                      clustered_ham,
                                      nbody,
                                      verbose,
                                      thresh_foi,
                                      max_number,
                                      prescreen,
                                      threaded_worker,
                                      blas_threads) where {T,N,R}
    blas_threads === nothing || BLAS.set_num_threads(blas_threads)
    local_ops = _spt_materialize_cluster_ops(
        cluster_ops, _spt_needed_cluster_indices_from_jobs(jobs_chunk))
    sigma_thread = [zeros(T, R) for _ in 1:(threaded_worker ? Threads.maxthreadid() : 1)]

    elapsed = @elapsed begin
        if threaded_worker
            Threads.@threads :static for idx in eachindex(jobs_chunk)
                tid = Threads.threadid()
                fock_sig, job = jobs_chunk[idx]
                sigma_thread[tid] .+= _pt2_job_sigma_norm_blockwise(
                    fock_sig, job, ref_vec, local_ops, clustered_ham,
                    nbody, verbose, thresh_foi, max_number, prescreen)
            end
        else
            for (fock_sig, job) in jobs_chunk
                sigma_thread[1] .+= _pt2_job_sigma_norm_blockwise(
                    fock_sig, job, ref_vec, local_ops, clustered_ham,
                    nbody, verbose, thresh_foi, max_number, prescreen)
            end
        end
    end
    return (sigma2=sum(sigma_thread), seconds=elapsed, njobs=length(jobs_chunk))
end

"""
    compute_spt_sigma_norm_blockwise_distributed(ref, cluster_ops, clustered_ham; ...)

Multinode version of `compute_spt_sigma_norm_blockwise`. It computes the FOIS
approximation to `<σ|σ>` by distributing Fock-sector jobs over Julia workers.
Returns the per-root vector `sigma2`.
"""
function compute_spt_sigma_norm_blockwise_distributed(ref::SPTstate{T,N,R},
                                                      cluster_ops,
                                                      clustered_ham;
                                                      H0="Hcmf",
                                                      nbody=4,
                                                      thresh_foi=1e-6,
                                                      max_number=nothing,
                                                      opt_ref=true,
                                                      ci_tol=1e-6,
                                                      verbose=1,
                                                      prescreen=false,
                                                      workers=Distributed.workers(),
                                                      threaded_worker=true,
                                                      blas_threads=1,
                                                      strategy::Symbol=:balanced) where {T,N,R}
    println()
    println(" |.......................SPT-sigma*sigma (blockwise, multinode)................")
    verbose < 1 || println(" H0          : ", H0)
    verbose < 1 || println(" nbody       : ", nbody)
    verbose < 1 || println(" thresh_foi  : ", thresh_foi)
    verbose < 1 || println(" max_number  : ", max_number)
    verbose < 1 || println(" opt_ref     : ", opt_ref)
    verbose < 1 || println(" ci_tol      : ", ci_tol)
    verbose < 1 || println(" workers     : ", workers)
    verbose < 1 || println(" threaded    : ", threaded_worker)
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
            ref_vec, cluster_ops, clustered_ham; nbody=nbody,
            workers=pids, blas_threads=blas_threads, strategy=strategy)
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
                    _spt_sigma_norm_worker_chunk, pid, chunks[pid],
                    ref_vec, cluster_ops, clustered_ham, nbody, verbose,
                    thresh_foi, max_number, prescreen, threaded_worker,
                    blas_threads)
            end
        end
    end

    sigma2 = zeros(T, R)
    for pid in pids
        sigma2 .+= results[pid].sigma2
    end

    @printf(" %-48s%10.1f s\n", "Wall time spent computing sigma*sigma: ", t_total)
    if verbose > 0
        for pid in pids
            @printf(" Worker %5i: %8i jobs  %10.1f s\n",
                    pid, results[pid].njobs, results[pid].seconds)
        end
    end

    for r in 1:R
        @printf(" Root %3i: <σ|σ> = %14.8f\n", r, sigma2[r])
    end
    return sigma2
end

const spt_variance_multinode = compute_spt_sigma_norm_blockwise_distributed
