"""
Multinode wrappers for TPSCI one-electron properties and RDMs.

These routines intentionally reuse the existing local TPSCI property kernels.
Work is split by transition root-pair blocks, and each worker materializes only
the cluster-operator tables it needs from `DistributedClusterOps`.
"""

_tpsci_property_materialize_cluster_ops(cluster_ops) = cluster_ops

function _tpsci_property_materialize_cluster_ops(cluster_ops::DistributedClusterOps)
    needed = [c.idx for c in cluster_ops.clusters]
    return _materialize_cluster_ops_for_indices(cluster_ops, needed)
end

function _tpsci_property_root_pair_chunks(R1::Int, R2::Int, pids;
                                          strategy::Symbol=:balanced)
    isempty(pids) && error("No workers supplied for multinode TPSCI property calculation")
    chunks = Dict(pid => Tuple{Int,Int}[] for pid in pids)
    loads = Dict(pid => 0 for pid in pids)
    for r2 in 1:R2, r1 in 1:R1
        pid = if strategy == :balanced
            _tpsci_least_loaded_pid(loads, pids)
        elseif strategy == :hash
            pids[Int(mod(hash((r1, r2)), UInt(length(pids)))) + 1]
        else
            error("Unknown TPSCI property multinode sharding strategy: $strategy")
        end
        push!(chunks[pid], (r1, r2))
        loads[pid] += 1
    end
    return chunks
end

function _tpsci_1rdm_root_pair_worker(kind::Symbol, root_pairs,
                                      bra::TPSCIstate{T,N,R1},
                                      ket::TPSCIstate{T,N,R2},
                                      cluster_ops,
                                      threaded_worker::Bool,
                                      blas_threads) where {T,N,R1,R2}
    blas_threads === nothing || BLAS.set_num_threads(blas_threads)
    local_ops = _tpsci_property_materialize_cluster_ops(cluster_ops)
    norb = sum(length(c) for c in bra.clusters)
    gamma_1 = zeros(T, norb, norb, R1, R2)
    gamma_2 = zeros(T, norb, norb, R1, R2)

    elapsed = @elapsed begin
        for (r1, r2) in root_pairs
            bra_r = extract_chosen_root(bra, r1)
            ket_r = extract_chosen_root(ket, r2)
            if kind == :number
                g1, g2 = threaded_worker ?
                    compute_1rdm_threaded(bra_r, ket_r, local_ops) :
                    compute_1rdm(bra_r, ket_r, local_ops)
            elseif kind == :spinflip
                g1, g2 = threaded_worker ?
                    compute_1rdm_sf_threaded(bra_r, ket_r, local_ops) :
                    compute_1rdm_sf(bra_r, ket_r, local_ops)
            else
                error("Unknown TPSCI 1-RDM root-pair worker kind: $kind")
            end
            gamma_1[:, :, r1, r2] .= @view g1[:, :, 1, 1]
            gamma_2[:, :, r1, r2] .= @view g2[:, :, 1, 1]
        end
    end
    return (gamma_1=gamma_1, gamma_2=gamma_2,
            npairs=length(root_pairs), seconds=elapsed)
end

function _tpsci_property_root_pair_worker(root_pairs,
                                          bra::TPSCIstate{T,N,R1},
                                          ket::TPSCIstate{T,N,R2},
                                          cluster_ops,
                                          h_prop::AbstractMatrix{T},
                                          blas_threads) where {T,N,R1,R2}
    blas_threads === nothing || BLAS.set_num_threads(blas_threads)
    local_ops = _tpsci_property_materialize_cluster_ops(cluster_ops)
    prop = zeros(T, R1, R2)

    elapsed = @elapsed begin
        for (r1, r2) in root_pairs
            bra_r = extract_chosen_root(bra, r1)
            ket_r = extract_chosen_root(ket, r2)
            p = compute_1e_property_direct(bra_r, ket_r, local_ops, h_prop)
            prop[r1, r2] = p[1, 1]
        end
    end
    return (property=prop, npairs=length(root_pairs), seconds=elapsed)
end

function _tpsci_2rdm_root_pair_worker(root_pairs,
                                      bra::TPSCIstate{T,N,R1},
                                      ket::TPSCIstate{T,N,R2},
                                      cluster_ops,
                                      threaded_worker::Bool,
                                      blas_threads) where {T,N,R1,R2}
    blas_threads === nothing || BLAS.set_num_threads(blas_threads)
    local_ops = _tpsci_property_materialize_cluster_ops(cluster_ops)
    blocks = Tuple{Int,Int,Array{T,4}}[]

    elapsed = @elapsed begin
        for (r1, r2) in root_pairs
            bra_r = extract_chosen_root(bra, r1)
            ket_r = extract_chosen_root(ket, r2)
            Gamma = threaded_worker ?
                compute_2rdm_threaded(bra_r, ket_r, local_ops) :
                compute_2rdm(bra_r, ket_r, local_ops)
            push!(blocks, (r1, r2, copy(@view Gamma[:, :, :, :, 1, 1])))
        end
    end
    return (blocks=blocks, npairs=length(root_pairs), seconds=elapsed)
end

function _tpsci_print_property_worker_report(results, pids, verbose)
    verbose > 0 || return
    for pid in pids
        @printf(" Worker %5i: %6i root pairs  %10.1f s\n",
                pid, results[pid].npairs, results[pid].seconds)
    end
end

"""
    compute_1rdm_distributed(bra, ket, cluster_ops; workers=workers(), ...)

Compute TPSCI transition 1-RDMs with root-pair blocks split across Julia
distributed workers. Returns `(gamma_aa, gamma_bb)`.
"""
function compute_1rdm_distributed(bra::TPSCIstate{T,N,R1},
                                  ket::TPSCIstate{T,N,R2},
                                  cluster_ops;
                                  workers=Distributed.workers(),
                                  threaded_worker=true,
                                  blas_threads=1,
                                  strategy::Symbol=:balanced,
                                  verbose=1) where {T,N,R1,R2}
    pids = ensure_tpsci_multinode_workers!(workers=workers)
    chunks = _tpsci_property_root_pair_chunks(R1, R2, pids; strategy=strategy)
    results = Dict{Int,Any}()

    t_total = @elapsed begin
        @sync for pid in pids
            @async begin
                results[pid] = Distributed.remotecall_fetch(
                    _tpsci_1rdm_root_pair_worker, pid, :number, chunks[pid],
                    bra, ket, cluster_ops, threaded_worker, blas_threads)
            end
        end
    end

    norb = sum(length(c) for c in bra.clusters)
    gamma_aa = zeros(T, norb, norb, R1, R2)
    gamma_bb = zeros(T, norb, norb, R1, R2)
    for pid in pids
        gamma_aa .+= results[pid].gamma_1
        gamma_bb .+= results[pid].gamma_2
    end

    if verbose > 0
        @printf(" TPSCI 1-RDM multinode root-pair wall time: %.1f s\n", t_total)
        _tpsci_print_property_worker_report(results, pids, verbose)
    end
    return gamma_aa, gamma_bb
end

function compute_1rdm_distributed(psi::TPSCIstate{T,N,R}, cluster_ops;
                                  kwargs...) where {T,N,R}
    return compute_1rdm_distributed(psi, psi, cluster_ops; kwargs...)
end

"""
    compute_1rdm_sf_distributed(bra, ket, cluster_ops; workers=workers(), ...)

Compute TPSCI spin-flip transition 1-RDMs with root-pair blocks split across
distributed workers. Returns `(gamma_ab, gamma_ba)`.
"""
function compute_1rdm_sf_distributed(bra::TPSCIstate{T,N,R1},
                                     ket::TPSCIstate{T,N,R2},
                                     cluster_ops;
                                     workers=Distributed.workers(),
                                     threaded_worker=true,
                                     blas_threads=1,
                                     strategy::Symbol=:balanced,
                                     verbose=1) where {T,N,R1,R2}
    pids = ensure_tpsci_multinode_workers!(workers=workers)
    chunks = _tpsci_property_root_pair_chunks(R1, R2, pids; strategy=strategy)
    results = Dict{Int,Any}()

    t_total = @elapsed begin
        @sync for pid in pids
            @async begin
                results[pid] = Distributed.remotecall_fetch(
                    _tpsci_1rdm_root_pair_worker, pid, :spinflip, chunks[pid],
                    bra, ket, cluster_ops, threaded_worker, blas_threads)
            end
        end
    end

    norb = sum(length(c) for c in bra.clusters)
    gamma_ab = zeros(T, norb, norb, R1, R2)
    gamma_ba = zeros(T, norb, norb, R1, R2)
    for pid in pids
        gamma_ab .+= results[pid].gamma_1
        gamma_ba .+= results[pid].gamma_2
    end

    if verbose > 0
        @printf(" TPSCI spin-flip 1-RDM multinode root-pair wall time: %.1f s\n", t_total)
        _tpsci_print_property_worker_report(results, pids, verbose)
    end
    return gamma_ab, gamma_ba
end

function compute_1rdm_sf_distributed(psi::TPSCIstate{T,N,R}, cluster_ops;
                                     kwargs...) where {T,N,R}
    return compute_1rdm_sf_distributed(psi, psi, cluster_ops; kwargs...)
end

"""
    compute_1e_property_direct_distributed(bra, ket, cluster_ops, h_prop; ...)

Compute one-electron TPSCI property matrix `P[r1,r2]` with transition root-pair
blocks split across distributed workers.
"""
function compute_1e_property_direct_distributed(bra::TPSCIstate{T,N,R1},
                                                ket::TPSCIstate{T,N,R2},
                                                cluster_ops,
                                                h_prop::AbstractMatrix{T};
                                                workers=Distributed.workers(),
                                                blas_threads=1,
                                                strategy::Symbol=:balanced,
                                                verbose=1) where {T,N,R1,R2}
    pids = ensure_tpsci_multinode_workers!(workers=workers)
    chunks = _tpsci_property_root_pair_chunks(R1, R2, pids; strategy=strategy)
    results = Dict{Int,Any}()

    t_total = @elapsed begin
        @sync for pid in pids
            @async begin
                results[pid] = Distributed.remotecall_fetch(
                    _tpsci_property_root_pair_worker, pid, chunks[pid],
                    bra, ket, cluster_ops, h_prop, blas_threads)
            end
        end
    end

    prop = zeros(T, R1, R2)
    for pid in pids
        prop .+= results[pid].property
    end

    if verbose > 0
        @printf(" TPSCI one-electron property multinode root-pair wall time: %.1f s\n",
                t_total)
        _tpsci_print_property_worker_report(results, pids, verbose)
    end
    return prop
end

function compute_1e_property_direct_distributed(psi::TPSCIstate{T,N,R},
                                                cluster_ops,
                                                h_prop::AbstractMatrix{T};
                                                kwargs...) where {T,N,R}
    return compute_1e_property_direct_distributed(psi, psi, cluster_ops, h_prop;
                                                  kwargs...)
end

"""
    compute_2rdm_distributed(bra, ket, cluster_ops; workers=workers(), ...)

Compute the spin-free TPSCI 2-RDM with transition root-pair blocks split across
distributed workers. The supplied operator tables must include `Ppqsr`; build
them with `compute_cluster_ops_2rdm`, `add_spinfree_2rdm_ops!`, or
`add_spinfree_2rdm_ops_distributed!`.
"""
function compute_2rdm_distributed(bra::TPSCIstate{T,N,R1},
                                  ket::TPSCIstate{T,N,R2},
                                  cluster_ops;
                                  workers=Distributed.workers(),
                                  threaded_worker=true,
                                  blas_threads=1,
                                  strategy::Symbol=:balanced,
                                  verbose=1) where {T,N,R1,R2}
    pids = ensure_tpsci_multinode_workers!(workers=workers)
    chunks = _tpsci_property_root_pair_chunks(R1, R2, pids; strategy=strategy)
    results = Dict{Int,Any}()

    t_total = @elapsed begin
        @sync for pid in pids
            @async begin
                results[pid] = Distributed.remotecall_fetch(
                    _tpsci_2rdm_root_pair_worker, pid, chunks[pid],
                    bra, ket, cluster_ops, threaded_worker, blas_threads)
            end
        end
    end

    norb = sum(length(c) for c in bra.clusters)
    Gamma = zeros(T, norb, norb, norb, norb, R1, R2)
    for pid in pids
        for (r1, r2, block) in results[pid].blocks
            Gamma[:, :, :, :, r1, r2] .= block
        end
    end

    if verbose > 0
        @printf(" TPSCI 2-RDM multinode root-pair wall time: %.1f s\n", t_total)
        _tpsci_print_property_worker_report(results, pids, verbose)
    end
    return Gamma
end

function compute_2rdm_distributed(psi::TPSCIstate{T,N,R}, cluster_ops;
                                  kwargs...) where {T,N,R}
    return compute_2rdm_distributed(psi, psi, cluster_ops; kwargs...)
end

function _add_spinfree_2rdm_ops_chunk!(id::Symbol, assigned_bases;
                                       verbose=0, blas_threads=1)
    blas_threads === nothing || BLAS.set_num_threads(blas_threads)
    local_ops = _get_cluster_ops_chunk(id)
    nbytes = OrderedDict{Int,Int}()
    for cb in assigned_bases
        ci = cb.cluster
        haskey(local_ops, ci.idx) ||
            error("Worker $(Distributed.myid()) does not own ClusterOps for cluster $(ci.idx)")
        verbose < 1 || (display(ci); flush(stdout))

        Ppqsr = tdm_spinfree_2rdm(cb)
        for ftrans in keys(Ppqsr)
            data = Ppqsr[ftrans]
            dim1 = prod(size(data)[1:(ndims(data)-2)])
            dim2 = size(data, ndims(data)-1)
            dim3 = size(data, ndims(data))
            Ppqsr[ftrans] = copy(reshape(data, (dim1, dim2, dim3)))
        end
        local_ops[ci.idx]["Ppqsr"] = Ppqsr
        nbytes[ci.idx] = _cluster_ops_nbytes(local_ops[ci.idx])
    end
    return nbytes
end

"""
    add_spinfree_2rdm_ops_distributed!(ops, cluster_bases; ...)

Add exact local spin-free 2-RDM transition tables (`Ppqsr`) to a
`DistributedClusterOps` object in place.
"""
function add_spinfree_2rdm_ops_distributed!(ops::DistributedClusterOps,
                                            cluster_bases;
                                            verbose=0,
                                            blas_threads=1)
    chunks = Dict(pid => Vector{eltype(cluster_bases)}() for pid in ops.workers)
    for cb in cluster_bases
        pid = ops.owners[cb.cluster.idx]
        push!(chunks[pid], cb)
    end

    nbyte_maps = Dict{Int,Any}()
    @sync for pid in ops.workers
        @async begin
            nbyte_maps[pid] = Distributed.remotecall_fetch(
                _add_spinfree_2rdm_ops_chunk!, pid, ops.id, chunks[pid];
                verbose=verbose, blas_threads=blas_threads)
        end
    end

    for pid in ops.workers
        ops.local_nbytes[pid] = sum(values(nbyte_maps[pid]); init=0)
    end
    return ops
end

"""
    compute_cluster_ops_2rdm_distributed(cluster_bases, ints; ...)

Build distributed cluster operator tables and attach the `Ppqsr` tables needed
by `compute_2rdm_distributed`.
"""
function compute_cluster_ops_2rdm_distributed(cluster_bases, ints::InCoreInts{T};
                                              workers=Distributed.workers(),
                                              id=nothing,
                                              strategy::Symbol=:balanced,
                                              verbose=1,
                                              blas_threads=1) where {T}
    ops = compute_cluster_ops_distributed(cluster_bases, ints;
                                          workers=workers, id=id,
                                          strategy=strategy, verbose=verbose,
                                          blas_threads=blas_threads)
    add_spinfree_2rdm_ops_distributed!(ops, cluster_bases;
                                       verbose=verbose,
                                       blas_threads=blas_threads)
    return ops
end
