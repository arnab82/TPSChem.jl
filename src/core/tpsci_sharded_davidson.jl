"""
Across-node variational diagonalization for TPSCI.

This file adds a Davidson eigensolver whose subspace vectors are themselves
sharded `DistributedTPSCIstate`s and are never gathered onto the master. It is
the missing piece for diagonalizing the variational Hamiltonian when the CI
vector is larger than a single node's memory.

Two Hamiltonian-application backends are provided (see `examples/multinode/SHARDED_DAVIDSON_PLAN.md`):

  * Tier A (`:blocks`): the Hamiltonian is built once as distributed, block-sparse
    dense Fock-pair blocks and reused via block GEMV on every matvec. This
    recovers the per-iteration speed of `tps_ci_direct` without ever forming a
    `dim × dim` dense matrix or gathering a vector.
  * Tier B (`:matrixfree`): each matvec re-contracts the cluster operators via the
    existing `tps_ci_matvec_sharded`. Minimal memory, slower per iteration.

`tps_ci_davidson_sharded` chooses between them (`h_storage=:auto`) from an
estimate of `nnz(H)` against a memory budget, or you can force either tier.

All subspace linear algebra reuses the sharded primitives already defined in
`tpsci_multinode.jl` (`overlap`, `add_scaled!`, `norm`, `scale!`,
`copy_sharded_state`, `similar_sharded_state`, `tps_ci_matvec_sharded`, ...).
"""

# ---------------------------------------------------------------------------
# Phase 0: cheap nnz(H) / memory estimator (metadata only, runs on master)
# ---------------------------------------------------------------------------

"""
    estimate_sharded_H_nnz_per_worker(ci_vector::DistributedTPSCIstate, clustered_ham)

Per-worker breakdown of the block-sparse Hamiltonian element count, keyed by pid.

`build_block_h_sharded` gives each worker the *row blocks* for the Fock-bra
sectors it owns, against every connected ket sector (see
`_build_block_h_chunk_typed!`). A worker's share is therefore

    Σ_{bra owned by pid} len(bra) * Σ_{ket : (bra-ket) ∈ clustered_ham} len(ket)

which is *not* proportional to the number of configurations it owns: connectivity
varies per sector, so a length-balanced shard layout can still be badly
memory-imbalanced. This is the quantity that decides whether a node OOMs, so it
is what `:auto` must test — the aggregate over all workers never sees the peak.
"""
function estimate_sharded_H_nnz_per_worker(ci_vector::DistributedTPSCIstate{T,N,R},
                                           clustered_ham::ClusteredOperator) where {T,N,R}
    focks = collect(keys(ci_vector.owners))
    lens = ci_vector.lengths
    per_worker = OrderedDict{Int,Float64}(pid => 0.0 for pid in ci_vector.workers)
    for fock_bra in focks
        lb = Float64(lens[fock_bra])
        row = 0.0
        for fock_ket in focks
            fock_trans = fock_bra - fock_ket
            haskey(clustered_ham, fock_trans) || continue
            row += Float64(lens[fock_ket])
        end
        owner = ci_vector.owners[fock_bra]
        per_worker[owner] = get(per_worker, owner, 0.0) + lb * row
    end
    return per_worker
end

"""
    estimate_sharded_H_nnz(ci_vector::DistributedTPSCIstate, clustered_ham)

Estimate the number of nonzero elements in the block-sparse variational
Hamiltonian for the space defined by `ci_vector`, counting every connected
`(fock_bra, fock_ket)` pair as a fully dense `n_bra × n_ket` block. Uses only the
per-Fock-sector lengths already tracked in the metadata, so it is O(nfocks^2) and
touches no coefficient data.
"""
function estimate_sharded_H_nnz(ci_vector::DistributedTPSCIstate{T,N,R},
                                clustered_ham::ClusteredOperator) where {T,N,R}
    return sum(values(estimate_sharded_H_nnz_per_worker(ci_vector, clustered_ham));
               init=0.0)
end

"""
    sharded_H_memory_report(ci_vector, clustered_ham)

Return a NamedTuple summarizing the block-sparse Hamiltonian size so a run can
decide between Tier A (stored H) and Tier B (matrix-free). `nnz` is the number of
stored matrix elements (summed over all workers), `bytes` its dense-block memory.

`per_worker_gb` gives the same figure split by owning pid, and `max_worker_gb` /
`max_worker_pid` identify the worker that has to hold the most. The aggregate
`gb` says whether the problem fits the *cluster*; only `max_worker_gb` says
whether it fits a *node*.
"""
function sharded_H_memory_report(ci_vector::DistributedTPSCIstate{T,N,R},
                                 clustered_ham::ClusteredOperator) where {T,N,R}
    per_worker_nnz = estimate_sharded_H_nnz_per_worker(ci_vector, clustered_ham)
    nnz = sum(values(per_worker_nnz); init=0.0)
    bytes = nnz * sizeof(T)
    per_worker_gb = OrderedDict{Int,Float64}(pid => n * sizeof(T) * 1e-9
                                             for (pid, n) in per_worker_nnz)
    max_worker_pid = isempty(per_worker_gb) ? 0 :
                     argmax(pid -> per_worker_gb[pid], keys(per_worker_gb))
    max_worker_gb = isempty(per_worker_gb) ? 0.0 : per_worker_gb[max_worker_pid]
    return (dim=length(ci_vector), nfocks=length(ci_vector.owners),
            nnz=nnz, bytes=bytes, gb=bytes * 1e-9,
            per_worker_gb=per_worker_gb,
            max_worker_gb=max_worker_gb, max_worker_pid=max_worker_pid)
end

# Resident set size of the calling process, in bytes. On Linux read it from
# /proc/self/statm (the truth, and what the OOM killer scores). Elsewhere fall
# back to the live Julia heap, which understates RSS but is the best portable
# proxy. Deliberately NOT Sys.free_memory(): that reads MemFree and excludes
# reclaimable page cache, so on a node that has just streamed a large JLD2 file
# it reports almost nothing free and would veto runs that fit comfortably.
function _process_rss_bytes()
    if Sys.islinux()
        try
            fields = split(read("/proc/self/statm", String))
            length(fields) >= 2 && return parse(Int, fields[2]) * Int(Sys.PAGESIZE)
        catch
            # fall through to the portable estimate
        end
    end
    return Int(Base.gc_live_bytes())
end

"""
    probe_worker_memory(workers=Distributed.workers())

Ask every worker how much memory its machine actually has, how much this job is
already holding there, and how many processes share that machine.

Returns an `OrderedDict` pid => NamedTuple with `host`, `total_gb`, `free_gb`,
`rss_gb` (this process), `host_rss_gb` (all of the job's processes on that host,
master included), `nprocs_on_host`, and `master_on_host`.

`Sys.total_memory()` is cgroup-aware on Linux, so under SLURM it reflects the
job's memory allocation rather than the bare hardware. `free_gb` is reported for
information only; the budget in `sharded_H_fit_report` is built from
`total_gb - host_rss_gb`, which is insensitive to page cache.
"""
function probe_worker_memory(workers=Distributed.workers())
    pids = collect(workers)
    probe = () -> (gethostname(), Sys.total_memory(), Sys.free_memory(),
                   _process_rss_bytes())
    raw = Dict{Int,Tuple{String,UInt64,UInt64,Int}}()
    @sync for pid in pids
        @async raw[pid] = Distributed.remotecall_fetch(probe, pid)
    end
    # The master is usually co-located with a worker (addprocs over a SLURM node
    # list includes the master's own host unless TPSCHEM_SKIP_MASTER_NODE=1), and
    # it holds the cluster bases and integrals. Charge its RSS to its host.
    master = probe()
    master_host = master[1]

    host_counts = Dict{String,Int}()
    host_rss = Dict{String,Int}()
    for pid in pids
        host = raw[pid][1]
        host_counts[host] = get(host_counts, host, 0) + 1
        host_rss[host] = get(host_rss, host, 0) + raw[pid][4]
    end
    host_rss[master_host] = get(host_rss, master_host, 0) + master[4]

    out = OrderedDict{Int,Any}()
    for pid in pids
        host, total, free, rss = raw[pid]
        out[pid] = (host=host,
                    total_gb=Float64(total) * 1e-9,
                    free_gb=Float64(free) * 1e-9,
                    rss_gb=Float64(rss) * 1e-9,
                    host_rss_gb=Float64(host_rss[host]) * 1e-9,
                    nprocs_on_host=host_counts[host],
                    master_on_host=(host == master_host))
    end
    return out
end

"""
    sharded_H_fit_report(rep, workers; headroom=0.85)

Combine a `sharded_H_memory_report` with a live `probe_worker_memory` and decide,
per worker, whether the stored block-sparse H actually fits in RAM.

A host's spare capacity is `total_gb * headroom - host_rss_gb`: its memory
allocation, less what this job already holds there (all co-located workers plus
the master), with `headroom` left for the solver's Krylov vectors and transient
build allocations. That spare capacity is split evenly among the workers on the
host, since `build_block_h_sharded` fills them concurrently.

Returns `(fits, worst_pid, worst_deficit_gb, rows)`; `rows` carries the
per-worker numbers for printing.
"""
function sharded_H_fit_report(rep, workers; headroom=0.85)
    mem = probe_worker_memory(workers)
    rows = NamedTuple[]
    fits = true
    worst_pid = 0
    worst_deficit = -Inf
    for (pid, info) in mem
        need = get(rep.per_worker_gb, pid, 0.0)
        spare = info.total_gb * headroom - info.host_rss_gb
        budget = spare / max(info.nprocs_on_host, 1)
        deficit = need - budget
        ok = deficit <= 0
        fits &= ok
        if deficit > worst_deficit
            worst_deficit = deficit
            worst_pid = pid
        end
        push!(rows, (pid=pid, host=info.host, need_gb=need,
                     total_gb=info.total_gb, free_gb=info.free_gb,
                     rss_gb=info.rss_gb, host_rss_gb=info.host_rss_gb,
                     nprocs_on_host=info.nprocs_on_host,
                     master_on_host=info.master_on_host,
                     budget_gb=budget, ok=ok))
    end
    return (fits=fits, worst_pid=worst_pid, worst_deficit_gb=worst_deficit, rows=rows)
end

"""
    print_sharded_H_fit(rep, fit; label="H")

Print the per-worker feasibility table for a stored block-sparse Hamiltonian.
"""
function print_sharded_H_fit(rep, fit; label="H")
    @printf(" Sharded %s memory feasibility  (dim=%i, nfocks=%i, aggregate=%.3f GB over %i workers)\n",
            label, rep.dim, rep.nfocks, rep.gb, length(fit.rows))
    @printf("   %-5s %-20s %11s %10s %10s %11s %9s %5s\n",
            "pid", "host", "needs(GB)", "node(GB)", "used(GB)", "budget(GB)",
            "procs", "fits")
    for r in fit.rows
        @printf("   %-5i %-20s %11.2f %10.1f %10.1f %11.2f %9s %5s\n",
                r.pid, first(r.host, 20), r.need_gb, r.total_gb, r.host_rss_gb,
                r.budget_gb,
                string(r.nprocs_on_host, r.master_on_host ? "+mstr" : ""),
                r.ok ? "yes" : "NO")
    end
    if !fit.fits
        @printf("   >> worker %i is short by %.2f GB; stored %s would be OOM-killed\n",
                fit.worst_pid, fit.worst_deficit_gb, label)
    end
    flush(stdout)
    return nothing
end

# Shared `:auto` decision. `max_mem_H` stays an *aggregate* GB budget so existing
# callers keep their meaning; the live per-worker probe is an additional gate that
# can only make the choice more conservative. Returns the chosen tier.
function _sharded_H_auto_tier(ci_vector, clustered_ham, workers, max_mem_H;
                              headroom=0.85, verbose=1, label="H")
    rep = sharded_H_memory_report(ci_vector, clustered_ham)
    fit = sharded_H_fit_report(rep, workers; headroom=headroom)
    verbose > 0 && print_sharded_H_fit(rep, fit; label=label)
    tier = (rep.gb <= max_mem_H && fit.fits) ? :blocks : :matrixfree
    if verbose > 0
        reason = rep.gb > max_mem_H ? "aggregate over max_mem_H" :
                 (fit.fits ? "fits" : "a worker would exceed node RAM")
        @printf(" %s storage :auto -> :%s  (aggregate %.3f GB vs max_mem_H %.1f GB; per-worker peak %.3f GB on pid %i; %s)\n",
                label, tier, rep.gb, max_mem_H, rep.max_worker_gb,
                rep.max_worker_pid, reason)
        flush(stdout)
    end
    return tier, rep, fit
end

# ---------------------------------------------------------------------------
# Phase 1 primitives: sharded root extraction / recombination, diagonal,
# preconditioner. All keep the input Fock layout and ownership so the resulting
# vectors are compatible with the existing sharded `overlap`/`add_scaled!`/etc.
# ---------------------------------------------------------------------------

function _tpsci_sharded_local_extract_root!(dest_id::Symbol, src_id::Symbol, root::Int)
    return _tpsci_sharded_local_extract_root_typed!(dest_id,
                                                    _tpsci_sharded_get_state(src_id),
                                                    root)
end

function _tpsci_sharded_local_extract_root_typed!(dest_id::Symbol,
                                                  src::TPSCIstate{T,N,R},
                                                  root::Int) where {T,N,R}
    out = TPSCIstate(src.clusters, T=T, R=1)
    for (fock, configs) in src.data
        add_fockconfig!(out, fock)
        for (config, coeffs) in configs
            out[fock][config] = MVector{1,T}(coeffs[root])
        end
    end
    _tpsci_sharded_store_state!(dest_id, out)
    return _tpsci_sharded_local_fock_lengths(dest_id)
end

"""
    extract_root_sharded(state::DistributedTPSCIstate, root) -> DistributedTPSCIstate{T,N,1}

Extract a single root from a multi-root sharded state, preserving Fock ownership
so the result is compatible with the source in later sharded operations.
"""
function extract_root_sharded(state::DistributedTPSCIstate{T,N,R}, root::Integer;
                              id=nothing) where {T,N,R}
    1 <= root <= R || throw(BoundsError(state, root))
    output_id = id === nothing ? gensym(:tpsci_shard_root) : Symbol(id)
    length_maps = Dict{Int,Any}()
    @sync for pid in state.workers
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(
                _tpsci_sharded_local_extract_root!, pid, output_id, state.id, Int(root))
        end
    end
    return _tpsci_sharded_metadata_from_lengths(output_id, state.clusters,
                                                state.workers, length_maps,
                                                collect(keys(state.owners)),
                                                T, Val(1))
end

function _tpsci_sharded_local_combine_roots!(dest_id::Symbol, src_ids::Vector{Symbol})
    srcs = [_tpsci_sharded_get_state(id) for id in src_ids]
    return _tpsci_sharded_local_combine_roots_typed!(dest_id, srcs)
end

function _tpsci_sharded_local_combine_roots_typed!(dest_id::Symbol,
                                                   srcs::Vector{<:TPSCIstate{T,N,1}}) where {T,N}
    R = length(srcs)
    out = TPSCIstate(srcs[1].clusters, T=T, R=R)
    for (fock, configs) in srcs[1].data
        add_fockconfig!(out, fock)
        for (config, _) in configs
            mv = MVector{R,T}(undef)
            for r in 1:R
                mv[r] = srcs[r].data[fock][config][1]
            end
            out[fock][config] = mv
        end
    end
    _tpsci_sharded_store_state!(dest_id, out)
    return _tpsci_sharded_local_fock_lengths(dest_id)
end

"""
    combine_roots_sharded(vecs::Vector{<:DistributedTPSCIstate}) -> DistributedTPSCIstate{T,N,R}

Stack `R` single-root sharded states (sharing Fock ownership) into one
`R`-root sharded state.
"""
function combine_roots_sharded(vecs::Vector{<:DistributedTPSCIstate{T,N,1}};
                               id=nothing) where {T,N}
    R = length(vecs)
    R >= 1 || error("combine_roots_sharded needs at least one vector")
    workers = vecs[1].workers
    for v in vecs
        v.workers == workers || error("combine_roots_sharded requires identical worker lists")
    end
    src_ids = [v.id for v in vecs]
    output_id = id === nothing ? gensym(:tpsci_shard_combine) : Symbol(id)
    length_maps = Dict{Int,Any}()
    @sync for pid in workers
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(
                _tpsci_sharded_local_combine_roots!, pid, output_id, src_ids)
        end
    end
    return _tpsci_sharded_metadata_from_lengths(output_id, vecs[1].clusters,
                                                workers, length_maps,
                                                collect(keys(vecs[1].owners)),
                                                T, Val(R))
end

function _tpsci_sharded_local_compute_diagonal!(dest_id::Symbol, src_id::Symbol)
    state = _tpsci_sharded_get_state(src_id)
    cluster_ops = _TPSCI_MULTINODE_CLUSTER_OPS[]
    clustered_ham = _TPSCI_MULTINODE_CLUSTERED_HAM[]
    cluster_ops === nothing && error("TPSCI sharded cluster-ops cache is empty")
    clustered_ham === nothing && error("TPSCI sharded Hamiltonian cache is empty")
    if cluster_ops isa DistributedClusterOps
        cluster_ops = _materialize_cluster_ops_for_indices(
            cluster_ops, [c.idx for c in state.clusters])
    end
    return _tpsci_sharded_local_compute_diagonal_typed!(dest_id, state, cluster_ops,
                                                        clustered_ham)
end

function _tpsci_sharded_local_compute_diagonal_typed!(dest_id::Symbol,
                                                      state::TPSCIstate{T,N,R},
                                                      cluster_ops,
                                                      clustered_ham::ClusteredOperator) where {T,N,R}
    out = TPSCIstate(state.clusters, T=T, R=1)
    zero_trans = TransferConfig([(0, 0) for _ in 1:N])
    has_diag = haskey(clustered_ham, zero_trans)
    for (fock, configs) in state.data
        add_fockconfig!(out, fock)
        for (config, _) in configs
            hd = zero(T)
            if has_diag
                for term in clustered_ham[zero_trans]
                    check_term(term, fock, config, fock, config) || continue
                    hd += contract_matrix_element(term, cluster_ops, fock, config,
                                                  fock, config)
                end
            end
            out[fock][config] = MVector{1,T}(hd)
        end
    end
    _tpsci_sharded_store_state!(dest_id, out)
    return _tpsci_sharded_local_fock_lengths(dest_id)
end

"""
    compute_diagonal_sharded(ci_vector, cluster_ops, clustered_ham) -> DistributedTPSCIstate{T,N,1}

Sharded diagonal of the full clustered Hamiltonian in the `ci_vector` basis, used
as the Davidson preconditioner. Each worker computes the diagonal for its owned
configurations only; nothing is gathered.
"""
function compute_diagonal_sharded(ci_vector::DistributedTPSCIstate{T,N,R},
                                  cluster_ops, clustered_ham::ClusteredOperator;
                                  workers=ci_vector.workers,
                                  blas_threads=1, id=nothing) where {T,N,R}
    _tpsci_sharded_cache_operator_problem!(cluster_ops, clustered_ham;
                                           workers=workers, blas_threads=blas_threads)
    output_id = id === nothing ? gensym(:tpsci_shard_diag) : Symbol(id)
    length_maps = Dict{Int,Any}()
    @sync for pid in ci_vector.workers
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(
                _tpsci_sharded_local_compute_diagonal!, pid, output_id, ci_vector.id)
        end
    end
    return _tpsci_sharded_metadata_from_lengths(output_id, ci_vector.clusters,
                                                ci_vector.workers, length_maps,
                                                collect(keys(ci_vector.owners)),
                                                T, Val(1))
end

function _tpsci_sharded_local_precondition!(dest_id::Symbol, res_id::Symbol,
                                            diag_id::Symbol, theta)
    return _tpsci_sharded_local_precondition_typed!(dest_id,
                                                    _tpsci_sharded_get_state(res_id),
                                                    _tpsci_sharded_get_state(diag_id),
                                                    theta)
end

function _tpsci_sharded_local_precondition_typed!(dest_id::Symbol,
                                                  res::TPSCIstate{T,N,1},
                                                  diag::TPSCIstate{T,N,1},
                                                  theta) where {T,N}
    thetaT = T(theta)
    out = TPSCIstate(res.clusters, T=T, R=1)
    for (fock, configs) in res.data
        add_fockconfig!(out, fock)
        for (config, coeffs) in configs
            d0 = thetaT - diag.data[fock][config][1]
            scale = max(abs(thetaT), abs(diag.data[fock][config][1]), one(T))
            floor = sqrt(eps(T)) * scale
            d = abs(d0) < floor ? copysign(floor, iszero(d0) ? one(T) : d0) : d0
            out[fock][config] = MVector{1,T}(coeffs[1] / d)
        end
    end
    _tpsci_sharded_store_state!(dest_id, out)
    return _tpsci_sharded_local_fock_lengths(dest_id)
end

"""
    precondition_sharded(res, Hdiag, theta) -> DistributedTPSCIstate{T,N,1}

Diagonal (Davidson) preconditioner: `t = res / (theta - Hdiag)`, applied
worker-locally. `res` and `Hdiag` must share Fock ownership.
"""
function precondition_sharded(res::DistributedTPSCIstate{T,N,1},
                              Hdiag::DistributedTPSCIstate{T,N,1}, theta;
                              id=nothing) where {T,N}
    res.workers == Hdiag.workers || error("precondition_sharded requires identical worker lists")
    output_id = id === nothing ? gensym(:tpsci_shard_precond) : Symbol(id)
    length_maps = Dict{Int,Any}()
    @sync for pid in res.workers
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(
                _tpsci_sharded_local_precondition!, pid, output_id, res.id, Hdiag.id, theta)
        end
    end
    return _tpsci_sharded_metadata_from_lengths(output_id, res.clusters,
                                                res.workers, length_maps,
                                                collect(keys(res.owners)),
                                                T, Val(1))
end

# ---------------------------------------------------------------------------
# Tier B operator: matrix-free H*v via the existing sharded matvec.
# ---------------------------------------------------------------------------

struct MatrixFreeShardedH{H}
    cluster_ops
    clustered_ham::H
    workers::Vector{Int}
    threaded_worker::Bool
    blas_threads::Int
end

function apply_sharded_H(op::MatrixFreeShardedH, v::DistributedTPSCIstate{T,N,R};
                         eshift=zero(T), id=nothing) where {T,N,R}
    # verbose=0: this is called once per Davidson direction / MINRES iteration;
    # per-call banners would swamp the solver log.
    out = tps_ci_matvec_sharded(v, op.cluster_ops, op.clustered_ham;
                                workers=op.workers,
                                threaded_worker=op.threaded_worker,
                                blas_threads=op.blas_threads,
                                id=id,
                                verbose=0)
    # The re-contracted matvec builds its own output state, so the shift cannot be
    # folded into it the way the stored-block path does; it costs one extra
    # fan-out, still far cheaper than the matvec itself.
    iszero(eshift) || add_scaled!(out, -eshift, v)
    return out
end

# ---------------------------------------------------------------------------
# Tier A operator: build the block-sparse H once, reuse via block GEMV.
# ---------------------------------------------------------------------------

const _TPSCI_SHARDED_BLOCK_H = Dict{Symbol,Any}()

struct ShardedBlockData{T,N}
    owned_bras::Vector{FockConfig{N}}
    bra_order::Dict{FockConfig{N},Vector{ClusterConfig{N}}}
    ket_order::Dict{FockConfig{N},Vector{ClusterConfig{N}}}
    connections::Dict{FockConfig{N},Vector{FockConfig{N}}}
    mats::Dict{Tuple{FockConfig{N},FockConfig{N}},Matrix{T}}
    # Ket sectors this chunk's row blocks read. Fixed for the life of the
    # operator (it follows the stored sparsity, not the vector), so the matvec
    # can fetch them in one batch per owner instead of rediscovering them a
    # blocking round trip at a time.
    needed_kets::Vector{FockConfig{N}}
end

# ---------------------------------------------------------------------------
# Term bucketing: which terms can connect a given pair of configurations.
#
# `check_term` is O(n_clusters) and the naive build calls it once per
# (bra, ket, term); with hundreds of terms per Fock transition almost every call
# fails, and the failures dominate the build. Two facts remove that work:
#
#   * whether a term's *Fock* pattern fits a (fock_bra, fock_ket) pair does not
#     depend on the configurations, so it is decided once per block;
#   * a term contributes only if the bra and ket configs agree on every cluster it
#     does not act on, i.e. only if the set of clusters where the two configs
#     differ is a subset of the term's active clusters.
#
# So the surviving terms are bucketed by difference pattern (each term is filed
# under every subset of its active-cluster mask, at most 2^4 = 16 of them) and the
# pair loop computes one difference mask -- with an early exit as soon as more
# clusters differ than any term can bridge -- followed by a single lookup.
# ---------------------------------------------------------------------------

@inline function _term_active_mask(term::ClusteredTerm)
    mask = UInt64(0)
    for c in term.clusters
        mask |= UInt64(1) << (c.idx - 1)
    end
    return mask
end

# Configuration-independent half of `check_term`. It depends on the Fock sectors
# only through their difference, which is what makes the buckets below reusable
# across every (bra, ket) sector pair with the same transfer.
function _term_fock_ok(term::ClusteredTerm, fock_trans::TransferConfig{N}) where {N}
    active = _term_active_mask(term)
    for ci in 1:N
        (active >> (ci - 1)) & UInt64(1) == UInt64(1) && continue
        fock_trans[ci] == (0, 0) || return false
    end
    for (i, c) in enumerate(term.clusters)
        fock_trans[c.idx] == term.delta[i] || return false
    end
    return true
end

const ShardedTermBuckets = Dict{UInt64,Vector{ClusteredTerm}}

"""
Bucket the terms of one Fock transfer by the cluster-difference patterns they can
bridge. Returns `(buckets, maxdiff)` where `buckets` maps a difference bitmask to
the terms that can contribute to a configuration pair with exactly that mask, and
`maxdiff` is the largest number of differing clusters any surviving term spans
(0 if no term's Fock pattern fits this transfer at all).
"""
function _bucket_terms_by_diff(terms, fock_trans::TransferConfig{N}) where {N}
    buckets = ShardedTermBuckets()
    maxdiff = -1
    for term in terms
        _term_fock_ok(term, fock_trans) || continue
        mask = _term_active_mask(term)
        maxdiff = max(maxdiff, count_ones(mask))
        sub = mask
        while true
            push!(get!(() -> ClusteredTerm[], buckets, sub), term)
            sub == UInt64(0) && break
            sub = (sub - UInt64(1)) & mask
        end
    end
    return buckets, max(maxdiff, 0)
end

# Bitmask of the clusters where `a` and `b` differ. Bails out (returning a
# popcount above `maxdiff`) as soon as the pair is too far apart for any term.
@inline function _config_diff_mask(a::ClusterConfig{N}, b::ClusterConfig{N},
                                   maxdiff::Int) where {N}
    mask = UInt64(0)
    pop = 0
    ac = a.config
    bc = b.config
    @inbounds for i in 1:N
        if ac[i] != bc[i]
            pop += 1
            pop <= maxdiff || return (mask, pop)
            mask |= UInt64(1) << (i - 1)
        end
    end
    return (mask, pop)
end

# Fill one row of a (fock_bra, fock_ket) block. `Hblk[bi, :]` is owned by the
# caller's thread, so the row loop over this is race free.
function _fill_block_h_row!(Hblk::Matrix{T}, bi::Int, config_bra::ClusterConfig{N},
                            ket_cfgs::Vector{ClusterConfig{N}}, buckets, maxdiff::Int,
                            cluster_ops, fock_bra::FockConfig{N},
                            fock_ket::FockConfig{N}) where {T,N}
    @inbounds for ki in eachindex(ket_cfgs)
        config_ket = ket_cfgs[ki]
        mask, pop = _config_diff_mask(config_bra, config_ket, maxdiff)
        pop <= maxdiff || continue
        cand = get(buckets, mask, nothing)
        cand === nothing && continue
        acc = zero(T)
        for term in cand
            acc += contract_matrix_element(term, cluster_ops, fock_bra, config_bra,
                                           fock_ket, config_ket)
        end
        Hblk[bi, ki] = acc
    end
    return nothing
end

function _fill_block_h!(Hblk::Matrix{T}, bra_cfgs::Vector{ClusterConfig{N}},
                        ket_cfgs::Vector{ClusterConfig{N}}, buckets, maxdiff::Int,
                        cluster_ops, fock_bra::FockConfig{N},
                        fock_ket::FockConfig{N}, threaded::Bool) where {T,N}
    if threaded && length(bra_cfgs) > 1 && Threads.nthreads() > 1
        Threads.@threads :dynamic for bi in eachindex(bra_cfgs)
            _fill_block_h_row!(Hblk, bi, bra_cfgs[bi], ket_cfgs, buckets, maxdiff,
                               cluster_ops, fock_bra, fock_ket)
        end
    else
        for bi in eachindex(bra_cfgs)
            _fill_block_h_row!(Hblk, bi, bra_cfgs[bi], ket_cfgs, buckets, maxdiff,
                               cluster_ops, fock_bra, fock_ket)
        end
    end
    return Hblk
end

# One (fock_bra, fock_ket) block of work: the destination and everything needed
# to fill it. Collected first so the whole chunk can be threaded at whichever
# granularity actually has parallelism -- a sharded space is usually many small
# Fock sectors (thread across blocks) but can be a few large ones (thread across
# the rows of a block).
struct ShardedBlockJob{T,N}
    fock_bra::FockConfig{N}
    fock_ket::FockConfig{N}
    Hblk::Matrix{T}
    bra_cfgs::Vector{ClusterConfig{N}}
    ket_cfgs::Vector{ClusterConfig{N}}
    buckets::ShardedTermBuckets
    maxdiff::Int
    reuse::Union{Nothing,Tuple{Matrix{T},Vector{Int},Vector{Int},Vector{Int}}}
end

function _run_block_h_job!(job::ShardedBlockJob{T,N}, cluster_ops,
                           thread_rows::Bool) where {T,N}
    if job.reuse === nothing
        _fill_block_h!(job.Hblk, job.bra_cfgs, job.ket_cfgs, job.buckets, job.maxdiff,
                       cluster_ops, job.fock_bra, job.fock_ket, thread_rows)
        return nothing
    end
    M_old, old_row, old_col, todo = job.reuse
    fill_row = bi -> begin
        bio = old_row[bi]
        if bio == 0
            _fill_block_h_row!(job.Hblk, bi, job.bra_cfgs[bi], job.ket_cfgs,
                               job.buckets, job.maxdiff, cluster_ops, job.fock_bra,
                               job.fock_ket)
        else
            _fill_block_h_row_reuse!(job.Hblk, bi, bio, M_old, old_col, todo,
                                     job.bra_cfgs[bi], job.ket_cfgs, job.buckets,
                                     job.maxdiff, cluster_ops, job.fock_bra,
                                     job.fock_ket)
        end
    end
    if thread_rows && length(job.bra_cfgs) > 1 && Threads.nthreads() > 1
        Threads.@threads :dynamic for bi in eachindex(job.bra_cfgs)
            fill_row(bi)
        end
    else
        for bi in eachindex(job.bra_cfgs)
            fill_row(bi)
        end
    end
    return nothing
end

function _run_block_h_jobs!(jobs::Vector{ShardedBlockJob{T,N}}, cluster_ops,
                            threaded::Bool) where {T,N}
    nt = Threads.nthreads()
    if !threaded || nt <= 1 || length(jobs) <= 1
        for job in jobs
            _run_block_h_job!(job, cluster_ops, threaded && nt > 1)
        end
    elseif length(jobs) >= 2 * nt
        # Plenty of blocks: one thread per block, which avoids paying the
        # fork/join cost on each of the (often tiny) inner row loops.
        Threads.@threads :dynamic for i in eachindex(jobs)
            _run_block_h_job!(jobs[i], cluster_ops, false)
        end
    else
        for job in jobs
            _run_block_h_job!(job, cluster_ops, true)
        end
    end
    return jobs
end

# The ket sectors this chunk's row blocks read (see `ShardedBlockData`). Which
# worker owns each of them is deliberately NOT baked in: `apply_sharded_H` may be
# handed a vector distributed differently from the one H was built on.
function _block_h_needed_kets(owned_bras, connections, ::Val{N}) where {N}
    needed = Set{FockConfig{N}}()
    for fock_bra in owned_bras
        push!(needed, fock_bra)          # the diagonal shift reads this sector too
        for fock_ket in get(connections, fock_bra, FockConfig{N}[])
            push!(needed, fock_ket)
        end
    end
    return collect(needed)
end

mutable struct ShardedBlockH{T,N}
    id::Symbol
    workers::Vector{Int}
    clusters::Vector{MOCluster}
    nnz::Float64
    # Thread the per-apply GEMV across owned bra sectors. Carried on the operator
    # so every apply inherits the same setting the build was given.
    threaded::Bool
end

function _tpsci_sharded_store_block_h!(id::Symbol, data)
    _TPSCI_SHARDED_BLOCK_H[id] = data
    return true
end

function _tpsci_sharded_delete_block_h!(id::Symbol)
    delete!(_TPSCI_SHARDED_BLOCK_H, id)
    return true
end

function _tpsci_sharded_get_block_h(id::Symbol)
    haskey(_TPSCI_SHARDED_BLOCK_H, id) ||
        error("No sharded block-H cached with id $id on worker $(Distributed.myid())")
    return _TPSCI_SHARDED_BLOCK_H[id]
end

function _build_block_h_chunk!(block_id::Symbol, ci_vector::DistributedTPSCIstate,
                               owned_bras, threaded_worker::Bool=true)
    cluster_ops = _TPSCI_MULTINODE_CLUSTER_OPS[]
    clustered_ham = _TPSCI_MULTINODE_CLUSTERED_HAM[]
    cluster_ops === nothing && error("TPSCI sharded cluster-ops cache is empty")
    clustered_ham === nothing && error("TPSCI sharded Hamiltonian cache is empty")
    if cluster_ops isa DistributedClusterOps
        cluster_ops = _materialize_cluster_ops_for_indices(
            cluster_ops, [c.idx for c in ci_vector.clusters])
    end
    return _build_block_h_chunk_typed!(block_id, ci_vector, owned_bras,
                                       cluster_ops, clustered_ham, threaded_worker)
end

function _build_block_h_chunk_typed!(block_id::Symbol,
                                     ci_vector::DistributedTPSCIstate{T,N,R},
                                     owned_bras, cluster_ops,
                                     clustered_ham::ClusteredOperator,
                                     threaded::Bool=true) where {T,N,R}
    local_state = _tpsci_sharded_get_state(ci_vector.id)
    # Pull every ket sector this chunk needs up front, one concurrent burst per
    # remote owner. The old lazy per-sector fetch inside the loop paid a blocking
    # network round trip for each of the (often hundreds of) Fock sectors.
    ket_cache = _tpsci_sharded_prefetch_ket_cache(ci_vector, clustered_ham, owned_bras)
    bra_order = Dict{FockConfig{N},Vector{ClusterConfig{N}}}()
    ket_order = Dict{FockConfig{N},Vector{ClusterConfig{N}}}()
    connections = Dict{FockConfig{N},Vector{FockConfig{N}}}()
    mats = Dict{Tuple{FockConfig{N},FockConfig{N}},Matrix{T}}()
    bucket_cache = Dict{TransferConfig{N},Tuple{ShardedTermBuckets,Int}}()
    jobs = ShardedBlockJob{T,N}[]

    for fock_bra in owned_bras
        bra_cfgs = collect(keys(local_state.data[fock_bra]))
        bra_order[fock_bra] = bra_cfgs
        ket_order[fock_bra] = bra_cfgs      # needed by the shifted apply
        conns = FockConfig{N}[]
        for fock_ket in keys(ci_vector.owners)
            fock_trans = fock_bra - fock_ket
            haskey(clustered_ham, fock_trans) || continue
            buckets, maxdiff = get!(bucket_cache, fock_trans) do
                _bucket_terms_by_diff(clustered_ham[fock_trans], fock_trans)
            end
            configs_ket = _tpsci_sharded_get_ket_configs(ci_vector, fock_ket, ket_cache)
            ket_cfgs = get!(() -> collect(keys(configs_ket)), ket_order, fock_ket)
            # No term's Fock pattern fits this transfer: the block is identically
            # zero, so neither store it nor GEMV against it later.
            isempty(buckets) && continue
            Hblk = zeros(T, length(bra_cfgs), length(ket_cfgs))
            push!(jobs, ShardedBlockJob{T,N}(fock_bra, fock_ket, Hblk, bra_cfgs,
                                             ket_cfgs, buckets, maxdiff, nothing))
            mats[(fock_bra, fock_ket)] = Hblk
            push!(conns, fock_ket)
        end
        connections[fock_bra] = conns
    end
    _run_block_h_jobs!(jobs, cluster_ops, threaded)
    # The job list only holds references to blocks that `mats` already owns, but
    # release it before the caller starts sizing the next allocation.
    empty!(jobs)
    empty!(bucket_cache)

    data = ShardedBlockData{T,N}(collect(owned_bras), bra_order, ket_order,
                                 connections, mats,
                                 _block_h_needed_kets(owned_bras, connections, Val(N)))
    _tpsci_sharded_store_block_h!(block_id, data)
    nbytes = 0
    for (_, m) in mats
        nbytes += length(m)
    end
    return nbytes
end

"""
    build_block_h_sharded(ci_vector, cluster_ops, clustered_ham) -> ShardedBlockH

Build the variational Hamiltonian once as distributed, block-sparse dense blocks.
Each worker owns the row-blocks for the Fock-bra sectors it owns in `ci_vector`.
Reused by `apply_sharded_H` for fast block-GEMV matvecs.
"""
function build_block_h_sharded(ci_vector::DistributedTPSCIstate{T,N,R},
                               cluster_ops, clustered_ham::ClusteredOperator;
                               workers=ci_vector.workers, blas_threads=1,
                               threaded_worker=true,
                               id=nothing, verbose=1) where {T,N,R}
    _tpsci_sharded_cache_operator_problem!(cluster_ops, clustered_ham;
                                           workers=workers, blas_threads=blas_threads)
    block_id = id === nothing ? gensym(:tpsci_shard_blockH) : Symbol(id)
    chunks = Dict(pid => FockConfig{N}[] for pid in ci_vector.workers)
    for (fock, owner) in ci_vector.owners
        push!(chunks[owner], fock)
    end

    if verbose > 0
        rep = sharded_H_memory_report(ci_vector, clustered_ham)
        @printf(" Build sharded block-sparse H: dim=%i  nfocks=%i  nnz=%.3e  mem=%.3f GB\n",
                rep.dim, rep.nfocks, rep.nnz, rep.gb)
        flush(stdout)
    end

    nbyte_maps = Dict{Int,Int}()
    @sync for pid in ci_vector.workers
        @async begin
            nbyte_maps[pid] = Distributed.remotecall_fetch(
                _build_block_h_chunk!, pid, block_id, ci_vector, chunks[pid],
                threaded_worker)
        end
    end
    nnz = sum(values(nbyte_maps); init=0)
    return ShardedBlockH{T,N}(block_id, ci_vector.workers, ci_vector.clusters,
                              Float64(nnz), threaded_worker)
end

function _update_block_h_chunk!(block_id::Symbol, ci_vector::DistributedTPSCIstate,
                                owned_bras, threaded_worker::Bool=true)
    cluster_ops = _TPSCI_MULTINODE_CLUSTER_OPS[]
    clustered_ham = _TPSCI_MULTINODE_CLUSTERED_HAM[]
    cluster_ops === nothing && error("TPSCI sharded cluster-ops cache is empty")
    clustered_ham === nothing && error("TPSCI sharded Hamiltonian cache is empty")
    if cluster_ops isa DistributedClusterOps
        cluster_ops = _materialize_cluster_ops_for_indices(
            cluster_ops, [c.idx for c in ci_vector.clusters])
    end
    return _update_block_h_chunk_typed!(block_id, ci_vector, owned_bras,
                                        cluster_ops, clustered_ham, threaded_worker)
end

# Fill row `bi` restricted to the ket columns listed in `todo`, copying every
# other column from the corresponding row `bio` of the previous block.
function _fill_block_h_row_reuse!(Hblk::Matrix{T}, bi::Int, bio::Int,
                                  M_old::Matrix{T}, old_col::Vector{Int},
                                  todo::Vector{Int},
                                  config_bra::ClusterConfig{N},
                                  ket_cfgs::Vector{ClusterConfig{N}}, buckets,
                                  maxdiff::Int, cluster_ops,
                                  fock_bra::FockConfig{N},
                                  fock_ket::FockConfig{N}) where {T,N}
    @inbounds for ki in eachindex(ket_cfgs)
        kio = old_col[ki]
        kio > 0 && (Hblk[bi, ki] = M_old[bio, kio])
    end
    @inbounds for ki in todo
        config_ket = ket_cfgs[ki]
        mask, pop = _config_diff_mask(config_bra, config_ket, maxdiff)
        pop <= maxdiff || continue
        cand = get(buckets, mask, nothing)
        cand === nothing && continue
        acc = zero(T)
        for term in cand
            acc += contract_matrix_element(term, cluster_ops, fock_bra, config_bra,
                                           fock_ket, config_ket)
        end
        Hblk[bi, ki] = acc
    end
    return nothing
end

function _update_block_h_chunk_typed!(block_id::Symbol,
                                      ci_vector::DistributedTPSCIstate{T,N,R},
                                      owned_bras, cluster_ops,
                                      clustered_ham::ClusteredOperator,
                                      threaded::Bool=true) where {T,N,R}
    old = haskey(_TPSCI_SHARDED_BLOCK_H, block_id) ?
          _tpsci_sharded_get_block_h(block_id) : nothing
    local_state = _tpsci_sharded_get_state(ci_vector.id)
    ket_cache = _tpsci_sharded_prefetch_ket_cache(ci_vector, clustered_ham, owned_bras)
    bra_order = Dict{FockConfig{N},Vector{ClusterConfig{N}}}()
    ket_order = Dict{FockConfig{N},Vector{ClusterConfig{N}}}()
    connections = Dict{FockConfig{N},Vector{FockConfig{N}}}()
    mats = Dict{Tuple{FockConfig{N},FockConfig{N}},Matrix{T}}()
    bucket_cache = Dict{TransferConfig{N},Tuple{ShardedTermBuckets,Int}}()
    jobs = ShardedBlockJob{T,N}[]

    # Old-position maps per sector, built lazily and shared across the chunk.
    old_ket_col = Dict{FockConfig{N},Vector{Int}}()
    old_ket_todo = Dict{FockConfig{N},Vector{Int}}()

    for fock_bra in owned_bras
        bra_cfgs = collect(keys(local_state.data[fock_bra]))
        bra_order[fock_bra] = bra_cfgs
        ket_order[fock_bra] = bra_cfgs
        old_bra = old === nothing ? ClusterConfig{N}[] :
                  get(old.bra_order, fock_bra, ClusterConfig{N}[])
        bra_pos_old = Dict(c => i for (i, c) in enumerate(old_bra))
        old_row = Int[get(bra_pos_old, c, 0) for c in bra_cfgs]
        conns = FockConfig{N}[]
        for fock_ket in keys(ci_vector.owners)
            fock_trans = fock_bra - fock_ket
            haskey(clustered_ham, fock_trans) || continue
            buckets, maxdiff = get!(bucket_cache, fock_trans) do
                _bucket_terms_by_diff(clustered_ham[fock_trans], fock_trans)
            end
            configs_ket = _tpsci_sharded_get_ket_configs(ci_vector, fock_ket, ket_cache)
            ket_cfgs = get!(() -> collect(keys(configs_ket)), ket_order, fock_ket)
            isempty(buckets) && continue
            M_old = old === nothing ? nothing :
                    get(old.mats, (fock_bra, fock_ket), nothing)
            reuse = nothing
            if M_old !== nothing
                # Which ket columns are new is the same for every row and every
                # bra sector, so the reuse pattern is worked out once per sector.
                if !haskey(old_ket_col, fock_ket)
                    oldk = get(old.ket_order, fock_ket, ClusterConfig{N}[])
                    kpos = Dict(c => i for (i, c) in enumerate(oldk))
                    col = Int[get(kpos, c, 0) for c in ket_cfgs]
                    old_ket_col[fock_ket] = col
                    old_ket_todo[fock_ket] = Int[ki for ki in eachindex(col) if col[ki] == 0]
                end
                reuse = (M_old, old_row, old_ket_col[fock_ket], old_ket_todo[fock_ket])
            end
            Hblk = zeros(T, length(bra_cfgs), length(ket_cfgs))
            push!(jobs, ShardedBlockJob{T,N}(fock_bra, fock_ket, Hblk, bra_cfgs,
                                             ket_cfgs, buckets, maxdiff, reuse))
            mats[(fock_bra, fock_ket)] = Hblk
            push!(conns, fock_ket)
        end
        connections[fock_bra] = conns
    end
    _run_block_h_jobs!(jobs, cluster_ops, threaded)
    empty!(jobs)
    empty!(bucket_cache)
    empty!(old_ket_col)
    empty!(old_ket_todo)

    data = ShardedBlockData{T,N}(collect(owned_bras), bra_order, ket_order,
                                 connections, mats,
                                 _block_h_needed_kets(owned_bras, connections, Val(N)))
    _tpsci_sharded_store_block_h!(block_id, data)
    nbytes = 0
    for (_, m) in mats
        nbytes += length(m)
    end
    return nbytes
end

"""
    update_block_h_sharded!(op::ShardedBlockH, ci_vector, cluster_ops, clustered_ham; ...)

Extend a stored block-sparse Hamiltonian in place after the space `ci_vector`
has GROWN (a selected-CI iteration): previously computed matrix elements are
copied from the existing blocks, and only the rows/columns belonging to new
configurations — plus whole blocks for newly appearing Fock sectors — are
contracted. This is the sharded analog of `tps_ci_direct`'s `H_old`/`v_old`
incremental rebuild: per block the contraction cost drops from
`n_bra·n_ket` to `n_bra_new·n_ket + n_bra_old·n_ket_new`.

Requires the same worker list and (for existing sectors) the same ownership as
when the operator was built — both guaranteed inside `tpsci_ci_sharded`, where
ownership is never reshuffled. Transiently holds old + new blocks on each
worker while swapping.
"""
function update_block_h_sharded!(op::ShardedBlockH{T,N},
                                 ci_vector::DistributedTPSCIstate{T,N,R},
                                 cluster_ops, clustered_ham::ClusteredOperator;
                                 workers=ci_vector.workers,
                                 blas_threads=1,
                                 threaded_worker=true,
                                 verbose=0) where {T,N,R}
    op.workers == ci_vector.workers ||
        error("update_block_h_sharded! requires the block-H and state on the same workers")
    _tpsci_sharded_cache_operator_problem!(cluster_ops, clustered_ham;
                                           workers=workers,
                                           blas_threads=blas_threads)
    chunks = Dict(pid => FockConfig{N}[] for pid in ci_vector.workers)
    for (fock, owner) in ci_vector.owners
        push!(chunks[owner], fock)
    end

    nbyte_maps = Dict{Int,Int}()
    @sync for pid in ci_vector.workers
        @async begin
            nbyte_maps[pid] = Distributed.remotecall_fetch(
                _update_block_h_chunk!, pid, op.id, ci_vector, chunks[pid],
                threaded_worker)
        end
    end
    old_nnz = op.nnz
    op.nnz = Float64(sum(values(nbyte_maps); init=0))
    verbose > 0 &&
        @printf(" Updated sharded block-sparse H incrementally: nnz %.3e -> %.3e\n",
                old_nnz, op.nnz)
    return op
end

# Dense coefficients of one Fock sector, in the order the stored H expects. The
# fast path walks the `OrderedDict` once and compares keys (all vectors over a
# fixed Q space are built in the same insertion order, so this matches); if the
# orders ever diverge it falls back to keyed lookup, which is what the old code
# always did.
function _dense_ket_in_order(::Type{T}, configs, want::Vector{ClusterConfig{N}}) where {T,N}
    n = length(want)
    out = Vector{T}(undef, n)
    length(configs) == n ||
        error("sharded H apply: vector has $(length(configs)) configs where the stored H expects $n")
    i = 0
    aligned = true
    @inbounds for (config, coeffs) in configs
        i += 1
        if config != want[i]
            aligned = false
            break
        end
        out[i] = coeffs[1]
    end
    if !aligned
        @inbounds for j in 1:n
            out[j] = configs[want[j]][1]
        end
    end
    return out
end

function _block_h_local_gather_kets(state_id::Symbol, focks, want, ::Type{T}) where {T}
    state = _tpsci_sharded_get_state(state_id)
    return [_dense_ket_in_order(T, state.data[focks[i]], want[i]) for i in eachindex(focks)]
end

# Collect the ket coefficients of every sector this chunk's row blocks connect
# to. Sectors owned elsewhere are fetched one call per remote owner (all owners
# concurrently) as dense `Vector{T}`s, instead of one blocking round trip per
# Fock sector carrying a whole `OrderedDict` of configurations.
function _block_h_gather_kets(block::ShardedBlockData{T,N},
                              v::DistributedTPSCIstate{Tv,N,1}) where {T,N,Tv}
    myid = Distributed.myid()
    kets = Dict{FockConfig{N},Vector{T}}()
    remote = Dict{Int,Vector{FockConfig{N}}}()
    local_state = nothing
    for fock in block.needed_kets
        owner = v.owners[fock]
        if owner == myid
            local_state === nothing && (local_state = _tpsci_sharded_get_state(v.id))
            kets[fock] = _dense_ket_in_order(T, local_state.data[fock],
                                             block.ket_order[fock])
        else
            push!(get!(() -> FockConfig{N}[], remote, owner), fock)
        end
    end
    isempty(remote) && return kets
    fetched = Dict{Int,Vector{Vector{T}}}()
    @sync for (owner, focks) in remote
        @async fetched[owner] = Distributed.remotecall_fetch(
            _block_h_local_gather_kets, owner, v.id, focks,
            [block.ket_order[f] for f in focks], T)
    end
    for (owner, focks) in remote
        blocks = fetched[owner]
        for i in eachindex(focks)
            kets[focks[i]] = blocks[i]
        end
    end
    return kets
end

function _apply_block_h_chunk!(out_id::Symbol, block_id::Symbol,
                               v::DistributedTPSCIstate, eshift,
                               threaded::Bool=true)
    return _apply_block_h_chunk_typed!(out_id, _tpsci_sharded_get_block_h(block_id),
                                       v, eshift, threaded)
end

# One row block: sigma_F = sum_F' H[F,F'] v_F', with the diagonal shift folded in.
# Split out of the chunk loop so the loop body is thread-safe -- it touches only
# this bra sector's own output buffer and reads the shared (immutable) ket cache.
function _apply_block_h_row(block::ShardedBlockData{T,N}, kets,
                            fock_bra::FockConfig{N}, shift::T) where {T,N}
    sig = zeros(T, length(block.bra_order[fock_bra]))
    for fock_ket in block.connections[fock_bra]
        mul!(sig, block.mats[(fock_bra, fock_ket)], kets[fock_ket], one(T), one(T))
    end
    # (H - eshift*I) v in the same pass: the CEPA/Davidson solvers always want
    # the shifted operator, and doing it here saves a whole extra fan-out.
    iszero(shift) || axpy!(-shift, kets[fock_bra], sig)
    return sig
end

function _apply_block_h_chunk_typed!(out_id::Symbol, block::ShardedBlockData{T,N},
                                     v::DistributedTPSCIstate{Tv,N,1},
                                     eshift, threaded::Bool=true) where {T,N,Tv}
    kets = _block_h_gather_kets(block, v)
    shift = T(eshift)
    bras = block.owned_bras

    # The GEMV is ~97% of a stored-H solve and every bra sector is independent,
    # so thread across them. `out` is assembled serially afterwards: a TPSCIstate
    # is a Dict of Dicts and cannot be grown safely from several threads, but
    # that assembly is a pointer shuffle next to the GEMV it follows.
    sigs = Vector{Vector{T}}(undef, length(bras))
    if threaded && Threads.nthreads() > 1 && length(bras) > 1
        Threads.@threads :dynamic for i in eachindex(bras)
            sigs[i] = _apply_block_h_row(block, kets, bras[i], shift)
        end
    else
        for i in eachindex(bras)
            sigs[i] = _apply_block_h_row(block, kets, bras[i], shift)
        end
    end

    out = TPSCIstate(v.clusters, T=T, R=1)
    for (i, fock_bra) in enumerate(bras)
        bra_cfgs = block.bra_order[fock_bra]
        sig = sigs[i]
        add_fockconfig!(out, fock_bra)
        for (bi, config_bra) in enumerate(bra_cfgs)
            out[fock_bra][config_bra] = MVector{1,T}(sig[bi])
        end
    end
    _tpsci_sharded_store_state!(out_id, out)
    return _tpsci_sharded_local_fock_lengths(out_id)
end

"""
    apply_sharded_H(op::ShardedBlockH, v; eshift=0, id=nothing)

Apply the stored block-sparse Hamiltonian to a sharded single-root vector,
optionally shifted: the result is `(H - eshift*I) v`.
"""
function apply_sharded_H(op::ShardedBlockH{T,N}, v::DistributedTPSCIstate{Tv,N,1};
                         eshift=zero(T), id=nothing) where {T,N,Tv}
    op.workers == v.workers || error("apply_sharded_H requires the block-H and vector on the same workers")
    output_id = id === nothing ? gensym(:tpsci_shard_blockHv) : Symbol(id)
    length_maps = Dict{Int,Any}()
    @sync for pid in op.workers
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(
                _apply_block_h_chunk!, pid, output_id, op.id, v, eshift, op.threaded)
        end
    end
    return _tpsci_sharded_metadata_from_lengths(output_id, op.clusters, op.workers,
                                                length_maps,
                                                collect(keys(v.owners)),
                                                T, Val(1))
end

function _block_h_local_diagonal!(diag_id::Symbol, block_id::Symbol,
                                  state_id::Symbol)
    return _block_h_local_diagonal_typed!(diag_id,
                                          _tpsci_sharded_get_block_h(block_id),
                                          _tpsci_sharded_get_state(state_id))
end

function _block_h_local_diagonal_typed!(diag_id::Symbol,
                                        block::ShardedBlockData{T,N},
                                        state::TPSCIstate{Ts,N,R}) where {T,N,Ts,R}
    out = TPSCIstate(state.clusters, T=T, R=1)
    for fock in block.owned_bras
        # A sector with no Fock-feasible term stores no block at all: its
        # Hamiltonian diagonal is exactly zero.
        Hff = get(block.mats, (fock, fock), nothing)
        bra_cfgs = block.bra_order[fock]
        add_fockconfig!(out, fock)
        if Hff === nothing
            for cfg in bra_cfgs
                out[fock][cfg] = MVector{1,T}(zero(T))
            end
            continue
        end
        ket_cfgs = block.ket_order[fock]
        ket_pos = Dict(c => i for (i, c) in enumerate(ket_cfgs))
        for (bi, cfg) in enumerate(bra_cfgs)
            ki = get(ket_pos, cfg, 0)
            ki > 0 || error("Stored block-H diagonal block is missing config $cfg in Fock sector $fock")
            out[fock][cfg] = MVector{1,T}(Hff[bi, ki])
        end
    end
    _tpsci_sharded_store_state!(diag_id, out)
    return _tpsci_sharded_local_fock_lengths(diag_id)
end

"""
    compute_diagonal_sharded(block_h, ci_vector) -> DistributedTPSCIstate{T,N,1}

Extract the Hamiltonian diagonal directly from an already-built stored
block-H. This avoids a second matrix-element contraction pass when the
Davidson preconditioner is used with `h_storage=:blocks`.
"""
function compute_diagonal_sharded(op::ShardedBlockH{T,N},
                                  ci_vector::DistributedTPSCIstate{Tv,N,R};
                                  id=nothing) where {T,N,Tv,R}
    op.workers == ci_vector.workers ||
        error("compute_diagonal_sharded(block_h, state) requires matching workers")
    output_id = id === nothing ? gensym(:tpsci_shard_blockHdiag) : Symbol(id)
    length_maps = Dict{Int,Any}()
    @sync for pid in op.workers
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(
                _block_h_local_diagonal!, pid, output_id, op.id, ci_vector.id)
        end
    end
    return _tpsci_sharded_metadata_from_lengths(output_id, op.clusters,
                                                op.workers, length_maps,
                                                collect(keys(ci_vector.owners)),
                                                T, Val(1))
end

function _tpsci_sharded_local_dense_coefficients(state_id::Symbol, owned_focks)
    state = _tpsci_sharded_get_state(state_id)
    return _tpsci_sharded_local_dense_coefficients_typed(state, owned_focks)
end

function _tpsci_sharded_local_dense_coefficients_typed(state::TPSCIstate{T,N,R},
                                                       owned_focks) where {T,N,R}
    chunks = Dict{FockConfig{N},Matrix{T}}()
    for fock in owned_focks
        configs = state.data[fock]
        X = zeros(T, length(configs), R)
        for (i, (_, coeffs)) in enumerate(configs)
            @views X[i, :] .= coeffs
        end
        chunks[fock] = X
    end
    return chunks
end

"""
    dense_coefficients_sharded(state) -> Matrix

Gather only the dense coefficient matrix for a sharded TPSCI state. This does
not gather the Hamiltonian or configuration metadata beyond the already-known
master-side offsets, and is used by the stored-H sharded direct solver where the
Davidson subspace is intentionally kept as dense coefficient matrices.
"""
function dense_coefficients_sharded(state::DistributedTPSCIstate{T,N,R}) where {T,N,R}
    chunks_by_pid = Dict(pid => FockConfig{N}[] for pid in state.workers)
    for (fock, owner) in state.owners
        push!(chunks_by_pid[owner], fock)
    end

    partials = Dict{Int,Dict{FockConfig{N},Matrix{T}}}()
    @sync for pid in state.workers
        @async begin
            partials[pid] = Distributed.remotecall_fetch(
                _tpsci_sharded_local_dense_coefficients, pid,
                state.id, chunks_by_pid[pid])
        end
    end

    X = zeros(T, length(state), R)
    for pid in state.workers
        for (fock, Xf) in partials[pid]
            off = state.offsets[fock]
            len = state.lengths[fock]
            @views X[off:off + len - 1, :] .= Xf
        end
    end
    return X
end

function _dense_coeff_chunks_from_global(state::DistributedTPSCIstate{T,N,R},
                                         X::AbstractMatrix) where {T,N,R}
    size(X, 1) == length(state) || throw(DimensionMismatch("coefficient row count does not match sharded state length"))
    chunks = Dict{FockConfig{N},Matrix{T}}()
    for (fock, len) in state.lengths
        off = state.offsets[fock]
        chunks[fock] = Matrix{T}(@view X[off:off + len - 1, :])
    end
    return chunks
end

function _tpsci_sharded_local_set_dense_coefficients!(state_id::Symbol, chunks)
    state = _tpsci_sharded_get_state(state_id)
    return _tpsci_sharded_local_set_dense_coefficients_typed!(state, chunks)
end

function _tpsci_sharded_local_set_dense_coefficients_typed!(state::TPSCIstate{T,N,R},
                                                            chunks) where {T,N,R}
    for (fock, Xf) in chunks
        haskey(state.data, fock) ||
            error("Destination sharded state is missing Fock sector $fock")
        size(Xf, 2) == R ||
            throw(DimensionMismatch("coefficient root count does not match destination state"))
        configs = state.data[fock]
        size(Xf, 1) == length(configs) ||
            throw(DimensionMismatch("coefficient block length does not match Fock sector $fock"))
        for (i, (_, coeffs)) in enumerate(configs)
            @views coeffs .= Xf[i, :]
        end
    end
    return true
end

function set_dense_coefficients_sharded!(state::DistributedTPSCIstate{T,N,R},
                                         X::AbstractMatrix) where {T,N,R}
    size(X) == (length(state), R) ||
        throw(DimensionMismatch("coefficient matrix size $(size(X)) does not match sharded state size $(size(state))"))
    chunks_by_pid = Dict(pid => Dict{FockConfig{N},Matrix{T}}() for pid in state.workers)
    for (fock, owner) in state.owners
        off = state.offsets[fock]
        len = state.lengths[fock]
        chunks_by_pid[owner][fock] = Matrix{T}(@view X[off:off + len - 1, :])
    end
    @sync for pid in state.workers
        @async Distributed.remotecall_fetch(_tpsci_sharded_local_set_dense_coefficients!,
                                            pid, state.id, chunks_by_pid[pid])
    end
    return state
end

function _block_h_local_apply_dense(block_id::Symbol, x_chunks)
    block = _tpsci_sharded_get_block_h(block_id)
    return _block_h_local_apply_dense_typed(block, x_chunks)
end

function _block_h_local_apply_dense_typed(block::ShardedBlockData{T,N},
                                          x_chunks) where {T,N}
    nrhs = isempty(x_chunks) ? 0 : size(first(values(x_chunks)), 2)
    y_chunks = Dict{FockConfig{N},Matrix{T}}()
    for fock_bra in block.owned_bras
        bra_cfgs = block.bra_order[fock_bra]
        Yb = zeros(T, length(bra_cfgs), nrhs)
        for fock_ket in block.connections[fock_bra]
            haskey(x_chunks, fock_ket) ||
                error("Dense sharded H apply is missing ket coefficient block for $fock_ket")
            Hblk = block.mats[(fock_bra, fock_ket)]
            Xk = x_chunks[fock_ket]
            mul!(Yb, Hblk, Xk, one(T), one(T))
        end
        y_chunks[fock_bra] = Yb
    end
    return y_chunks
end

"""
    apply_sharded_H_dense(block_h, ci_vector, X)

Apply a stored sharded block Hamiltonian to one or more dense coefficient
vectors. `X` lives on the master as `dim × nrhs`; the Hamiltonian blocks stay on
workers. Each worker receives the small dense coefficient blocks and performs
batched GEMM for its owned row blocks, then returns only its output rows.
"""
function apply_sharded_H_dense(op::ShardedBlockH{T,N},
                               ci_vector::DistributedTPSCIstate{Tv,N,R},
                               X::Union{AbstractVector,AbstractMatrix}) where {T,N,Tv,R}
    op.workers == ci_vector.workers ||
        error("apply_sharded_H_dense requires the block-H and state on the same workers")
    isvec = X isa AbstractVector
    Xmat = isvec ? reshape(Vector{T}(X), :, 1) : Matrix{T}(X)
    size(Xmat, 1) == length(ci_vector) ||
        throw(DimensionMismatch("coefficient row count does not match sharded state length"))
    x_chunks = _dense_coeff_chunks_from_global(ci_vector, Xmat)

    partials = Dict{Int,Dict{FockConfig{N},Matrix{T}}}()
    @sync for pid in op.workers
        @async begin
            partials[pid] = Distributed.remotecall_fetch(
                _block_h_local_apply_dense, pid, op.id, x_chunks)
        end
    end

    Y = zeros(T, length(ci_vector), size(Xmat, 2))
    for pid in op.workers
        for (fock, Yf) in partials[pid]
            off = ci_vector.offsets[fock]
            len = ci_vector.lengths[fock]
            @views Y[off:off + len - 1, :] .= Yf
        end
    end
    return isvec ? Y[:, 1] : Y
end

function _block_h_local_dense_diagonal(block_id::Symbol)
    block = _tpsci_sharded_get_block_h(block_id)
    return _block_h_local_dense_diagonal_typed(block)
end

function _block_h_local_dense_diagonal_typed(block::ShardedBlockData{T,N}) where {T,N}
    chunks = Dict{FockConfig{N},Vector{T}}()
    for fock in block.owned_bras
        Hff = get(block.mats, (fock, fock), nothing)
        Hff === nothing && error("Stored block-H is missing diagonal block for Fock sector $fock")
        bra_cfgs = block.bra_order[fock]
        ket_cfgs = block.ket_order[fock]
        ket_pos = Dict(c => i for (i, c) in enumerate(ket_cfgs))
        d = zeros(T, length(bra_cfgs))
        for (bi, cfg) in enumerate(bra_cfgs)
            ki = get(ket_pos, cfg, 0)
            ki > 0 || error("Stored block-H diagonal block is missing config $cfg in Fock sector $fock")
            d[bi] = Hff[bi, ki]
        end
        chunks[fock] = d
    end
    return chunks
end

function compute_diagonal_dense_sharded(op::ShardedBlockH{T,N},
                                        ci_vector::DistributedTPSCIstate{Tv,N,R}) where {T,N,Tv,R}
    op.workers == ci_vector.workers ||
        error("compute_diagonal_dense_sharded requires the block-H and state on the same workers")
    partials = Dict{Int,Dict{FockConfig{N},Vector{T}}}()
    @sync for pid in op.workers
        @async begin
            partials[pid] = Distributed.remotecall_fetch(
                _block_h_local_dense_diagonal, pid, op.id)
        end
    end
    d = zeros(T, length(ci_vector))
    for pid in op.workers
        for (fock, df) in partials[pid]
            off = ci_vector.offsets[fock]
            len = ci_vector.lengths[fock]
            @views d[off:off + len - 1] .= df
        end
    end
    return d
end

"""
    tps_ci_direct_sharded(ci_vector, cluster_ops, clustered_ham; ...)

Stored-H sharded analogue of `tps_ci_direct`. The Hamiltonian is kept as
distributed dense Fock-pair blocks (`ShardedBlockH`), while the small Davidson
subspace is held as dense coefficient matrices on the master. Hamiltonian
applications use batched worker-local GEMM, so neither the full Hamiltonian nor
the full TPSCI state/configuration payload is gathered onto one node.

For large spaces this intentionally mirrors `tps_ci_direct`'s practical solver:
stored Hamiltonian plus Davidson iterations. It is not the matrix-free sharded
Davidson path.
"""
function tps_ci_direct_sharded(ci_vector::DistributedTPSCIstate{T,N,R},
                               cluster_ops, clustered_ham::ClusteredOperator;
                               nroots::Int=R,
                               conv_thresh=1e-5,
                               lindep_thresh=1e-12,
                               max_ss_vecs=12,
                               max_iter=40,
                               precond=true,
                               h_storage::Symbol=:blocks,
                               max_mem_H=50.0,
                               block_h::Union{Nothing,ShardedBlockH}=nothing,
                               workers=ci_vector.workers,
                               blas_threads=1,
                               compute_s2=true,
                               verbose=1,
                               id=nothing) where {T,N,R}
    nroots <= R ||
        error("tps_ci_direct_sharded needs the guess to carry at least nroots ($nroots) roots; got R=$R")
    worker_ids = ensure_tpsci_multinode_workers!(workers=workers)
    worker_ids == ci_vector.workers ||
        error("tps_ci_direct_sharded requires the guess on the requested workers")

    tier = h_storage
    if block_h === nothing
        if tier == :auto
            rep = sharded_H_memory_report(ci_vector, clustered_ham)
            fit = sharded_H_fit_report(rep, worker_ids)
            verbose > 0 && print_sharded_H_fit(rep, fit)
            rep.gb <= max_mem_H ||
                error("tps_ci_direct_sharded requires stored block-H, but the estimate is " *
                      "$(round(rep.gb; digits=3)) GB above max_mem_H=$(max_mem_H) GB. " *
                      "Increase max_mem_H/nodes or use tps_ci_davidson_sharded with h_storage=:matrixfree explicitly.")
            fit.fits ||
                error("tps_ci_direct_sharded requires stored block-H, but worker " *
                      "$(fit.worst_pid) is short by $(round(fit.worst_deficit_gb; digits=2)) GB " *
                      "(per-worker peak $(round(rep.max_worker_gb; digits=2)) GB on pid " *
                      "$(rep.max_worker_pid)). Add nodes or put one worker per node.")
            tier = :blocks
            verbose > 0 &&
                @printf(" H storage :auto -> :blocks  (block-sparse H ~ %.3f GB, budget %.1f GB)\n",
                        rep.gb, max_mem_H)
        elseif tier != :blocks
            error("tps_ci_direct_sharded only supports stored block-H (:blocks or fitting :auto); got h_storage=:$tier")
        end
        block_h = build_block_h_sharded(ci_vector, cluster_ops, clustered_ham;
                                        workers=worker_ids,
                                        blas_threads=blas_threads,
                                        verbose=verbose)
    else
        block_h.workers == worker_ids ||
            error("supplied block_h lives on different workers than the guess")
    end

    verbose > 0 && begin
        println()
        @printf(" |== Sharded Direct Tensor Product State CI ========================\n")
        @printf(" Hamiltonian matrix dimension = %i\n", length(ci_vector))
        @printf(" nroots = %i   max_ss_vecs = %i   conv_thresh = %.1e\n",
                nroots, max_ss_vecs, conv_thresh)
        flush(stdout)
    end

    X0 = dense_coefficients_sharded(ci_vector)
    X0 = Matrix{T}(@view X0[:, 1:nroots])
    dim = length(ci_vector)

    function matvec(v::Vector)
        return apply_sharded_H_dense(block_h, ci_vector, v)
    end
    function matvec(v::Matrix)
        return apply_sharded_H_dense(block_h, ci_vector, v)
    end

    Hmap = LinOpMat{T}(matvec, dim, true)
    davidson = Davidson(Hmap, v0=X0,
                        max_iter=max_iter, max_ss_vecs=max_ss_vecs,
                        nroots=nroots, tol=conv_thresh,
                        lindep_thresh=lindep_thresh)

    e = nothing
    v = nothing
    elapsed = if precond
        verbose > 0 && @printf(" %-50s", "Extract stored-H diagonal: ")
        diag_time = @elapsed Hd = compute_diagonal_dense_sharded(block_h, ci_vector)
        verbose > 0 && @printf("%10.6f seconds\n", diag_time)
        verbose > 0 && (println(" Now iterate with sharded stored-H GEMM: "); flush(stdout))
        @elapsed e, v = BlockDavidson.eigs(davidson, Adiag=Hd)
    else
        verbose > 0 && (println(" Now iterate with sharded stored-H GEMM: "); flush(stdout))
        @elapsed e, v = BlockDavidson.eigs(davidson)
    end

    vmat = v isa AbstractMatrix ? Matrix{T}(@view v[:, 1:nroots]) :
           Matrix{T}(hcat(v[1:nroots]...))
    vec_out = similar_sharded_state(ci_vector; roots=nroots, id=id)
    set_dense_coefficients_sharded!(vec_out, vmat)

    if verbose > 0
        @printf(" %-50s%10.6f seconds\n", "Diagonalization time: ", elapsed)
    end

    if compute_s2
        clustered_S2 = extract_S2(ci_vector.clusters, T=T)
        verbose > 0 && @printf(" %-50s", "Compute S2 expectation values: ")
        s2_time = @elapsed begin
            sig_s2 = tps_ci_matvec_sharded(vec_out, cluster_ops, clustered_S2;
                                           workers=worker_ids,
                                           blas_threads=blas_threads,
                                           verbose=0)
            s2 = LinearAlgebra.diag(overlap(vec_out, sig_s2))
            destroy!(sig_s2)
        end
        if verbose > 0
            @printf("%10.6f seconds\n", s2_time)
            @printf(" %5s %12s %12s\n", "Root", "Energy", "S2")
            for r in 1:nroots
                @printf(" %5i %12.8f %12.8f\n", r, e[r], abs(s2[r]))
            end
        end
    elseif verbose > 0
        @printf(" %5s %18s\n", "Root", "Energy")
        for r in 1:nroots
            @printf(" %5i %18.10f\n", r, e[r])
        end
    end

    verbose > 0 && @printf(" ==================================================================|\n")
    return Vector{T}(e[1:nroots]), vec_out, block_h
end

function destroy!(op::ShardedBlockH)
    @sync for pid in op.workers
        @async Distributed.remotecall_fetch(_tpsci_sharded_delete_block_h!, pid, op.id)
    end
    op.nnz = 0.0
    return op
end

# ---------------------------------------------------------------------------
# The sharded Davidson driver.
# ---------------------------------------------------------------------------

# Modified Gram-Schmidt: orthogonalize `t` (a sharded R=1 vector) against every
# vector in `basis`, in place. Returns the norm of the surviving component.
# Orthogonalize `t` against the (orthonormal) `basis` in place, then return its
# surviving norm. Two passes of *classical* Gram-Schmidt (CGS-2 / DGKS): for an
# orthonormal basis this is mathematically identical to modified GS in exact
# arithmetic, and the second pass restores stability against round-off. Each pass
# is exactly two network round-trips (one batched overlap, one batched axpy),
# versus 2*|basis| for the per-vector MGS it replaces.
function _mgs_against!(t::AbstractShardedState{T,N,1}, basis) where {T,N}
    if !isempty(basis)
        for _ in 1:2
            coeffs = overlap_batch(t, basis)      # ⟨t|v_i⟩ for all i, one fan-out
            add_scaled_multi!(t, -coeffs, basis)  # t -= Σ ⟨t|v_i⟩ v_i, one fan-out
        end
    end
    return norm(t)[1]
end

function _restart_from_ritz_basis!(ritz_x::Vector{<:AbstractShardedState{T,N,1}},
                                   apply_H, lindep_thresh) where {T,N}
    S = eltype(ritz_x)
    V = S[]
    HV = S[]
    Hss = zeros(T, 0, 0)
    for x in ritz_x
        nrm = _mgs_against!(x, V)
        if nrm <= lindep_thresh
            destroy!(x)
            continue
        end
        scale!(x, one(T) / nrm)
        Hss = _append_direction!(V, HV, Hss, x, apply_H)
    end
    return V, HV, Hss
end

# Append an orthonormal direction `t` to the subspace: apply H, grow the small
# projected matrix `Hss`, and push into `V`/`HV`. Returns the enlarged `Hss`.
function _append_direction!(V, HV, Hss::Matrix{T}, t, apply_H) where {T}
    hv = apply_H(t)
    m = length(V)
    # One batched fan-out for the whole projected-matrix column: newcol[i] =
    # ⟨V[i]|hv⟩ = ⟨hv|V[i]⟩ (real, symmetric H), instead of m separate overlaps.
    newcol = overlap_batch(hv, V)
    d = overlap(t, hv)[1, 1]
    push!(V, t)
    push!(HV, hv)
    Hnew = zeros(T, m + 1, m + 1)
    Hnew[1:m, 1:m] .= Hss
    Hnew[1:m, m + 1] .= newcol
    Hnew[m + 1, 1:m] .= newcol
    Hnew[m + 1, m + 1] = d
    return Hnew
end

function _ritz_vector(V, y_col, ::Type{T}) where {T}
    x = similar_sharded_state(V[1])
    for i in 1:length(V)
        add_scaled!(x, y_col[i], V[i])
    end
    return x
end

"""
    _tps_ci_davidson_sharded_core(seeds, apply_H, Hdiag; nroots, ...)

Block Davidson over sharded subspace vectors. `seeds` is a vector of `nroots`
orthonormal sharded R=1 vectors, `apply_H(v)` returns a sharded `H*v`, and
`Hdiag` is the sharded diagonal preconditioner. Returns `(eigvals, ritz_vectors)`
where `ritz_vectors` is a `Vector{DistributedTPSCIstate{T,N,1}}` of length
`nroots`. All transient sharded vectors are `destroy!`ed; the returned Ritz
vectors are the caller's to free.
"""
function _tps_ci_davidson_sharded_core(seeds::Vector{<:AbstractShardedState{T,N,1}},
                                       apply_H, Hdiag::AbstractShardedState{T,N,1};
                                       nroots::Int, conv_thresh::Float64,
                                       max_ss_vecs::Int, max_iter::Int,
                                       lindep_thresh::Float64, verbose::Int) where {T,N}
    S = eltype(seeds)
    V = S[]
    HV = S[]
    Hss = zeros(T, 0, 0)
    for s in seeds
        Hss = _append_direction!(V, HV, Hss, s, apply_H)
    end

    cap = max_ss_vecs * nroots
    theta = zeros(T, nroots)
    ritz_x = S[]
    converged = false

    for iter in 1:max_iter
        F = eigen(Symmetric(Hss))
        order = sortperm(real.(F.values))[1:nroots]
        theta = T.(real.(F.values[order]))
        y = F.vectors[:, order]

        # cleanup previous iteration's Ritz vectors
        for x in ritz_x
            destroy!(x)
        end
        ritz_x = S[]
        ritz_Hx = S[]
        resnorms = zeros(T, nroots)
        newdirs = S[]

        for s in 1:nroots
            x = _ritz_vector(V, view(y, :, s), T)
            hx = _ritz_vector(HV, view(y, :, s), T)
            push!(ritz_x, x)
            push!(ritz_Hx, hx)
            r = copy_sharded_state(hx)
            add_scaled!(r, -theta[s], x)
            resnorms[s] = norm(r)[1]
            if resnorms[s] > conv_thresh
                push!(newdirs, precondition_sharded(r, Hdiag, theta[s]))
            end
            destroy!(r)
        end

        if verbose > 0
            @printf(" Sharded Davidson iter %3i  ss=%3i  E:", iter, length(V))
            for s in 1:nroots
                @printf(" %14.8f", theta[s])
            end
            @printf("  |R|:")
            for s in 1:nroots
                @printf(" %8.2e", resnorms[s])
            end
            println()
            flush(stdout)
        end

        if all(resnorms .<= conv_thresh)
            converged = true
            for hx in ritz_Hx
                destroy!(hx)
            end
            for v in V
                destroy!(v)
            end
            for v in HV
                destroy!(v)
            end
            return theta, ritz_x
        end

        # Would the subspace exceed the cap? If so (or nothing new to add),
        # collapse to the current Ritz basis, reusing the already-computed
        # Ritz H-vectors so no extra matvec is needed.
        collapse = (length(V) + length(newdirs) > cap) || isempty(newdirs)
        if collapse
            for v in V
                destroy!(v)
            end
            for v in HV
                destroy!(v)
            end
            for hx in ritz_Hx
                destroy!(hx)
            end
            V, HV, Hss = _restart_from_ritz_basis!(ritz_x, apply_H, lindep_thresh)
            length(V) >= nroots ||
                error("Sharded Davidson restart lost Ritz vectors to linear dependence")
            if length(V) > nroots
                error("Internal Sharded Davidson restart error: kept too many Ritz vectors")
            end
            # Ritz vectors are now the basis; do not free them here. Mark ritz_x
            # empty so the next iteration's cleanup does not double-free.
            ritz_x = S[]
            ritz_Hx = S[]
        else
            for hx in ritz_Hx
                destroy!(hx)
            end
        end

        # Orthonormalize new directions against the (possibly collapsed) subspace
        # and against each other, then append the survivors.
        for t in newdirs
            nrm = _mgs_against!(t, V)
            if nrm <= lindep_thresh
                destroy!(t)
                continue
            end
            scale!(t, one(T) / nrm)
            Hss = _append_direction!(V, HV, Hss, t, apply_H)
        end
    end

    if !converged && verbose >= 0
        @printf(" Sharded Davidson did not converge in %i iterations\n", max_iter)
        flush(stdout)
    end
    # ritz_x currently holds the latest Ritz vectors (unless we just collapsed,
    # in which case they are in V). Recompute cleanly to be safe.
    for x in ritz_x
        destroy!(x)
    end
    F = eigen(Symmetric(Hss))
    order = sortperm(real.(F.values))[1:nroots]
    theta = T.(real.(F.values[order]))
    y = F.vectors[:, order]
    ritz_x = [_ritz_vector(V, view(y, :, s), T) for s in 1:nroots]
    for v in V
        destroy!(v)
    end
    for v in HV
        destroy!(v)
    end
    return theta, ritz_x
end

"""
    tps_ci_davidson_sharded(ci_vector, cluster_ops, clustered_ham; nroots, ...)

Diagonalize the variational Hamiltonian in the space of a sharded `ci_vector`
without ever gathering a full vector onto the master. `ci_vector` supplies both
the space and the initial guess (its `R` roots); `nroots` defaults to `R`.

`h_storage`:
  * `:auto`   — build stored block-sparse H (Tier A) if its estimated memory is
                below `max_mem_H` GB (per aggregate), else matrix-free (Tier B).
  * `:blocks` — force Tier A (fast, needs the block-sparse H to fit).
  * `:matrixfree` — force Tier B (minimal memory, slower).

Returns `(eigvals::Vector, ci_out::DistributedTPSCIstate{T,N,nroots})`. Call
`destroy!(ci_out)` when finished.
"""
function tps_ci_davidson_sharded(ci_vector::DistributedTPSCIstate{T,N,R},
                                 cluster_ops, clustered_ham::ClusteredOperator;
                                 nroots::Int=R,
                                 conv_thresh=1e-5,
                                 lindep_thresh=1e-12,
                                 max_ss_vecs=4,
                                 max_iter=100,
                                 h_storage::Symbol=:auto,
                                 max_mem_H=50.0,
                                 block_h::Union{Nothing,ShardedBlockH}=nothing,
                                 workers=ci_vector.workers,
                                 threaded_worker=true,
                                 blas_threads=1,
                                 verbose=1, id=nothing) where {T,N,R}
    nroots <= R || error("tps_ci_davidson_sharded needs the guess to carry at least nroots ($nroots) roots; got R=$R")
    worker_ids = ensure_tpsci_multinode_workers!(workers=workers)
    worker_ids == ci_vector.workers ||
        error("tps_ci_davidson_sharded requires the guess on the requested workers")

    if verbose > 0
        println()
        @printf(" |== Sharded Tensor Product State CI ===============================\n")
        @printf(" Hamiltonian matrix dimension = %i\n", length(ci_vector))
        @printf(" nroots = %i   max_ss_vecs = %i   conv_thresh = %.1e\n",
                nroots, max_ss_vecs, conv_thresh)
        flush(stdout)
    end

    owns_block_h = false
    if block_h !== nothing
        # Caller-managed stored H (e.g. tpsci_ci_sharded reusing/updating one
        # block-sparse H across selected-CI iterations). Use it, don't free it.
        block_h.workers == worker_ids ||
            error("supplied block_h lives on different workers than the guess")
        apply_H = v -> apply_sharded_H(block_h, v)
        Hdiag = compute_diagonal_sharded(block_h, ci_vector)
    else
        # Decide storage tier.
        tier = h_storage
        if tier == :auto
            tier, _, _ = _sharded_H_auto_tier(ci_vector, clustered_ham, worker_ids,
                                              max_mem_H; verbose=verbose)
        elseif tier == :blocks
            rep = sharded_H_memory_report(ci_vector, clustered_ham)
            fit = sharded_H_fit_report(rep, worker_ids)
            verbose > 0 && print_sharded_H_fit(rep, fit)
            fit.fits ||
                error("h_storage=:blocks needs $(round(rep.max_worker_gb; digits=2)) GB on " *
                      "worker $(rep.max_worker_pid), but worker $(fit.worst_pid) is short by " *
                      "$(round(fit.worst_deficit_gb; digits=2)) GB. Add nodes, put one worker " *
                      "per node, or use h_storage=:matrixfree.")
        end

        if tier == :blocks
            block_h = build_block_h_sharded(ci_vector, cluster_ops, clustered_ham;
                                            workers=worker_ids, blas_threads=blas_threads,
                                            verbose=verbose)
            owns_block_h = true
            apply_H = v -> apply_sharded_H(block_h, v)
            Hdiag = compute_diagonal_sharded(block_h, ci_vector)
        elseif tier == :matrixfree
            # Diagonal preconditioner also primes the operator cache on workers.
            Hdiag = compute_diagonal_sharded(ci_vector, cluster_ops, clustered_ham;
                                             workers=worker_ids,
                                             blas_threads=blas_threads)
            op = MatrixFreeShardedH(cluster_ops, clustered_ham, worker_ids,
                                    threaded_worker, Int(blas_threads))
            apply_H = v -> apply_sharded_H(op, v)
        else
            error("Unknown h_storage: $h_storage (use :auto, :blocks, or :matrixfree)")
        end
    end

    # Seed: extract nroots orthonormal guesses from the input state.
    seeds = DistributedTPSCIstate{T,N,1}[]
    for r in 1:nroots
        s = extract_root_sharded(ci_vector, r)
        nrm = _mgs_against!(s, seeds)
        nrm > lindep_thresh || error("initial guess root $r is linearly dependent")
        scale!(s, one(T) / nrm)
        push!(seeds, s)
    end

    tstart = time()
    eigvals, ritz = _tps_ci_davidson_sharded_core(seeds, apply_H, Hdiag;
                                                  nroots=nroots,
                                                  conv_thresh=Float64(conv_thresh),
                                                  max_ss_vecs=Int(max_ss_vecs),
                                                  max_iter=Int(max_iter),
                                                  lindep_thresh=Float64(lindep_thresh),
                                                  verbose=Int(verbose))
    elapsed = time() - tstart

    ci_out = combine_roots_sharded(ritz; id=id)
    for x in ritz
        destroy!(x)
    end
    destroy!(Hdiag)
    owns_block_h && destroy!(block_h)

    if verbose > 0
        @printf(" Sharded Davidson finished in %.2f s\n", elapsed)
        @printf(" %5s %18s\n", "Root", "Energy")
        for r in 1:nroots
            @printf(" %5i %18.10f\n", r, eigvals[r])
        end
        @printf(" ==================================================================|\n")
        flush(stdout)
    end
    return eigvals, ci_out
end
