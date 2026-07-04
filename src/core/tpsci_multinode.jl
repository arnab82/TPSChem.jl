"""
Multinode helpers for TPSCI.

These routines intentionally keep the existing TPSCI data structures and add a
distributed execution backend around the large job loops. This first backend
still replicates the TPSCI vector on each worker; its memory win comes from
avoiding dense Hamiltonian storage and from splitting FOIS/matvec scratch across
workers. A CI vector larger than per-node memory needs a sharded TPSCI state and
a distributed Davidson solver.

Intended production layout:
  * one Julia distributed worker per node for memory ownership
  * many Julia threads inside each worker for on-node CPU parallelism
  * a sharded `DistributedTPSCIstate` so `vec_var`/`vec_pt` are never gathered
    on one node during selected-CI growth
"""

const _TPSCI_MULTINODE_CI = Ref{Any}(nothing)
const _TPSCI_MULTINODE_CLUSTER_OPS = Ref{Any}(nothing)
const _TPSCI_MULTINODE_CLUSTERED_HAM = Ref{Any}(nothing)
const _TPSCI_SHARDED_STATES = Dict{Symbol,Any}()
const _TPSCI_SHARDED_CLUSTER_OPS = Dict{Symbol,Any}()
const _TPSCI_SHARDED_CLUSTER_H0_DIAGS = Dict{Tuple{Symbol,String},Any}()

"""
    DistributedTPSCIstate{T,N,R}

Metadata handle for a TPSCI vector whose coefficient/configuration blocks live
on distributed workers. The master stores only ownership, lengths, and offsets;
each worker stores an ordinary local `TPSCIstate` containing the Fock sectors it
owns.
"""
mutable struct DistributedTPSCIstate{T,N,R} <: AbstractState
    id::Symbol
    clusters::Vector{MOCluster}
    workers::Vector{Int}
    owners::OrderedDict{FockConfig{N},Int}
    lengths::OrderedDict{FockConfig{N},Int}
    offsets::OrderedDict{FockConfig{N},Int}
    local_lengths::OrderedDict{Int,Int}
    total_length::Int
end

Base.length(s::DistributedTPSCIstate) = s.total_length
Base.size(s::DistributedTPSCIstate{T,N,R}) where {T,N,R} = (s.total_length, R)
Base.haskey(s::DistributedTPSCIstate, fock) = haskey(s.owners, fock)

"""
    DistributedClusterOps{T}

Metadata handle for cluster-local operator tables whose `ClusterOps` payloads
live on distributed workers. Ownership is by cluster index, which matches the
natural independence of `compute_cluster_ops`.
"""
mutable struct DistributedClusterOps{T}
    id::Symbol
    clusters::Vector{MOCluster}
    workers::Vector{Int}
    owners::OrderedDict{Int,Int}
    local_cluster_indices::OrderedDict{Int,Vector{Int}}
    local_nbytes::OrderedDict{Int,Int}
    h_sectors::OrderedDict{Int,Set{Tuple{Int16,Int16}}}
end

Base.length(ops::DistributedClusterOps) = length(ops.clusters)

struct ClusterH0DiagonalCache{T}
    opstring::String
    data::Vector{Dict{Tuple{Int16,Int16},Vector{T}}}
end

function _cluster_ops_nbytes(ops_i::ClusterOps)
    nbytes = 0
    for (_, sectors) in ops_i.data
        for (_, array) in sectors
            nbytes += Base.summarysize(array)
        end
    end
    return nbytes
end

function _cluster_ops_h_sectors(ops_i::ClusterOps)
    sectors = Set{Tuple{Int16,Int16}}()
    if haskey(ops_i, "H")
        for key in keys(ops_i["H"])
            push!(sectors, key[1])
        end
    end
    return sectors
end

function _cluster_basis_work_estimate(cb)
    work = 0
    for (_, basis) in cb
        n = size(basis, 2)
        work += n * n
    end
    return max(work, 1)
end

function _cluster_ops_chunks(cluster_bases, pids; strategy::Symbol=:balanced)
    chunks = Dict(pid => Vector{eltype(cluster_bases)}() for pid in pids)
    owners = OrderedDict{Int,Int}()
    loads = Dict(pid => 0 for pid in pids)

    for cb in cluster_bases
        ci = cb.cluster
        pid = if strategy == :balanced
            _tpsci_least_loaded_pid(loads, pids)
        elseif strategy == :hash
            pids[Int(mod(hash(ci.idx), UInt(length(pids)))) + 1]
        else
            error("Unknown DistributedClusterOps sharding strategy: $strategy")
        end
        push!(chunks[pid], cb)
        owners[ci.idx] = pid
        loads[pid] += _cluster_basis_work_estimate(cb)
    end
    return chunks, owners
end

function _store_cluster_ops_chunk!(id::Symbol, local_ops)
    _TPSCI_SHARDED_CLUSTER_OPS[id] = local_ops
    return true
end

function _delete_cluster_ops_chunk!(id::Symbol)
    delete!(_TPSCI_SHARDED_CLUSTER_OPS, id)
    _clear_cluster_h0_diag_cache!(id)
    return true
end

function _get_cluster_ops_chunk(id::Symbol)
    haskey(_TPSCI_SHARDED_CLUSTER_OPS, id) ||
        error("No DistributedClusterOps chunk cached with id $id on worker $(Distributed.myid())")
    return _TPSCI_SHARDED_CLUSTER_OPS[id]
end

function _compute_cluster_ops_chunk!(id::Symbol, assigned_bases, ints;
                                     verbose=1, blas_threads=1)
    blas_threads === nothing || BLAS.set_num_threads(blas_threads)
    local_ops = Dict{Int,ClusterOps{typeof(ints.h0)}}()
    nbytes = OrderedDict{Int,Int}()
    h_sectors = OrderedDict{Int,Set{Tuple{Int16,Int16}}}()
    for cb in assigned_bases
        ops_i = _compute_cluster_ops_for_cluster(cb, ints, verbose=verbose)
        local_ops[cb.cluster.idx] = ops_i
        nbytes[cb.cluster.idx] = _cluster_ops_nbytes(ops_i)
        h_sectors[cb.cluster.idx] = _cluster_ops_h_sectors(ops_i)
    end
    _store_cluster_ops_chunk!(id, local_ops)
    return (nbytes=nbytes, h_sectors=h_sectors)
end

function _cluster_ops_local_summary(id::Symbol)
    local_ops = _get_cluster_ops_chunk(id)
    nbytes = 0
    indices = Int[]
    for (idx, ops_i) in local_ops
        push!(indices, idx)
        nbytes += _cluster_ops_nbytes(ops_i)
    end
    sort!(indices)
    return (Distributed.myid(), indices, nbytes)
end

"""
    compute_cluster_ops_distributed(cluster_bases, ints; workers=workers())

Build cluster-local operator tables on distributed workers, sharded by cluster
index. This avoids materializing the full `Vector{ClusterOps}` on every node.
Use `collect_cluster_ops` only for small debugging comparisons.
"""
function compute_cluster_ops_distributed(cluster_bases, ints::InCoreInts{T};
                                         workers=Distributed.workers(),
                                         id=nothing,
                                         strategy::Symbol=:balanced,
                                         verbose=1,
                                         blas_threads=1) where {T}
    pids = ensure_tpsci_multinode_workers!(workers=workers)
    ops_id = id === nothing ? gensym(:cluster_ops_shard) : Symbol(id)
    chunks, owners = _cluster_ops_chunks(cluster_bases, pids, strategy=strategy)
    clusters = [cb.cluster for cb in cluster_bases]

    metadata_maps = Dict{Int,Any}()
    @sync for pid in pids
        @async begin
            metadata_maps[pid] = Distributed.remotecall_fetch(_compute_cluster_ops_chunk!,
                                                              pid, ops_id, chunks[pid],
                                                              ints; verbose=verbose,
                                                              blas_threads=blas_threads)
        end
    end

    local_cluster_indices = OrderedDict{Int,Vector{Int}}()
    local_nbytes = OrderedDict{Int,Int}()
    h_sectors = OrderedDict{Int,Set{Tuple{Int16,Int16}}}()
    for pid in pids
        local_cluster_indices[pid] = sort([cb.cluster.idx for cb in chunks[pid]])
        local_nbytes[pid] = sum(values(metadata_maps[pid].nbytes); init=0)
        for (idx, sectors) in metadata_maps[pid].h_sectors
            h_sectors[idx] = sectors
        end
    end

    return DistributedClusterOps{T}(ops_id, clusters, pids, owners,
                                   local_cluster_indices, local_nbytes,
                                   h_sectors)
end

function cluster_ops_summary(ops::DistributedClusterOps)
    rows = []
    for pid in ops.workers
        push!(rows, Distributed.remotecall_fetch(_cluster_ops_local_summary,
                                                 pid, ops.id))
    end
    return rows
end

function destroy!(ops::DistributedClusterOps)
    @sync for pid in ops.workers
        @async Distributed.remotecall_fetch(_delete_cluster_ops_chunk!, pid, ops.id)
    end
    ops.owners = empty(ops.owners)
    ops.local_cluster_indices = empty(ops.local_cluster_indices)
    ops.local_nbytes = empty(ops.local_nbytes)
    ops.h_sectors = empty(ops.h_sectors)
    return ops
end

function _collect_cluster_ops_chunk(id::Symbol)
    local_ops = _get_cluster_ops_chunk(id)
    return Dict(idx => ops_i for (idx, ops_i) in local_ops)
end

function _cluster_ops_local_copy_index(id::Symbol, idx::Integer)
    idx = Int(idx)
    local_ops = _get_cluster_ops_chunk(id)
    haskey(local_ops, idx) ||
        error("Worker $(Distributed.myid()) does not own ClusterOps for cluster $idx")
    return local_ops[idx]
end

function _clear_cluster_h0_diag_cache!(id::Symbol)
    for key in collect(keys(_TPSCI_SHARDED_CLUSTER_H0_DIAGS))
        key[1] == id && delete!(_TPSCI_SHARDED_CLUSTER_H0_DIAGS, key)
    end
    return true
end

function _cluster_ops_h0_diagonal_index(ops_i::ClusterOps{T},
                                        opstring::String) where {T}
    haskey(ops_i, opstring) ||
        error("Cluster $(ops_i.cluster.idx) is missing operator $opstring")
    out = Dict{Tuple{Int16,Int16},Vector{T}}()
    for ((fock_l, fock_r), mat) in ops_i[opstring]
        fock_l == fock_r || continue
        out[fock_l] = LinearAlgebra.diag(mat)
    end
    return out
end

function _cluster_ops_local_h0_diagonal_index(id::Symbol, idx::Integer,
                                              opstring::String)
    return _cluster_ops_h0_diagonal_index(_cluster_ops_local_copy_index(id, idx),
                                          opstring)
end

function _materialize_cluster_h0_diagonals(ops::DistributedClusterOps{T},
                                           opstring::String) where {T}
    data = Vector{Dict{Tuple{Int16,Int16},Vector{T}}}(undef, length(ops.clusters))
    for c in ops.clusters
        owner = ops.owners[c.idx]
        data[c.idx] = if owner == Distributed.myid()
            _cluster_ops_h0_diagonal_index(_get_cluster_ops_chunk(ops.id)[c.idx],
                                           opstring)
        else
            Distributed.remotecall_fetch(_cluster_ops_local_h0_diagonal_index,
                                         owner, ops.id, Int(c.idx), opstring)
        end
    end
    return ClusterH0DiagonalCache{T}(opstring, data)
end

function _materialize_cluster_h0_diagonals_cached(ops::DistributedClusterOps,
                                                  opstring::String)
    key = (ops.id, opstring)
    if !haskey(_TPSCI_SHARDED_CLUSTER_H0_DIAGS, key)
        _TPSCI_SHARDED_CLUSTER_H0_DIAGS[key] =
            _materialize_cluster_h0_diagonals(ops, opstring)
    end
    return _TPSCI_SHARDED_CLUSTER_H0_DIAGS[key]
end

function _materialize_cluster_ops_for_indices(ops::DistributedClusterOps{T},
                                              needed_indices) where {T}
    local_ops = Vector{ClusterOps{T}}(undef, length(ops.clusters))
    needed = Set(needed_indices)
    for c in ops.clusters
        if c.idx in needed
            owner = ops.owners[c.idx]
            local_ops[c.idx] = if owner == Distributed.myid()
                _get_cluster_ops_chunk(ops.id)[c.idx]
            else
                Distributed.remotecall_fetch(_cluster_ops_local_copy_index,
                                             owner, ops.id, Int(c.idx))
            end
        else
            local_ops[c.idx] = ClusterOps(c, T=T)
        end
    end
    return local_ops
end

"""
    collect_cluster_ops(ops::DistributedClusterOps)

Debug/testing helper that gathers sharded cluster operator tables into the
traditional local `Vector{ClusterOps}`. Do not use for production runs whose
operator tables exceed node memory.
"""
function collect_cluster_ops(ops::DistributedClusterOps{T}) where {T}
    out = Vector{ClusterOps{T}}(undef, length(ops.clusters))
    for pid in ops.workers
        local_ops = Distributed.remotecall_fetch(_collect_cluster_ops_chunk,
                                                 pid, ops.id)
        for (idx, ops_i) in local_ops
            out[idx] = ops_i
        end
    end
    return out
end

function _add_cmf_operators_chunk!(id::Symbol, assigned_bases, ints, Da, Db;
                                   verbose=0, blas_threads=1)
    blas_threads === nothing || BLAS.set_num_threads(blas_threads)
    local_ops = _get_cluster_ops_chunk(id)
    nbytes = OrderedDict{Int,Int}()
    for cb in assigned_bases
        ci = cb.cluster
        haskey(local_ops, ci.idx) ||
            error("Worker $(Distributed.myid()) does not own ClusterOps for cluster $(ci.idx)")
        _add_cmf_operator_for_cluster!(local_ops[ci.idx], cb, ints, Da, Db;
                                       verbose=verbose)
        nbytes[ci.idx] = _cluster_ops_nbytes(local_ops[ci.idx])
    end
    return nbytes
end

"""
    add_cmf_operators_distributed!(ops, cluster_bases, ints, Da, Db)

Add `Hcmf` tables to sharded `DistributedClusterOps` in place, computing each
cluster's CMF operator only on the worker that owns that cluster.
"""
function add_cmf_operators_distributed!(ops::DistributedClusterOps, cluster_bases,
                                        ints, Da, Db; verbose=0,
                                        blas_threads=1)
    chunks = Dict(pid => Vector{eltype(cluster_bases)}() for pid in ops.workers)
    for cb in cluster_bases
        pid = ops.owners[cb.cluster.idx]
        push!(chunks[pid], cb)
    end

    nbyte_maps = Dict{Int,Any}()
    @sync for pid in ops.workers
        @async begin
            nbyte_maps[pid] = Distributed.remotecall_fetch(_add_cmf_operators_chunk!,
                                                           pid, ops.id, chunks[pid],
                                                           ints, Da, Db;
                                                           verbose=verbose,
                                                           blas_threads=blas_threads)
        end
    end

    for pid in ops.workers
        ops.local_nbytes[pid] = sum(values(nbyte_maps[pid]); init=0)
    end
    @sync for pid in ops.workers
        @async Distributed.remotecall_fetch(_clear_cluster_h0_diag_cache!, pid,
                                            ops.id)
    end
    return ops
end

function _tpsci_sharded_store_state!(id::Symbol, state)
    _TPSCI_SHARDED_STATES[id] = state
    return length(state)
end

function _tpsci_sharded_delete_state!(id::Symbol)
    delete!(_TPSCI_SHARDED_STATES, id)
    return true
end

function _tpsci_sharded_get_state(id::Symbol)
    haskey(_TPSCI_SHARDED_STATES, id) || error("No sharded TPSCI state cached with id $id on worker $(Distributed.myid())")
    return _TPSCI_SHARDED_STATES[id]
end

function _tpsci_sharded_local_fock_lengths(id::Symbol)
    state = _tpsci_sharded_get_state(id)
    out = OrderedDict()
    for (fock, configs) in state.data
        out[fock] = length(configs)
    end
    return out
end

function _tpsci_sharded_local_summary(id::Symbol)
    state = _tpsci_sharded_get_state(id)
    return (Distributed.myid(), length(state), length(state.data))
end

function _tpsci_copy_fock_block(configs::OrderedDict{ClusterConfig{N},MVector{R,T}}) where {T,N,R}
    out = OrderedDict{ClusterConfig{N},MVector{R,T}}()
    for (config, coeffs) in configs
        out[config] = copy(coeffs)
    end
    return out
end

function _tpsci_owner_for_fock(fock, pids, strategy::Symbol)
    strategy == :hash && return pids[Int(mod(hash(fock), UInt(length(pids)))) + 1]
    error("Unknown TPSCI sharding strategy: $strategy")
end

function _tpsci_least_loaded_pid(loads, pids)
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

function _tpsci_sharded_empty_local_state(::Type{T}, ::Val{R}, clusters) where {T,R}
    return TPSCIstate(clusters, T=T, R=R)
end

function _tpsci_sharded_metadata(id::Symbol, clusters, pids, local_states::Dict{Int,TPSCIstate{T,N,R}},
                                 ordered_focks) where {T,N,R}
    owners = OrderedDict{FockConfig{N},Int}()
    lengths = OrderedDict{FockConfig{N},Int}()
    offsets = OrderedDict{FockConfig{N},Int}()
    local_lengths = OrderedDict{Int,Int}(pid => 0 for pid in pids)

    offset = 1
    for fock in ordered_focks
        for pid in pids
            state = local_states[pid]
            haskey(state.data, fock) || continue
            len = length(state.data[fock])
            len == 0 && continue
            owners[fock] = pid
            lengths[fock] = len
            offsets[fock] = offset
            local_lengths[pid] += len
            offset += len
            break
        end
    end
    return DistributedTPSCIstate{T,N,R}(id, clusters, pids, owners, lengths,
                                       offsets, local_lengths, offset - 1)
end

"""
    distribute_tpsci_state(ci_vector; workers=workers(), id=nothing, strategy=:balanced)

Shard a local `TPSCIstate` across workers. With `strategy=:hash`, ownership is a
stable function of `FockConfig`, so future states with the same Fock sectors can
be distributed compatibly without sending a global vector to every node. With
`strategy=:balanced`, initial Fock sectors are assigned greedily by current
configuration counts; later selected-CI growth should preserve existing owners
and greedily assign new sectors.
"""
function distribute_tpsci_state(ci_vector::TPSCIstate{T,N,R};
                                workers=Distributed.workers(),
                                id=nothing,
                                strategy::Symbol=:balanced,
                                blas_threads=1) where {T,N,R}
    pids = ensure_tpsci_multinode_workers!(workers=workers)
    state_id = id === nothing ? gensym(:tpsci_shard) : Symbol(id)
    local_states = Dict(pid => TPSCIstate(ci_vector.clusters, T=T, R=R) for pid in pids)
    loads = Dict(pid => 0 for pid in pids)
    ordered_focks = FockConfig{N}[]

    for (fock, configs) in ci_vector.data
        push!(ordered_focks, fock)
        pid = if strategy == :balanced
            _tpsci_least_loaded_pid(loads, pids)
        else
            _tpsci_owner_for_fock(fock, pids, strategy)
        end
        local_states[pid].data[fock] = _tpsci_copy_fock_block(configs)
        loads[pid] += length(configs)
    end

    @sync for pid in pids
        @async begin
            if pid == Distributed.myid()
                _tpsci_sharded_store_state!(state_id, local_states[pid])
                blas_threads === nothing || BLAS.set_num_threads(blas_threads)
            else
                Distributed.remotecall_fetch(_tpsci_sharded_store_state!, pid,
                                             state_id, local_states[pid])
                blas_threads === nothing ||
                    Distributed.remotecall_fetch(BLAS.set_num_threads, pid, blas_threads)
            end
        end
    end

    return _tpsci_sharded_metadata(state_id, ci_vector.clusters, pids,
                                  local_states, ordered_focks)
end

function _tpsci_sharded_refresh_metadata!(state::DistributedTPSCIstate{T,N,R}) where {T,N,R}
    summaries = Dict{Int,OrderedDict{FockConfig{N},Int}}()
    @sync for pid in state.workers
        @async begin
            summaries[pid] = Distributed.remotecall_fetch(_tpsci_sharded_local_fock_lengths,
                                                          pid, state.id)
        end
    end

    owners = OrderedDict{FockConfig{N},Int}()
    lengths = OrderedDict{FockConfig{N},Int}()
    offsets = OrderedDict{FockConfig{N},Int}()
    local_lengths = OrderedDict{Int,Int}(pid => 0 for pid in state.workers)
    seen = Set{FockConfig{N}}()
    ordered = collect(keys(state.owners))

    for fock in ordered
        for pid in state.workers
            haskey(summaries[pid], fock) || continue
            len = summaries[pid][fock]
            len == 0 && continue
            push!(seen, fock)
            owners[fock] = pid
            lengths[fock] = len
            break
        end
    end

    for pid in state.workers
        for (fock, len) in summaries[pid]
            fock in seen && continue
            len == 0 && continue
            push!(seen, fock)
            owners[fock] = pid
            lengths[fock] = len
        end
    end

    offset = 1
    for (fock, len) in lengths
        offsets[fock] = offset
        local_lengths[owners[fock]] += len
        offset += len
    end

    state.owners = owners
    state.lengths = lengths
    state.offsets = offsets
    state.local_lengths = local_lengths
    state.total_length = offset - 1
    return state
end

function sharded_state_summary(state::DistributedTPSCIstate)
    rows = []
    for pid in state.workers
        push!(rows, Distributed.remotecall_fetch(_tpsci_sharded_local_summary,
                                                 pid, state.id))
    end
    return rows
end

function destroy!(state::DistributedTPSCIstate)
    @sync for pid in state.workers
        @async Distributed.remotecall_fetch(_tpsci_sharded_delete_state!, pid, state.id)
    end
    state.owners = empty(state.owners)
    state.lengths = empty(state.lengths)
    state.offsets = empty(state.offsets)
    state.local_lengths = empty(state.local_lengths)
    state.total_length = 0
    return state
end

function _tpsci_sharded_local_normsq(id::Symbol)
    state = _tpsci_sharded_get_state(id)
    return _tpsci_sharded_local_normsq_typed(state)
end

function _tpsci_sharded_local_normsq_typed(state::TPSCIstate{T,N,R}) where {T,N,R}
    norms = zeros(T, R)
    for (_, configs) in state.data
        for (_, coeffs) in configs
            for r in 1:R
                norms[r] += coeffs[r] * coeffs[r]
            end
        end
    end
    return norms
end

function LinearAlgebra.norm(state::DistributedTPSCIstate{T,N,R}) where {T,N,R}
    norms = zeros(T, R)
    for pid in state.workers
        norms .+= Distributed.remotecall_fetch(_tpsci_sharded_local_normsq,
                                               pid, state.id)
    end
    return sqrt.(norms)
end

function _tpsci_sharded_local_overlap(id1::Symbol, id2::Symbol)
    s1 = _tpsci_sharded_get_state(id1)
    s2 = _tpsci_sharded_get_state(id2)
    return overlap(s1, s2)
end

function overlap(s1::DistributedTPSCIstate{T,N,R}, s2::DistributedTPSCIstate{T,N,R}) where {T,N,R}
    s1.workers == s2.workers || error("DistributedTPSCIstate overlap currently requires identical worker lists")
    out = zeros(T, R, R)
    for pid in s1.workers
        out .+= Distributed.remotecall_fetch(_tpsci_sharded_local_overlap,
                                             pid, s1.id, s2.id)
    end
    return out
end

function _tpsci_sharded_local_zero!(id::Symbol)
    zero!(_tpsci_sharded_get_state(id))
    return true
end

function zero!(state::DistributedTPSCIstate)
    @sync for pid in state.workers
        @async Distributed.remotecall_fetch(_tpsci_sharded_local_zero!, pid, state.id)
    end
    return state
end

function _tpsci_sharded_local_clip!(id::Symbol, thresh)
    clip!(_tpsci_sharded_get_state(id), thresh=thresh)
    return true
end

function clip!(state::DistributedTPSCIstate; thresh=1e-5)
    @sync for pid in state.workers
        @async Distributed.remotecall_fetch(_tpsci_sharded_local_clip!, pid,
                                            state.id, thresh)
    end
    return _tpsci_sharded_refresh_metadata!(state)
end

function _tpsci_sharded_local_mult!(id::Symbol, C)
    mult!(_tpsci_sharded_get_state(id), C)
    return true
end

function orthonormalize!(state::DistributedTPSCIstate{T,N,R}) where {T,N,R}
    S = Symmetric(overlap(state, state))
    F = eigen(S)
    minimum(F.values) > eps(T) || error("Cannot orthonormalize sharded TPSCI state with linearly dependent roots")
    X = F.vectors * Diagonal(1 ./ sqrt.(F.values)) * F.vectors'
    @sync for pid in state.workers
        @async Distributed.remotecall_fetch(_tpsci_sharded_local_mult!, pid,
                                            state.id, X)
    end
    return state
end

function _tpsci_sharded_local_add!(dest_id::Symbol, src_id::Symbol)
    add!(_tpsci_sharded_get_state(dest_id), _tpsci_sharded_get_state(src_id))
    return true
end

function add!(dest::DistributedTPSCIstate{T,N,R}, src::DistributedTPSCIstate{T,N,R}) where {T,N,R}
    dest.workers == src.workers || error("DistributedTPSCIstate add! currently requires identical worker lists")
    for (fock, owner) in src.owners
        haskey(dest.owners, fock) && dest.owners[fock] != owner &&
            error("Cannot add sharded states with different owners for Fock sector $fock")
    end
    @sync for pid in dest.workers
        @async Distributed.remotecall_fetch(_tpsci_sharded_local_add!, pid,
                                            dest.id, src.id)
    end
    return _tpsci_sharded_refresh_metadata!(dest)
end

function _tpsci_sharded_local_project_out!(v_id::Symbol, w_id::Symbol)
    project_out!(_tpsci_sharded_get_state(v_id), _tpsci_sharded_get_state(w_id))
    return true
end

function project_out!(v::DistributedTPSCIstate{T,N,R}, w::DistributedTPSCIstate{T,N,R}) where {T,N,R}
    v.workers == w.workers || error("DistributedTPSCIstate project_out! currently requires identical worker lists")
    for (fock, owner_w) in w.owners
        haskey(v.owners, fock) || continue
        v.owners[fock] == owner_w ||
            error("Cannot project out Fock sector $fock because sharded states use different owners")
    end
    @sync for pid in v.workers
        @async Distributed.remotecall_fetch(_tpsci_sharded_local_project_out!, pid,
                                            v.id, w.id)
    end
    return _tpsci_sharded_refresh_metadata!(v)
end

"""
    collect_tpsci_state(state::DistributedTPSCIstate)

Debug/testing helper that gathers a sharded state back to one local
`TPSCIstate`. Do not use this on production states that exceed node memory.
"""
function collect_tpsci_state(state::DistributedTPSCIstate{T,N,R}) where {T,N,R}
    locals = Dict{Int,TPSCIstate{T,N,R}}()
    for pid in state.workers
        locals[pid] = Distributed.remotecall_fetch(_tpsci_sharded_get_state,
                                                   pid, state.id)
    end
    out = TPSCIstate(state.clusters, T=T, R=R)
    for (fock, pid) in state.owners
        haskey(locals[pid].data, fock) || continue
        out.data[fock] = _tpsci_copy_fock_block(locals[pid].data[fock])
    end
    return out
end

function _tpsci_sharded_set_operator_cache!(cluster_ops, clustered_ham; blas_threads=1)
    _TPSCI_MULTINODE_CLUSTER_OPS[] = cluster_ops
    _TPSCI_MULTINODE_CLUSTERED_HAM[] = clustered_ham
    blas_threads === nothing || BLAS.set_num_threads(blas_threads)
    return Distributed.myid()
end

function _tpsci_sharded_cache_operator_problem!(cluster_ops, clustered_ham;
                                                workers=Distributed.workers(),
                                                blas_threads=1)
    pids = ensure_tpsci_multinode_workers!(workers=workers)
    @sync for pid in pids
        @async begin
            if pid == Distributed.myid()
                _tpsci_sharded_set_operator_cache!(cluster_ops, clustered_ham;
                                                   blas_threads=blas_threads)
            else
                Distributed.remotecall_fetch(_tpsci_sharded_set_operator_cache!, pid,
                                             cluster_ops, clustered_ham;
                                             blas_threads=blas_threads)
            end
        end
    end
    return pids
end

function _tpsci_fois_fock_configs(ci_vector::DistributedTPSCIstate{T,N,R},
                                  cluster_ops,
                                  clustered_ham::ClusteredOperator) where {T,N,R}
    clusters = ci_vector.clusters
    fock_bras = FockConfig{N}[]
    seen = Set{FockConfig{N}}()
    for fock_ket in keys(ci_vector.owners)
        for (ftrans, _) in clustered_ham
            fock_bra = ftrans + fock_ket
            _tpsci_valid_fock(fock_bra, clusters) || continue
            _tpsci_has_cluster_space(fock_bra, clusters, cluster_ops) || continue
            fock_bra in seen && continue
            push!(seen, fock_bra)
            push!(fock_bras, fock_bra)
        end
    end
    return fock_bras
end

function _tpsci_sharded_local_copy_fock_block(id::Symbol, fock)
    state = _tpsci_sharded_get_state(id)
    haskey(state.data, fock) || return nothing
    return _tpsci_copy_fock_block(state.data[fock])
end

function _tpsci_sharded_get_ket_configs(input_state::DistributedTPSCIstate{T,N,R},
                                        fock_ket::FockConfig{N}, ket_cache) where {T,N,R}
    haskey(ket_cache, fock_ket) && return ket_cache[fock_ket]
    owner = input_state.owners[fock_ket]
    configs = if owner == Distributed.myid()
        _tpsci_sharded_get_state(input_state.id).data[fock_ket]
    else
        Distributed.remotecall_fetch(_tpsci_sharded_local_copy_fock_block,
                                     owner, input_state.id, fock_ket)
    end
    configs === nothing && error("Missing sharded ket Fock block $fock_ket on owner $owner")
    ket_cache[fock_ket] = configs
    return configs
end

function _tpsci_sharded_open_job_for_fock(input_state::DistributedTPSCIstate{T,N,R},
                                          clustered_ham, fock_bra::FockConfig{N},
                                          ket_cache) where {T,N,R}
    job = Tuple[]
    for fock_ket in keys(input_state.owners)
        fock_trans = fock_bra - fock_ket
        haskey(clustered_ham, fock_trans) || continue
        configs_ket = _tpsci_sharded_get_ket_configs(input_state, fock_ket, ket_cache)
        push!(job, (clustered_ham[fock_trans], fock_ket, configs_ket))
    end
    return job
end

function _tpsci_needed_cluster_indices(input_state::DistributedTPSCIstate{T,N,R},
                                       clustered_ham, fock_bras) where {T,N,R}
    needed = Set{Int}()
    for fock_bra in fock_bras
        for fock_ket in keys(input_state.owners)
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

function _tpsci_sharded_prefetch_ket_cache(input_state::DistributedTPSCIstate{T,N,R},
                                           clustered_ham, fock_bras) where {T,N,R}
    ket_cache = Dict{FockConfig{N},Any}()
    for fock_bra in fock_bras
        for fock_ket in keys(input_state.owners)
            fock_trans = fock_bra - fock_ket
            haskey(clustered_ham, fock_trans) || continue
            _tpsci_sharded_get_ket_configs(input_state, fock_ket, ket_cache)
        end
    end
    return ket_cache
end

function _tpsci_sharded_open_output_chunk_core!(input_state::DistributedTPSCIstate{T,N,R},
                                                output_id::Symbol, fock_bras,
                                                cluster_ops, clustered_ham,
                                                nbody, thresh, prescreen,
                                                threaded_worker) where {T,N,R}
    nt = threaded_worker ? Threads.maxthreadid() : 1
    partial = [TPSCIstate(input_state.clusters, T=T, R=R) for _ in 1:nt]
    ket_cache = _tpsci_sharded_prefetch_ket_cache(input_state, clustered_ham, fock_bras)
    ket_caches = [ket_cache for _ in 1:nt]
    scr_f, scr_i, scr_m = _tpsci_open_scratch(T, Val(N), nt)

    if threaded_worker
        Threads.@threads :static for idx in eachindex(fock_bras)
            tid = Threads.threadid()
            fock_bra = fock_bras[idx]
            job = _tpsci_sharded_open_job_for_fock(input_state, clustered_ham,
                                                   fock_bra, ket_caches[tid])
            _open_matvec_thread_job(job, fock_bra, cluster_ops, nbody, thresh,
                                    partial[tid], scr_f[tid], scr_i[tid], scr_m[tid],
                                    prescreen)
        end
    else
        for fock_bra in fock_bras
            job = _tpsci_sharded_open_job_for_fock(input_state, clustered_ham,
                                                   fock_bra, ket_caches[1])
            _open_matvec_thread_job(job, fock_bra, cluster_ops, nbody, thresh,
                                    partial[1], scr_f[1], scr_i[1], scr_m[1],
                                    prescreen)
        end
    end

    sig = TPSCIstate(input_state.clusters, T=T, R=R)
    for part in partial
        _tpsci_collect_fock_blocks!(sig, part)
    end
    _tpsci_sharded_store_state!(output_id, sig)
    return _tpsci_sharded_local_fock_lengths(output_id)
end

function _tpsci_sharded_open_output_chunk!(input_state::DistributedTPSCIstate{T,N,R},
                                           output_id::Symbol, fock_bras,
                                           nbody, thresh, prescreen,
                                           threaded_worker) where {T,N,R}
    cluster_ops = _TPSCI_MULTINODE_CLUSTER_OPS[]
    clustered_ham = _TPSCI_MULTINODE_CLUSTERED_HAM[]
    cluster_ops === nothing && error("TPSCI sharded operator cache is empty")
    clustered_ham === nothing && error("TPSCI sharded Hamiltonian cache is empty")
    return _tpsci_sharded_open_output_chunk_core!(input_state, output_id, fock_bras,
                                                  cluster_ops, clustered_ham,
                                                  nbody, thresh, prescreen,
                                                  threaded_worker)
end

function _tpsci_sharded_open_output_chunk_distops!(input_state::DistributedTPSCIstate{T,N,R},
                                                   output_id::Symbol, fock_bras,
                                                   cluster_ops::DistributedClusterOps,
                                                   clustered_ham,
                                                   nbody, thresh, prescreen,
                                                   threaded_worker) where {T,N,R}
    needed = _tpsci_needed_cluster_indices(input_state, clustered_ham, fock_bras)
    local_cluster_ops = _materialize_cluster_ops_for_indices(cluster_ops, needed)
    return _tpsci_sharded_open_output_chunk_core!(input_state, output_id, fock_bras,
                                                  local_cluster_ops, clustered_ham,
                                                  nbody, thresh, prescreen,
                                                  threaded_worker)
end

function _tpsci_sharded_metadata_from_lengths(id::Symbol, clusters, pids,
                                              length_maps::Dict{Int,Any},
                                              ordered_focks::Vector{FockConfig{N}},
                                              ::Type{T}, ::Val{R}) where {T,N,R}
    owners = OrderedDict{FockConfig{N},Int}()
    lengths = OrderedDict{FockConfig{N},Int}()
    offsets = OrderedDict{FockConfig{N},Int}()
    local_lengths = OrderedDict{Int,Int}(pid => 0 for pid in pids)

    offset = 1
    for fock in ordered_focks
        for pid in pids
            lenmap = length_maps[pid]
            haskey(lenmap, fock) || continue
            len = lenmap[fock]
            len == 0 && continue
            owners[fock] = pid
            lengths[fock] = len
            offsets[fock] = offset
            local_lengths[pid] += len
            offset += len
            break
        end
    end

    return DistributedTPSCIstate{T,N,R}(id, clusters, pids, owners, lengths,
                                       offsets, local_lengths, offset - 1)
end

function _tpsci_sharded_output_chunks(input_state::DistributedTPSCIstate{T,N,R},
                                      fock_bras::Vector{FockConfig{N}}, pids,
                                      strategy::Symbol) where {T,N,R}
    chunks = Dict(pid => FockConfig{N}[] for pid in pids)
    assigned = OrderedDict{FockConfig{N},Int}()
    loads = Dict(pid => get(input_state.local_lengths, pid, 0) for pid in pids)

    for fock in fock_bras
        pid = if haskey(input_state.owners, fock)
            input_state.owners[fock]
        elseif strategy == :balanced
            _tpsci_least_loaded_pid(loads, pids)
        else
            _tpsci_owner_for_fock(fock, pids, strategy)
        end
        assigned[fock] = pid
        push!(chunks[pid], fock)
        loads[pid] += 1
    end

    return chunks, assigned
end

"""
    open_matvec_sharded(ci_vector::DistributedTPSCIstate, cluster_ops, clustered_ham; ...)

Construct the open TPSCI matvec/FOIS vector while keeping the input CI vector
sharded. Each output Fock sector is assigned to its owner worker, and that
worker fetches only the connected ket Fock blocks needed for its local jobs.
"""
function open_matvec_sharded(ci_vector::DistributedTPSCIstate{T,N,R}, cluster_ops,
                             clustered_ham::ClusteredOperator;
                             thresh=1e-9,
                             prescreen=true,
                             nbody=4,
                             workers=ci_vector.workers,
                             threaded_worker=true,
                             blas_threads=1,
                             id=nothing,
                             strategy::Symbol=:balanced) where {T,N,R}
    pids = _tpsci_sharded_cache_operator_problem!(cluster_ops, clustered_ham;
                                                  workers=workers,
                                                  blas_threads=blas_threads)
    pids == ci_vector.workers || error("open_matvec_sharded currently requires the same worker list as the input state")
    output_id = id === nothing ? gensym(:tpsci_shard_sig) : Symbol(id)
    fock_bras = _tpsci_fois_fock_configs(ci_vector, cluster_ops, clustered_ham)
    chunks, _ = _tpsci_sharded_output_chunks(ci_vector, fock_bras, pids, strategy)

    println(" In open_matvec_sharded")
    println(" Number of output Fock jobs: ", length(fock_bras))
    println(" Number of workers:          ", length(pids))
    println(" Threads per worker enabled: ", threaded_worker)
    flush(stdout)

    length_maps = Dict{Int,Any}()
    @sync for pid in pids
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(
                _tpsci_sharded_open_output_chunk!, pid, ci_vector, output_id,
                chunks[pid], nbody, thresh, prescreen, threaded_worker)
        end
    end

    return _tpsci_sharded_metadata_from_lengths(output_id, ci_vector.clusters,
                                                pids, length_maps, fock_bras,
                                                T, Val(R))
end

function open_matvec_sharded(ci_vector::DistributedTPSCIstate{T,N,R},
                             cluster_ops::DistributedClusterOps{T},
                             clustered_ham::ClusteredOperator;
                             thresh=1e-9,
                             prescreen=true,
                             nbody=4,
                             workers=ci_vector.workers,
                             threaded_worker=true,
                             blas_threads=1,
                             id=nothing,
                             strategy::Symbol=:balanced) where {T,N,R}
    pids = ensure_tpsci_multinode_workers!(workers=workers)
    pids == ci_vector.workers || error("open_matvec_sharded currently requires the same worker list as the input state")
    pids == cluster_ops.workers || error("open_matvec_sharded currently requires DistributedClusterOps on the same worker list")
    output_id = id === nothing ? gensym(:tpsci_shard_sig) : Symbol(id)
    fock_bras = _tpsci_fois_fock_configs(ci_vector, cluster_ops, clustered_ham)
    chunks, _ = _tpsci_sharded_output_chunks(ci_vector, fock_bras, pids, strategy)

    println(" In open_matvec_sharded with DistributedClusterOps")
    println(" Number of output Fock jobs: ", length(fock_bras))
    println(" Number of workers:          ", length(pids))
    println(" Threads per worker enabled: ", threaded_worker)
    flush(stdout)

    length_maps = Dict{Int,Any}()
    @sync for pid in pids
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(
                _tpsci_sharded_open_output_chunk_distops!, pid, ci_vector,
                output_id, chunks[pid], cluster_ops, clustered_ham,
                nbody, thresh, prescreen, threaded_worker)
        end
    end

    return _tpsci_sharded_metadata_from_lengths(output_id, ci_vector.clusters,
                                                pids, length_maps, fock_bras,
                                                T, Val(R))
end

function _tpsci_sharded_matvec_config!(sig_out::MVector{R,T},
                                       input_state::DistributedTPSCIstate{T,N,R},
                                       clustered_ham, cluster_ops,
                                       fock_bra::FockConfig{N},
                                       config_bra::ClusterConfig{N},
                                       ket_cache) where {T,N,R}
    for (fock_trans, terms) in clustered_ham
        fock_ket = fock_bra - fock_trans
        haskey(input_state.owners, fock_ket) || continue
        configs_ket = _tpsci_sharded_get_ket_configs(input_state, fock_ket,
                                                     ket_cache)
        for (config_ket, coeff_ket) in configs_ket
            for term in terms
                check_term(term, fock_bra, config_bra, fock_ket, config_ket) ||
                    continue
                me = contract_matrix_element(term, cluster_ops, fock_bra,
                                             config_bra, fock_ket, config_ket)
                @simd for r in 1:R
                    @inbounds sig_out[r] += me * coeff_ket[r]
                end
            end
        end
    end
    return sig_out
end

function _tpsci_sharded_matvec_output_chunk_core!(input_state::DistributedTPSCIstate{T,N,R},
                                                  output_id::Symbol, fock_bras,
                                                  cluster_ops, clustered_ham,
                                                  threaded_worker) where {T,N,R}
    local_state = _tpsci_sharded_get_state(input_state.id)
    sig = TPSCIstate(local_state, T=T, R=R)
    jobs = Tuple{FockConfig{N},ClusterConfig{N}}[]
    for fock_bra in fock_bras
        haskey(local_state.data, fock_bra) || continue
        for (config_bra, _) in local_state.data[fock_bra]
            push!(jobs, (fock_bra, config_bra))
        end
    end

    ket_cache = _tpsci_sharded_prefetch_ket_cache(input_state, clustered_ham,
                                                  fock_bras)
    ket_caches = [ket_cache for _ in 1:(threaded_worker ? Threads.maxthreadid() : 1)]

    if threaded_worker
        Threads.@threads :static for idx in eachindex(jobs)
            tid = Threads.threadid()
            fock_bra, config_bra = jobs[idx]
            _tpsci_sharded_matvec_config!(sig.data[fock_bra][config_bra],
                                          input_state, clustered_ham, cluster_ops,
                                          fock_bra, config_bra, ket_caches[tid])
        end
    else
        for (fock_bra, config_bra) in jobs
            _tpsci_sharded_matvec_config!(sig.data[fock_bra][config_bra],
                                          input_state, clustered_ham, cluster_ops,
                                          fock_bra, config_bra, ket_caches[1])
        end
    end

    _tpsci_sharded_store_state!(output_id, sig)
    return _tpsci_sharded_local_fock_lengths(output_id)
end

function _tpsci_sharded_matvec_output_chunk!(input_state::DistributedTPSCIstate{T,N,R},
                                             output_id::Symbol, fock_bras,
                                             threaded_worker) where {T,N,R}
    cluster_ops = _TPSCI_MULTINODE_CLUSTER_OPS[]
    clustered_ham = _TPSCI_MULTINODE_CLUSTERED_HAM[]
    cluster_ops === nothing && error("TPSCI sharded operator cache is empty")
    clustered_ham === nothing && error("TPSCI sharded Hamiltonian cache is empty")
    return _tpsci_sharded_matvec_output_chunk_core!(input_state, output_id,
                                                    fock_bras, cluster_ops,
                                                    clustered_ham,
                                                    threaded_worker)
end

function _tpsci_sharded_matvec_output_chunk_distops!(input_state::DistributedTPSCIstate{T,N,R},
                                                     output_id::Symbol, fock_bras,
                                                     cluster_ops::DistributedClusterOps,
                                                     clustered_ham,
                                                     threaded_worker) where {T,N,R}
    needed = _tpsci_needed_cluster_indices(input_state, clustered_ham, fock_bras)
    local_cluster_ops = _materialize_cluster_ops_for_indices(cluster_ops, needed)
    return _tpsci_sharded_matvec_output_chunk_core!(input_state, output_id,
                                                    fock_bras, local_cluster_ops,
                                                    clustered_ham,
                                                    threaded_worker)
end

function _tpsci_owned_fock_chunks(state::DistributedTPSCIstate{T,N,R}, pids) where {T,N,R}
    chunks = Dict(pid => FockConfig{N}[] for pid in pids)
    for (fock, owner) in state.owners
        push!(chunks[owner], fock)
    end
    return chunks
end

"""
    tps_ci_matvec_sharded(ci_vector, cluster_ops, clustered_ham; ...)

Apply the clustered Hamiltonian inside the current sharded TPSCI variational
space. The returned `DistributedTPSCIstate` has the same Fock sectors and
ownership as the input state, so no full CI vector is gathered on the master.
"""
function tps_ci_matvec_sharded(ci_vector::DistributedTPSCIstate{T,N,R},
                               cluster_ops, clustered_ham::ClusteredOperator;
                               workers=ci_vector.workers,
                               threaded_worker=true,
                               blas_threads=1,
                               id=nothing) where {T,N,R}
    pids = _tpsci_sharded_cache_operator_problem!(cluster_ops, clustered_ham;
                                                  workers=workers,
                                                  blas_threads=blas_threads)
    pids == ci_vector.workers || error("tps_ci_matvec_sharded currently requires the same worker list as the input state")
    output_id = id === nothing ? gensym(:tpsci_shard_hv) : Symbol(id)
    chunks = _tpsci_owned_fock_chunks(ci_vector, pids)

    println(" In tps_ci_matvec_sharded")
    println(" Number of input Fock blocks: ", length(ci_vector.owners))
    println(" Number of workers:           ", length(pids))
    println(" Threads per worker enabled:  ", threaded_worker)
    flush(stdout)

    length_maps = Dict{Int,Any}()
    @sync for pid in pids
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(
                _tpsci_sharded_matvec_output_chunk!, pid, ci_vector, output_id,
                chunks[pid], threaded_worker)
        end
    end

    return _tpsci_sharded_metadata_from_lengths(output_id, ci_vector.clusters,
                                                pids, length_maps,
                                                collect(keys(ci_vector.owners)),
                                                T, Val(R))
end

function tps_ci_matvec_sharded(ci_vector::DistributedTPSCIstate{T,N,R},
                               cluster_ops::DistributedClusterOps{T},
                               clustered_ham::ClusteredOperator;
                               workers=ci_vector.workers,
                               threaded_worker=true,
                               blas_threads=1,
                               id=nothing) where {T,N,R}
    pids = ensure_tpsci_multinode_workers!(workers=workers)
    pids == ci_vector.workers || error("tps_ci_matvec_sharded currently requires the same worker list as the input state")
    pids == cluster_ops.workers || error("tps_ci_matvec_sharded currently requires DistributedClusterOps on the same worker list")
    if blas_threads !== nothing
        @sync for pid in pids
            @async Distributed.remotecall_fetch(BLAS.set_num_threads, pid,
                                                blas_threads)
        end
    end
    output_id = id === nothing ? gensym(:tpsci_shard_hv) : Symbol(id)
    chunks = _tpsci_owned_fock_chunks(ci_vector, pids)

    println(" In tps_ci_matvec_sharded with DistributedClusterOps")
    println(" Number of input Fock blocks: ", length(ci_vector.owners))
    println(" Number of workers:           ", length(pids))
    println(" Threads per worker enabled:  ", threaded_worker)
    flush(stdout)

    length_maps = Dict{Int,Any}()
    @sync for pid in pids
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(
                _tpsci_sharded_matvec_output_chunk_distops!, pid, ci_vector,
                output_id, chunks[pid], cluster_ops, clustered_ham,
                threaded_worker)
        end
    end

    return _tpsci_sharded_metadata_from_lengths(output_id, ci_vector.clusters,
                                                pids, length_maps,
                                                collect(keys(ci_vector.owners)),
                                                T, Val(R))
end

function _tpsci_sharded_local_copy_state!(dest_id::Symbol, src_id::Symbol)
    _tpsci_sharded_store_state!(dest_id, deepcopy(_tpsci_sharded_get_state(src_id)))
    return _tpsci_sharded_local_fock_lengths(dest_id)
end

function copy_sharded_state(state::DistributedTPSCIstate{T,N,R};
                            id=nothing) where {T,N,R}
    output_id = id === nothing ? gensym(:tpsci_shard_copy) : Symbol(id)
    length_maps = Dict{Int,Any}()
    @sync for pid in state.workers
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(
                _tpsci_sharded_local_copy_state!, pid, output_id, state.id)
        end
    end
    return _tpsci_sharded_metadata_from_lengths(output_id, state.clusters,
                                                state.workers, length_maps,
                                                collect(keys(state.owners)),
                                                T, Val(R))
end

function _tpsci_sharded_local_similar_state!(dest_id::Symbol, src_id::Symbol,
                                             ::Type{T}, roots::Integer) where {T}
    src = _tpsci_sharded_get_state(src_id)
    _tpsci_sharded_store_state!(dest_id, TPSCIstate(src, T=T, R=Int(roots)))
    return _tpsci_sharded_local_fock_lengths(dest_id)
end

function similar_sharded_state(state::DistributedTPSCIstate{T,N,R};
                               roots::Integer=R,
                               id=nothing) where {T,N,R}
    roots = Int(roots)
    output_id = id === nothing ? gensym(:tpsci_shard_similar) : Symbol(id)
    length_maps = Dict{Int,Any}()
    @sync for pid in state.workers
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(
                _tpsci_sharded_local_similar_state!, pid, output_id, state.id,
                T, roots)
        end
    end
    return _tpsci_sharded_metadata_from_lengths(output_id, state.clusters,
                                                state.workers, length_maps,
                                                collect(keys(state.owners)),
                                                T, Val(roots))
end

function _tpsci_sharded_local_scale!(id::Symbol, alpha)
    state = _tpsci_sharded_get_state(id)
    for (_, configs) in state.data
        for (_, coeffs) in configs
            coeffs .*= alpha
        end
    end
    return true
end

function scale!(state::DistributedTPSCIstate, alpha)
    @sync for pid in state.workers
        @async Distributed.remotecall_fetch(_tpsci_sharded_local_scale!, pid,
                                            state.id, alpha)
    end
    return state
end

function _tpsci_sharded_local_add_scaled!(dest_id::Symbol, alpha, src_id::Symbol)
    dest = _tpsci_sharded_get_state(dest_id)
    src = _tpsci_sharded_get_state(src_id)
    for (fock, configs) in src.data
        haskey(dest.data, fock) ||
            error("Destination sharded state is missing Fock sector $fock")
        for (config, coeffs) in configs
            haskey(dest.data[fock], config) ||
                error("Destination sharded state is missing config $config in Fock sector $fock")
            dest.data[fock][config] .+= alpha .* coeffs
        end
    end
    return true
end

function add_scaled!(dest::DistributedTPSCIstate{T,N,R}, alpha,
                     src::DistributedTPSCIstate{T,N,R}) where {T,N,R}
    dest.workers == src.workers ||
        error("add_scaled! requires identical worker lists")
    for (fock, owner) in src.owners
        haskey(dest.owners, fock) && dest.owners[fock] == owner ||
            error("add_scaled! requires matching Fock ownership")
    end
    @sync for pid in dest.workers
        @async Distributed.remotecall_fetch(_tpsci_sharded_local_add_scaled!,
                                            pid, dest.id, alpha, src.id)
    end
    return dest
end

function _tpsci_sharded_local_lincomb!(dest_id::Symbol, alpha,
                                       src_id::Symbol, beta)
    dest = _tpsci_sharded_get_state(dest_id)
    src = _tpsci_sharded_get_state(src_id)
    for (fock, configs) in dest.data
        haskey(src.data, fock) ||
            error("Source sharded state is missing Fock sector $fock")
        for (config, coeffs) in configs
            haskey(src.data[fock], config) ||
                error("Source sharded state is missing config $config in Fock sector $fock")
            coeffs .= alpha .* coeffs .+ beta .* src.data[fock][config]
        end
    end
    return true
end

function linear_combination!(dest::DistributedTPSCIstate{T,N,R}, alpha,
                             src::DistributedTPSCIstate{T,N,R}, beta) where {T,N,R}
    dest.workers == src.workers ||
        error("linear_combination! requires identical worker lists")
    for (fock, owner) in dest.owners
        haskey(src.owners, fock) && src.owners[fock] == owner ||
            error("linear_combination! requires matching Fock ownership")
    end
    @sync for pid in dest.workers
        @async Distributed.remotecall_fetch(_tpsci_sharded_local_lincomb!,
                                            pid, dest.id, alpha, src.id, beta)
    end
    return dest
end

function _tpsci_sharded_restrict_to_basis_chunk!(output_id::Symbol,
                                                 source::DistributedTPSCIstate{T,N,Rs},
                                                 basis::DistributedTPSCIstate{Tb,N,Rb}) where {T,Tb,N,Rs,Rb}
    basis_local = _tpsci_sharded_get_state(basis.id)
    out = TPSCIstate(basis.clusters, T=T, R=Rs)
    source_local = haskey(_TPSCI_SHARDED_STATES, source.id) ?
        _tpsci_sharded_get_state(source.id) : nothing
    source_cache = Dict{FockConfig{N},Any}()

    for (fock, basis_configs) in basis_local.data
        add_fockconfig!(out, fock)
        source_configs = nothing
        if haskey(source.owners, fock)
            owner = source.owners[fock]
            source_configs = if owner == Distributed.myid()
                source_local.data[fock]
            else
                get!(source_cache, fock) do
                    Distributed.remotecall_fetch(_tpsci_sharded_local_copy_fock_block,
                                                 owner, source.id, fock)
                end
            end
        end
        for (config, _) in basis_configs
            out[fock][config] =
                source_configs !== nothing && haskey(source_configs, config) ?
                copy(source_configs[config]) : zeros(T, Rs)
        end
    end

    _tpsci_sharded_store_state!(output_id, out)
    return _tpsci_sharded_local_fock_lengths(output_id)
end

"""
    restrict_to_basis_sharded(source, basis)

Return a sharded state on `basis`'s Fock/configuration space, copying
coefficients from `source` where present and inserting zeros elsewhere.
Ownership follows `basis`, which is useful for Q-space projections.
"""
function restrict_to_basis_sharded(source::DistributedTPSCIstate{T,N,Rs},
                                   basis::DistributedTPSCIstate{Tb,N,Rb};
                                   id=nothing) where {T,Tb,N,Rs,Rb}
    source.workers == basis.workers ||
        error("restrict_to_basis_sharded requires identical worker lists")
    output_id = id === nothing ? gensym(:tpsci_shard_restrict) : Symbol(id)
    length_maps = Dict{Int,Any}()
    @sync for pid in basis.workers
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(
                _tpsci_sharded_restrict_to_basis_chunk!, pid, output_id,
                source, basis)
        end
    end
    return _tpsci_sharded_metadata_from_lengths(output_id, basis.clusters,
                                                basis.workers, length_maps,
                                                collect(keys(basis.owners)),
                                                T, Val(Rs))
end

function _tpsci_cepa_apply_shifted_hq(v::DistributedTPSCIstate{T,N,1},
                                      cluster_ops, clustered_ham, eshift;
                                      workers=v.workers,
                                      threaded_worker=true,
                                      blas_threads=1) where {T,N}
    out = tps_ci_matvec_sharded(v, cluster_ops, clustered_ham;
                                workers=workers,
                                threaded_worker=threaded_worker,
                                blas_threads=blas_threads)
    add_scaled!(out, -eshift, v)
    return out
end

function _tpsci_sharded_cg_linsolve(b::DistributedTPSCIstate{T,N,1},
                                    cluster_ops, clustered_ham;
                                    eshift,
                                    tol=1e-5,
                                    maxiter=300,
                                    workers=b.workers,
                                    threaded_worker=true,
                                    blas_threads=1,
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
        Ap = _tpsci_cepa_apply_shifted_hq(p, cluster_ops, clustered_ham, eshift;
                                          workers=workers,
                                          threaded_worker=threaded_worker,
                                          blas_threads=blas_threads)
        pAp = overlap(p, Ap)[1, 1]
        abs(pAp) > eps(T) ||
            error("Sharded CEPA CG encountered a near-zero denominator")
        alpha = rr / pAp
        add_scaled!(x, alpha, p)
        add_scaled!(r, -alpha, Ap)
        destroy!(Ap)

        rr_new = overlap(r, r)[1, 1]
        resid = sqrt(abs(rr_new))
        if verbose > 1
            @printf("   sharded CG iter %4i residual %12.4e\n", it, resid)
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

function _tpsci_givens(a::T, b::T) where {T}
    if iszero(b)
        return one(T), zero(T), a
    elseif iszero(a)
        return zero(T), one(T), b
    else
        r = hypot(a, b)
        return a / r, b / r, r
    end
end

function _tpsci_sharded_minres_linsolve(b::DistributedTPSCIstate{T,N,1},
                                        cluster_ops, clustered_ham;
                                        eshift,
                                        tol=1e-5,
                                        abstol=zero(T),
                                        maxiter=300,
                                        workers=b.workers,
                                        threaded_worker=true,
                                        blas_threads=1,
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
        v_next = _tpsci_cepa_apply_shifted_hq(v_curr, cluster_ops,
                                              clustered_ham, eshift;
                                              workers=workers,
                                              threaded_worker=threaded_worker,
                                              blas_threads=blas_threads)
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
            error("Sharded CEPA MINRES encountered a near-zero Hessenberg pivot")
        scale!(w_next, inv(H[3]))

        add_scaled!(x, rhs[1], w_next)

        v_prev, v_curr, v_next = v_curr, v_next, v_prev
        w_prev, w_curr, w_next = w_curr, w_next, w_prev
        c_prev, s_prev, c_curr, s_curr = c_curr, s_curr, c, s
        rhs[1] = rhs[2]
        H[2] = H[4]

        resnorm = abs(rhs[2])
        if verbose > 1
            @printf("   sharded MINRES iter %4i residual %12.4e\n", it, resnorm)
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

"""
    tpsci_cepa_solve_sharded(ref_vector, e0, cepa_vector, cluster_ops, clustered_ham; ...)

Solve the TPSCI CEPA equations with the Q-space amplitudes stored as sharded
`DistributedTPSCIstate`s. This avoids gathering the CEPA vector on the master.
"""
function tpsci_cepa_solve_sharded(ref_vector::TPSCIstate{T,N,R}, e0::Vector,
                                  cepa_vector::DistributedTPSCIstate{T,N,Rq},
                                  cluster_ops, clustered_ham,
                                  cepa_shift="cepa",
                                  cepa_mit=50;
                                  tol=1e-5,
                                  cg_maxiter=300,
                                  nbody=4,
                                  thresh_sigma=1e-8,
                                  solver=:minres,
                                  workers=cepa_vector.workers,
                                  threaded_worker=true,
                                  blas_threads=1,
                                  verbose=0) where {T,N,R,Rq}
    worker_ids = ensure_tpsci_multinode_workers!(workers=workers)
    worker_ids == cepa_vector.workers ||
        error("tpsci_cepa_solve_sharded requires the CEPA vector on the requested workers")
    n_clusters = length(ref_vector.clusters)

    @printf(" Sharded CEPA solver: dim_q=%i, R=%i, shift=%s\n",
            length(cepa_vector), R, cepa_shift)

    Ec = zeros(T, R)

    for i in 1:R
        @printf(" Compute sharded coupling vector h for root %i\n", i)
        ref_i = extract_chosen_root(ref_vector, i)
        dref_i = distribute_tpsci_state(ref_i; workers=worker_ids,
                                        strategy=:balanced,
                                        blas_threads=blas_threads)
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
                Cd_i, history = _tpsci_sharded_minres_linsolve(
                    rhs, cluster_ops, clustered_ham;
                    eshift=e0[i] + shift,
                    tol=tol,
                    maxiter=cg_maxiter,
                    workers=worker_ids,
                    threaded_worker=threaded_worker,
                    blas_threads=blas_threads,
                    verbose=verbose)
            elseif solver == :cg || solver == :krylov
                Cd_i, history = _tpsci_sharded_cg_linsolve(
                    rhs, cluster_ops, clustered_ham;
                    eshift=e0[i] + shift,
                    tol=tol,
                    maxiter=cg_maxiter,
                    workers=worker_ids,
                    threaded_worker=threaded_worker,
                    blas_threads=blas_threads,
                    verbose=verbose)
            else
                error("Unknown sharded CEPA solver: $solver")
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
    return Ec, e0 .+ Ec
end

"""
    do_fois_cepa_sharded(ref, cluster_ops, clustered_ham; ...)

Build the CEPA FOIS and solve CEPA with the Q-space vector sharded across
workers. The returned second object is a `DistributedTPSCIstate`; call
`destroy!` on it when finished.
"""
function do_fois_cepa_sharded(ref::TPSCIstate{T,N,R}, cluster_ops, clustered_ham;
                              cepa_shift="cepa",
                              cepa_mit=30,
                              nbody=4,
                              thresh_foi=1e-6,
                              thresh_clip=1e-5,
                              tol=1e-8,
                              thresh_sigma=1e-8,
                              compress=false,
                              reference_solver=:direct,
                              solver=:minres,
                              e0=nothing,
                              ci_max_iter=50,
                              ci_max_ss_vecs=12,
                              ci_lindep_thresh=1e-12,
                              cg_maxiter=300,
                              workers=Distributed.workers(),
                              threaded_worker=true,
                              blas_threads=1,
                              verbose=1) where {T,N,R}
    worker_ids = ensure_tpsci_multinode_workers!(workers=workers)
    @printf("\n-------------------------------------------------------\n")
    @printf(" Do Sharded CEPA\n")
    @printf("   thresh_foi              = %-8.1e\n", thresh_foi)
    @printf("   nbody                   = %-i\n", nbody)
    @printf("\n")
    @printf("   Length of Reference     = %-i\n", length(ref))
    @printf("   Calculation type        = %s\n", cepa_shift)
    @printf("   Workers                 = %s\n", worker_ids)
    @printf("\n-------------------------------------------------------\n")

    ref_vec = deepcopy(ref)
    if e0 === nothing
        cluster_ops isa DistributedClusterOps &&
            error("do_fois_cepa_sharded with DistributedClusterOps needs a pre-solved reference and e0 keyword, or a future sharded Davidson reference solve")
        @printf(" Solve zeroth-order problem. Dimension = %10i\n", length(ref_vec))
        if reference_solver == :direct
            @time e0, ref_vec = tps_ci_direct(ref_vec, cluster_ops, clustered_ham,
                                              conv_thresh=tol)
        elseif reference_solver == :distributed_davidson
            orthonormalize!(ref_vec)
            @time e0, ref_vec = tps_ci_davidson_distributed(
                ref_vec, cluster_ops, clustered_ham;
                conv_thresh=tol,
                max_iter=ci_max_iter,
                max_ss_vecs=ci_max_ss_vecs,
                lindep_thresh=ci_lindep_thresh,
                workers=worker_ids,
                blas_threads=blas_threads)
        else
            error("Unknown reference_solver: $reference_solver")
        end
    else
        e0 = Vector{T}(e0)
    end

    println()
    println(" Compute sharded FOIS. Reference space dim = ", length(ref_vec))
    dref = distribute_tpsci_state(ref_vec; workers=worker_ids,
                                  strategy=:balanced,
                                  blas_threads=blas_threads)
    pt1_vec = open_matvec_sharded(dref, cluster_ops, clustered_ham;
                                  nbody=nbody,
                                  thresh=thresh_foi,
                                  prescreen=false,
                                  workers=worker_ids,
                                  threaded_worker=threaded_worker,
                                  blas_threads=blas_threads)
    project_out!(pt1_vec, dref)

    if compress
        norms1 = norm(pt1_vec)
        dim1 = length(pt1_vec)
        clip!(pt1_vec, thresh=thresh_clip)
        norms2 = norm(pt1_vec)
        dim2 = length(pt1_vec)
        @printf(" %-50s%10i -> %-10i (thresh = %8.1e)\n",
                "FOIS Compressed from: ", dim1, dim2, thresh_clip)
        for i in 1:R
            @printf(" %-50s%10.2e -> %-10.2e\n",
                    "Norm of |1>: ", norms1[i], norms2[i])
        end
    end

    S10 = overlap(pt1_vec, dref)
    for i in 1:R
        @printf(" %-50s%10.6f\n", "Overlap between <1|0>: ", S10[i, i])
    end
    destroy!(dref)

    println()
    println(" Do sharded CEPA: shared FOIS dim = ", length(pt1_vec))
    @time Ec, e_cepa = tpsci_cepa_solve_sharded(
        ref_vec, e0, pt1_vec, cluster_ops, clustered_ham,
        cepa_shift, cepa_mit;
        tol=tol,
        cg_maxiter=cg_maxiter,
        nbody=nbody,
        thresh_sigma=thresh_sigma,
        solver=solver,
        workers=worker_ids,
        threaded_worker=threaded_worker,
        blas_threads=blas_threads,
        verbose=verbose)

    for i in 1:R
        @printf(" E(cepa) root %i  corr= %12.8f  total= %12.8f\n",
                i, Ec[i], e_cepa[i])
    end

    return e_cepa, pt1_vec
end

function _tpsci_local_h0_diagonal(cluster_ops, opstring::String, clusters, fock, config, ::Type{T}) where {T}
    hd = zero(T)
    for c in clusters
        mat = cluster_ops[c.idx][opstring][(fock[c.idx], fock[c.idx])]
        hd += mat[config[c.idx], config[c.idx]]
    end
    return hd
end

function _tpsci_local_h0_diagonal(h0::ClusterH0DiagonalCache{T},
                                  opstring::String, clusters, fock, config,
                                  ::Type{T}) where {T}
    hd = zero(T)
    for c in clusters
        sector_diags = h0.data[c.idx]
        fock_i = fock[c.idx]
        haskey(sector_diags, fock_i) ||
            error("Cluster $(c.idx) is missing diagonal $opstring sector $fock_i")
        hd += sector_diags[fock_i][Int(config[c.idx])]
    end
    return hd
end

function _tpsci_sharded_local_h0_expectation(id::Symbol, opstring::String)
    state = _tpsci_sharded_get_state(id)
    cluster_ops = _TPSCI_MULTINODE_CLUSTER_OPS[]
    cluster_ops === nothing && error("TPSCI sharded cluster-ops cache is empty")
    if cluster_ops isa DistributedClusterOps
        cluster_ops = _materialize_cluster_h0_diagonals_cached(cluster_ops,
                                                               opstring)
    end
    return _tpsci_sharded_local_h0_expectation_typed(state, cluster_ops, opstring)
end

function _tpsci_sharded_local_h0_expectation_typed(state::TPSCIstate{T,N,R},
                                                   cluster_ops, opstring::String) where {T,N,R}
    out = zeros(T, R)
    for (fock, configs) in state.data
        for (config, coeffs) in configs
            hd = _tpsci_local_h0_diagonal(cluster_ops, opstring, state.clusters,
                                          fock, config, T)
            for r in 1:R
                out[r] += coeffs[r] * coeffs[r] * hd
            end
        end
    end
    return out
end

"""
    compute_expectation_value_sharded_h0(ci_vector, cluster_ops; opstring="Hcmf")

Compute a diagonal zeroth-order expectation value for a sharded TPSCI state.
This keeps only the small root-wise expectation vector on the master.
"""
function compute_expectation_value_sharded_h0(ci_vector::DistributedTPSCIstate{T,N,R},
                                             cluster_ops; opstring::String="Hcmf",
                                             workers=ci_vector.workers,
                                             blas_threads=1) where {T,N,R}
    _tpsci_sharded_cache_operator_problem!(cluster_ops, nothing;
                                           workers=workers,
                                           blas_threads=blas_threads)
    out = zeros(T, R)
    for pid in ci_vector.workers
        out .+= Distributed.remotecall_fetch(_tpsci_sharded_local_h0_expectation,
                                             pid, ci_vector.id, opstring)
    end
    return out
end

function _tpsci_sharded_local_apply_pt1_denominator!(sig_id::Symbol,
                                                     opstring::String,
                                                     E0, norms)
    sig = _tpsci_sharded_get_state(sig_id)
    cluster_ops = _TPSCI_MULTINODE_CLUSTER_OPS[]
    cluster_ops === nothing && error("TPSCI sharded cluster-ops cache is empty")
    if cluster_ops isa DistributedClusterOps
        cluster_ops = _materialize_cluster_h0_diagonals_cached(cluster_ops,
                                                               opstring)
    end
    return _tpsci_sharded_local_apply_pt1_denominator_typed!(sig, cluster_ops,
                                                             opstring, E0, norms)
end

function _tpsci_sharded_local_apply_pt1_denominator_typed!(sig::TPSCIstate{T,N,R},
                                                           cluster_ops, opstring,
                                                           E0, norms) where {T,N,R}
    e2 = zeros(T, R)
    for (fock, configs) in sig.data
        for (config, coeffs) in configs
            hd = _tpsci_local_h0_diagonal(cluster_ops, opstring, sig.clusters,
                                          fock, config, T)
            for r in 1:R
                amp = coeffs[r]
                denom = E0[r] / (norms[r] * norms[r]) - hd
                coeffs[r] = amp / denom
                e2[r] += amp * coeffs[r]
            end
        end
    end
    return e2
end

"""
    compute_pt1_wavefunction_sharded(ci_vector, cluster_ops, clustered_ham; ...)

Sharded PT1 construction for selected TPSCI. The FOIS vector, projection,
diagonal denominator application, clipping, and later addition can all happen
without gathering the CI vector on the master.
"""
function compute_pt1_wavefunction_sharded(ci_vector::DistributedTPSCIstate{T,N,R},
                                          cluster_ops, clustered_ham::ClusteredOperator;
                                          nbody=4,
                                          H0::String="Hcmf",
                                          E0=nothing,
                                          thresh_foi=1e-8,
                                          prescreen=false,
                                          workers=ci_vector.workers,
                                          threaded_worker=true,
                                          blas_threads=1) where {T,N,R}
    println()
    println(" |.........................do sharded PT1..........................")
    println(" thresh_foi    :", thresh_foi)
    println(" prescreen     :", prescreen)
    println(" H0            :", H0)
    println(" nbody         :", nbody)

    norms = norm(ci_vector)
    println(" Norms of input states")
    [@printf(" %12.8f\n", i) for i in norms]

    if E0 === nothing
        @printf(" %-50s", "Compute sharded <0|H0|0>:")
        @time E0 = compute_expectation_value_sharded_h0(ci_vector, cluster_ops,
                                                        opstring=H0,
                                                        workers=workers,
                                                        blas_threads=blas_threads)
        flush(stdout)
    end

    sig = open_matvec_sharded(ci_vector, cluster_ops, clustered_ham,
                              nbody=nbody, thresh=thresh_foi,
                              prescreen=prescreen, workers=workers,
                              threaded_worker=threaded_worker,
                              blas_threads=blas_threads)
    println(" Length of FOIS vector: ", length(sig))

    project_out!(sig, ci_vector)
    println(" Length of projected FOIS vector: ", length(sig))

    e2 = zeros(T, R)
    @printf(" %-50s", "Apply sharded PT1 denominators:")
    @time begin
        _tpsci_sharded_cache_operator_problem!(cluster_ops, clustered_ham;
                                               workers=workers,
                                               blas_threads=blas_threads)
        for pid in sig.workers
            e2 .+= Distributed.remotecall_fetch(_tpsci_sharded_local_apply_pt1_denominator!,
                                                pid, sig.id, H0, E0, norms)
        end
    end
    flush(stdout)

    println()
    @printf(" %5s %12s %12s\n", "Root", "E(0)", "E(2)")
    for r in 1:R
        @printf(" %5s %12.8f %12.8f\n", r, E0[r] / (norms[r] * norms[r]),
                E0[r] / (norms[r] * norms[r]) + e2[r])
    end
    println(" ..................................................................|")
    return e2, sig
end

function _tpsci_worker_ids(worker_pool)
    pids = collect(worker_pool)
    isempty(pids) && error("No distributed workers available. Start Julia with workers, e.g. addprocs(...; exeflags=\"--project\").")
    return pids
end

function ensure_tpsci_multinode_workers!(; workers=Distributed.workers())
    pids = _tpsci_worker_ids(workers)
    project_root = dirname(dirname(@__DIR__))
    @sync for pid in pids
        pid == Distributed.myid() && continue
        @async Distributed.remotecall_fetch(
            Core.eval, pid, Main,
            :(begin
                  import Pkg
                  Pkg.activate($project_root; io=devnull)
                  using TPSChem
              end))
    end
    return pids
end

function _tpsci_multinode_set_cache!(ci_vector, cluster_ops, clustered_ham; blas_threads=1)
    _TPSCI_MULTINODE_CI[] = ci_vector
    _TPSCI_MULTINODE_CLUSTER_OPS[] = cluster_ops
    _TPSCI_MULTINODE_CLUSTERED_HAM[] = clustered_ham
    blas_threads === nothing || BLAS.set_num_threads(blas_threads)
    return (Distributed.myid(), length(ci_vector))
end

function _tpsci_multinode_cache_problem!(ci_vector, cluster_ops, clustered_ham;
                                         workers=Distributed.workers(),
                                         blas_threads=1)
    pids = ensure_tpsci_multinode_workers!(workers=workers)
    @sync for pid in pids
        @async begin
            if pid == Distributed.myid()
                _tpsci_multinode_set_cache!(ci_vector, cluster_ops, clustered_ham;
                                            blas_threads=blas_threads)
            else
                Distributed.remotecall_fetch(_tpsci_multinode_set_cache!, pid,
                                             ci_vector, cluster_ops, clustered_ham;
                                             blas_threads=blas_threads)
            end
        end
    end
    return pids
end

function _tpsci_partition_round_robin(items, pids)
    chunks = [Vector{eltype(items)}() for _ in pids]
    isempty(items) && return chunks
    for (i, item) in enumerate(items)
        push!(chunks[mod1(i, length(pids))], item)
    end
    return chunks
end

function _tpsci_partition_jobs(jobs, pids)
    chunks = [Vector{eltype(jobs)}() for _ in pids]
    isempty(jobs) && return chunks
    for (i, job) in enumerate(jobs)
        push!(chunks[mod1(i, length(pids))], job)
    end
    return chunks
end

function _tpsci_valid_fock(fock, clusters)
    all(f[1] >= 0 for f in fock) || return false
    all(f[2] >= 0 for f in fock) || return false
    all(f[1] <= length(clusters[fi]) for (fi, f) in enumerate(fock)) || return false
    all(f[2] <= length(clusters[fi]) for (fi, f) in enumerate(fock)) || return false
    return true
end

function _tpsci_has_cluster_space(fock, clusters, cluster_ops)
    for c in clusters
        haskey(cluster_ops[c.idx]["H"], (fock[c.idx], fock[c.idx])) || return false
    end
    return true
end

function _tpsci_has_cluster_space(fock, clusters, cluster_ops::DistributedClusterOps)
    for c in clusters
        haskey(cluster_ops.h_sectors, c.idx) || return false
        fock[c.idx] in cluster_ops.h_sectors[c.idx] || return false
    end
    return true
end

function _tpsci_fois_fock_configs(ci_vector::TPSCIstate{T,N,R}, cluster_ops,
                                  clustered_ham::ClusteredOperator) where {T,N,R}
    clusters = ci_vector.clusters
    fock_bras = FockConfig{N}[]
    seen = Set{FockConfig{N}}()
    for (fock_ket, _) in ci_vector.data
        for (ftrans, _) in clustered_ham
            fock_bra = ftrans + fock_ket
            _tpsci_valid_fock(fock_bra, clusters) || continue
            _tpsci_has_cluster_space(fock_bra, clusters, cluster_ops) || continue
            fock_bra in seen && continue
            push!(seen, fock_bra)
            push!(fock_bras, fock_bra)
        end
    end
    return fock_bras
end

function _tpsci_bra_jobs(ci_vector::TPSCIstate{T,N,R}) where {T,N,R}
    jobs = Tuple{Int,FockConfig{N},ClusterConfig{N}}[]
    bra_idx = 0
    for (fock_bra, configs_bra) in ci_vector.data
        for (config_bra, _) in configs_bra
            bra_idx += 1
            push!(jobs, (bra_idx, fock_bra, config_bra))
        end
    end
    return jobs
end

function _tpsci_zero_mvector(::Val{N}) where {N}
    return MVector{N,Int16}(ntuple(_ -> Int16(0), N))
end

function _tpsci_open_scratch(::Type{T}, ::Val{N}, nt::Int) where {T,N}
    nscr = 20
    scr_f = Vector{Vector{Vector{T}}}()
    scr_i = Vector{Vector{Vector{Int16}}}()
    scr_m = Vector{Vector{MVector{N,Int16}}}()
    for _ in 1:nt
        push!(scr_f, [zeros(T, 10000) for _ in 1:nscr])
        push!(scr_i, [zeros(Int16, 10000) for _ in 1:nscr])
        push!(scr_m, [_tpsci_zero_mvector(Val(N)) for _ in 1:nscr])
    end
    return scr_f, scr_i, scr_m
end

function _tpsci_open_job_for_fock(ci_vector, clustered_ham, fock_bra)
    job = Tuple[]
    for (fock_ket, configs_ket) in ci_vector.data
        fock_trans = fock_bra - fock_ket
        haskey(clustered_ham, fock_trans) || continue
        push!(job, (clustered_ham[fock_trans], fock_ket, configs_ket))
    end
    return job
end

function _tpsci_collect_fock_blocks!(dest, src)
    for (fock, configs) in src.data
        haskey(dest, fock) && error("Duplicate Fock block while collecting distributed TPSCI result")
        dest[fock] = configs
    end
    return dest
end

function _tpsci_open_matvec_chunk_cached(fock_bras, nbody, thresh, prescreen,
                                         threaded_worker)
    ci_vector = _TPSCI_MULTINODE_CI[]
    cluster_ops = _TPSCI_MULTINODE_CLUSTER_OPS[]
    clustered_ham = _TPSCI_MULTINODE_CLUSTERED_HAM[]
    ci_vector === nothing && error("TPSCI multinode worker cache is empty")
    return _tpsci_open_matvec_chunk_cached_typed(ci_vector, cluster_ops, clustered_ham,
                                                fock_bras, nbody, thresh,
                                                prescreen, threaded_worker)
end

function _tpsci_open_matvec_chunk_cached_typed(ci_vector::TPSCIstate{T,N,R},
                                              cluster_ops, clustered_ham,
                                              fock_bras, nbody, thresh,
                                              prescreen, threaded_worker) where {T,N,R}
    nt = threaded_worker ? Threads.maxthreadid() : 1
    partial = [TPSCIstate(ci_vector.clusters, T=T, R=R) for _ in 1:nt]
    scr_f, scr_i, scr_m = _tpsci_open_scratch(T, Val(N), nt)

    if threaded_worker
        Threads.@threads :static for idx in eachindex(fock_bras)
            tid = Threads.threadid()
            fock_bra = fock_bras[idx]
            job = _tpsci_open_job_for_fock(ci_vector, clustered_ham, fock_bra)
            _open_matvec_thread_job(job, fock_bra, cluster_ops, nbody, thresh,
                                    partial[tid], scr_f[tid], scr_i[tid], scr_m[tid],
                                    prescreen)
        end
    else
        for fock_bra in fock_bras
            job = _tpsci_open_job_for_fock(ci_vector, clustered_ham, fock_bra)
            _open_matvec_thread_job(job, fock_bra, cluster_ops, nbody, thresh,
                                    partial[1], scr_f[1], scr_i[1], scr_m[1],
                                    prescreen)
        end
    end

    sig = TPSCIstate(ci_vector.clusters, T=T, R=R)
    for part in partial
        _tpsci_collect_fock_blocks!(sig, part)
    end
    return sig
end

"""
    open_matvec_distributed(ci_vector, cluster_ops, clustered_ham; ...)

Distributed version of `open_matvec_thread`. Work is split by output Fock
sector. Each worker caches the large read-only TPSCI objects once for this call
and returns only its owned Fock blocks.
"""
function open_matvec_distributed(ci_vector::TPSCIstate{T,N,R}, cluster_ops,
                                 clustered_ham::ClusteredOperator;
                                 thresh=1e-9,
                                 prescreen=true,
                                 nbody=4,
                                 workers=Distributed.workers(),
                                 threaded_worker=true,
                                 blas_threads=1) where {T,N,R}
    pids = _tpsci_multinode_cache_problem!(ci_vector, cluster_ops, clustered_ham;
                                           workers=workers,
                                           blas_threads=blas_threads)
    fock_bras = _tpsci_fois_fock_configs(ci_vector, cluster_ops, clustered_ham)
    chunks = _tpsci_partition_round_robin(fock_bras, pids)

    println(" In open_matvec_distributed")
    println(" Number of output Fock jobs: ", length(fock_bras))
    println(" Number of workers:          ", length(pids))
    println(" Threads per worker enabled: ", threaded_worker)
    flush(stdout)

    results = Any[nothing for _ in pids]
    @sync for (i, pid) in enumerate(pids)
        isempty(chunks[i]) && continue
        @async begin
            results[i] = Distributed.remotecall_fetch(_tpsci_open_matvec_chunk_cached,
                                                      pid, chunks[i], nbody, thresh,
                                                      prescreen, threaded_worker)
        end
    end

    sig = TPSCIstate(ci_vector.clusters, T=T, R=R)
    for result in results
        result === nothing && continue
        _tpsci_collect_fock_blocks!(sig, result)
    end
    return sig
end

function _tpsci_matvec_chunk_cached(jobs, coeffs)
    ci_template = _TPSCI_MULTINODE_CI[]
    cluster_ops = _TPSCI_MULTINODE_CLUSTER_OPS[]
    clustered_ham = _TPSCI_MULTINODE_CLUSTERED_HAM[]
    ci_template === nothing && error("TPSCI multinode worker cache is empty")
    return _tpsci_matvec_chunk_cached_typed(ci_template, cluster_ops, clustered_ham,
                                           jobs, coeffs)
end

function _tpsci_matvec_chunk_cached_typed(ci_template::TPSCIstate{T,N,R0},
                                         cluster_ops, clustered_ham,
                                         jobs, coeffs) where {T,N,R0}
    nroots = coeffs isa AbstractVector ? 1 : size(coeffs, 2)
    ci_vector = TPSCIstate(ci_template, T=T, R=nroots)
    if coeffs isa AbstractVector
        set_vector!(ci_vector, Vector{T}(coeffs))
    else
        set_vector!(ci_vector, Matrix{T}(coeffs))
    end

    rows = Vector{Int}(undef, length(jobs))
    vals = zeros(T, length(jobs), nroots)

    for (local_idx, job) in enumerate(jobs)
        bra_idx, fock_bra, config_bra = job
        rows[local_idx] = bra_idx
        sig_out = view(vals, local_idx, :)

        for (fock_trans, _) in clustered_ham
            fock_ket = fock_bra - fock_trans
            haskey(ci_vector.data, fock_ket) || continue
            configs_ket = ci_vector[fock_ket]

            for (config_ket, coeff_ket) in configs_ket
                for term in clustered_ham[fock_trans]
                    check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue
                    me = contract_matrix_element(term, cluster_ops, fock_bra, config_bra,
                                                 fock_ket, config_ket)
                    @simd for r in 1:nroots
                        @inbounds sig_out[r] += me * coeff_ket[r]
                    end
                end
            end
        end
    end

    return rows, vals
end

function _tpsci_matvec_distributed_cached(ci_template::TPSCIstate{T,N,R},
                                          coeffs;
                                          workers=Distributed.workers()) where {T,N,R}
    pids = _tpsci_worker_ids(workers)
    jobs = _tpsci_bra_jobs(ci_template)
    chunks = _tpsci_partition_jobs(jobs, pids)
    nroots = coeffs isa AbstractVector ? 1 : size(coeffs, 2)
    sigv = zeros(T, length(ci_template), nroots)

    results = Any[nothing for _ in pids]
    @sync for (i, pid) in enumerate(pids)
        isempty(chunks[i]) && continue
        @async begin
            results[i] = Distributed.remotecall_fetch(_tpsci_matvec_chunk_cached,
                                                      pid, chunks[i], coeffs)
        end
    end

    for result in results
        result === nothing && continue
        rows, vals = result
        sigv[rows, :] .= vals
    end
    return coeffs isa AbstractVector ? sigv[:, 1] : sigv
end

"""
    tps_ci_matvec_distributed(ci_vector, cluster_ops, clustered_ham; workers=...)

Matrix-free distributed TPSCI matvec. The dense Hamiltonian is never built.
"""
function tps_ci_matvec_distributed(ci_vector::TPSCIstate{T,N,R}, cluster_ops,
                                   clustered_ham::ClusteredOperator;
                                   workers=Distributed.workers(),
                                   blas_threads=1) where {T,N,R}
    pids = _tpsci_multinode_cache_problem!(ci_vector, cluster_ops, clustered_ham;
                                           workers=workers,
                                           blas_threads=blas_threads)
    return _tpsci_matvec_distributed_cached(ci_vector, get_vector(ci_vector);
                                            workers=pids)
end

"""
    tps_ci_davidson_distributed(ci_vector, cluster_ops, clustered_ham; ...)

Distributed, matrix-free Davidson solve for the current TPSCI variational
space. This is the multinode replacement for building/storing the dense CI
Hamiltonian when memory per node matters.
"""
function tps_ci_davidson_distributed(ci_vector::TPSCIstate{T,N,R}, cluster_ops,
                                     clustered_ham::ClusteredOperator;
                                     conv_thresh=1e-5,
                                     lindep_thresh=1e-12,
                                     max_ss_vecs=12,
                                     max_iter=40,
                                     precond=true,
                                     verbose=0,
                                     workers=Distributed.workers(),
                                     blas_threads=1) where {T,N,R}
    println()
    @printf(" |== Multinode Tensor Product State CI =============================\n")
    @printf(" Hamiltonian matrix dimension = %5i\n", length(ci_vector))
    flush(stdout)

    pids = _tpsci_multinode_cache_problem!(ci_vector, cluster_ops, clustered_ham;
                                           workers=workers,
                                           blas_threads=blas_threads)
    vec_out = deepcopy(ci_vector)
    dim = length(ci_vector)
    iters = 0

    function matvec(v::Vector)
        iters += 1
        return _tpsci_matvec_distributed_cached(ci_vector, v; workers=pids)
    end
    function matvec(v::Matrix)
        iters += 1
        return _tpsci_matvec_distributed_cached(ci_vector, v; workers=pids)
    end

    Hmap = LinOpMat{T}(matvec, dim, true)
    davidson = Davidson(Hmap, v0=get_vector(ci_vector),
                        max_iter=max_iter, max_ss_vecs=max_ss_vecs,
                        nroots=R, tol=conv_thresh,
                        lindep_thresh=lindep_thresh)

    e = nothing
    v = nothing
    if precond
        @printf(" %-50s", "Compute diagonal: ")
        @time Hd = compute_diagonal(ci_vector, cluster_ops, clustered_ham)
        @printf(" Now iterate on distributed matvec: \n")
        flush(stdout)
        @time e, v = BlockDavidson.eigs(davidson, Adiag=Hd)
    else
        @time e, v = BlockDavidson.eigs(davidson)
    end
    set_vector!(vec_out, v)

    clustered_S2 = extract_S2(ci_vector.clusters)
    @printf(" %-50s", "Compute S2 expectation values: ")
    @time s2 = compute_expectation_value_parallel(vec_out, cluster_ops, clustered_S2)
    flush(stdout)
    @printf(" %5s %12s %12s\n", "Root", "Energy", "S2")
    for r in 1:R
        @printf(" %5s %12.8f %12.8f\n", r, e[r], abs(s2[r]))
    end

    if verbose > 1
        for r in 1:R
            display(vec_out, root=r)
        end
    end

    @printf(" ==================================================================|\n")
    return e, vec_out
end

"""
    compute_pt1_wavefunction_distributed(ci_vector, cluster_ops, clustered_ham; ...)

Distributed PT1/FOIS construction. The open matvec scratch is split across
workers; the selected PT1 vector is collected on the master for the next TPSCI
iteration.
"""
function compute_pt1_wavefunction_distributed(ci_vector::TPSCIstate{T,N,R}, cluster_ops,
                                              clustered_ham::ClusteredOperator;
                                              nbody=4,
                                              H0="Hcmf",
                                              E0=nothing,
                                              thresh_foi=1e-8,
                                              prescreen=false,
                                              verbose=1,
                                              workers=Distributed.workers(),
                                              threaded_worker=true,
                                              blas_threads=1) where {T,N,R}
    println()
    println(" |.......................do distributed PT1........................")
    println(" thresh_foi    :", thresh_foi)
    println(" prescreen     :", prescreen)
    println(" H0            :", H0)
    println(" nbody         :", nbody)

    e2 = zeros(T, R)
    norms = norm(ci_vector)
    println(" Norms of input states")
    [@printf(" %12.8f\n", i) for i in norms]
    println(" Compute distributed FOIS vector")

    sig = open_matvec_distributed(ci_vector, cluster_ops, clustered_ham,
                                  nbody=nbody, thresh=thresh_foi,
                                  prescreen=prescreen, workers=workers,
                                  threaded_worker=threaded_worker,
                                  blas_threads=blas_threads)
    println(" Length of FOIS vector: ", length(sig))

    clustered_ham_0 = extract_1body_operator(clustered_ham, op_string=H0)

    project_out!(sig, ci_vector)
    println(" Length of projected FOIS vector: ", length(sig))

    @printf(" %-50s", "Compute diagonal")
    @time Hd = compute_diagonal(sig, cluster_ops, clustered_ham_0)

    if E0 === nothing
        @printf(" %-50s", "Compute <0|H0|0>:")
        @time E0 = compute_expectation_value_parallel(ci_vector, cluster_ops, clustered_ham_0)
        flush(stdout)
    end

    @printf(" %-50s", "Compute <0|H|0>:")
    @time Evar = compute_expectation_value_parallel(ci_vector, cluster_ops, clustered_ham)
    flush(stdout)

    sig_v = get_vector(sig)
    v_pt = zeros(T, size(sig_v))

    println()
    @printf(" %5s %12s %12s\n", "Root", "E(0)", "E(2)")
    for r in 1:R
        denom = 1.0 ./ (E0[r] / (norms[r] * norms[r]) .- Hd)
        v_pt[:, r] .= denom .* sig_v[:, r]
        e2[r] = sum(sig_v[:, r] .* v_pt[:, r])
        @printf(" %5s %12.8f %12.8f\n", r, Evar[r] / norms[r],
                Evar[r] / (norms[r] * norms[r]) + e2[r])
    end

    set_vector!(sig, v_pt)
    println(" ..................................................................|")
    return e2, sig
end

"""
    tpsci_ci_multinode(ci_vector, cluster_ops, clustered_ham; ...)

Run selected TPSCI using multinode kernels with the replicated TPSCI-state
backend. This path always uses the matrix-free distributed Davidson solver
instead of building the dense CI Hamiltonian, but it still caches the full
`ci_vector` on each worker. For CI vectors larger than per-node memory, use this
only as a stepping stone toward a sharded distributed state.
"""
function tpsci_ci_multinode(ci_vector::TPSCIstate{T,N,R}, cluster_ops,
                            clustered_ham::ClusteredOperator;
                            thresh_cipsi=1e-2,
                            thresh_foi=1e-6,
                            thresh_asci=nothing,
                            thresh_var=nothing,
                            thresh_spin=nothing,
                            max_iter=10,
                            conv_thresh=1e-4,
                            nbody=4,
                            incremental=false,
                            ci_conv=1e-5,
                            ci_max_iter=50,
                            ci_max_ss_vecs=12,
                            ci_lindep_thresh=1e-12,
                            max_mem_ci=20.0,
                            prescreen=true,
                            workers=Distributed.workers(),
                            threaded_worker=true,
                            blas_threads=1) where {T,N,R}
    worker_ids = ensure_tpsci_multinode_workers!(workers=workers)
    incremental && @warn("incremental=true keeps the previous FOIS vector on the master; incremental=false is recommended for lower memory.")
    @warn("tpsci_ci_multinode currently uses a replicated TPSCIstate backend. It avoids dense H storage, but it does not yet shard a >node-memory CI vector.")

    vec_var = deepcopy(ci_vector)
    vec_pt = deepcopy(ci_vector)
    length(ci_vector) > 0 || error(" input vector has zero length")
    zero!(vec_pt)
    e0 = zeros(T, R)
    e2 = zeros(T, R)
    e0_last = zeros(T, R)

    clustered_S2 = extract_S2(ci_vector.clusters)
    clustered_ham_0 = extract_1body_operator(clustered_ham, op_string="Hcmf")

    println(" ci_vector        : ", size(ci_vector))
    println(" thresh_cipsi     : ", thresh_cipsi)
    println(" thresh_foi       : ", thresh_foi)
    println(" thresh_asci      : ", thresh_asci)
    println(" thresh_var       : ", thresh_var)
    println(" thresh_spin      : ", thresh_spin)
    println(" max_iter         : ", max_iter)
    println(" conv_thresh      : ", conv_thresh)
    println(" nbody            : ", nbody)
    println(" incremental      : ", incremental)
    println(" ci_conv          : ", ci_conv)
    println(" ci_max_iter      : ", ci_max_iter)
    println(" ci_max_ss_vecs   : ", ci_max_ss_vecs)
    println(" ci_lindep_thresh : ", ci_lindep_thresh)
    println(" max_mem_ci       : ", max_mem_ci)
    println(" workers          : ", worker_ids)
    println(" threaded_worker  : ", threaded_worker)

    vec_asci_old = TPSCIstate(ci_vector.clusters, R=R, T=T)
    sig = TPSCIstate(ci_vector.clusters, R=R, T=T)
    sig_old = TPSCIstate(ci_vector.clusters, R=R, T=T)
    to = TimerOutput()

    for it in 1:max_iter
        println()
        println()
        println(" ===================================================================")
        @printf("     Multinode Selected CI Iteration: %4i epsilon: %12.8f\n", it, thresh_cipsi)
        println(" ===================================================================")

        if it > 1
            if thresh_var !== nothing
                l1 = length(vec_var)
                clip!(vec_var, thresh=thresh_var)
                l2 = length(vec_var)
                @printf(" Clip values < %8.1e         %6i -> %6i\n", thresh_var, l1, l2)
            end

            l1 = length(vec_var)
            zero!(vec_pt)
            add!(vec_var, vec_pt)
            l2 = length(vec_var)
            @printf("%-50s%6i -> %6i\n", " Add pt vector to current space", l1, l2)
        end

        @timeit to "s2 extension" if thresh_spin !== nothing
            spin_residual = open_matvec_distributed(vec_var, cluster_ops, clustered_S2,
                                                    nbody=nbody, thresh=thresh_spin,
                                                    prescreen=prescreen,
                                                    workers=worker_ids,
                                                    threaded_worker=threaded_worker,
                                                    blas_threads=blas_threads)
            spin_expval = TPSChem.overlap(vec_var, spin_residual)
            spin_residual = spin_residual - (vec_var * spin_expval)
            for r in 1:R
                @printf(" S^2 Residual %12.8f\n", dot(spin_residual, spin_residual, r, r))
            end
            zero!(spin_residual)

            l1 = size(vec_var)[1]
            add!(vec_var, spin_residual)
            l2 = size(vec_var)[1]
            @printf("%-50s%6i -> %6i\n", " Add spin completing states", l1, l2)
            flush(stdout)
        end

        mem_needed = sizeof(T) * length(vec_var) * length(vec_var) * 1e-9
        @printf(" Dense CI matrix would need: %12.8f (Gb); multinode path uses matrix-free Davidson\n", mem_needed)
        flush(stdout)

        @timeit to "distributed ci" begin
            orthonormalize!(vec_var)
            e0, vec_var = tps_ci_davidson_distributed(vec_var, cluster_ops, clustered_ham,
                                                       conv_thresh=ci_conv,
                                                       max_iter=ci_max_iter,
                                                       max_ss_vecs=ci_max_ss_vecs,
                                                       lindep_thresh=ci_lindep_thresh,
                                                       workers=worker_ids,
                                                       blas_threads=blas_threads)
        end
        flush(stdout)

        Efock = compute_expectation_value_parallel(vec_var, cluster_ops, clustered_ham_0)
        flush(stdout)

        vec_asci = deepcopy(vec_var)
        if thresh_asci !== nothing
            l1 = length(vec_asci)
            clip!(vec_asci, thresh=thresh_asci)
            l2 = length(vec_asci)
            @printf("%-50s%6i -> %6i\n", " Length of ASCI vector", l1, l2)
        end

        if incremental
            S = overlap(vec_asci_old, vec_asci)
            println(" Overlap between old and new eigenvectors:")
            for i in 1:size(S, 1)
                for j in 1:size(S, 2)
                    @printf(" %6.3f", S[i, j])
                end
                println()
            end
            println()

            @timeit to "sig rotate" sig = sig_old * S
            @timeit to "vec rotate" del_v0 = vec_asci - (vec_asci_old * S)

            println(" Norm of new projection:")
            [@printf(" %12.8f\n", i) for i in norm(del_v0)]
            println()

            @timeit to "copy" vec_asci_old = deepcopy(vec_asci)
            @timeit to "distributed matvec" del_sig_it = open_matvec_distributed(
                del_v0, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi,
                prescreen=prescreen, workers=worker_ids,
                threaded_worker=threaded_worker, blas_threads=blas_threads)
            @timeit to "sig update" add!(sig, del_sig_it)
            @timeit to "copy" sig_old = deepcopy(sig)

            @timeit to "project out" project_out!(sig, vec_asci)
            println(" Length of FOIS vector: ", length(sig))

            @printf(" %-50s", "Compute diagonal: ")
            flush(stdout)
            @timeit to "diagonal" @time Hd = compute_diagonal(sig, cluster_ops, "Hcmf")
            println()
            flush(stdout)

            sig_v = get_vector(sig)
            v_pt = zeros(T, size(sig_v))

            norms = norm(vec_asci)
            println()
            @printf(" %5s %12s %12s\n", "Root", "E(0)", "E(2)")
            for r in 1:R
                denom = 1.0 ./ (Efock[r] / (norms[r] * norms[r]) .- Hd)
                v_pt[:, r] .= denom .* sig_v[:, r]
                e2[r] = sum(sig_v[:, r] .* v_pt[:, r])
                @printf(" %5s %12.8f %12.8f\n", r, e0[r] / norms[r],
                        e0[r] / (norms[r] * norms[r]) + e2[r])
            end

            @timeit to "copy" vec_pt = deepcopy(sig)
            set_vector!(vec_pt, Matrix{T}(v_pt))
        else
            @timeit to "distributed pt1" e2, vec_pt = compute_pt1_wavefunction_distributed(
                vec_asci, cluster_ops, clustered_ham, E0=Efock,
                thresh_foi=thresh_foi, prescreen=prescreen,
                nbody=nbody, workers=worker_ids,
                threaded_worker=threaded_worker, blas_threads=blas_threads)
        end
        flush(stdout)

        l1 = length(vec_pt)
        clip!(vec_pt, thresh=thresh_cipsi)
        l2 = length(vec_pt)
        @printf("%-50s%6i -> %6i\n", " Length of PT1  vector", l1, l2)

        if maximum(abs.(e0_last .- e0)) < conv_thresh
            print_tpsci_iter(vec_var, it, e0, true)
            break
        else
            print_tpsci_iter(vec_var, it, e0, false)
            e0_last .= e0
        end
        flush(stdout)
    end

    println("")
    show(to)
    println("")
    flush(stdout)
    return e0, vec_var
end
