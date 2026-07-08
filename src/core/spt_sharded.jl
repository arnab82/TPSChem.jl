"""
Never-gather (sharded) SPT: a `DistributedSPTstate` whose per-Fock-sector Tucker
blocks live across workers and are never gathered onto the master, plus a
distributed Tucker CI solver (`spt_ci_davidson_sharded`).

Motivation (see `examples/multinode/SHARDED_SPT_PLAN.md`): Tucker-compressed SPT states are big in
practice, so the existing `subspace_product_tucker_multinode` — which gathers the
full FOIS/variational state on the master and runs a *local* `ci_solve` Davidson
holding many full vectors — is master-memory-bound. Here the variational vector
stays sharded through the whole CI solve.

Design: the block Davidson driver `_tps_ci_davidson_sharded_core`
(`tpsci_sharded_davidson.jl`) is generic over `AbstractShardedState`; this file
supplies the SPT container and the sharded primitives it calls (`overlap`,
`norm`, `scale!`, `add_scaled!`, `copy_sharded_state`, `similar_sharded_state`,
`precondition_sharded`, `destroy!`) plus the SPT matvec (`build_sigma_sharded`)
and the CMF diagonal preconditioner.

Within a CI solve the Tucker basis is fixed (the factors are set once by the
pseudo-canonical rotation, then only cores change), so `orth_dot`-style core
arithmetic is exact and every subspace vector shares the same factors.
"""

const _SPT_SHARDED_STATES = Dict{Symbol,Any}()

function _spt_sharded_store_state!(id::Symbol, state)
    _SPT_SHARDED_STATES[id] = state
    return length(state)
end

function _spt_sharded_delete_state!(id::Symbol)
    delete!(_SPT_SHARDED_STATES, id)
    return true
end

function _spt_sharded_get_state(id::Symbol)
    haskey(_SPT_SHARDED_STATES, id) ||
        error("No sharded SPT state cached with id $id on worker $(Distributed.myid())")
    return _SPT_SHARDED_STATES[id]
end

"""
    DistributedSPTstate{T,N,R} <: AbstractShardedState{T,N,R}

Metadata handle for an `SPTstate` whose Fock-sector Tucker blocks live on
distributed workers. The master keeps ownership/lengths/offsets and the (small)
p/q subspace definitions; each worker holds an ordinary local `SPTstate` with the
Fock sectors it owns. `lengths` are coefficient counts (Σ over TuckerConfigs of
`length(tuck)`), matching `get_vector`.
"""
mutable struct DistributedSPTstate{T,N,R} <: AbstractShardedState{T,N,R}
    id::Symbol
    clusters::Vector{MOCluster}
    p_spaces::Vector{ClusterSubspace}
    q_spaces::Vector{ClusterSubspace}
    workers::Vector{Int}
    owners::OrderedDict{FockConfig{N},Int}
    lengths::OrderedDict{FockConfig{N},Int}
    offsets::OrderedDict{FockConfig{N},Int}
    local_lengths::OrderedDict{Int,Int}
    total_length::Int
end

Base.length(s::DistributedSPTstate) = s.total_length
Base.size(s::DistributedSPTstate{T,N,R}) where {T,N,R} = (s.total_length, R)
Base.haskey(s::DistributedSPTstate, fock) = haskey(s.owners, fock)
nroots(s::DistributedSPTstate{T,N,R}) where {T,N,R} = R

# ---------------------------------------------------------------------------
# Local (worker-side) building blocks
# ---------------------------------------------------------------------------

function _spt_block_length(tconfigs)
    l = 0
    for (_, tuck) in tconfigs
        l += length(tuck)
    end
    return l
end

function _spt_copy_fock_block(tconfigs::OrderedDict{TuckerConfig{N},Tucker{T,N,R}}) where {T,N,R}
    out = OrderedDict{TuckerConfig{N},Tucker{T,N,R}}()
    for (tc, tuck) in tconfigs
        out[tc] = deepcopy(tuck)
    end
    return out
end

function _spt_empty_local_state(::Type{T}, ::Val{R}, clusters, p_spaces, q_spaces, ::Val{N}) where {T,R,N}
    data = OrderedDict{FockConfig{N},OrderedDict{TuckerConfig{N},Tucker{T,N,R}}}()
    return SPTstate{T,N,R}(clusters, data, p_spaces, q_spaces)
end

function _spt_sharded_local_fock_lengths(id::Symbol)
    state = _spt_sharded_get_state(id)
    out = OrderedDict{keytype(state.data),Int}()
    for (fock, tconfigs) in state.data
        out[fock] = _spt_block_length(tconfigs)
    end
    return out
end

function _spt_sharded_local_summary(id::Symbol)
    state = _spt_sharded_get_state(id)
    return (Distributed.myid(), length(state), length(state.data))
end

# ---------------------------------------------------------------------------
# Metadata assembly
# ---------------------------------------------------------------------------

function _spt_sharded_metadata_from_lengths(id::Symbol, clusters, p_spaces, q_spaces,
                                            pids, length_maps, ordered_focks,
                                            ::Type{T}, ::Val{R}, ::Val{N}) where {T,R,N}
    owners = OrderedDict{FockConfig{N},Int}()
    lengths = OrderedDict{FockConfig{N},Int}()
    offsets = OrderedDict{FockConfig{N},Int}()
    local_lengths = OrderedDict{Int,Int}(pid => 0 for pid in pids)

    offset = 1
    for fock in ordered_focks
        for pid in pids
            lm = length_maps[pid]
            haskey(lm, fock) || continue
            len = lm[fock]
            len == 0 && continue
            owners[fock] = pid
            lengths[fock] = len
            offsets[fock] = offset
            local_lengths[pid] += len
            offset += len
            break
        end
    end
    return DistributedSPTstate{T,N,R}(id, clusters, p_spaces, q_spaces, pids,
                                      owners, lengths, offsets, local_lengths,
                                      offset - 1)
end

function _spt_sharded_refresh_metadata!(state::DistributedSPTstate{T,N,R}) where {T,N,R}
    length_maps = Dict{Int,Any}()
    @sync for pid in state.workers
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(
                _spt_sharded_local_fock_lengths, pid, state.id)
        end
    end
    ordered = collect(keys(state.owners))
    for pid in state.workers
        for fock in keys(length_maps[pid])
            fock in ordered || push!(ordered, fock)
        end
    end
    fresh = _spt_sharded_metadata_from_lengths(state.id, state.clusters,
                                               state.p_spaces, state.q_spaces,
                                               state.workers, length_maps, ordered,
                                               T, Val(R), Val(N))
    state.owners = fresh.owners
    state.lengths = fresh.lengths
    state.offsets = fresh.offsets
    state.local_lengths = fresh.local_lengths
    state.total_length = fresh.total_length
    return state
end

# ---------------------------------------------------------------------------
# distribute / collect / destroy
# ---------------------------------------------------------------------------

"""
    distribute_spt_state(v::SPTstate; workers, id, strategy=:balanced) -> DistributedSPTstate

Shard a local `SPTstate` by Fock sector across workers. `:balanced` assigns
sectors greedily by coefficient count, preserving existing owners on growth;
`:hash` makes ownership a stable function of the Fock sector.
"""
function distribute_spt_state(v::SPTstate{T,N,R};
                              workers=Distributed.workers(),
                              id=nothing,
                              strategy::Symbol=:balanced,
                              blas_threads=1) where {T,N,R}
    pids = ensure_tpsci_multinode_workers!(workers=workers)
    state_id = id === nothing ? gensym(:spt_shard) : Symbol(id)
    local_states = Dict(pid => _spt_empty_local_state(T, Val(R), v.clusters,
                                                      v.p_spaces, v.q_spaces, Val(N))
                        for pid in pids)
    loads = Dict(pid => 0 for pid in pids)
    ordered_focks = FockConfig{N}[]

    for (fock, tconfigs) in v.data
        push!(ordered_focks, fock)
        pid = if strategy == :balanced
            _tpsci_least_loaded_pid(loads, pids)
        elseif strategy == :hash
            pids[Int(mod(hash(fock), UInt(length(pids)))) + 1]
        else
            error("Unknown SPT sharding strategy: $strategy")
        end
        local_states[pid].data[fock] = _spt_copy_fock_block(tconfigs)
        loads[pid] += _spt_block_length(tconfigs)
    end

    length_maps = Dict{Int,Any}()
    @sync for pid in pids
        @async begin
            if pid == Distributed.myid()
                _spt_sharded_store_state!(state_id, local_states[pid])
                blas_threads === nothing || BLAS.set_num_threads(blas_threads)
                length_maps[pid] = _spt_sharded_local_fock_lengths(state_id)
            else
                Distributed.remotecall_fetch(_spt_sharded_store_state!, pid,
                                             state_id, local_states[pid])
                blas_threads === nothing ||
                    Distributed.remotecall_fetch(BLAS.set_num_threads, pid, blas_threads)
                length_maps[pid] = Distributed.remotecall_fetch(
                    _spt_sharded_local_fock_lengths, pid, state_id)
            end
        end
    end

    return _spt_sharded_metadata_from_lengths(state_id, v.clusters, v.p_spaces,
                                              v.q_spaces, pids, length_maps,
                                              ordered_focks, T, Val(R), Val(N))
end

"""
    collect_spt_state(state::DistributedSPTstate) -> SPTstate

Debug/testing helper: gather a sharded SPT state back to one local `SPTstate`.
Do not use on production states that exceed node memory.
"""
function collect_spt_state(state::DistributedSPTstate{T,N,R}) where {T,N,R}
    locals = Dict{Int,SPTstate{T,N,R}}()
    for pid in state.workers
        locals[pid] = Distributed.remotecall_fetch(_spt_sharded_get_state,
                                                   pid, state.id)
    end
    out = _spt_empty_local_state(T, Val(R), state.clusters, state.p_spaces,
                                 state.q_spaces, Val(N))
    for (fock, pid) in state.owners
        haskey(locals[pid].data, fock) || continue
        out.data[fock] = _spt_copy_fock_block(locals[pid].data[fock])
    end
    return out
end

function destroy!(state::DistributedSPTstate)
    @sync for pid in state.workers
        @async Distributed.remotecall_fetch(_spt_sharded_delete_state!, pid, state.id)
    end
    state.owners = empty(state.owners)
    state.lengths = empty(state.lengths)
    state.offsets = empty(state.offsets)
    state.local_lengths = empty(state.local_lengths)
    state.total_length = 0
    return state
end

function sharded_spt_summary(state::DistributedSPTstate)
    rows = []
    for pid in state.workers
        push!(rows, Distributed.remotecall_fetch(_spt_sharded_local_summary,
                                                 pid, state.id))
    end
    return rows
end

# ---------------------------------------------------------------------------
# Sharded linear algebra (same-basis core arithmetic; reduced on the master)
# ---------------------------------------------------------------------------

function _spt_orth_overlap(s1::SPTstate{T,N,R1}, s2::SPTstate{T,N,R2}) where {T,N,R1,R2}
    out = zeros(T, R1, R2)
    for (fock, tconfigs) in s2.data
        haskey(s1.data, fock) || continue
        for (tc, tuck2) in tconfigs
            haskey(s1.data[fock], tc) || continue
            c1 = s1.data[fock][tc].core
            c2 = tuck2.core
            for r2 in 1:R2
                for r1 in 1:R1
                    out[r1, r2] += sum(c1[r1] .* c2[r2])
                end
            end
        end
    end
    return out
end

function _spt_sharded_local_overlap(id1::Symbol, id2::Symbol)
    return _spt_orth_overlap(_spt_sharded_get_state(id1), _spt_sharded_get_state(id2))
end

function overlap(s1::DistributedSPTstate{T,N,R1}, s2::DistributedSPTstate{T,N,R2}) where {T,N,R1,R2}
    s1.workers == s2.workers || error("DistributedSPTstate overlap requires identical worker lists")
    partials = Dict{Int,Matrix{T}}()
    @sync for pid in s1.workers
        @async begin
            partials[pid] = Distributed.remotecall_fetch(_spt_sharded_local_overlap,
                                                         pid, s1.id, s2.id)
        end
    end
    out = zeros(T, R1, R2)
    for pid in s1.workers
        out .+= partials[pid]
    end
    return out
end

function _spt_local_normsq(state::SPTstate{T,N,R}) where {T,N,R}
    norms = zeros(T, R)
    for (_, tconfigs) in state.data
        for (_, tuck) in tconfigs
            for r in 1:R
                norms[r] += sum(abs2, tuck.core[r])
            end
        end
    end
    return norms
end

_spt_sharded_local_normsq(id::Symbol) = _spt_local_normsq(_spt_sharded_get_state(id))

function LinearAlgebra.norm(state::DistributedSPTstate{T,N,R}) where {T,N,R}
    partials = Dict{Int,Vector{T}}()
    @sync for pid in state.workers
        @async begin
            partials[pid] = Distributed.remotecall_fetch(_spt_sharded_local_normsq,
                                                         pid, state.id)
        end
    end
    norms = zeros(T, R)
    for pid in state.workers
        norms .+= partials[pid]
    end
    return sqrt.(norms)
end

# scale! by a scalar: worker-local per-block core scaling.
function _spt_local_scale!(state::SPTstate{T,N,R}, alpha) where {T,N,R}
    a = T(alpha)
    for (_, tconfigs) in state.data
        for (_, tuck) in tconfigs
            for r in 1:R
                tuck.core[r] .*= a
            end
        end
    end
    return true
end

_spt_sharded_local_scale_typed!(id::Symbol, alpha) = _spt_local_scale!(_spt_sharded_get_state(id), alpha)

function scale!(state::DistributedSPTstate, alpha)
    @sync for pid in state.workers
        @async Distributed.remotecall_fetch(_spt_sharded_local_scale_typed!, pid,
                                            state.id, alpha)
    end
    return state
end

function _spt_sharded_local_zero!(id::Symbol)
    zero!(_spt_sharded_get_state(id))
    return true
end

function zero!(state::DistributedSPTstate)
    @sync for pid in state.workers
        @async Distributed.remotecall_fetch(_spt_sharded_local_zero!, pid, state.id)
    end
    return state
end

function _spt_local_add_scaled!(dest::SPTstate{T,N,R}, alpha, src::SPTstate{T,N,R}) where {T,N,R}
    a = T(alpha)
    for (fock, tconfigs) in src.data
        haskey(dest.data, fock) || (dest.data[fock] = OrderedDict{TuckerConfig{N},Tucker{T,N,R}}())
        for (tc, tuck) in tconfigs
            if haskey(dest.data[fock], tc)
                dc = dest.data[fock][tc].core
                for r in 1:R
                    dc[r] .+= a .* tuck.core[r]
                end
            else
                newtuck = deepcopy(tuck)
                for r in 1:R
                    newtuck.core[r] .*= a
                end
                dest.data[fock][tc] = newtuck
            end
        end
    end
    return true
end

function _spt_sharded_local_add_scaled!(dest_id::Symbol, alpha, src_id::Symbol)
    return _spt_local_add_scaled!(_spt_sharded_get_state(dest_id), alpha,
                                  _spt_sharded_get_state(src_id))
end

function add_scaled!(dest::DistributedSPTstate{T,N,R}, alpha,
                     src::DistributedSPTstate{T,N,R}) where {T,N,R}
    dest.workers == src.workers || error("DistributedSPTstate add_scaled! requires identical worker lists")
    @sync for pid in dest.workers
        @async Distributed.remotecall_fetch(_spt_sharded_local_add_scaled!, pid,
                                            dest.id, alpha, src.id)
    end
    return dest
end

function _spt_sharded_local_copy_state!(dest_id::Symbol, src_id::Symbol)
    _spt_sharded_store_state!(dest_id, deepcopy(_spt_sharded_get_state(src_id)))
    return _spt_sharded_local_fock_lengths(dest_id)
end

function copy_sharded_state(state::DistributedSPTstate{T,N,R}; id=nothing) where {T,N,R}
    output_id = id === nothing ? gensym(:spt_shard_copy) : Symbol(id)
    length_maps = Dict{Int,Any}()
    @sync for pid in state.workers
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(
                _spt_sharded_local_copy_state!, pid, output_id, state.id)
        end
    end
    return _spt_sharded_metadata_from_lengths(output_id, state.clusters,
                                              state.p_spaces, state.q_spaces,
                                              state.workers, length_maps,
                                              collect(keys(state.owners)),
                                              T, Val(R), Val(N))
end

function _spt_local_similar(src::SPTstate{T,N,R}) where {T,N,R}
    out = deepcopy(src)
    zero!(out)
    return out
end

function _spt_sharded_local_similar_state!(dest_id::Symbol, src_id::Symbol)
    _spt_sharded_store_state!(dest_id, _spt_local_similar(_spt_sharded_get_state(src_id)))
    return _spt_sharded_local_fock_lengths(dest_id)
end

function similar_sharded_state(state::DistributedSPTstate{T,N,R}; id=nothing) where {T,N,R}
    output_id = id === nothing ? gensym(:spt_shard_similar) : Symbol(id)
    length_maps = Dict{Int,Any}()
    @sync for pid in state.workers
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(
                _spt_sharded_local_similar_state!, pid, output_id, state.id)
        end
    end
    return _spt_sharded_metadata_from_lengths(output_id, state.clusters,
                                              state.p_spaces, state.q_spaces,
                                              state.workers, length_maps,
                                              collect(keys(state.owners)),
                                              T, Val(R), Val(N))
end

# R x R symmetric orthonormalization of the roots (mult! cores by X).
function _spt_local_mult!(state::SPTstate{T,N,R}, X) where {T,N,R}
    for (_, tconfigs) in state.data
        for (_, tuck) in tconfigs
            old = ntuple(r -> copy(tuck.core[r]), R)
            for r in 1:R
                nc = tuck.core[r]
                fill!(nc, zero(T))
                for s in 1:R
                    nc .+= old[s] .* X[s, r]
                end
            end
        end
    end
    return true
end

_spt_sharded_local_mult!(id::Symbol, X) = _spt_local_mult!(_spt_sharded_get_state(id), X)

function orthonormalize!(state::DistributedSPTstate{T,N,R}) where {T,N,R}
    S = Symmetric(overlap(state, state))
    F = eigen(S)
    minimum(F.values) > eps(T) ||
        error("Cannot orthonormalize sharded SPT state with linearly dependent roots")
    X = F.vectors * Diagonal(1 ./ sqrt.(F.values)) * F.vectors'
    @sync for pid in state.workers
        @async Distributed.remotecall_fetch(_spt_sharded_local_mult!, pid, state.id, X)
    end
    return state
end

# ---------------------------------------------------------------------------
# Root extract / combine
# ---------------------------------------------------------------------------

function _spt_extract_root(src::SPTstate{T,N,R}, root::Int) where {T,N,R}
    data = OrderedDict{FockConfig{N},OrderedDict{TuckerConfig{N},Tucker{T,N,1}}}()
    for (fock, tconfigs) in src.data
        d = OrderedDict{TuckerConfig{N},Tucker{T,N,1}}()
        for (tc, tuck) in tconfigs
            d[tc] = Tucker{T,N,1}((copy(tuck.core[root]),), deepcopy(tuck.factors))
        end
        data[fock] = d
    end
    return SPTstate{T,N,1}(src.clusters, data, src.p_spaces, src.q_spaces)
end

function _spt_sharded_local_extract_root!(dest_id::Symbol, src_id::Symbol, root::Int)
    _spt_sharded_store_state!(dest_id, _spt_extract_root(_spt_sharded_get_state(src_id), root))
    return _spt_sharded_local_fock_lengths(dest_id)
end

function extract_root_sharded(state::DistributedSPTstate{T,N,R}, root::Integer;
                              id=nothing) where {T,N,R}
    1 <= root <= R || throw(BoundsError(state, root))
    output_id = id === nothing ? gensym(:spt_shard_root) : Symbol(id)
    length_maps = Dict{Int,Any}()
    @sync for pid in state.workers
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(
                _spt_sharded_local_extract_root!, pid, output_id, state.id, Int(root))
        end
    end
    return _spt_sharded_metadata_from_lengths(output_id, state.clusters,
                                              state.p_spaces, state.q_spaces,
                                              state.workers, length_maps,
                                              collect(keys(state.owners)),
                                              T, Val(1), Val(N))
end

function _spt_combine_roots(srcs::Vector{<:SPTstate{T,N,1}}) where {T,N}
    R = length(srcs)
    ref = srcs[1]
    data = OrderedDict{FockConfig{N},OrderedDict{TuckerConfig{N},Tucker{T,N,R}}}()
    for (fock, tconfigs) in ref.data
        d = OrderedDict{TuckerConfig{N},Tucker{T,N,R}}()
        for (tc, tuck) in tconfigs
            cores = ntuple(r -> copy(srcs[r].data[fock][tc].core[1]), R)
            d[tc] = Tucker{T,N,R}(cores, deepcopy(tuck.factors))
        end
        data[fock] = d
    end
    return SPTstate{T,N,R}(ref.clusters, data, ref.p_spaces, ref.q_spaces)
end

function _spt_sharded_local_combine_roots!(dest_id::Symbol, src_ids::Vector{Symbol})
    srcs = [_spt_sharded_get_state(id) for id in src_ids]
    _spt_sharded_store_state!(dest_id, _spt_combine_roots(srcs))
    return _spt_sharded_local_fock_lengths(dest_id)
end

function combine_roots_sharded(vecs::Vector{<:DistributedSPTstate{T,N,1}}; id=nothing) where {T,N}
    R = length(vecs)
    R >= 1 || error("combine_roots_sharded needs at least one vector")
    workers = vecs[1].workers
    for v in vecs
        v.workers == workers || error("combine_roots_sharded requires identical worker lists")
    end
    src_ids = [v.id for v in vecs]
    output_id = id === nothing ? gensym(:spt_shard_combine) : Symbol(id)
    length_maps = Dict{Int,Any}()
    @sync for pid in workers
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(
                _spt_sharded_local_combine_roots!, pid, output_id, src_ids)
        end
    end
    return _spt_sharded_metadata_from_lengths(output_id, vecs[1].clusters,
                                              vecs[1].p_spaces, vecs[1].q_spaces,
                                              workers, length_maps,
                                              collect(keys(vecs[1].owners)),
                                              T, Val(R), Val(N))
end

# ---------------------------------------------------------------------------
# Diagonal (Davidson) preconditioner: t = res / (theta - Fdiag), per core entry
# ---------------------------------------------------------------------------

function _spt_local_precondition(res::SPTstate{T,N,1}, diag::SPTstate{T,N,1}, theta) where {T,N}
    thetaT = T(theta)
    out = deepcopy(res)
    for (fock, tconfigs) in out.data
        for (tc, tuck) in tconfigs
            rc = res.data[fock][tc].core[1]
            dc = diag.data[fock][tc].core[1]
            oc = tuck.core[1]
            for i in eachindex(oc)
                d0 = thetaT - dc[i]
                sc = max(abs(thetaT), abs(dc[i]), one(T))
                floor = sqrt(eps(T)) * sc
                d = abs(d0) < floor ? copysign(floor, iszero(d0) ? one(T) : d0) : d0
                oc[i] = rc[i] / d
            end
        end
    end
    return out
end

function _spt_sharded_local_precondition!(dest_id::Symbol, res_id::Symbol,
                                          diag_id::Symbol, theta)
    out = _spt_local_precondition(_spt_sharded_get_state(res_id),
                                  _spt_sharded_get_state(diag_id), theta)
    _spt_sharded_store_state!(dest_id, out)
    return _spt_sharded_local_fock_lengths(dest_id)
end

function precondition_sharded(res::DistributedSPTstate{T,N,1},
                              Fdiag::DistributedSPTstate{T,N,1}, theta;
                              id=nothing) where {T,N}
    res.workers == Fdiag.workers || error("precondition_sharded requires identical worker lists")
    output_id = id === nothing ? gensym(:spt_shard_precond) : Symbol(id)
    length_maps = Dict{Int,Any}()
    @sync for pid in res.workers
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(
                _spt_sharded_local_precondition!, pid, output_id, res.id, Fdiag.id, theta)
        end
    end
    return _spt_sharded_metadata_from_lengths(output_id, res.clusters,
                                              res.p_spaces, res.q_spaces,
                                              res.workers, length_maps,
                                              collect(keys(res.owners)),
                                              T, Val(1), Val(N))
end

# ---------------------------------------------------------------------------
# Sharded CMF diagonal + pseudo-canonical basis rotation (fixes the basis).
# Runs form_1body_operator_diagonal! per shard: rotates each block's factors in
# place (per-block, so identical to the single-node result) and returns Fdiag.
# ---------------------------------------------------------------------------

function _spt_sharded_local_diagonal!(diag_id::Symbol, state_id::Symbol, H0::String)
    state = _spt_sharded_get_state(state_id)
    cluster_ops = _TPSCI_MULTINODE_CLUSTER_OPS[]
    cluster_ops === nothing && error("SPT sharded cluster-ops cache is empty")
    if cluster_ops isa DistributedClusterOps
        cluster_ops = _materialize_cluster_ops_for_indices(
            cluster_ops, [c.idx for c in state.clusters])
    end
    Fdiag = SPTstate(state, R=1)
    form_1body_operator_diagonal!(state, Fdiag, cluster_ops; H0=H0, pseudo_canon=true)
    _spt_sharded_store_state!(diag_id, Fdiag)
    return _spt_sharded_local_fock_lengths(diag_id)
end

function spt_diagonal_sharded!(state::DistributedSPTstate{T,N,R}; H0="Hcmf", id=nothing) where {T,N,R}
    diag_id = id === nothing ? gensym(:spt_shard_diag) : Symbol(id)
    length_maps = Dict{Int,Any}()
    @sync for pid in state.workers
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(
                _spt_sharded_local_diagonal!, pid, diag_id, state.id, String(H0))
        end
    end
    return _spt_sharded_metadata_from_lengths(diag_id, state.clusters,
                                              state.p_spaces, state.q_spaces,
                                              state.workers, length_maps,
                                              collect(keys(state.owners)),
                                              T, Val(1), Val(N))
end

# ---------------------------------------------------------------------------
# Never-gather SPT matvec: H*v in the fixed Tucker basis, output stays sharded.
# ---------------------------------------------------------------------------

function _spt_sharded_local_copy_fock_block(id::Symbol, fock)
    state = _spt_sharded_get_state(id)
    haskey(state.data, fock) || return nothing
    return _spt_copy_fock_block(state.data[fock])
end

function _spt_sharded_get_ket_block(v::DistributedSPTstate{T,N,R}, fock, ket_cache) where {T,N,R}
    haskey(ket_cache, fock) && return ket_cache[fock]
    owner = v.owners[fock]
    block = if owner == Distributed.myid()
        _spt_copy_fock_block(_spt_sharded_get_state(v.id).data[fock])
    else
        Distributed.remotecall_fetch(_spt_sharded_local_copy_fock_block, owner, v.id, fock)
    end
    block === nothing && error("Missing sharded ket Fock block $fock on owner $owner")
    ket_cache[fock] = block
    return block
end

# Worker kernel: build the sigma blocks for `owned_bras` (in the bra state's
# fixed Tucker basis) from a possibly different ket state, fetching the connected
# ket Tucker blocks. Reads the operator cache. When bra===ket this is H*v; when
# they differ it is <bra-basis|H|ket> (used by PT1's <X|H|0>).
function _spt_build_sigma_into_chunk!(out_id::Symbol, bra::DistributedSPTstate{T,N,R},
                                      ket::DistributedSPTstate{T,N,R}, owned_bras,
                                      nbody, blas_threads) where {T,N,R}
    blas_threads === nothing || BLAS.set_num_threads(blas_threads)
    cluster_ops = _TPSCI_MULTINODE_CLUSTER_OPS[]
    clustered_ham = _TPSCI_MULTINODE_CLUSTERED_HAM[]
    cluster_ops === nothing && error("SPT sharded cluster-ops cache is empty")
    clustered_ham === nothing && error("SPT sharded Hamiltonian cache is empty")
    if cluster_ops isa DistributedClusterOps
        cluster_ops = _materialize_cluster_ops_for_indices(
            cluster_ops, [c.idx for c in bra.clusters])
    end

    local_bra = _spt_sharded_get_state(bra.id)

    # Output: zeroed copy of this worker's owned bra sectors (fixed basis).
    sig = _spt_empty_local_state(T, Val(R), bra.clusters, bra.p_spaces, bra.q_spaces, Val(N))
    for fock_bra in owned_bras
        sig.data[fock_bra] = _spt_copy_fock_block(local_bra.data[fock_bra])
    end
    zero!(sig)

    # Ket: every ket Fock sector connected by H to one of our owned bras.
    ketl = _spt_empty_local_state(T, Val(R), ket.clusters, ket.p_spaces, ket.q_spaces, Val(N))
    ket_cache = Dict{FockConfig{N},Any}()
    for fock_bra in owned_bras
        for fock_ket in keys(ket.owners)
            fock_trans = fock_bra - fock_ket
            haskey(clustered_ham, fock_trans) || continue
            haskey(ketl.data, fock_ket) && continue
            ketl.data[fock_ket] = _spt_sharded_get_ket_block(ket, fock_ket, ket_cache)
        end
    end

    build_sigma!(sig, ketl, cluster_ops, clustered_ham; nbody=nbody, verbose=0)
    _spt_sharded_store_state!(out_id, sig)
    return _spt_sharded_local_fock_lengths(out_id)
end

function _spt_build_sigma_into(bra::DistributedSPTstate{T,N,R},
                               ket::DistributedSPTstate{T,N,R}, nbody, blas_threads,
                               id) where {T,N,R}
    output_id = id === nothing ? gensym(:spt_shard_sigma) : Symbol(id)
    chunks = Dict(pid => FockConfig{N}[] for pid in bra.workers)
    for (fock, owner) in bra.owners
        push!(chunks[owner], fock)
    end
    length_maps = Dict{Int,Any}()
    @sync for pid in bra.workers
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(
                _spt_build_sigma_into_chunk!, pid, output_id, bra, ket, chunks[pid],
                nbody, blas_threads)
        end
    end
    return _spt_sharded_metadata_from_lengths(output_id, bra.clusters, bra.p_spaces,
                                              bra.q_spaces, bra.workers, length_maps,
                                              collect(keys(bra.owners)),
                                              T, Val(R), Val(N))
end

"""
    build_sigma_sharded(v, cluster_ops, clustered_ham; nbody, ...) -> DistributedSPTstate

Never-gather SPT matvec: apply `H` to a sharded SPT vector in its own (fixed)
Tucker basis. Each worker builds the sigma blocks for the Fock sectors it owns,
fetching only the connected ket Tucker blocks it does not own. The output keeps
`v`'s Fock ownership. Assumes the operator cache is primed
(`_tpsci_sharded_cache_operator_problem!`) — this is the hot-loop matvec, so it
does not re-cache.
"""
function build_sigma_sharded(v::DistributedSPTstate{T,N,R}, cluster_ops, clustered_ham;
                             nbody=4, workers=v.workers, threaded_worker=true,
                             blas_threads=1, id=nothing) where {T,N,R}
    return _spt_build_sigma_into(v, v, nbody, blas_threads, id)
end

"""
    build_sigma_into_sharded(bra_basis, ket, cluster_ops, clustered_ham; nbody, ...) -> DistributedSPTstate

Never-gather `<bra_basis|H|ket>`: apply `H` to `ket` and express the result in
the fixed Tucker basis of `bra_basis` (a possibly different sharded state, e.g.
the FOIS). Output keeps `bra_basis`'s ownership. Caches the passed operator (not
a hot-loop call), so it is safe to use with a different operator than the one the
solver cached.
"""
function build_sigma_into_sharded(bra_basis::DistributedSPTstate{T,N,R},
                                  ket::DistributedSPTstate{T,N,R}, cluster_ops,
                                  clustered_ham; nbody=4, workers=bra_basis.workers,
                                  blas_threads=1, id=nothing) where {T,N,R}
    _tpsci_sharded_cache_operator_problem!(cluster_ops, clustered_ham;
                                           workers=workers, blas_threads=blas_threads)
    return _spt_build_sigma_into(bra_basis, ket, nbody, blas_threads, id)
end

# ---------------------------------------------------------------------------
# Never-gather FOIS: build the first-order interacting space as a sharded state.
# Each output Fock sector is built entirely by its owner, which fetches the
# reference (ket) Tucker blocks feeding it. No output sector is split across
# workers, so no cross-worker merge is needed.
# ---------------------------------------------------------------------------

function _spt_fois_sharded_chunk!(out_id::Symbol, ref::DistributedSPTstate{T,N,R},
                                  assignments, nbody, thresh, max_number,
                                  prescreen, compress_twice, blas_threads) where {T,N,R}
    blas_threads === nothing || BLAS.set_num_threads(blas_threads)
    cluster_ops = _TPSCI_MULTINODE_CLUSTER_OPS[]
    clustered_ham = _TPSCI_MULTINODE_CLUSTERED_HAM[]
    cluster_ops === nothing && error("SPT sharded cluster-ops cache is empty")
    clustered_ham === nothing && error("SPT sharded Hamiltonian cache is empty")
    if cluster_ops isa DistributedClusterOps
        cluster_ops = _materialize_cluster_ops_for_indices(
            cluster_ops, [c.idx for c in ref.clusters])
    end

    σ = _spt_empty_local_state(T, Val(R), ref.clusters, ref.p_spaces, ref.q_spaces, Val(N))
    ket_cache = Dict{FockConfig{N},Any}()
    scr_v = [zeros(T, 1000) for _ in 1:N]

    for (fock_bra, fock_kets) in assignments
        _spt_has_cluster_space(fock_bra, ref.clusters, cluster_ops) || continue
        jobs = Tuple[]
        for fock_ket in fock_kets
            ftrans = fock_bra - fock_ket
            haskey(clustered_ham, ftrans) || continue
            ket_block = _spt_sharded_get_ket_block(ref, fock_ket, ket_cache)
            push!(jobs, (clustered_ham[ftrans], fock_ket, ket_block))
        end
        isempty(jobs) && continue
        _build_compressed_1st_order_state_job(fock_bra, jobs, σ, cluster_ops, nbody,
                                              thresh, max_number, prescreen,
                                              compress_twice, scr_v)
    end
    _spt_sharded_store_state!(out_id, σ)
    return _spt_sharded_local_fock_lengths(out_id)
end

"""
    build_compressed_1st_order_state_sharded(ref, cluster_ops, clustered_ham; ...) -> DistributedSPTstate

Never-gather SPT FOIS builder. The first-order interacting space of the sharded
reference `ref` is produced directly as a `DistributedSPTstate`: the master
enumerates the output Fock sectors (metadata only) and assigns each to a worker,
which fetches the reference Tucker blocks feeding that sector and runs the
existing local kernel. The full FOIS is never assembled on any single node —
this is the piece that removes the master-side FOIS gather of
`build_compressed_1st_order_state_distributed`.

`existing_owners` (optional `FockConfig => pid`) pins sectors already present in
a variational state to their current owner, so a later `add!`-style merge is
ownership-consistent.
"""
function build_compressed_1st_order_state_sharded(ref::DistributedSPTstate{T,N,R},
                                                  cluster_ops, clustered_ham;
                                                  thresh=1e-7,
                                                  max_number=nothing,
                                                  nbody=4,
                                                  prescreen=false,
                                                  compress_twice=true,
                                                  workers=ref.workers,
                                                  blas_threads=1,
                                                  strategy::Symbol=:balanced,
                                                  existing_owners=nothing,
                                                  id=nothing) where {T,N,R}
    worker_ids = ensure_tpsci_multinode_workers!(workers=workers)
    _tpsci_sharded_cache_operator_problem!(cluster_ops, clustered_ham;
                                           workers=worker_ids, blas_threads=blas_threads)

    # Enumerate candidate output sectors and their contributing reference kets
    # from metadata alone (no coefficients touched on the master).
    cand = OrderedDict{FockConfig{N},Vector{FockConfig{N}}}()
    for fock_ket in keys(ref.owners)
        for (ftrans, _) in clustered_ham
            fock_bra = ftrans + fock_ket
            _spt_valid_fock(fock_bra, ref.clusters) || continue
            if haskey(cand, fock_bra)
                push!(cand[fock_bra], fock_ket)
            else
                cand[fock_bra] = FockConfig{N}[fock_ket]
            end
        end
    end

    assign = Dict(pid => Tuple{FockConfig{N},Vector{FockConfig{N}}}[] for pid in worker_ids)
    loads = Dict(pid => 0 for pid in worker_ids)
    for (fock_bra, kets) in cand
        pid = if existing_owners !== nothing && haskey(existing_owners, fock_bra)
            existing_owners[fock_bra]
        elseif strategy == :balanced
            _tpsci_least_loaded_pid(loads, worker_ids)
        elseif strategy == :hash
            worker_ids[Int(mod(hash(fock_bra), UInt(length(worker_ids)))) + 1]
        else
            error("Unknown SPT FOIS sharding strategy: $strategy")
        end
        push!(assign[pid], (fock_bra, kets))
        loads[pid] += length(kets)
    end

    output_id = id === nothing ? gensym(:spt_shard_fois) : Symbol(id)
    length_maps = Dict{Int,Any}()
    @sync for pid in worker_ids
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(
                _spt_fois_sharded_chunk!, pid, output_id, ref, assign[pid], nbody,
                thresh, max_number, prescreen, compress_twice, blas_threads)
        end
    end
    return _spt_sharded_metadata_from_lengths(output_id, ref.clusters, ref.p_spaces,
                                              ref.q_spaces, worker_ids, length_maps,
                                              collect(keys(cand)), T, Val(R), Val(N))
end

struct MatrixFreeShardedSPT{H}
    cluster_ops
    clustered_ham::H
    nbody::Int
    workers::Vector{Int}
    threaded_worker::Bool
    blas_threads::Int
end

function apply_sharded_H(op::MatrixFreeShardedSPT, v::DistributedSPTstate)
    return build_sigma_sharded(v, op.cluster_ops, op.clustered_ham;
                               nbody=op.nbody, workers=op.workers,
                               threaded_worker=op.threaded_worker,
                               blas_threads=op.blas_threads)
end

# ---------------------------------------------------------------------------
# The distributed Tucker CI solver.
# ---------------------------------------------------------------------------

"""
    spt_ci_davidson_sharded(dv, cluster_ops, clustered_ham; nroots=R, ...) -> (eigvals, DistributedSPTstate)

Diagonalize `H` in the fixed Tucker basis of a sharded `DistributedSPTstate`,
never gathering a full vector. Mirrors the single-node `ci_solve`: orthonormalize
the roots, rotate to the pseudo-canonical CMF basis (which also gives the
diagonal preconditioner `Fdiag`), then run the shared sharded block Davidson with
a matrix-free `build_sigma_sharded` matvec.

Note: `dv` is rotated in place (its shards' Tucker factors) so that all subspace
vectors share one basis. Returns `(eigvals, ci_out)`; `destroy!(ci_out)` when
done. `ci_out` is expressed in that rotated basis (unitarily equivalent).
"""
function spt_ci_davidson_sharded(dv::DistributedSPTstate{T,N,R}, cluster_ops, clustered_ham;
                                 nroots::Int=R,
                                 conv_thresh=1e-5,
                                 lindep_thresh=1e-10,
                                 max_ss_vecs=12,
                                 max_iter=100,
                                 nbody=4,
                                 H0="Hcmf",
                                 workers=dv.workers,
                                 threaded_worker=true,
                                 blas_threads=1,
                                 verbose=1, id=nothing) where {T,N,R}
    nroots <= R || error("spt_ci_davidson_sharded needs the guess to carry at least nroots ($nroots) roots; got R=$R")
    worker_ids = ensure_tpsci_multinode_workers!(workers=workers)
    worker_ids == dv.workers ||
        error("spt_ci_davidson_sharded requires the guess on the requested workers")

    if verbose > 0
        println()
        @printf(" |== Sharded SPT CI ================================================\n")
        @printf(" Hamiltonian matrix dimension = %i\n", length(dv))
        @printf(" nroots = %i   max_ss_vecs = %i   conv_thresh = %.1e\n",
                nroots, max_ss_vecs, conv_thresh)
        flush(stdout)
    end

    _tpsci_sharded_cache_operator_problem!(cluster_ops, clustered_ham;
                                           workers=worker_ids, blas_threads=blas_threads)

    orthonormalize!(dv)
    # Rotate to pseudo-canonical CMF basis; Fdiag is the diagonal preconditioner.
    Fdiag = spt_diagonal_sharded!(dv; H0=H0)

    op = MatrixFreeShardedSPT(cluster_ops, clustered_ham, Int(nbody), worker_ids,
                              threaded_worker, Int(blas_threads))
    apply_H = v -> apply_sharded_H(op, v)

    seeds = DistributedSPTstate{T,N,1}[]
    for r in 1:nroots
        s = extract_root_sharded(dv, r)
        nrm = _mgs_against!(s, seeds)
        nrm > lindep_thresh || error("initial guess root $r is linearly dependent")
        scale!(s, one(T) / nrm)
        push!(seeds, s)
    end

    tstart = time()
    eigvals, ritz = _tps_ci_davidson_sharded_core(seeds, apply_H, Fdiag;
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
    destroy!(Fdiag)

    if verbose > 0
        @printf(" Sharded SPT Davidson finished in %.2f s\n", elapsed)
        @printf(" %5s %18s\n", "Root", "Energy")
        for r in 1:nroots
            @printf(" %5i %18.10f\n", r, eigvals[r])
        end
        @printf(" ==================================================================|\n")
        flush(stdout)
    end
    return eigvals, ci_out
end

# ---------------------------------------------------------------------------
# Driver-level sharded ops. These assume :hash ownership so that any two states
# sharing a Fock sector place it on the same worker — making compress /
# nonorth_add! / project_into_new_basis fully worker-local (only the matvec and
# FOIS need cross-Fock fetches). The never-gather SPT driver enforces :hash.
# ---------------------------------------------------------------------------

function _spt_compress_chunk!(out_id::Symbol, src_id::Symbol, thresh, max_number)
    _spt_sharded_store_state!(out_id,
        compress(_spt_sharded_get_state(src_id); thresh=thresh, max_number=max_number))
    return _spt_sharded_local_fock_lengths(out_id)
end

"""
    compress_sharded(dv; thresh, max_number) -> DistributedSPTstate

Tucker-compress each shard locally (compression is per-`(fock,tconfig)`
independent, so no communication). Returns a fresh sharded state; `destroy!` the
input if no longer needed.
"""
function compress_sharded(dv::DistributedSPTstate{T,N,R}; thresh=-1,
                          max_number=nothing, id=nothing) where {T,N,R}
    output_id = id === nothing ? gensym(:spt_shard_compress) : Symbol(id)
    length_maps = Dict{Int,Any}()
    @sync for pid in dv.workers
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(
                _spt_compress_chunk!, pid, output_id, dv.id, thresh, max_number)
        end
    end
    return _spt_sharded_metadata_from_lengths(output_id, dv.clusters, dv.p_spaces,
                                              dv.q_spaces, dv.workers, length_maps,
                                              collect(keys(dv.owners)),
                                              T, Val(R), Val(N))
end

function _spt_nonorth_add_chunk!(dest_id::Symbol, src_id::Symbol)
    nonorth_add!(_spt_sharded_get_state(dest_id), _spt_sharded_get_state(src_id))
    return _spt_sharded_local_fock_lengths(dest_id)
end

"""
    nonorth_add_sharded!(dest, src)

Worker-local `nonorth_add!` of `src` into `dest`. Requires `:hash`-consistent
ownership (so a Fock sector present in both, or new from `src`, lands on the same
worker). Grows `dest` in place; metadata is refreshed.
"""
function nonorth_add_sharded!(dest::DistributedSPTstate{T,N,R},
                              src::DistributedSPTstate{T,N,R}) where {T,N,R}
    dest.workers == src.workers || error("nonorth_add_sharded! requires identical worker lists")
    @sync for pid in dest.workers
        @async Distributed.remotecall_fetch(_spt_nonorth_add_chunk!, pid, dest.id, src.id)
    end
    return _spt_sharded_refresh_metadata!(dest)
end

function _spt_project_chunk!(out_id::Symbol, v1_id::Symbol, v2_id::Symbol)
    _spt_sharded_store_state!(out_id,
        project_into_new_basis(_spt_sharded_get_state(v1_id),
                               _spt_sharded_get_state(v2_id)))
    return _spt_sharded_local_fock_lengths(out_id)
end

"""
    project_into_new_basis_sharded(v1, v2) -> DistributedSPTstate

Project `v1` into the Tucker basis of `v2`, worker-locally. Output takes `v2`'s
basis and ownership. Requires `:hash`-consistent ownership so each `v2` Fock
sector and the matching `v1` block are co-located.
"""
function project_into_new_basis_sharded(v1::DistributedSPTstate{T,N,R},
                                        v2::DistributedSPTstate{T,N,R}) where {T,N,R}
    v1.workers == v2.workers || error("project_into_new_basis_sharded requires identical worker lists")
    output_id = gensym(:spt_shard_project)
    length_maps = Dict{Int,Any}()
    @sync for pid in v2.workers
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(
                _spt_project_chunk!, pid, output_id, v1.id, v2.id)
        end
    end
    return _spt_sharded_metadata_from_lengths(output_id, v2.clusters, v2.p_spaces,
                                              v2.q_spaces, v2.workers, length_maps,
                                              collect(keys(v2.owners)),
                                              T, Val(R), Val(N))
end

"""
    spt_expectation_sharded(dv, cluster_ops, op; nbody) -> Vector

Per-root expectation value `<dv|op|dv>` (e.g. energy or S²) without gathering:
apply `op` in `dv`'s basis, then take the sharded overlap diagonal. Re-caches the
operator, so it is safe to call with a different operator than the solver used.
"""
function spt_expectation_sharded(dv::DistributedSPTstate{T,N,R}, cluster_ops, op;
                                 nbody=4, workers=dv.workers, blas_threads=1) where {T,N,R}
    _tpsci_sharded_cache_operator_problem!(cluster_ops, op;
                                           workers=workers, blas_threads=blas_threads)
    sig = build_sigma_sharded(dv, cluster_ops, op; nbody=nbody, workers=workers,
                              blas_threads=blas_threads)
    e = LinearAlgebra.diag(overlap(dv, sig))
    destroy!(sig)
    return e
end

# Shard-local PT1 assembly: res = XH0 - XF0 - Sx .* (E0-F0)_r  (same basis).
function _spt_pt1_assemble_chunk!(res_id::Symbol, xh0_id::Symbol, xf0_id::Symbol,
                                  sx_id::Symbol, coef)
    xh0 = _spt_sharded_get_state(xh0_id)
    xf0 = _spt_sharded_get_state(xf0_id)
    sx = _spt_sharded_get_state(sx_id)
    out = deepcopy(xh0)
    for (fock, tcs) in out.data
        for (tc, tuck) in tcs
            for r in eachindex(tuck.core)
                oc = tuck.core[r]
                oc .-= xf0.data[fock][tc].core[r]
                oc .-= coef[r] .* sx.data[fock][tc].core[r]
            end
        end
    end
    _spt_sharded_store_state!(res_id, out)
    return _spt_sharded_local_fock_lengths(res_id)
end

# Shard-local PT1 denominators: psi1 = res / (F0_r - Fdiag) per core entry.
function _spt_pt1_denominator_chunk!(psi1_id::Symbol, res_id::Symbol,
                                     fdiag_id::Symbol, F0)
    res = _spt_sharded_get_state(res_id)
    fdiag = _spt_sharded_get_state(fdiag_id)
    out = deepcopy(res)
    for (fock, tcs) in out.data
        for (tc, tuck) in tcs
            dcore = fdiag.data[fock][tc].core[1]
            for r in eachindex(tuck.core)
                oc = tuck.core[r]
                rc = res.data[fock][tc].core[r]
                for i in eachindex(oc)
                    oc[i] = rc[i] / (F0[r] - dcore[i] + 1e-12)
                end
            end
        end
    end
    _spt_sharded_store_state!(psi1_id, out)
    return _spt_sharded_local_fock_lengths(psi1_id)
end

function _spt_dispatch_local!(fn, out_id, workers, ordered_focks, clusters,
                              p_spaces, q_spaces, ::Type{T}, ::Val{R}, ::Val{N},
                              args...) where {T,R,N}
    length_maps = Dict{Int,Any}()
    @sync for pid in workers
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(fn, pid, out_id, args...)
        end
    end
    return _spt_sharded_metadata_from_lengths(out_id, clusters, p_spaces, q_spaces,
                                              workers, length_maps, ordered_focks,
                                              T, Val(R), Val(N))
end

"""
    compute_pt1_wavefunction_sharded(fois, psi0, cluster_ops, clustered_ham; H0, nbody, ...)
        -> (psi1::DistributedSPTstate, E2::Vector, ecorr::Vector)

Never-gather SPT-PT1/PT2. `fois` is the (sharded) first-order interacting space;
`psi0` the sharded reference. Rotates the FOIS to its pseudo-canonical basis
(shard-local, giving the diagonal `Fdiag`), forms `Sx = P_psi0` in that basis,
builds `<X|H|0>` and `<X|F|0>` with the never-gather matvec, then assembles the
first-order wavefunction and PT2 energy entirely from shard-local core
arithmetic and sharded reductions. No full FOIS/PT1 vector is ever gathered.
Requires `:hash`-consistent ownership between `fois` and `psi0`.
"""
function compute_pt1_wavefunction_sharded(fois::DistributedSPTstate{T,N,R},
                                          psi0::DistributedSPTstate{T,N,R},
                                          cluster_ops, clustered_ham;
                                          H0="Hcmf", nbody=4,
                                          workers=fois.workers, blas_threads=1,
                                          verbose=1) where {T,N,R}
    worker_ids = ensure_tpsci_multinode_workers!(workers=workers)
    clustered_ham_0 = extract_1body_operator(clustered_ham, op_string=H0)

    # Zeroth-order energies (never-gather expectation values).
    E0 = spt_expectation_sharded(psi0, cluster_ops, clustered_ham;
                                 nbody=nbody, workers=worker_ids, blas_threads=blas_threads)
    F0 = spt_expectation_sharded(psi0, cluster_ops, clustered_ham_0;
                                 nbody=1, workers=worker_ids, blas_threads=blas_threads)

    # Fixed FOIS basis + diagonal preconditioner (rotates `sigma` in place).
    sigma = copy_sharded_state(fois)
    zero!(sigma)
    Fdiag = spt_diagonal_sharded!(sigma; H0=H0)

    # Sx = psi0 projected into the (rotated) FOIS basis.
    Sx = project_into_new_basis_sharded(psi0, sigma)

    # <X|H|0> and <X|F|0> in the FOIS basis.
    XH0 = build_sigma_into_sharded(sigma, psi0, cluster_ops, clustered_ham;
                                   nbody=nbody, workers=worker_ids, blas_threads=blas_threads)
    XF0 = build_sigma_into_sharded(sigma, psi0, cluster_ops, clustered_ham_0;
                                   nbody=1, workers=worker_ids, blas_threads=blas_threads)

    # res = <X|H|0> - <X|F|0> - Sx (E0 - F0)
    coef = E0 .- F0
    res_id = gensym(:spt_shard_pt1res)
    res = _spt_dispatch_local!(_spt_pt1_assemble_chunk!, res_id, worker_ids,
                               collect(keys(sigma.owners)), sigma.clusters,
                               sigma.p_spaces, sigma.q_spaces, T, Val(R), Val(N),
                               XH0.id, XF0.id, Sx.id, coef)

    # psi1 = res / (F0 - Fdiag)
    psi1_id = gensym(:spt_shard_pt1)
    psi1 = _spt_dispatch_local!(_spt_pt1_denominator_chunk!, psi1_id, worker_ids,
                                collect(keys(res.owners)), res.clusters,
                                res.p_spaces, res.q_spaces, T, Val(R), Val(N),
                                res.id, Fdiag.id, F0)

    ecorr = LinearAlgebra.diag(overlap(res, psi1))
    E2 = E0 .+ ecorr

    if verbose > 0
        @printf(" %5s %14s %14s\n", "Root", "E(0)", "E(2)")
        for r in 1:R
            @printf(" %5i %14.8f %14.8f\n", r, E0[r], E2[r])
        end
        flush(stdout)
    end

    destroy!(sigma); destroy!(Fdiag); destroy!(Sx)
    destroy!(XH0); destroy!(XF0); destroy!(res)
    return psi1, E2, ecorr
end

"""
    subspace_product_tucker_sharded(input_vec, cluster_ops, clustered_ham; ...) -> (e_var, DistributedSPTstate)

Never-gather variational SPT (SPT / `subspace_product_tucker`). The variational
vector, FOIS, and PT1 all stay sharded as `DistributedSPTstate`s for the whole
loop — the FOIS/PT1 are never assembled on the master, and the variational solve
uses `spt_ci_davidson_sharded` instead of a node-local `ci_solve`. This is the
end-to-end memory-scalable SPT driver.

Ownership is `:hash` throughout so growth/compression/projection stay worker-
local. Not supported here (use `subspace_product_tucker` / `spt_multinode`): the
S² spin extension (`thresh_spin`). Returns `(e_var, ref)` with `ref` a
`DistributedSPTstate`; `collect_spt_state(ref)` only if it fits one node.
"""
function subspace_product_tucker_sharded(input_vec::SPTstate{T,N,R}, cluster_ops,
                                         clustered_ham;
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
                                         workers=Distributed.workers(),
                                         blas_threads=1,
                                         verbose=1) where {T,N,R}
    thresh_spin === nothing ||
        error("subspace_product_tucker_sharded does not support the S² spin extension (thresh_spin); use subspace_product_tucker / spt_multinode")
    worker_ids = ensure_tpsci_multinode_workers!(workers=workers)
    clustered_S2 = extract_S2(input_vec.clusters)

    println()
    println(" |== Never-gather Sharded SPT ======================================")
    println(" max_iter    : ", max_iter)
    println(" nbody       : ", nbody)
    println(" thresh_var  : ", thresh_var)
    println(" thresh_foi  : ", thresh_foi)
    println(" thresh_pt   : ", thresh_pt)
    println(" do_pt       : ", do_pt)
    println(" workers     : ", worker_ids)
    flush(stdout)

    ref = distribute_spt_state(deepcopy(input_vec); workers=worker_ids,
                               strategy=:hash, blas_threads=blas_threads)
    var_old = distribute_spt_state(deepcopy(input_vec); workers=worker_ids,
                                   strategy=:hash, blas_threads=blas_threads)
    e_last = zeros(T, R)
    e_var = zeros(T, R)
    e0 = zeros(T, R)
    converged = false

    for iter in 1:max_iter
        println()
        println(" ===================================================================")
        @printf("     Sharded SPT Iteration: %4i  thresh_var: %12.8f\n", iter, thresh_var)
        println(" ===================================================================")

        # Compress + orthonormalize the reference.
        d1 = length(ref)
        ref2 = compress_sharded(ref; thresh=thresh_var); destroy!(ref); ref = ref2
        orthonormalize!(ref)
        @printf(" %-40s %8i -> %-8i\n", "Ref compressed", d1, length(ref))

        # Zeroth-order energy.
        if resolve_ss
            e0, refnew = spt_ci_davidson_sharded(ref, cluster_ops, clustered_ham;
                nroots=R, conv_thresh=ci_conv, max_iter=ci_max_iter,
                max_ss_vecs=ci_max_ss_vecs, lindep_thresh=ci_lindep_thresh,
                nbody=nbody, workers=worker_ids, verbose=0)
            destroy!(ref); ref = refnew
        else
            e0 = spt_expectation_sharded(ref, cluster_ops, clustered_ham;
                                         nbody=nbody, workers=worker_ids, blas_threads=blas_threads)
        end
        iter == 1 && (e_last = copy(e0))
        s2 = spt_expectation_sharded(ref, cluster_ops, clustered_S2;
                                     nbody=nbody, workers=worker_ids, blas_threads=blas_threads)
        @printf(" %5s %14s %12s\n", "Root", "Energy", "S2")
        for r in 1:R
            @printf(" %5i %14.8f %12.8f\n", r, e0[r], abs(s2[r]))
        end
        flush(stdout)

        # First-order interacting space (never gathered). Pin reference sectors
        # to their owners so the later nonorth_add stays worker-local.
        fois = build_compressed_1st_order_state_sharded(ref, cluster_ops, clustered_ham;
            nbody=nbody, thresh=thresh_foi, workers=worker_ids, strategy=:hash,
            existing_owners=ref.owners, blas_threads=blas_threads)

        if do_pt
            psi1, e_pt2, _ = compute_pt1_wavefunction_sharded(fois, ref, cluster_ops,
                clustered_ham; H0=H0, nbody=nbody, workers=worker_ids,
                blas_threads=blas_threads, verbose=verbose)
            destroy!(fois); fois = psi1
        end

        d1 = length(fois)
        fois2 = compress_sharded(fois; thresh=thresh_foi); destroy!(fois); fois = fois2
        @printf(" %-40s %8i -> %-8i\n", "FOIS compressed", d1, length(fois))

        # Grow the variational space: var = ref + FOIS.
        var = copy_sharded_state(ref)
        d1 = length(var)
        nonorth_add_sharded!(var, fois); destroy!(fois)
        var2 = compress_sharded(var; thresh=thresh_pt); destroy!(var); var = var2
        orthonormalize!(var)
        @printf(" %-40s %8i -> %-8i\n", "Variational space", d1, length(var))

        # Carry the previous solution as the guess in the new basis.
        varp = project_into_new_basis_sharded(var_old, var); destroy!(var); var = varp
        orthonormalize!(var)

        # Variational solve (never-gather Tucker CI).
        e_var, varnew = spt_ci_davidson_sharded(var, cluster_ops, clustered_ham;
            nroots=R, conv_thresh=ci_conv, max_iter=ci_max_iter,
            max_ss_vecs=ci_max_ss_vecs, lindep_thresh=ci_lindep_thresh,
            nbody=nbody, workers=worker_ids, verbose=(verbose > 1 ? 1 : 0))
        destroy!(var); var = varnew

        destroy!(var_old); var_old = copy_sharded_state(var)
        destroy!(ref); ref = var        # ref_vec = var_vec (var handle transferred)

        @printf(" Sharded SPT Iter %3i  dim=%9i  E:", iter, length(ref))
        for r in 1:R
            @printf(" %14.8f", e_var[r])
        end
        if maximum(abs.(e_last .- e_var)) < tol_tucker
            println("   *converged*")
            converged = true
            flush(stdout)
            break
        end
        println()
        e_last = copy(e_var)
        flush(stdout)
    end

    destroy!(var_old)
    converged ||
        @printf(" subspace_product_tucker_sharded did not converge in %i iterations\n", max_iter)
    @printf(" ==================================================================|\n")
    flush(stdout)
    return e_var, ref
end
