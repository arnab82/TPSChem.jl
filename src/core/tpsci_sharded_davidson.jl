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
    estimate_sharded_H_nnz(ci_vector::DistributedTPSCIstate, clustered_ham)

Estimate the number of nonzero elements in the block-sparse variational
Hamiltonian for the space defined by `ci_vector`, counting every connected
`(fock_bra, fock_ket)` pair as a fully dense `n_bra × n_ket` block. Uses only the
per-Fock-sector lengths already tracked in the metadata, so it is O(nfocks^2) and
touches no coefficient data.
"""
function estimate_sharded_H_nnz(ci_vector::DistributedTPSCIstate{T,N,R},
                                clustered_ham::ClusteredOperator) where {T,N,R}
    focks = collect(keys(ci_vector.owners))
    lens = ci_vector.lengths
    nnz = 0.0
    for fock_bra in focks
        lb = lens[fock_bra]
        for fock_ket in focks
            fock_trans = fock_bra - fock_ket
            haskey(clustered_ham, fock_trans) || continue
            nnz += Float64(lb) * Float64(lens[fock_ket])
        end
    end
    return nnz
end

"""
    sharded_H_memory_report(ci_vector, clustered_ham; ::Type{T}=Float64)

Return a NamedTuple summarizing the block-sparse Hamiltonian size so a run can
decide between Tier A (stored H) and Tier B (matrix-free). `nnz` is the number of
stored matrix elements (summed over all workers), `bytes` its dense-block memory.
"""
function sharded_H_memory_report(ci_vector::DistributedTPSCIstate{T,N,R},
                                 clustered_ham::ClusteredOperator) where {T,N,R}
    nnz = estimate_sharded_H_nnz(ci_vector, clustered_ham)
    bytes = nnz * sizeof(T)
    return (dim=length(ci_vector), nfocks=length(ci_vector.owners),
            nnz=nnz, bytes=bytes, gb=bytes * 1e-9)
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

function apply_sharded_H(op::MatrixFreeShardedH, v::DistributedTPSCIstate)
    # verbose=0: this is called once per Davidson direction / MINRES iteration;
    # per-call banners would swamp the solver log.
    return tps_ci_matvec_sharded(v, op.cluster_ops, op.clustered_ham;
                                 workers=op.workers,
                                 threaded_worker=op.threaded_worker,
                                 blas_threads=op.blas_threads,
                                 verbose=0)
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
end

mutable struct ShardedBlockH{T,N}
    id::Symbol
    workers::Vector{Int}
    clusters::Vector{MOCluster}
    nnz::Float64
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
                               owned_bras)
    cluster_ops = _TPSCI_MULTINODE_CLUSTER_OPS[]
    clustered_ham = _TPSCI_MULTINODE_CLUSTERED_HAM[]
    cluster_ops === nothing && error("TPSCI sharded cluster-ops cache is empty")
    clustered_ham === nothing && error("TPSCI sharded Hamiltonian cache is empty")
    if cluster_ops isa DistributedClusterOps
        cluster_ops = _materialize_cluster_ops_for_indices(
            cluster_ops, [c.idx for c in ci_vector.clusters])
    end
    return _build_block_h_chunk_typed!(block_id, ci_vector, owned_bras,
                                       cluster_ops, clustered_ham)
end

function _build_block_h_chunk_typed!(block_id::Symbol,
                                     ci_vector::DistributedTPSCIstate{T,N,R},
                                     owned_bras, cluster_ops,
                                     clustered_ham::ClusteredOperator) where {T,N,R}
    local_state = _tpsci_sharded_get_state(ci_vector.id)
    ket_cache = Dict{FockConfig{N},Any}()
    bra_order = Dict{FockConfig{N},Vector{ClusterConfig{N}}}()
    ket_order = Dict{FockConfig{N},Vector{ClusterConfig{N}}}()
    connections = Dict{FockConfig{N},Vector{FockConfig{N}}}()
    mats = Dict{Tuple{FockConfig{N},FockConfig{N}},Matrix{T}}()

    for fock_bra in owned_bras
        bra_cfgs = collect(keys(local_state.data[fock_bra]))
        bra_order[fock_bra] = bra_cfgs
        conns = FockConfig{N}[]
        for fock_ket in keys(ci_vector.owners)
            fock_trans = fock_bra - fock_ket
            haskey(clustered_ham, fock_trans) || continue
            configs_ket = _tpsci_sharded_get_ket_configs(ci_vector, fock_ket, ket_cache)
            ket_cfgs = collect(keys(configs_ket))
            ket_order[fock_ket] = ket_cfgs
            Hblk = zeros(T, length(bra_cfgs), length(ket_cfgs))
            terms = clustered_ham[fock_trans]
            for (bi, config_bra) in enumerate(bra_cfgs)
                for (ki, config_ket) in enumerate(ket_cfgs)
                    acc = zero(T)
                    for term in terms
                        check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue
                        acc += contract_matrix_element(term, cluster_ops, fock_bra,
                                                       config_bra, fock_ket, config_ket)
                    end
                    Hblk[bi, ki] = acc
                end
            end
            mats[(fock_bra, fock_ket)] = Hblk
            push!(conns, fock_ket)
        end
        connections[fock_bra] = conns
    end

    data = ShardedBlockData{T,N}(collect(owned_bras), bra_order, ket_order,
                                 connections, mats)
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
                _build_block_h_chunk!, pid, block_id, ci_vector, chunks[pid])
        end
    end
    nnz = sum(values(nbyte_maps); init=0)
    return ShardedBlockH{T,N}(block_id, ci_vector.workers, ci_vector.clusters,
                              Float64(nnz))
end

function _update_block_h_chunk!(block_id::Symbol, ci_vector::DistributedTPSCIstate,
                                owned_bras)
    cluster_ops = _TPSCI_MULTINODE_CLUSTER_OPS[]
    clustered_ham = _TPSCI_MULTINODE_CLUSTERED_HAM[]
    cluster_ops === nothing && error("TPSCI sharded cluster-ops cache is empty")
    clustered_ham === nothing && error("TPSCI sharded Hamiltonian cache is empty")
    if cluster_ops isa DistributedClusterOps
        cluster_ops = _materialize_cluster_ops_for_indices(
            cluster_ops, [c.idx for c in ci_vector.clusters])
    end
    return _update_block_h_chunk_typed!(block_id, ci_vector, owned_bras,
                                        cluster_ops, clustered_ham)
end

function _update_block_h_chunk_typed!(block_id::Symbol,
                                      ci_vector::DistributedTPSCIstate{T,N,R},
                                      owned_bras, cluster_ops,
                                      clustered_ham::ClusteredOperator) where {T,N,R}
    old = haskey(_TPSCI_SHARDED_BLOCK_H, block_id) ?
          _tpsci_sharded_get_block_h(block_id) : nothing
    local_state = _tpsci_sharded_get_state(ci_vector.id)
    ket_cache = Dict{FockConfig{N},Any}()
    bra_order = Dict{FockConfig{N},Vector{ClusterConfig{N}}}()
    ket_order = Dict{FockConfig{N},Vector{ClusterConfig{N}}}()
    connections = Dict{FockConfig{N},Vector{FockConfig{N}}}()
    mats = Dict{Tuple{FockConfig{N},FockConfig{N}},Matrix{T}}()

    # Old-column position maps per ket sector, built lazily and shared across
    # all bra sectors in this chunk.
    old_ket_pos = Dict{FockConfig{N},Dict{ClusterConfig{N},Int}}()

    for fock_bra in owned_bras
        bra_cfgs = collect(keys(local_state.data[fock_bra]))
        bra_order[fock_bra] = bra_cfgs
        old_bra = old === nothing ? ClusterConfig{N}[] :
                  get(old.bra_order, fock_bra, ClusterConfig{N}[])
        bra_pos_old = Dict(c => i for (i, c) in enumerate(old_bra))
        conns = FockConfig{N}[]
        for fock_ket in keys(ci_vector.owners)
            fock_trans = fock_bra - fock_ket
            haskey(clustered_ham, fock_trans) || continue
            configs_ket = _tpsci_sharded_get_ket_configs(ci_vector, fock_ket, ket_cache)
            ket_cfgs = collect(keys(configs_ket))
            ket_order[fock_ket] = ket_cfgs
            kpos_old = get!(old_ket_pos, fock_ket) do
                oldk = old === nothing ? ClusterConfig{N}[] :
                       get(old.ket_order, fock_ket, ClusterConfig{N}[])
                Dict(c => i for (i, c) in enumerate(oldk))
            end
            M_old = old === nothing ? nothing :
                    get(old.mats, (fock_bra, fock_ket), nothing)
            terms = clustered_ham[fock_trans]
            Hblk = zeros(T, length(bra_cfgs), length(ket_cfgs))
            for (bi, config_bra) in enumerate(bra_cfgs)
                bio = M_old === nothing ? 0 : get(bra_pos_old, config_bra, 0)
                for (ki, config_ket) in enumerate(ket_cfgs)
                    kio = bio == 0 ? 0 : get(kpos_old, config_ket, 0)
                    if bio > 0 && kio > 0
                        # previously computed element: copy, don't re-contract
                        Hblk[bi, ki] = M_old[bio, kio]
                    else
                        acc = zero(T)
                        for term in terms
                            check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue
                            acc += contract_matrix_element(term, cluster_ops, fock_bra,
                                                           config_bra, fock_ket, config_ket)
                        end
                        Hblk[bi, ki] = acc
                    end
                end
            end
            mats[(fock_bra, fock_ket)] = Hblk
            push!(conns, fock_ket)
        end
        connections[fock_bra] = conns
    end

    data = ShardedBlockData{T,N}(collect(owned_bras), bra_order, ket_order,
                                 connections, mats)
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
                _update_block_h_chunk!, pid, op.id, ci_vector, chunks[pid])
        end
    end
    old_nnz = op.nnz
    op.nnz = Float64(sum(values(nbyte_maps); init=0))
    verbose > 0 &&
        @printf(" Updated sharded block-sparse H incrementally: nnz %.3e -> %.3e\n",
                old_nnz, op.nnz)
    return op
end

function _apply_block_h_chunk!(out_id::Symbol, block_id::Symbol,
                               v::DistributedTPSCIstate)
    return _apply_block_h_chunk_typed!(out_id, _tpsci_sharded_get_block_h(block_id), v)
end

function _apply_block_h_chunk_typed!(out_id::Symbol, block::ShardedBlockData{T,N},
                                     v::DistributedTPSCIstate{Tv,N,1}) where {T,N,Tv}
    vket_cache = Dict{FockConfig{N},Any}()
    out = TPSCIstate(v.clusters, T=T, R=1)
    for fock_bra in block.owned_bras
        bra_cfgs = block.bra_order[fock_bra]
        sig = zeros(T, length(bra_cfgs))
        for fock_ket in block.connections[fock_bra]
            Hblk = block.mats[(fock_bra, fock_ket)]
            ket_cfgs = block.ket_order[fock_ket]
            configs_ket = _tpsci_sharded_get_ket_configs(v, fock_ket, vket_cache)
            vk = Vector{T}(undef, length(ket_cfgs))
            for (ki, config_ket) in enumerate(ket_cfgs)
                vk[ki] = configs_ket[config_ket][1]
            end
            mul!(sig, Hblk, vk, one(T), one(T))
        end
        add_fockconfig!(out, fock_bra)
        for (bi, config_bra) in enumerate(bra_cfgs)
            out[fock_bra][config_bra] = MVector{1,T}(sig[bi])
        end
    end
    _tpsci_sharded_store_state!(out_id, out)
    return _tpsci_sharded_local_fock_lengths(out_id)
end

function apply_sharded_H(op::ShardedBlockH{T,N}, v::DistributedTPSCIstate{Tv,N,1};
                         id=nothing) where {T,N,Tv}
    op.workers == v.workers || error("apply_sharded_H requires the block-H and vector on the same workers")
    output_id = id === nothing ? gensym(:tpsci_shard_blockHv) : Symbol(id)
    length_maps = Dict{Int,Any}()
    @sync for pid in op.workers
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(
                _apply_block_h_chunk!, pid, output_id, op.id, v)
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
        Hff = get(block.mats, (fock, fock), nothing)
        Hff === nothing && error("Stored block-H is missing diagonal block for Fock sector $fock")
        bra_cfgs = block.bra_order[fock]
        ket_cfgs = block.ket_order[fock]
        ket_pos = Dict(c => i for (i, c) in enumerate(ket_cfgs))
        add_fockconfig!(out, fock)
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
            rep = sharded_H_memory_report(ci_vector, clustered_ham)
            tier = rep.gb <= max_mem_H ? :blocks : :matrixfree
            verbose > 0 && @printf(" H storage :auto -> :%s  (block-sparse H ~ %.3f GB, budget %.1f GB)\n",
                                   tier, rep.gb, max_mem_H)
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
