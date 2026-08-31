# ─────────────────────────────────────────────────────────────────────────────
# Blocked-root TPS-CEPA across nodes: all R roots in one solver pass.
#
# Nothing here makes the method multiroot — it always was; R roots in, R energies
# out. What changes is that the roots used to be solved one *attempt* at a time
# and now share a single one.
#
# The sharded CEPA solver used to walk the roots one at a time: every root paid
# its own Hamiltonian apply, its own fan-out per vector operation, and its own
# round of remote ket gathers. None of that work is root-specific — the stored
# H blocks, the Fock-sector routing and the communication pattern are identical
# for all R roots — so this file batches it. One apply, one fan-out, R roots.
#
# What is *not* shared: each root keeps its own shift, its own Krylov scalars and
# its own convergence test. That is forced on us anyway (a different `eshift` per
# root rules out true block CG/MINRES), and it is also what makes the batched
# solvers reproduce the root-at-a-time ones exactly.
#
# Layouts. Two different ones, each chosen by what the kernel underneath wants:
#   * on the workers, the stored-H apply keeps kets as `n_ket × R` matrices so the
#     block GEMV becomes a BLAS GEMM with no repacking;
#   * the fused vector algebra stages coefficients root-major, `R × n`, which is
#     exactly how `MVector{R,T}` already lays them out per config, so gather and
#     scatter stay a straight copy and the per-root scalar loops stay contiguous.
# ─────────────────────────────────────────────────────────────────────────────


# ── Stored block-H: apply to R roots at once (GEMV → GEMM) ───────────────────

"""
    _dense_ket_block_in_order(T, configs, want, Val(R)) -> Matrix{T}

Gather one Fock sector's coefficients into a dense `n_ket × R` matrix in the
order the stored H block expects. Column-major with the root index second is what
`mul!` wants for the GEMM, so the ket block goes straight into BLAS.

Mirrors `_dense_ket_in_order`, including its fast path: vectors over a fixed space
are built in the same insertion order, so the ordered walk almost always succeeds
and the keyed re-walk is only a fallback.
"""
function _dense_ket_block_in_order(::Type{T}, configs, want::Vector{ClusterConfig{N}},
                                   ::Val{R}) where {T,N,R}
#={{{=#
    n = length(want)
    out = Matrix{T}(undef, n, R)
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
        for c in 1:R
            out[i, c] = coeffs[c]
        end
    end
    if !aligned
        @inbounds for j in 1:n
            coeffs = configs[want[j]]
            for c in 1:R
                out[j, c] = coeffs[c]
            end
        end
    end
    return out
end
#=}}}=#

function _block_h_local_gather_kets_block(state_id::Symbol, focks, want,
                                          ::Type{T}, ::Val{R}) where {T,R}
    state = _tpsci_sharded_get_state(state_id)
    return [_dense_ket_block_in_order(T, state.data[focks[i]], want[i], Val(R))
            for i in eachindex(focks)]
end

"""
    _block_h_gather_kets_block(block, v) -> Dict{FockConfig,Matrix}

R-root version of `_block_h_gather_kets`. The remote fetches are the part that
matters here: the set of sectors to fetch and the number of round trips are fixed
by the stored sparsity, not by R, so pulling R roots costs one message per remote
owner exactly as one root did — only the payload grows.
"""
function _block_h_gather_kets_block(block::ShardedBlockData{T,N},
                                    v::DistributedTPSCIstate{Tv,N,R}) where {T,N,Tv,R}
#={{{=#
    myid = Distributed.myid()
    kets = Dict{FockConfig{N},Matrix{T}}()
    remote = Dict{Int,Vector{FockConfig{N}}}()
    local_state = nothing
    for fock in block.needed_kets
        owner = v.owners[fock]
        if owner == myid
            local_state === nothing && (local_state = _tpsci_sharded_get_state(v.id))
            kets[fock] = _dense_ket_block_in_order(T, local_state.data[fock],
                                                   block.ket_order[fock], Val(R))
        else
            push!(get!(() -> FockConfig{N}[], remote, owner), fock)
        end
    end
    isempty(remote) && return kets
    fetched = Dict{Int,Vector{Matrix{T}}}()
    @sync for (owner, focks) in remote
        @async fetched[owner] = Distributed.remotecall_fetch(
            _block_h_local_gather_kets_block, owner, v.id, focks,
            [block.ket_order[f] for f in focks], T, Val(R))
    end
    for (owner, focks) in remote
        blocks = fetched[owner]
        for i in eachindex(focks)
            kets[focks[i]] = blocks[i]
        end
    end
    return kets
end
#=}}}=#

# One row block for R roots: sigma_F = sum_F' H[F,F'] V_F', with each root's own
# shift folded in. The GEMV of the single-root path becomes a GEMM over the same
# stored matrix — the matrix is read once for all R roots instead of R times,
# which is the whole reason blocking pays here.
function _apply_block_h_row_block(block::ShardedBlockData{T,N}, kets,
                                  fock_bra::FockConfig{N}, shifts::Vector{T},
                                  ::Val{R}) where {T,N,R}
#={{{=#
    sig = zeros(T, length(block.bra_order[fock_bra]), R)
    for fock_ket in block.connections[fock_bra]
        mul!(sig, block.mats[(fock_bra, fock_ket)], kets[fock_ket], one(T), one(T))
    end
    # (H - eshift_c*I) v_c in the same pass, one shift per root.
    if !all(iszero, shifts)
        K = kets[fock_bra]
        for c in 1:R
            iszero(shifts[c]) && continue
            @views axpy!(-shifts[c], K[:, c], sig[:, c])
        end
    end
    return sig
end
#=}}}=#

function _apply_block_h_chunk_block!(out_id::Symbol, block_id::Symbol,
                                     v::DistributedTPSCIstate, shifts,
                                     threaded::Bool=true)
    return _apply_block_h_chunk_block_typed!(out_id, _tpsci_sharded_get_block_h(block_id),
                                             v, shifts, threaded)
end

function _apply_block_h_chunk_block_typed!(out_id::Symbol, block::ShardedBlockData{T,N},
                                           v::DistributedTPSCIstate{Tv,N,R},
                                           shifts, threaded::Bool=true) where {T,N,Tv,R}
#={{{=#
    kets = _block_h_gather_kets_block(block, v)
    sh = T[T(s) for s in shifts]
    length(sh) == R || throw(DimensionMismatch("got $(length(sh)) shifts for $R roots"))
    bras = block.owned_bras

    sigs = Vector{Matrix{T}}(undef, length(bras))
    if threaded && Threads.nthreads() > 1 && length(bras) > 1
        Threads.@threads :dynamic for i in eachindex(bras)
            sigs[i] = _apply_block_h_row_block(block, kets, bras[i], sh, Val(R))
        end
    else
        for i in eachindex(bras)
            sigs[i] = _apply_block_h_row_block(block, kets, bras[i], sh, Val(R))
        end
    end

    out = TPSCIstate(v.clusters, T=T, R=R)
    for (i, fock_bra) in enumerate(bras)
        bra_cfgs = block.bra_order[fock_bra]
        sig = sigs[i]
        add_fockconfig!(out, fock_bra)
        for (bi, config_bra) in enumerate(bra_cfgs)
            out[fock_bra][config_bra] = MVector{R,T}(ntuple(c -> sig[bi, c], Val(R)))
        end
    end
    _tpsci_sharded_store_state!(out_id, out)
    return _tpsci_sharded_local_fock_lengths(out_id)
end
#=}}}=#

"""
    apply_sharded_H_block(op, v, eshifts; id=nothing)

Apply `(H - eshifts[c]*I)` to root `c` of a sharded R-root vector, for every root,
in one pass. `eshifts` has one entry per root.
"""
function apply_sharded_H_block(op::ShardedBlockH{T,N}, v::DistributedTPSCIstate{Tv,N,R},
                               eshifts::Vector; id=nothing) where {T,N,Tv,R}
#={{{=#
    op.workers == v.workers ||
        error("apply_sharded_H_block requires the block-H and vector on the same workers")
    length(eshifts) == R || throw(DimensionMismatch("got $(length(eshifts)) shifts for $R roots"))
    output_id = id === nothing ? gensym(:tpsci_shard_blockHV) : Symbol(id)
    length_maps = Dict{Int,Any}()
    @sync for pid in op.workers
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(
                _apply_block_h_chunk_block!, pid, output_id, op.id, v, eshifts, op.threaded)
        end
    end
    return _tpsci_sharded_metadata_from_lengths(output_id, op.clusters, op.workers,
                                                length_maps,
                                                collect(keys(v.owners)),
                                                T, Val(R))
end
#=}}}=#

function apply_sharded_H_block(op::MatrixFreeShardedH, v::DistributedTPSCIstate{T,N,R},
                               eshifts::Vector; id=nothing) where {T,N,R}
    # The re-contracted matvec is already multi-root; it just cannot fold a shift
    # into its output the way the stored path does, so the shift costs one fan-out.
    out = apply_sharded_H(op, v; eshift=zero(T), id=id)
    all(iszero, eshifts) ||
        sharded_block_fused_ops!([out, v], [svb_axpy(T, 1, T[-e for e in eshifts], 2)])
    return out
end

# Apply into a recycled buffer, mirroring `_tps_sharded_apply_into!`.
function _tps_sharded_apply_block_into!(op, v::DistributedTPSCIstate{T,N,R},
                                        eshifts::Vector,
                                        buf::DistributedTPSCIstate{T,N,R}) where {T,N,R}
    out = apply_sharded_H_block(op, v, eshifts; id=buf.id)
    out.id == buf.id || destroy!(buf)
    return out
end


# ── R-root fused sharded vector algebra ──────────────────────────────────────

"""
    ShardedBlockVecOp{T}

One elementwise operation in a fused *multi-root* sharded batch — the R-root
counterpart of [`ShardedVecOp`](@ref). The only structural difference is that the
coefficients are per-root vectors and the reductions return one value per root, so
each root can carry its own shift and its own Krylov scalars while sharing the
fan-out.

| `kind`   | effect (per root `c`)              | returns      |
|:---------|:-----------------------------------|:-------------|
| `:axpy`  | `dest[c] .+= a[c] .* src[c]`       | -            |
| `:axpby` | `dest[c] .= a[c].*dest[c] .+ b[c].*src[c]` | -    |
| `:scale` | `dest[c] .*= a[c]`                 | -            |
| `:copy`  | `dest[c] .= src[c]`                | -            |
| `:zero`  | `dest .= 0`                        | -            |
| `:dot`   | -                                  | `dest[c]·src[c]` |
| `:nrm2sq`| -                                  | `dest[c]·dest[c]`|
"""
struct ShardedBlockVecOp{T}
    kind::Symbol
    dest::Int
    src::Int
    a::Vector{T}
    b::Vector{T}
end

svb_axpy(::Type{T}, dest, a::AbstractVector, src) where {T} =
    ShardedBlockVecOp{T}(:axpy, dest, src, T[a...], T[])
svb_axpby(::Type{T}, dest, a::AbstractVector, b::AbstractVector, src) where {T} =
    ShardedBlockVecOp{T}(:axpby, dest, src, T[a...], T[b...])
svb_scale(::Type{T}, dest, a::AbstractVector) where {T} =
    ShardedBlockVecOp{T}(:scale, dest, 0, T[a...], T[])
svb_copy(::Type{T}, dest, src) where {T} =
    ShardedBlockVecOp{T}(:copy, dest, src, T[], T[])
svb_zero(::Type{T}, dest) where {T} =
    ShardedBlockVecOp{T}(:zero, dest, 0, T[], T[])
svb_dot(::Type{T}, a, b) where {T} =
    ShardedBlockVecOp{T}(:dot, a, b, T[], T[])
svb_nrm2sq(::Type{T}, a) where {T} =
    ShardedBlockVecOp{T}(:nrm2sq, a, a, T[], T[])

"""
    _tpsci_sharded_gather_block_flat!(out, state, ref)

Stage an R-root state into a root-major `R × n` buffer in `ref`'s config order.
Root-major is not an arbitrary choice: `MVector{R,T}` already stores a config's
roots contiguously, so this is a straight copy, and every per-root scalar loop
downstream runs over contiguous memory.

Same three-tier ordering strategy as `_tpsci_sharded_gather_flat!`: identity, then
an ordered walk with a key check, then a keyed re-walk.
"""
function _tpsci_sharded_gather_block_flat!(out::Matrix{T}, state::TPSCIstate{Ts,N,R},
                                           ref::TPSCIstate{Tr,N,Rr}) where {T,Ts,Tr,N,R,Rr}
#={{{=#
    n = size(out, 2)
    length(state) == n ||
        error("sharded_block_fused_ops!: vectors span different spaces on worker $(Distributed.myid())")
    if state === ref
        i = 0
        @inbounds for (_, configs) in state.data
            for (_, coeffs) in configs
                i += 1
                for c in 1:R
                    out[c, i] = coeffs[c]
                end
            end
        end
        return out
    end
    i = 0
    aligned = true
    for ((fs, cs), (fr, cr)) in zip(state.data, ref.data)
        if fs != fr || length(cs) != length(cr)
            aligned = false
            break
        end
        for ((ks, vs), (kr, _)) in zip(cs, cr)
            if ks != kr
                aligned = false
                break
            end
            i += 1
            @inbounds for c in 1:R
                out[c, i] = vs[c]
            end
        end
        aligned || break
    end
    aligned && i == n && return out
    i = 0
    for (fock, configs) in ref.data
        for (config, _) in configs
            i += 1
            coeffs = state.data[fock][config]
            @inbounds for c in 1:R
                out[c, i] = coeffs[c]
            end
        end
    end
    return out
end
#=}}}=#

function _tpsci_sharded_scatter_block_flat!(state::TPSCIstate{Ts,N,R},
                                            ref::TPSCIstate{Tr,N,Rr},
                                            vals::Matrix{T}) where {T,Ts,Tr,N,R,Rr}
#={{{=#
    n = size(vals, 2)
    i = 0
    if state === ref
        @inbounds for (_, configs) in state.data
            for (_, coeffs) in configs
                i += 1
                for c in 1:R
                    coeffs[c] = vals[c, i]
                end
            end
        end
        return state
    end
    aligned = true
    for ((fs, cs), (fr, cr)) in zip(state.data, ref.data)
        if fs != fr || length(cs) != length(cr)
            aligned = false
            break
        end
        for ((ks, vs), (kr, _)) in zip(cs, cr)
            if ks != kr
                aligned = false
                break
            end
            i += 1
            @inbounds for c in 1:R
                vs[c] = vals[c, i]
            end
        end
        aligned || break
    end
    aligned && i == n && return state
    i = 0
    for (fock, configs) in ref.data
        for (config, _) in configs
            i += 1
            coeffs = state.data[fock][config]
            @inbounds for c in 1:R
                coeffs[c] = vals[c, i]
            end
        end
    end
    return state
end
#=}}}=#

function _tpsci_sharded_block_fused_ops!(ids::Vector{Symbol},
                                         ops::Vector{ShardedBlockVecOp{T}}) where {T}
    return _tpsci_sharded_block_fused_ops_typed!(
        [_tpsci_sharded_get_state(id) for id in ids], ops)
end

function _tpsci_sharded_block_fused_ops_typed!(states::Vector,
                                               ops::Vector{ShardedBlockVecOp{T}}) where {T}
#={{{=#
    isempty(states) && return zeros(T, 0, 0)
    nst = length(states)
    used = falses(nst)
    written = falses(nst)
    for op in ops
        used[op.dest] = true
        op.src == 0 || (used[op.src] = true)
        (op.kind === :dot || op.kind === :nrm2sq) || (written[op.dest] = true)
    end
    refidx = findfirst(used)
    refidx === nothing && return zeros(T, 0, 0)
    ref = states[refidx]
    n = length(ref)
    R = _tpsci_state_nroots(ref)
    bufs = Vector{Matrix{T}}(undef, nst)
    for i in 1:nst
        used[i] || continue
        bufs[i] = _tpsci_sharded_gather_block_flat!(Matrix{T}(undef, R, n), states[i], ref)
    end

    nred = count(o -> o.kind === :dot || o.kind === :nrm2sq, ops)
    out = zeros(T, nred, R)
    ired = 0
    for op in ops
        d = bufs[op.dest]
        if op.kind === :scale
            a = op.a
            @inbounds for j in 1:n
                @simd for c in 1:R
                    d[c, j] *= a[c]
                end
            end
        elseif op.kind === :zero
            fill!(d, zero(T))
        else
            s = bufs[op.src]
            if op.kind === :axpy
                a = op.a
                @inbounds for j in 1:n
                    @simd for c in 1:R
                        d[c, j] += a[c] * s[c, j]
                    end
                end
            elseif op.kind === :axpby
                a = op.a; b = op.b
                @inbounds for j in 1:n
                    @simd for c in 1:R
                        d[c, j] = a[c] * d[c, j] + b[c] * s[c, j]
                    end
                end
            elseif op.kind === :copy
                copyto!(d, s)
            elseif op.kind === :dot || op.kind === :nrm2sq
                ired += 1
                @inbounds for j in 1:n
                    @simd for c in 1:R
                        out[ired, c] += d[c, j] * s[c, j]
                    end
                end
            else
                error("Unknown ShardedBlockVecOp kind: $(op.kind)")
            end
        end
    end

    for i in 1:nst
        written[i] && _tpsci_sharded_scatter_block_flat!(states[i], ref, bufs[i])
    end
    return out
end
#=}}}=#

_tpsci_state_nroots(::TPSCIstate{T,N,R}) where {T,N,R} = R

"""
    sharded_block_fused_ops!(states, ops) -> Matrix{T}

Run a batch of elementwise operations on a set of sharded R-root vectors in a
single fan-out, returning an `nred × R` matrix of the globally summed reduction
results in the order the reductions appear in `ops`.

This is the R-root counterpart of [`sharded_fused_ops!`](@ref), and the reason the
blocked solvers cost the same communication as the single-root ones: a batch is
one round trip per worker whether it carries one root or six.
"""
function sharded_block_fused_ops!(states::AbstractVector,
                                  ops::Vector{ShardedBlockVecOp{T}}) where {T}
#={{{=#
    isempty(ops) && return zeros(T, 0, 0)
    pids = states[1].workers
    owners = states[1].owners
    for s in states
        s.workers == pids || error("sharded_block_fused_ops! requires identical worker lists")
        s.owners == owners || error("sharded_block_fused_ops! requires matching Fock ownership")
    end
    ids = Symbol[s.id for s in states]
    allunique(ids) || error("sharded_block_fused_ops! requires distinct states")
    partials = Dict{Int,Matrix{T}}()
    @sync for pid in pids
        @async partials[pid] = Distributed.remotecall_fetch(
            _tpsci_sharded_block_fused_ops!, pid, ids, ops)
    end
    total = nothing
    for pid in pids
        p = partials[pid]
        size(p) == (0, 0) && continue
        total = total === nothing ? copy(p) : (total .+= p)
    end
    return total === nothing ? zeros(T, 0, 0) : total
end
#=}}}=#

# Per-root norms / dots as plain vectors, for readability at the call sites.
_svb_norms(states, idx, ::Type{T}) where {T} =
    vec(sqrt.(abs.(sharded_block_fused_ops!(states, [svb_nrm2sq(T, idx)]))))


# ── Jacobi preconditioner for R roots ────────────────────────────────────────

function _tpsci_sharded_local_precondition_block!(dest_id::Symbol, res_id::Symbol,
                                                  diag_id::Symbol, eshifts)
    return _tpsci_sharded_local_precondition_block_typed!(
        dest_id, _tpsci_sharded_get_state(res_id),
        _tpsci_sharded_get_state(diag_id), eshifts)
end

function _tpsci_sharded_local_precondition_block_typed!(dest_id::Symbol,
                                                        res::TPSCIstate{T,N,R},
                                                        diag::TPSCIstate{T,N,1},
                                                        eshifts) where {T,N,R}
#={{{=#
    sh = T[T(e) for e in eshifts]
    out = TPSCIstate(res.clusters, T=T, R=R)
    for (fock, configs) in res.data
        add_fockconfig!(out, fock)
        # Sharded states carry placeholder Fock sectors with no configurations;
        # the diagonal, built from the stored H's owned bras, does not. Mirror an
        # empty sector and move on. A *non-empty* sector missing from the diagonal
        # would mean the two states disagree about ownership — a real error, not a
        # placeholder — so say so rather than silently preconditioning with zeros.
        dfock = get(diag.data, fock, nothing)
        if dfock === nothing
            isempty(configs) ||
                error("precondition_block_sharded: Fock sector $fock holds " *
                      "$(length(configs)) configurations but is absent from the " *
                      "diagonal on worker $(Distributed.myid())")
            continue
        end
        for (config, coeffs) in configs
            d = dfock[config][1]
            vals = ntuple(Val(R)) do c
                d0 = d - sh[c]
                fl = sqrt(eps(T)) * max(abs(sh[c]), abs(d), one(T))
                den = abs(d0) < fl ? copysign(fl, iszero(d0) ? one(T) : d0) : d0
                coeffs[c] / den
            end
            out[fock][config] = MVector{R,T}(vals)
        end
    end
    _tpsci_sharded_store_state!(dest_id, out)
    return _tpsci_sharded_local_fock_lengths(dest_id)
end
#=}}}=#

"""
    precondition_block_sharded(res, Hdiag, eshifts; id=nothing)

`z[c] = res[c] / (diag(H) - eshifts[c])` for every root, with near-zero
denominators floored exactly as `precondition_sharded` floors them.

Note the sign: unlike the single-root `precondition_sharded`, which returns
`r/(theta - diag)` and leaves callers to flip it, this divides by
`diag - eshift` directly — the sign CG actually wants.
"""
function precondition_block_sharded(res::DistributedTPSCIstate{T,N,R},
                                    Hdiag::DistributedTPSCIstate{T,N,1},
                                    eshifts::Vector; id=nothing) where {T,N,R}
#={{{=#
    res.workers == Hdiag.workers ||
        error("precondition_block_sharded requires identical worker lists")
    length(eshifts) == R || throw(DimensionMismatch("got $(length(eshifts)) shifts for $R roots"))
    output_id = id === nothing ? gensym(:tpsci_shard_bprecond) : Symbol(id)
    length_maps = Dict{Int,Any}()
    @sync for pid in res.workers
        @async begin
            length_maps[pid] = Distributed.remotecall_fetch(
                _tpsci_sharded_local_precondition_block!, pid, output_id,
                res.id, Hdiag.id, eshifts)
        end
    end
    return _tpsci_sharded_metadata_from_lengths(output_id, res.clusters,
                                                res.workers, length_maps,
                                                collect(keys(res.owners)),
                                                T, Val(R))
end
#=}}}=#


# ── Blocked sharded linear solvers ───────────────────────────────────────────
#
# Both mirror the single-node `cepa_pcg_block` / `cepa_minres_block` exactly —
# same recurrences, same per-root bookkeeping, same freezing of finished roots —
# with the vector algebra swapped for fused sharded batches. An iteration costs
# one Hamiltonian apply plus three fan-outs for *all* R roots, which is what the
# single-root solvers spent on one.

"""
    tps_sharded_cepa_pcg_block(b, op, Hdiag, eshifts; tol, maxiter, verbose, X0)

Jacobi-preconditioned CG for `(H_QQ - eshifts[c]*I) x_c = b_c` over a sharded
R-root Q-space, all roots sharing one apply per iteration.

A root that meets a `p'Ap <= 0` direction is frozen and flagged in
`info.indefinite`; the caller redoes the block with MINRES. Roots that converge
early are frozen too — their search direction is zeroed, so they stop
contributing while the block keeps its shape.
"""
function tps_sharded_cepa_pcg_block(b::DistributedTPSCIstate{T,N,R}, op,
                                    Hdiag::DistributedTPSCIstate{T,N,1},
                                    eshifts::Vector{T};
                                    tol=1e-5,
                                    maxiter=300,
                                    verbose=0,
                                    x0=nothing) where {T,N,R}
#={{{=#
    x  = x0 === nothing ? similar_sharded_state(b) : x0
    r  = copy_sharded_state(b)
    Ap = similar_sharded_state(b)

    # Measure against ||b||, not the starting residual: a warm start makes the
    # latter small, which would silently tighten `tol` every macro-iteration.
    bnorm = _svb_norms([r], 1, T)
    threshold = T[tol * max(bnorm[c], one(T)) for c in 1:R]

    if x0 !== nothing
        Ax = _tps_sharded_apply_block_into!(op, x, eshifts, Ap)
        sharded_block_fused_ops!([r, Ax], [svb_axpy(T, 1, fill(-one(T), R), 2)])
        Ap = Ax
    end

    z = precondition_block_sharded(r, Hdiag, eshifts)
    p = copy_sharded_state(z)
    vecs = [x, r, p, Ap, z]
    IX, IR, IP, IAP, IZ = 1, 2, 3, 4, 5

    red = sharded_block_fused_ops!(vecs, [svb_nrm2sq(T, IR), svb_dot(T, IR, IZ)])
    resid = T[sqrt(abs(red[1, c])) for c in 1:R]
    rz    = T[red[2, c] for c in 1:R]

    converged  = Bool[resid[c] <= threshold[c] for c in 1:R]
    indefinite = fill(false, R)
    active     = Bool[!converged[c] for c in 1:R]
    iters      = zeros(Int, R)
    pAp0       = zeros(T, R)
    alpha      = zeros(T, R)
    beta       = zeros(T, R)
    mask       = T[active[c] ? one(T) : zero(T) for c in 1:R]

    all(converged) || sharded_block_fused_ops!(vecs, [svb_scale(T, IP, mask)])

    for it in 1:maxiter
        any(active) || break
        vecs[IAP] = _tps_sharded_apply_block_into!(op, vecs[IP], eshifts, vecs[IAP])
        pAp = vec(sharded_block_fused_ops!(vecs, [svb_dot(T, IP, IAP)]))

        changed = false
        for c in 1:R
            if !active[c]
                alpha[c] = zero(T)
                continue
            end
            iters[c] = it
            if pAp[c] <= zero(T)
                verbose > 0 &&
                    @printf("   TPS-CEPA block PCG root %i met an indefinite direction (pAp = %.3e)\n",
                            c, pAp[c])
                indefinite[c] = true
                active[c] = false
                alpha[c] = zero(T)
                changed = true
                continue
            end
            it == 1 && (pAp0[c] = pAp[c])
            # Underflow guard only: p'Ap shrinks like ||r||^2, so an eps*pAp0 cutoff
            # would cap CG at a relative residual of sqrt(eps) whatever `tol` asked.
            if pAp[c] <= eps(T)^2 * pAp0[c]
                verbose > 1 &&
                    @printf("   TPS-CEPA block PCG root %i stalled at fp precision (iter %i)\n", c, it)
                active[c] = false
                alpha[c] = zero(T)
                changed = true
                continue
            end
            alpha[c] = rz[c] / pAp[c]
        end

        red = sharded_block_fused_ops!(vecs,
                  [svb_axpy(T, IX, alpha, IP),
                   svb_axpy(T, IR, T[-a for a in alpha], IAP),
                   svb_nrm2sq(T, IR)])
        for c in 1:R
            resid[c] = sqrt(abs(red[1, c]))
        end

        for c in 1:R
            active[c] || continue
            if resid[c] <= threshold[c]
                converged[c] = true
                active[c] = false
                changed = true
            end
        end
        any(active) || break
        if changed
            for c in 1:R
                mask[c] = active[c] ? one(T) : zero(T)
            end
            sharded_block_fused_ops!(vecs, [svb_scale(T, IP, mask)])
        end

        vecs[IZ] = precondition_block_sharded(vecs[IR], Hdiag, eshifts; id=vecs[IZ].id)
        rz_new = vec(sharded_block_fused_ops!(vecs, [svb_dot(T, IR, IZ)]))
        for c in 1:R
            if active[c]
                beta[c] = rz_new[c] / rz[c]
                rz[c] = rz_new[c]
            else
                beta[c] = zero(T)
            end
        end
        # p .= mask*(z + beta*p): axpby writes a*dest + b*src with dest=p, src=z.
        sharded_block_fused_ops!(vecs, [svb_axpby(T, IP, beta, mask, IZ)])
    end

    destroy!(vecs[IR]); destroy!(vecs[IP]); destroy!(vecs[IAP]); destroy!(vecs[IZ])
    return vecs[IX], (iters=iters, residual=resid, converged=converged,
                      indefinite=indefinite)
end
#=}}}=#


"""
    tps_sharded_cepa_minres_block(b, op, eshifts; tol, maxiter, verbose, x0)

MINRES for `(H_QQ - eshifts[c]*I) x_c = b_c` over a sharded R-root Q-space.

Column-wise Paige–Saunders, identical in structure to the single-root
`tps_sharded_cepa_minres_linsolve` and to the single-node `cepa_minres_block`
that was checked against `IterativeSolvers` to 5e-16 per root — three fused
fan-outs and one Hamiltonian apply per iteration, now covering every root.
"""
function tps_sharded_cepa_minres_block(b::DistributedTPSCIstate{T,N,R}, op,
                                       eshifts::Vector{T};
                                       tol=1e-5,
                                       abstol=zero(T),
                                       maxiter=300,
                                       verbose=0,
                                       x0=nothing) where {T,N,R}
#={{{=#
    x      = x0 === nothing ? similar_sharded_state(b) : x0
    v_prev = similar_sharded_state(b)
    v_curr = copy_sharded_state(b)
    v_next = similar_sharded_state(b)
    w_prev = similar_sharded_state(b)
    w_curr = similar_sharded_state(b)
    w_next = similar_sharded_state(b)

    bnorm = _svb_norms([v_curr], 1, T)
    resnorm = copy(bnorm)
    if x0 !== nothing
        ax0 = apply_sharded_H_block(op, x0, eshifts)
        red = sharded_block_fused_ops!([v_curr, ax0],
                  [svb_axpy(T, 1, fill(-one(T), R), 2), svb_nrm2sq(T, 1)])
        destroy!(ax0)
        for c in 1:R
            resnorm[c] = sqrt(abs(red[1, c]))
        end
    end

    threshold = T[max(tol * bnorm[c], abstol) for c in 1:R]
    active    = Bool[resnorm[c] > threshold[c] && resnorm[c] > zero(T) for c in 1:R]
    converged = Bool[!active[c] for c in 1:R]
    iters     = zeros(Int, R)

    if !any(active)
        for tmp in (v_prev, v_curr, v_next, w_prev, w_curr, w_next)
            destroy!(tmp)
        end
        return x, (iters=iters, residual=resnorm, converged=converged)
    end

    # Normalize the first Krylov vector; dead roots are scaled to zero so the
    # shared apply can never grow their stale rows into an overflow.
    scal = T[active[c] ? inv(resnorm[c]) : zero(T) for c in 1:R]
    vecs = [v_next, v_prev, v_curr, w_next, w_curr, w_prev, x]
    IVN, IVP, IVC, IWN, IWC, IWP, IX = 1, 2, 3, 4, 5, 6, 7
    sharded_block_fused_ops!(vecs, [svb_scale(T, IVC, scal)])

    H1 = zeros(T, R); H2 = zeros(T, R); H3 = zeros(T, R); H4 = zeros(T, R)
    rhs1 = copy(resnorm); rhs2 = zeros(T, R)
    c_prev = ones(T, R);  s_prev = zeros(T, R)
    c_curr = ones(T, R);  s_curr = zeros(T, R)
    mask = T[active[c] ? one(T) : zero(T) for c in 1:R]

    for it in 1:maxiter
        any(active) || break
        vecs[IVN] = _tps_sharded_apply_block_into!(op, vecs[IVC], eshifts, vecs[IVN])

        # Fan-out 1: finish the Lanczos vector against v_prev, then <v_curr|v_next>.
        ops1 = ShardedBlockVecOp{T}[]
        it > 1 && push!(ops1, svb_axpy(T, IVN, T[-h for h in H2], IVP))
        push!(ops1, svb_dot(T, IVN, IVC))
        proj = vec(sharded_block_fused_ops!(vecs, ops1))
        for c in 1:R
            active[c] || (proj[c] = zero(T))
            H3[c] = proj[c]
        end

        # Fan-out 2: orthogonalize against v_curr and measure the new vector.
        red = sharded_block_fused_ops!(vecs,
                  [svb_axpy(T, IVN, T[-p for p in proj], IVC), svb_nrm2sq(T, IVN)])
        for c in 1:R
            H4[c] = sqrt(abs(red[1, c]))
        end

        changed = false
        for c in 1:R
            if !active[c]
                scal[c] = zero(T)
                continue
            end
            if iszero(H4[c])
                # Exact Krylov termination: this root's residual is already zero.
                scal[c] = zero(T)
                converged[c] = true
                active[c] = false
                resnorm[c] = zero(T)
                changed = true
                continue
            end
            scal[c] = inv(H4[c])
            iters[c] = it
            if it > 2
                H1[c] = s_prev[c] * H2[c]
                H2[c] = c_prev[c] * H2[c]
            end
            if it > 1
                tmp   = -s_curr[c] * H2[c] + c_curr[c] * H3[c]
                H2[c] =  c_curr[c] * H2[c] + s_curr[c] * H3[c]
                H3[c] = tmp
            end
            cc, ss, rr = _tpsci_givens(H3[c], H4[c])
            H3[c] = rr
            if abs(H3[c]) <= eps(T)
                # A vanishing Hessenberg pivot means the Krylov space is exhausted;
                # the single-root solver throws here, but with a block that would
                # take the healthy roots down with it. Freeze this one instead.
                verbose > 0 &&
                    @printf("   TPS-CEPA block MINRES root %i hit a near-zero pivot at iter %i\n", c, it)
                active[c] = false
                changed = true
                scal[c] = zero(T)
                continue
            end
            rhs2[c] = -ss * rhs1[c]
            rhs1[c] =  cc * rhs1[c]
            c_prev[c], s_prev[c] = c_curr[c], s_curr[c]
            c_curr[c], s_curr[c] = cc, ss
        end

        # Fan-out 3: normalize v_next, build w_next from the three-term recurrence,
        # take the solution step. Dead roots carry zero coefficients throughout.
        invH3 = T[active[c] ? inv(H3[c]) : zero(T) for c in 1:R]
        step  = T[active[c] ? rhs1[c] : zero(T) for c in 1:R]
        ops3 = ShardedBlockVecOp{T}[]
        push!(ops3, svb_scale(T, IVN, scal))
        push!(ops3, svb_copy(T, IWN, IVC))                              # w_next = v_curr
        it > 1 && push!(ops3, svb_axpy(T, IWN, T[-h for h in H2], IWC)) # -= H2*w_curr
        it > 2 && push!(ops3, svb_axpy(T, IWN, T[-h for h in H1], IWP)) # -= H1*w_prev
        push!(ops3, svb_scale(T, IWN, invH3))
        push!(ops3, svb_axpy(T, IX, step, IWN))                         # x += rhs1*w_next
        sharded_block_fused_ops!(vecs, ops3)

        vecs[IVP], vecs[IVC], vecs[IVN] = vecs[IVC], vecs[IVN], vecs[IVP]
        vecs[IWP], vecs[IWC], vecs[IWN] = vecs[IWC], vecs[IWN], vecs[IWP]

        for c in 1:R
            active[c] || continue
            rhs1[c] = rhs2[c]
            H2[c]   = H4[c]
            resnorm[c] = abs(rhs2[c])
            if resnorm[c] <= threshold[c]
                converged[c] = true
                active[c] = false
                changed = true
            end
        end
        if changed
            for c in 1:R
                mask[c] = active[c] ? one(T) : zero(T)
            end
            sharded_block_fused_ops!(vecs, [svb_scale(T, IVP, mask),
                                            svb_scale(T, IVC, mask),
                                            svb_scale(T, IWP, mask),
                                            svb_scale(T, IWC, mask)])
        end
        verbose > 2 &&
            @printf("   TPS-CEPA block MINRES iter %4i  max residual %12.4e\n",
                    it, maximum(resnorm))
    end

    for i in (IVN, IVP, IVC, IWN, IWC, IWP)
        destroy!(vecs[i])
    end
    return vecs[IX], (iters=iters, residual=resnorm, converged=converged)
end
#=}}}=#


# ── Blocked sharded CEPA driver ──────────────────────────────────────────────

"""
    _tps_sharded_cepa_solve_block(ref_vector, e0, cepa_vector, cluster_ops, clustered_ham,
                                  cepa_shift, cepa_mit; ...) -> (Ec, e0 .+ Ec)

Solve every root's amplitude equation in one pass: the Q-space Hamiltonian is
built once and all R equations ride on the same applies, so one apply and one set
of fan-outs serve all R roots instead of R of each.

Solver choice is made for the block as a unit. CG needs a positive-definite
shifted operator and MINRES does not, so PCG runs only when *every* root clears
the diagonal screen, and if any root then reports an indefinite direction the
whole block is redone with MINRES. Splitting the block by root would cost two
applies per iteration, and an apply costs the same whether one root or all R ride
on it — which would hand back exactly what blocking bought.
"""
function _tps_sharded_cepa_solve_block(ref_vector::TPSCIstate{T,N,R}, e0::Vector,
                                       cepa_vector::DistributedTPSCIstate{T,N,Rq},
                                       cluster_ops, clustered_ham,
                                       cepa_shift, cepa_mit;
                                       tol, cg_maxiter, nbody, thresh_sigma,
                                       solver, linsolve_tol, warm_start,
                                       h_storage, max_mem_H, workers,
                                       threaded_worker, blas_threads,
                                       verbose) where {T,N,R,Rq}
#={{{=#
    worker_ids = ensure_tpsci_multinode_workers!(workers=workers)
    worker_ids == cepa_vector.workers ||
        error("_tps_sharded_cepa_solve_block requires the CEPA vector on the requested workers")
    n_clusters = length(ref_vector.clusters)

    if verbose > 0
        @printf(" TPS-CEPA (stored-H) solver: dim_q=%i, R=%i, shift=%s, roots=block\n",
                length(cepa_vector), R, cepa_shift)
        flush(stdout)
    end

    hq_op, is_block = _tps_sharded_cepa_build_hq_op(cepa_vector, cluster_ops,
                                                    clustered_ham;
                                                    h_storage=h_storage,
                                                    max_mem_H=max_mem_H,
                                                    workers=worker_ids,
                                                    threaded_worker=threaded_worker,
                                                    blas_threads=blas_threads,
                                                    verbose=verbose)

    # The Q-space diagonal is root independent: build it once, reuse it for every
    # root and macro-iteration. Only the shift changes.
    Hdiag = nothing
    use_pcg = solver === :pcg
    if use_pcg
        if is_block
            Hdiag = compute_diagonal_sharded(hq_op, cepa_vector)
        else
            verbose > 0 &&
                @printf(" solver=:pcg needs a stored H for the diagonal; using :minres\n")
            use_pcg = false
        end
    end
    hdiag_min = Hdiag === nothing ? zero(T) : T(sharded_state_min(Hdiag))

    ltol = linsolve_tol === nothing ? tol : linsolve_tol
    Ec = zeros(T, R)
    Ec_prev = fill(T(Inf), R)
    h = nothing
    rhs = nothing
    Cd = nothing

    try
        # One sharded FOIS matvec for every root's coupling vector, instead of R.
        verbose > 1 && @printf(" Compute sharded coupling vectors for all %i roots\n", R)
        dref = distribute_tpsci_state(ref_vector; workers=worker_ids,
                                      strategy=:balanced, blas_threads=blas_threads)
        sig = open_matvec_sharded(dref, cluster_ops, clustered_ham;
                                  nbody=nbody, thresh=thresh_sigma, prescreen=false,
                                  workers=worker_ids, threaded_worker=threaded_worker,
                                  blas_threads=blas_threads,
                                  verbose=(verbose > 1 ? 1 : 0))
        h = restrict_to_basis_sharded(sig, cepa_vector)
        destroy!(sig)
        destroy!(dref)

        # The right-hand side does not change between macro-iterations; only the
        # shift does. Build it once.
        rhs = copy_sharded_state(h)
        sharded_block_fused_ops!([rhs], [svb_scale(T, 1, fill(-one(T), R))])

        for it in 1:cepa_mit
            shifts = zeros(T, R)
            for i in 1:R
                if     cepa_shift == "cepa";  shifts[i] = zero(T)
                elseif cepa_shift == "acpf";  shifts[i] = Ec[i] * 2.0 / n_clusters
                elseif cepa_shift == "aqcc"
                    shifts[i] = (1.0 - (n_clusters - 3.0) * (n_clusters - 2.0) /
                                 (n_clusters * (n_clusters - 1.0))) * Ec[i]
                elseif cepa_shift == "cisd";  shifts[i] = Ec[i]
                else;  error("Unknown cepa_shift: $cepa_shift")
                end
            end
            eshifts = T[e0[i] + shifts[i] for i in 1:R]

            if verbose > 1
                @printf(" CEPA Iter %3i  Shifts = %s\n", it,
                        join((@sprintf("%12.8f", s) for s in shifts), " "))
                flush(stdout)
            end

            x0 = (warm_start && Cd !== nothing) ? Cd : nothing
            Cd_new = nothing
            history = nothing
            used = :minres
            if use_pcg && all(hdiag_min - e > 0 for e in eshifts)
                Cd_new, history = tps_sharded_cepa_pcg_block(
                    rhs, hq_op, Hdiag, eshifts;
                    tol=ltol, maxiter=cg_maxiter, verbose=verbose, x0=x0)
                if any(history.indefinite)
                    verbose > 0 &&
                        @printf("   roots %s hit an indefinite direction; redoing the block with MINRES\n",
                                findall(history.indefinite))
                    Cd_new === nothing || x0 === nothing || Cd_new.id === x0.id || destroy!(Cd_new)
                    Cd_new = nothing
                else
                    used = :pcg
                end
            elseif use_pcg && verbose > 0
                @printf("   shifted diagonal is not positive for every root (min %.3e); using MINRES\n",
                        hdiag_min - maximum(eshifts))
            end
            if Cd_new === nothing
                Cd_new, history = tps_sharded_cepa_minres_block(
                    rhs, hq_op, eshifts;
                    tol=ltol, maxiter=cg_maxiter, verbose=verbose, x0=x0)
            end

            # A warm-started solve accumulates into the guess and hands the same
            # state back; only free the old amplitudes if a new one was allocated.
            Cd === nothing || Cd_new.id === Cd.id || destroy!(Cd)
            Cd = Cd_new

            ec = sharded_block_fused_ops!([Cd, h], [svb_dot(T, 1, 2)])
            for i in 1:R
                Ec[i] = ec[1, i]
            end
            if verbose > 0
                for i in 1:R
                    @printf(" Iter %3i  Root %i  [%s]  nops=%4i  res=%8.2e  E_corr = %16.12f%s\n",
                            it, i, used, history.iters[i], history.residual[i], Ec[i],
                            history.converged[i] ? "" : "   (not converged)")
                end
                flush(stdout)
            end

            cepa_shift == "cepa" && break
            maximum(abs.(Ec .- Ec_prev)) < tol && break
            Ec_prev .= Ec
        end
    finally
        Cd === nothing || destroy!(Cd)
        rhs === nothing || destroy!(rhs)
        h === nothing || destroy!(h)
        Hdiag === nothing || destroy!(Hdiag)
        is_block && destroy!(hq_op)
    end
    return Ec, e0 .+ Ec
end
#=}}}=#
