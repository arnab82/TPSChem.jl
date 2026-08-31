using TimerOutputs
using .BlockDavidson

"""
    build_full_H(ci_vector::TPSCIstate, cluster_ops, clustered_ham::ClusteredOperator)

Build full TPSCI Hamiltonian matrix in space spanned by `ci_vector`. This works in serial for the full matrix
"""
function build_full_H(ci_vector::TPSCIstate, cluster_ops, clustered_ham::ClusteredOperator)
#={{{=#
    dim = length(ci_vector)
    H = zeros(dim, dim)

    zero_fock = TransferConfig([(0,0) for i in ci_vector.clusters])
    bra_idx = 0
    for (fock_bra, configs_bra) in ci_vector.data
        for (config_bra, coeff_bra) in configs_bra
            bra_idx += 1
            ket_idx = 0
            for (fock_ket, configs_ket) in ci_vector.data
                fock_trans = fock_bra - fock_ket

                # check if transition is connected by H
                if haskey(clustered_ham, fock_trans) == false
                    ket_idx += length(configs_ket)
                    continue
                end

                for (config_ket, coeff_ket) in configs_ket
                    ket_idx += 1
                    ket_idx <= bra_idx || continue


                    for term in clustered_ham[fock_trans]
                    
                        check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue
                       
                        me = contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_ket, config_ket)
                        H[bra_idx, ket_idx] += me 
                    end

                    H[ket_idx, bra_idx] = H[bra_idx, ket_idx]

                end
            end
        end
    end
    return H
end
#=}}}=#


"""
    build_full_H_parallel(ci_vector::TPSCIstate, cluster_ops, clustered_ham::ClusteredOperator)

Build full TPSCI Hamiltonian matrix in space spanned by `ci_vector`. This works in serial for the full matrix
"""
function build_full_H_parallel( ci_vector_l::TPSCIstate{T,N,R}, ci_vector_r::TPSCIstate{T,N,R}, 
                                cluster_ops, clustered_ham::ClusteredOperator;
                                sym=false) where {T,N,R}
#={{{=#
    dim_l = length(ci_vector_l)
    dim_r = length(ci_vector_r)
    H = zeros(T, dim_l, dim_r)

    dim_l == dim_r || sym == false || error(" dim_l!=dim_r yet sym==true")

    if (dim_l == dim_r) && sym == false
        @warn(" are you missing sym=true?")
    end
    jobs = []

    zero_fock = TransferConfig([(0,0) for i in 1:N])
    bra_idx = 0

    for (fock_bra, configs_bra) in ci_vector_l.data
        for (config_bra, coeff_bra) in configs_bra
            bra_idx += 1
            #push!(jobs, (bra_idx, fock_bra, config_bra) )
            #push!(jobs, (bra_idx, fock_bra, config_bra, H[bra_idx,:]) )
            push!(jobs, (bra_idx, fock_bra, config_bra, zeros(dim_r)) )
        end
    end

    function do_job(job)
        fock_bra = job[2]
        config_bra = job[3]
        Hrow = job[4]
        ket_idx = 0

        for (fock_ket, configs_ket) in ci_vector_r.data
            fock_trans = fock_bra - fock_ket

            # check if transition is connected by H
            if haskey(clustered_ham, fock_trans) == false
                ket_idx += length(configs_ket)
                continue
            end

            for (config_ket, coeff_ket) in configs_ket
                ket_idx += 1
                ket_idx <= job[1] || sym == false || continue

                for term in clustered_ham[fock_trans]
                       
                    #length(term.clusters) <= 2 || continue
                    check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue
                    
                    me = contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_ket, config_ket)
                    #if term isa ClusteredTerm4B
                    #    @btime contract_matrix_element($term, $cluster_ops, $fock_bra, $config_bra, $fock_ket, $config_ket)
                    #end
                    Hrow[ket_idx] += me 
                    #H[job[1],ket_idx] += me 
                end

            end

        end
    end

    # because @threads divides evenly the loop, let's distribute thework more fairly
    #mid = length(jobs) ÷ 2
    #r = collect(1:length(jobs))
    #perm = [r[1:mid] reverse(r[mid+1:end])]'[:]
    #jobs = jobs[perm]
    
    #for job in jobs
    Threads.@threads :static for job in jobs
        do_job(job)
        #@btime $do_job($job)
    end

    for job in jobs
        H[job[1],:] .= job[4]
    end

    if sym
        for i in 1:dim_l
            @simd for j in i+1:dim_l
                @inbounds H[i,j] = H[j,i]
            end
        end
    end


    return H
end
#=}}}=#


"""
    build_H_qq(ci_vector::TPSCIstate, cluster_ops, clustered_ham)

Build the symmetric dim_q×dim_q Hamiltonian matrix for the Q-space defined by `ci_vector`.

Unlike `build_full_H_parallel`, each thread writes directly into its own row of H via a
view (no per-job scratch copy), halving peak memory from 2×dim_q²×8 B to 1×dim_q²×8 B.
Safe because each job owns a unique bra_idx row — no data race.
"""
function build_H_qq(ci_vector::TPSCIstate{T,N,R}, cluster_ops,
                    clustered_ham::ClusteredOperator) where {T,N,R}
#={{{=#
    dim = length(ci_vector)
    H = zeros(T, dim, dim)

    jobs = Vector{Tuple{Int, FockConfig{N}, ClusterConfig{N}}}()
    bra_idx = 0
    for (fock_bra, configs_bra) in ci_vector.data
        for (config_bra, _) in configs_bra
            bra_idx += 1
            push!(jobs, (bra_idx, fock_bra, config_bra))
        end
    end

    function do_job(job)
        fock_bra  = job[2]
        config_bra = job[3]
        Hrow = view(H, job[1], :)   # direct view — unique row, no race
        ket_idx = 0

        for (fock_ket, configs_ket) in ci_vector.data
            fock_trans = fock_bra - fock_ket
            if haskey(clustered_ham, fock_trans) == false
                ket_idx += length(configs_ket)
                continue
            end
            for (config_ket, _) in configs_ket
                ket_idx += 1
                ket_idx <= job[1] || continue   # lower triangle only

                for term in clustered_ham[fock_trans]
                    check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue
                    me = contract_matrix_element(term, cluster_ops, fock_bra, config_bra,
                                                 fock_ket, config_ket)
                    Hrow[ket_idx] += me
                end
            end
        end
    end

    Threads.@threads for job in jobs
        do_job(job)
    end

    # fill upper triangle
    for i in 1:dim
        @simd for j in i+1:dim
            @inbounds H[i,j] = H[j,i]
        end
    end

    return H
end
#=}}}=#


"""
    build_H_qq_sparse(ci_vector::TPSCIstate, cluster_ops, clustered_ham)

Build a sparse symmetric dim_q×dim_q Hamiltonian for the Q-space defined by `ci_vector`.

Each thread accumulates its own (I,J,V) triplet lists (no shared state, no locks), then
the full set is merged and passed to `sparse()`.  Peak memory is O(nnz) rather than
O(dim_q²), making this viable for dim_q >> 160K where the dense builders OOM.
"""
function build_H_qq_sparse(ci_vector::TPSCIstate{T,N,R}, cluster_ops,
                            clustered_ham::ClusteredOperator) where {T,N,R}
#={{{=#
    dim = length(ci_vector)
    nt  = Threads.maxthreadid()

    jobs = Vector{Tuple{Int, FockConfig{N}, ClusterConfig{N}}}()
    bra_idx = 0
    for (fock_bra, configs_bra) in ci_vector.data
        for (config_bra, _) in configs_bra
            bra_idx += 1
            push!(jobs, (bra_idx, fock_bra, config_bra))
        end
    end

    # Per-thread COO accumulators — no shared writes, no locks needed
    Is = [Vector{Int}()   for _ in 1:nt]
    Js = [Vector{Int}()   for _ in 1:nt]
    Vs = [Vector{T}()     for _ in 1:nt]

    Threads.@threads :static for job in jobs
        tid       = Threads.threadid()
        bra_idx_j = job[1]
        fock_bra  = job[2]
        config_bra = job[3]
        ket_idx   = 0

        for (fock_ket, configs_ket) in ci_vector.data
            fock_trans = fock_bra - fock_ket
            if !haskey(clustered_ham, fock_trans)
                ket_idx += length(configs_ket)
                continue
            end

            for (config_ket, _) in configs_ket
                ket_idx += 1
                ket_idx <= bra_idx_j || continue   # lower triangle only

                me = zero(T)
                for term in clustered_ham[fock_trans]
                    check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue
                    me += contract_matrix_element(term, cluster_ops, fock_bra, config_bra,
                                                  fock_ket, config_ket)
                end

                iszero(me) && continue

                push!(Is[tid], bra_idx_j); push!(Js[tid], ket_idx); push!(Vs[tid], me)
                if bra_idx_j != ket_idx   # off-diagonal: store transpose too
                    push!(Is[tid], ket_idx); push!(Js[tid], bra_idx_j); push!(Vs[tid], me)
                end
            end
        end
    end

    I_all = vcat(Is...)
    J_all = vcat(Js...)
    V_all = vcat(Vs...)
    return sparse(I_all, J_all, V_all, dim, dim)
end
#=}}}=#


"""
    _nnz_balanced_ranges(colptr, ncols, nchunks) -> Vector{UnitRange{Int}}

Split `1:ncols` into contiguous ranges holding roughly equal numbers of stored
entries, given a CSC `colptr`.

H_qq columns are very uneven — a Q-space config couples to as many others as its
Fock sector allows — so splitting the columns evenly by index leaves some threads
finishing long before others. Splitting by nnz keeps them all busy.
"""
function _nnz_balanced_ranges(colptr, ncols::Int, nchunks::Int)
#={{{=#
    ncols > 0 || return UnitRange{Int}[]
    nchunks = clamp(nchunks, 1, ncols)
    base  = colptr[1]
    total = colptr[ncols+1] - base
    ranges = UnitRange{Int}[]
    start = 1
    for c in 1:nchunks-1
        start > ncols && break
        goal = base + fld(total * c, nchunks)
        stop = clamp(searchsortedfirst(colptr, goal) - 1, start, ncols)
        push!(ranges, start:stop)
        start = stop + 1
    end
    start <= ncols && push!(ranges, start:ncols)
    return ranges
end
#=}}}=#


"""
    ThreadedSymSpMV(A::SparseMatrixCSC; chunks_per_thread=4)

Threaded matvec for a **symmetric** sparse matrix, callable as `op(x)` and usable
in place through `mul!(y, op, x)`.

`SparseArrays` runs `A*x` on a single thread, and its kernel *scatters* into
`y[rowvals(A)]`, which is why it cannot simply be threaded — two columns can hit
the same output row. H_qq is symmetric by construction, so its CSC arrays are
equally its CSR arrays: column j lists the entries of row j. That lets each thread
own a disjoint slice of `y` and *gather* into it — no locks, no per-thread
buffers, and the writes stay in cache.

The nnz-balanced column ranges are computed once, at construction.

!!! warning
    Only valid for a symmetric `A`. `build_H_qq`, `build_H_qq_sparse` and
    `build_full_H_parallel(..., sym=true)` all produce one.
"""
struct ThreadedSymSpMV{T,Ti}
    A::SparseMatrixCSC{T,Ti}
    ranges::Vector{UnitRange{Int}}
end

function ThreadedSymSpMV(A::SparseMatrixCSC{T,Ti}; chunks_per_thread::Int=4) where {T,Ti}
    # On one thread the gather kernel is ~1.5x slower than the SparseArrays scatter
    # kernel, so hand the work back to it: an empty range list means "delegate".
    Threads.nthreads() > 1 || return ThreadedSymSpMV{T,Ti}(A, UnitRange{Int}[])
    ranges = _nnz_balanced_ranges(SparseArrays.getcolptr(A), size(A, 2),
                                  chunks_per_thread * Threads.nthreads())
    return ThreadedSymSpMV{T,Ti}(A, ranges)
end

function LinearAlgebra.mul!(y::AbstractVector{T}, op::ThreadedSymSpMV{T}, x::AbstractVector{T}) where {T}
#={{{=#
    A = op.A
    length(y) == size(A, 1) && length(x) == size(A, 2) ||
        throw(DimensionMismatch("ThreadedSymSpMV: y=$(length(y)), A=$(size(A)), x=$(length(x))"))
    isempty(op.ranges) && return mul!(y, A, x)   # single-threaded: SparseArrays is faster
    nzv = nonzeros(A)
    rv  = rowvals(A)
    cp  = SparseArrays.getcolptr(A)
    Threads.@threads :dynamic for rng in op.ranges
        @inbounds for j in rng
            s = zero(T)
            for k in cp[j]:(cp[j+1]-1)
                s += nzv[k] * x[rv[k]]
            end
            y[j] = s
        end
    end
    return y
end
#=}}}=#

(op::ThreadedSymSpMV{T})(x::AbstractVector{T}) where {T} = mul!(similar(x, size(op.A, 1)), op, x)

Base.size(op::ThreadedSymSpMV) = size(op.A)
Base.eltype(::ThreadedSymSpMV{T}) where {T} = T


"""
    ThreadedSymDenseMV(A::Matrix; chunks_per_thread=4)

Threaded matvec for a **symmetric** dense matrix, callable as `op(x)`.

The dense H_qq builders hand back a plain `Matrix`, and `A*x` on it is a BLAS-2
`gemv` — which runs on however many threads BLAS was given. The CEPA drivers set
`BLAS.set_num_threads(1)` (and `open_matvec_thread` sets it again), so in practice
that matvec is serial. Threading it over Julia threads instead makes the dense
path independent of the BLAS thread setting, and comparable with
[`ThreadedSymSpMV`](@ref). Column j of a symmetric A is row j, so each thread
gathers a disjoint slice of `y` with a contiguous `dot`.
"""
struct ThreadedSymDenseMV{T}
    A::Matrix{T}
    ranges::Vector{UnitRange{Int}}
end

function ThreadedSymDenseMV(A::Matrix{T}; chunks_per_thread::Int=4) where {T}
    n = size(A, 2)
    # Single thread: BLAS gemv beats a column-at-a-time loop. Empty means "delegate".
    Threads.nthreads() > 1 || return ThreadedSymDenseMV{T}(A, UnitRange{Int}[])
    nchunks = clamp(chunks_per_thread * Threads.nthreads(), 1, max(n, 1))
    # Dense columns all cost the same, so an even split is already balanced.
    step = cld(n, nchunks)
    ranges = [i:min(i+step-1, n) for i in 1:step:n]
    return ThreadedSymDenseMV{T}(A, ranges)
end

function LinearAlgebra.mul!(y::AbstractVector{T}, op::ThreadedSymDenseMV{T}, x::AbstractVector{T}) where {T}
    A = op.A
    length(y) == size(A, 1) && length(x) == size(A, 2) ||
        throw(DimensionMismatch("ThreadedSymDenseMV: y=$(length(y)), A=$(size(A)), x=$(length(x))"))
    isempty(op.ranges) && return mul!(y, A, x)    # single-threaded: BLAS is faster
    Threads.@threads :dynamic for rng in op.ranges
        @inbounds for j in rng
            y[j] = dot(view(A, :, j), x)
        end
    end
    return y
end

(op::ThreadedSymDenseMV{T})(x::AbstractVector{T}) where {T} = mul!(similar(x, size(op.A, 1)), op, x)

Base.size(op::ThreadedSymDenseMV) = size(op.A)
Base.eltype(::ThreadedSymDenseMV{T}) where {T} = T


# ─────────────────────────────────────────────────────────────────────────────
# Block (multi-root) Q-space operators.
#
# Layout convention: block vectors are **root-major**, an `R × dim_q` matrix whose
# row `c` holds root `c`'s Q-space coefficients. That is what makes batching pay:
# the R numbers a given nonzero (or a given recomputed matrix element) has to touch
# sit next to each other in one cache line, so H_qq is streamed — or rebuilt — once
# for all R roots instead of once per root. The saving is in the *shared* work, not
# in flops, which is why it is large for these operators and nil for a plain
# `SparseArrays` `A*X` (that kernel loops the RHS columns outermost and re-streams A
# for each one).
#
# Every operator below provides
#   mul!(y, op, x)         single root,  x/y :: Vector      (length dim_q)
#   mul_block!(Y, op, X)   R roots,      X/Y :: Matrix      (R × dim_q)
# so the CEPA solver can pick a storage strategy and a root strategy independently.
# ─────────────────────────────────────────────────────────────────────────────

"""
    mul_block!(Y, op, X)

Apply a Q-space operator to a whole block of roots: `Y .= op * X`, with `X` and `Y`
root-major (`R × dim_q`). See the block-operator notes above for why the layout is
transposed relative to the usual `dim_q × R`.
"""
function mul_block! end


function mul_block!(Y::Matrix{T}, op::ThreadedSymSpMV{T}, X::Matrix{T}) where {T}
#={{{=#
    A = op.A
    R, n = size(X)
    size(Y) == (R, n) || throw(DimensionMismatch("Y is $(size(Y)), expected $((R, n))"))
    n == size(A, 2) || throw(DimensionMismatch("X has $n columns, A is $(size(A))"))
    nzv = nonzeros(A)
    rv  = rowvals(A)
    cp  = SparseArrays.getcolptr(A)
    ranges = isempty(op.ranges) ? [1:n] : op.ranges
    Threads.@threads :dynamic for rng in ranges
        s = zeros(T, R)
        @inbounds for j in rng
            fill!(s, zero(T))
            for k in cp[j]:(cp[j+1]-1)
                r = rv[k]
                v = nzv[k]
                @simd for c in 1:R
                    s[c] += v * X[c, r]
                end
            end
            @simd for c in 1:R
                Y[c, j] = s[c]
            end
        end
    end
    return Y
end
#=}}}=#


function mul_block!(Y::Matrix{T}, op::ThreadedSymDenseMV{T}, X::Matrix{T}) where {T}
#={{{=#
    A = op.A
    R, n = size(X)
    size(Y) == (R, n) || throw(DimensionMismatch("Y is $(size(Y)), expected $((R, n))"))
    n == size(A, 2) || throw(DimensionMismatch("X has $n columns, A is $(size(A))"))
    ranges = isempty(op.ranges) ? [1:n] : op.ranges
    Threads.@threads :dynamic for rng in ranges
        s = zeros(T, R)
        @inbounds for j in rng
            fill!(s, zero(T))
            for r in 1:n
                v = A[r, j]
                @simd for c in 1:R
                    s[c] += v * X[c, r]
                end
            end
            @simd for c in 1:R
                Y[c, j] = s[c]
            end
        end
    end
    return Y
end
#=}}}=#


"""
    StructuredHqq(ci_vector, cluster_ops, clustered_ham; nroots=1)

Q-space operator that stores no matrix and recomputes H_qq's entries on every
apply, the `build_hqq=:matvec` strategy. Peak memory is `O(nthreads × R × dim_q)`,
which is what makes it the only option once `dim_q` is large enough that even the
sparse matrix will not fit.

Recomputing `contract_matrix_element` dominates the cost completely, and that work
is shared by every root — so applying this to a block of R roots costs barely more
than applying it to one, and `mul_block!` is close to an R-fold saving.

The job list and the per-thread accumulation buffers are built once, at
construction, rather than on every apply.
"""
mutable struct StructuredHqq{T,N,R,H}
    ci_vector::TPSCIstate{T,N,1}
    cluster_ops
    clustered_ham::H
    jobs::Vector{Tuple{Int,FockConfig{N},ClusterConfig{N}}}
    scr::Vector{Matrix{T}}      # per-thread R × dim_q accumulators
    dim::Int
end

function StructuredHqq(ci_vector::TPSCIstate{T,N,R1}, cluster_ops,
                       clustered_ham::H; nroots::Int=1) where {T,N,R1,H}
#={{{=#
    v1 = R1 == 1 ? ci_vector : TPSCIstate(ci_vector, R=1)
    dim = length(v1)
    jobs = Vector{Tuple{Int,FockConfig{N},ClusterConfig{N}}}()
    bra_idx = 0
    for (fock_bra, configs_bra) in v1.data
        for (config_bra, _) in configs_bra
            bra_idx += 1
            push!(jobs, (bra_idx, fock_bra, config_bra))
        end
    end
    scr = [zeros(T, nroots, dim) for _ in 1:Threads.maxthreadid()]
    return StructuredHqq{T,N,nroots,H}(v1, cluster_ops, clustered_ham, jobs, scr, dim)
end
#=}}}=#

Base.size(op::StructuredHqq) = (op.dim, op.dim)
Base.eltype(::StructuredHqq{T}) where {T} = T


function mul_block!(Y::Matrix{T}, op::StructuredHqq{T,N,R}, X::Matrix{T}) where {T,N,R}
#={{{=#
    nr, n = size(X)
    n == op.dim || throw(DimensionMismatch("X has $n columns, expected $(op.dim)"))
    size(Y) == (nr, n) || throw(DimensionMismatch("Y is $(size(Y)), expected $((nr, n))"))
    nr <= size(op.scr[1], 1) ||
        throw(DimensionMismatch("operator was built for $(size(op.scr[1],1)) roots, got $nr"))

    ci_vector    = op.ci_vector
    cluster_ops  = op.cluster_ops
    clustered_ham = op.clustered_ham
    for s in op.scr
        fill!(s, zero(T))
    end

    Threads.@threads :static for job in op.jobs
        tid        = Threads.threadid()
        res        = op.scr[tid]
        bra_idx_j  = job[1]
        fock_bra   = job[2]
        config_bra = job[3]
        ket_idx    = 0

        for (fock_ket, configs_ket) in ci_vector.data
            fock_trans = fock_bra - fock_ket
            if !haskey(clustered_ham, fock_trans)
                ket_idx += length(configs_ket)
                continue
            end

            for (config_ket, _) in configs_ket
                ket_idx += 1
                ket_idx <= bra_idx_j || continue   # lower triangle only

                me = zero(T)
                for term in clustered_ham[fock_trans]
                    check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue
                    me += contract_matrix_element(term, cluster_ops, fock_bra, config_bra,
                                                  fock_ket, config_ket)
                end

                iszero(me) && continue

                # One recomputed matrix element, R roots updated from it. This is the
                # whole point of the block layout: both slices are contiguous.
                @inbounds @simd for c in 1:nr
                    res[c, bra_idx_j] += me * X[c, ket_idx]
                end
                if bra_idx_j != ket_idx
                    @inbounds @simd for c in 1:nr
                        res[c, ket_idx] += me * X[c, bra_idx_j]
                    end
                end
            end
        end
    end

    # The scratch is sized for the operator's root count, which can exceed the
    # block actually being applied — slice, don't copyto!, which would flatten.
    Y .= view(op.scr[1], 1:nr, :)
    for t in 2:length(op.scr)
        Y .+= view(op.scr[t], 1:nr, :)
    end
    return Y
end
#=}}}=#

function LinearAlgebra.mul!(y::Vector{T}, op::StructuredHqq{T}, x::Vector{T}) where {T}
    Y = reshape(y, 1, length(y))
    X = reshape(x, 1, length(x))
    mul_block!(Y, op, X)
    return y
end

(op::StructuredHqq{T})(x::Vector{T}) where {T} = mul!(similar(x, op.dim), op, x)


"""
    FoisHqq(cepa_vector, cluster_ops, clustered_ham; nbody=4, nroots=1)

Q-space operator that never touches H_qq at all: it applies H through
`open_matvec_thread` over the full first-order interacting space and projects the
result back onto Q. Peak memory is `O(dim_q)` per root plus the matvec's own
scratch, which is what lets it run when neither the sparse nor the dense matrix
fits — and it is also, by a wide margin, the slowest per apply.

`open_matvec_thread` is already multi-root: the Fock-transition job setup, the term
screening and `contract_matvec_thread` all carry the whole length-R coefficient
vector through in one pass. Applying it to a block of roots therefore amortises all
of that over R, which is where the batching win comes from on this path.

!!! note
    Screening inside the matvec uses `maximum(abs, coef_ket)` over the roots, so a
    block apply keeps the *union* of the roots' surviving terms. Against a
    root-at-a-time apply that retains strictly more terms — more accurate, but not
    bitwise identical.
"""
mutable struct FoisHqq{T,N,R,H}
    work::TPSCIstate{T,N,R}         # R-root Q-space scratch, reused every apply
    basis::TPSCIstate{T,N,R}        # defines the Q-space ordering
    cluster_ops
    clustered_ham::H
    nbody::Int
    dim::Int
end

function FoisHqq(cepa_vector::TPSCIstate{T,N,R1}, cluster_ops, clustered_ham::H;
                 nbody::Int=4, nroots::Int=1) where {T,N,R1,H}
    work = TPSCIstate(cepa_vector, R=nroots)
    return FoisHqq{T,N,nroots,H}(work, work, cluster_ops, clustered_ham, nbody,
                                 length(cepa_vector))
end

Base.size(op::FoisHqq) = (op.dim, op.dim)
Base.eltype(::FoisHqq{T}) where {T} = T

"""
    project_to_Q!(out, sig, basis)

Gather the coefficients of `sig` that live on `basis`'s configs into the root-major
block `out` (`R × dim_q`), zeroing anything `sig` does not carry.
"""
function project_to_Q!(out::Matrix{T}, sig::TPSCIstate{T,N,Rs},
                       basis::TPSCIstate{T,N,Rb}) where {T,N,Rs,Rb}
#={{{=#
    nr = size(out, 1)
    nr <= Rs || throw(DimensionMismatch("sig carries $Rs roots, out wants $nr"))
    fill!(out, zero(T))
    idx = 0
    for (fock, configs) in basis.data
        has_fock = haskey(sig, fock)
        for (config, _) in configs
            idx += 1
            has_fock || continue
            sig_f = sig[fock]
            haskey(sig_f, config) || continue
            coeffs = sig_f[config]
            @inbounds @simd for c in 1:nr
                out[c, idx] = coeffs[c]
            end
        end
    end
    return out
end
#=}}}=#

function mul_block!(Y::Matrix{T}, op::FoisHqq{T,N,R}, X::Matrix{T}) where {T,N,R}
#={{{=#
    nr, n = size(X)
    n == op.dim || throw(DimensionMismatch("X has $n columns, expected $(op.dim)"))
    nr == R || throw(DimensionMismatch("operator was built for $R roots, got $nr"))
    # set_vector! wants dim_q × R, the block kernels want R × dim_q. The transpose is
    # a couple of MB against a matvec that costs seconds — not worth avoiding.
    set_vector!(op.work, Matrix(transpose(X)))
    sig = open_matvec_thread(op.work, op.cluster_ops, op.clustered_ham,
                             nbody=op.nbody, thresh=0.0)
    return project_to_Q!(Y, sig, op.basis)
end
#=}}}=#

function LinearAlgebra.mul!(y::Vector{T}, op::FoisHqq{T,N,1}, x::Vector{T}) where {T,N}
    Y = reshape(y, 1, length(y))
    X = reshape(x, 1, length(x))
    mul_block!(Y, op, X)
    return y
end

(op::FoisHqq{T,N,1})(x::Vector{T}) where {T,N} = mul!(similar(x, op.dim), op, x)


"""
    matvec_H_qq(ci_vector::TPSCIstate, cluster_ops, clustered_ham, v) -> Vector

Apply H_qq to `v` without storing H_qq.

Structurally identical to `build_H_qq_sparse` but instead of accumulating COO triplets,
each thread accumulates H_{bra,ket}×v[ket] and H_{ket,bra}×v[bra] directly into a
per-thread result buffer. Peak memory is O(nthreads × dim_q) — a few hundred MB for
dim_q=262K — making this viable when both the sparse builder and open_matvec_thread OOM.
"""
function matvec_H_qq(ci_vector::TPSCIstate{T,N,R}, cluster_ops,
                     clustered_ham::ClusteredOperator, v::Vector{T}) where {T,N,R}
#={{{=#
    dim = length(ci_vector)
    length(v) == dim || throw(DimensionMismatch("v has length $(length(v)), expected $dim"))
    nt  = Threads.maxthreadid()

    jobs = Vector{Tuple{Int, FockConfig{N}, ClusterConfig{N}}}()
    bra_idx = 0
    for (fock_bra, configs_bra) in ci_vector.data
        for (config_bra, _) in configs_bra
            bra_idx += 1
            push!(jobs, (bra_idx, fock_bra, config_bra))
        end
    end

    # Per-thread result buffers — no shared writes, no locks needed
    res = [zeros(T, dim) for _ in 1:nt]

    Threads.@threads :static for job in jobs
        tid        = Threads.threadid()
        bra_idx_j  = job[1]
        fock_bra   = job[2]
        config_bra = job[3]
        ket_idx    = 0

        for (fock_ket, configs_ket) in ci_vector.data
            fock_trans = fock_bra - fock_ket
            if !haskey(clustered_ham, fock_trans)
                ket_idx += length(configs_ket)
                continue
            end

            for (config_ket, _) in configs_ket
                ket_idx += 1
                ket_idx <= bra_idx_j || continue   # lower triangle only

                me = zero(T)
                for term in clustered_ham[fock_trans]
                    check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue
                    me += contract_matrix_element(term, cluster_ops, fock_bra, config_bra,
                                                  fock_ket, config_ket)
                end

                iszero(me) && continue

                res[tid][bra_idx_j] += me * v[ket_idx]
                if bra_idx_j != ket_idx
                    res[tid][ket_idx] += me * v[bra_idx_j]
                end
            end
        end
    end

    # Reduce per-thread buffers
    result = res[1]
    for t in 2:nt
        result .+= res[t]
    end
    return result
end
#=}}}=#


"""
    function tps_ci_direct( ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator;
                        H_old    = nothing,
                        v_old    = nothing,
                        verbose   = 0) where {T,N,R}

# Solve for eigenvectors/values in the basis defined by `ci_vector`. Use direct diagonalization. 

If updating existing matrix, pass in H_old/v_old to avoid rebuilding that block
# Arguments
- `solver`: Which solver to use. Options = ["davidson", "krylovkit"]
"""
function tps_ci_direct( ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator;
                        conv_thresh = 1e-5,
                        lindep_thresh = 1e-12,
                        max_ss_vecs = 12,
                        max_iter    = 40,
                        shift       = nothing,
                        precond     = true,
                        H_old    = nothing,
                        v_old    = nothing,
                        verbose   = 0,
                        solver = "davidson") where {T,N,R}
    #={{{=#
    println()
    @printf(" |== Tensor Product State CI =======================================\n")
    vec_out = deepcopy(ci_vector)
    e0 = zeros(T,R)
    @printf(" Hamiltonian matrix dimension = %5i: \n", length(ci_vector))
    dim = length(ci_vector)
    flush(stdout)
   
    precond == true || println(" davidson not using preconditioning")

    H = zeros(T, 1,1)

    if H_old !== nothing
        v_old !== nothing || error(" can't specify H_old w/out v_old")
        v_tot = deepcopy(ci_vector)
        v_new = deepcopy(ci_vector)
        
        project_out!(v_new, v_old)
        
        #v_tot = copy(v_old)
        #add!(v_tot, v_new)

        dim_old = length(v_old)
        dim_new = length(v_new)
            

        # create indexing to find old indices in new space
        indices = OrderedDict{FockConfig{N}, OrderedDict{ClusterConfig{N}, Int}}()
   
        idx = 1
        for (fock,configs) in v_tot.data
            indices[fock] = OrderedDict{ClusterConfig{N}, Int}()
            for (config,coeff) in configs
                indices[fock][config] = idx
                idx += 1
            end
        end

        dim = dim_old + dim_new

        dim == length(v_tot) || error(" not adding up", dim_old, " ", dim_new, " ", length(v_tot))

        H = zeros(T, dim, dim)


        # add old H elements
        @printf(" %-50s", "Fill old/old Hamiltonian: ")
        flush(stdout)
        @time _fill_H_block!(H, H_old, v_old, v_old, indices)

        @printf(" %-50s", "Build old/new Hamiltonian matrix with dimension: ")
        flush(stdout)
        @time Htmp = build_full_H_parallel(v_old, v_new, cluster_ops, clustered_ham)
        _fill_H_block!(H, Htmp, v_old, v_new, indices)
        _fill_H_block!(H, Htmp', v_new, v_old, indices)

        @printf(" %-50s", "Build new/new Hamiltonian matrix with dimension: ")
        flush(stdout)
        @time Htmp = build_full_H_parallel(v_new, v_new, cluster_ops, clustered_ham, sym=true)
        _fill_H_block!(H, Htmp, v_new, v_new, indices)
        
        vec_out = deepcopy(v_tot)
    else
        @printf(" %-50s", "Build full Hamiltonian matrix with dimension: ")
        @time H = build_full_H_parallel(ci_vector, ci_vector, cluster_ops, clustered_ham, sym=true)
    end
        
        

    @printf(" Now diagonalize\n")
    flush(stdout)
    if length(vec_out) > 500
    
        if solver == "krylovkit"
            time = @elapsed e0,v, info = KrylovKit.eigsolve(H, R, :SR, 
                                                            verbosity=  verbose, 
                                                            maxiter=    max_iter, 
                                                            #krylovdim=20, 
                                                            issymmetric=true, 
                                                            ishermitian=true, 
                                                            tol=        conv_thresh)
            println()
            println(info)
            println()
            @printf(" %-50s%10.6f seconds\n", "Diagonalization time: ",time)
            v = hcat(v[1:R]...)

        elseif solver == "arpack"
            time = @elapsed e0,v = Arpack.eigs(H, nev = R, which=:SR)
        
        elseif solver == "davidson"
            davidson = Davidson(H, v0=get_vector(ci_vector), 
                                        max_iter=max_iter, max_ss_vecs=max_ss_vecs, nroots=R, tol=conv_thresh, lindep_thresh=lindep_thresh)
            # time = @elapsed e0,v = BlockDavidson.eigs(davidson);
            time = @elapsed e0,v = BlockDavidson.eigs(davidson, Adiag=diag(H), precond_start_thresh=1e-1);
        end
        @printf(" %-50s", "Diagonalization time: ")
        @printf("%10.6f seconds\n",time)
        if verbose > 0
            display(info)
        end
    else
        time = @elapsed F = eigen(H)
        e0 = F.values[1:R]
        v = F.vectors[:,1:R]
        @printf(" %-50s", "Diagonalization time: ")
        @printf("%10.6f seconds\n",time)
    end
    set_vector!(vec_out, v)

    clustered_S2 = extract_S2(ci_vector.clusters, T=T)
    @printf(" %-50s", "Compute S2 expectation values: ")
    @time s2 = compute_expectation_value_parallel(vec_out, cluster_ops, clustered_S2)
    #@timeit to "<S2>" s2 = compute_expectation_value_parallel(vec_out, cluster_ops, clustered_S2)
    flush(stdout)
    @printf(" %5s %12s %12s\n", "Root", "Energy", "S2") 
    for r in 1:R
        @printf(" %5s %12.8f %12.8f\n",r, e0[r], abs(s2[r]))
    end

    if verbose > 1
        for r in 1:R
            display(vec_out, root=r)
        end
    end

    @printf(" ==================================================================|\n")
    return e0, vec_out, H 
end
#=}}}=#

function _fill_H_block!(H_big, H_small, v_l,v_r, indices)
    #={{{=#
    # Fill H_big with elements from H_small
    idx_l = 1
    
    idx_l = zeros(Int,length(v_l))
    idx_r = zeros(Int,length(v_r))

    idx = 1
    for (fock,configs) in v_l.data
        for (config,coeff) in configs
            idx_l[idx] = indices[fock][config]
            idx += 1
        end
    end

    idx = 1
    for (fock,configs) in v_r.data
        for (config,coeff) in configs
            idx_r[idx] = indices[fock][config]
            idx += 1
        end
    end

    for (il,iil) in enumerate(idx_l)
        for (ir,iir) in enumerate(idx_r)
            H_big[iil,iir] = H_small[il,ir]
        end
    end
#    for (fock_l,configs_l) in v_l.data
#        for (config_l,coeff_l) in configs_l
#            idx_l_tot = indices[fock_l][config_l]
#
#            idx_r = 1
#            for (fock_r,configs_r) in v_r.data
#                for (config_r,coeff_r) in configs_r
#                    idx_r_tot = indices[fock_r][config_r]
#
#                    H_big[idx_l_tot, idx_r_tot] = H_small[idx_l, idx_r]
#
#                    idx_r += 1
#                end
#            end
#
#            idx_l += 1
#        end
#    end
end
#=}}}=#


"""
    tps_ci_davidson(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator) where {T,N,R}

# Solve for eigenvectors/values in the basis defined by `ci_vector`. Use iterative davidson solver. 
"""
function tps_ci_davidson(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator;
                        conv_thresh = 1e-5,
                        lindep_thresh = 1e-12,
                        max_ss_vecs = 12,
                        max_iter    = 40,
                        shift       = nothing,
                        precond     = true,
                        verbose     = 0) where {T,N,R}
    #={{{=#
    println()
    @printf(" |== Tensor Product State CI =======================================\n")
    vec_out = deepcopy(ci_vector)
    e0 = zeros(T,R) 
   
    dim = length(ci_vector)
    iters = 0

    
    function matvec(v::Vector) 
        iters += 1
        #in = deepcopy(ci_vector) 
        in = TPSCIstate(ci_vector, R=size(v,2))
        set_vector!(in, v)
        #sig = deepcopy(in)
        #zero!(sig)
        #build_sigma!(sig, ci_vector, cluster_ops, clustered_ham, cache=cache)
        return tps_ci_matvec(in, cluster_ops, clustered_ham)[:,1]
    end
    function matvec(v::Matrix)
        iters += 1
        #in = deepcopy(ci_vector) 
        in = TPSCIstate(ci_vector, R=size(v,2))
        set_vector!(in, v)
        #sig = deepcopy(in)
        #zero!(sig)
        #build_sigma!(sig, ci_vector, cluster_ops, clustered_ham, cache=cache)
        return tps_ci_matvec(in, cluster_ops, clustered_ham)
    end


    Hmap = LinOpMat{T}(matvec, dim, true)

    davidson = Davidson(Hmap, v0=get_vector(ci_vector), 
                                max_iter=max_iter, max_ss_vecs=max_ss_vecs, nroots=R, tol=conv_thresh, lindep_thresh=lindep_thresh)

    #time = @elapsed e0,v = Arpack.eigs(Hmap, nev = R, which=:SR)
    #time = @elapsed e0,v, info = KrylovKit.eigsolve(Hmap, R, :SR, 
    #                                                verbosity=  verbose, 
    #                                                maxiter=    max_iter, 
    #                                                #krylovdim=20, 
    #                                                issymmetric=true, 
    #                                                ishermitian=true, 
    #                                                tol=        conv_thresh)

    e = nothing
    v = nothing
    if precond
        @printf(" %-50s", "Compute diagonal: ")
        # clustered_ham_0 = extract_1body_operator(clustered_ham, op_string = "Hcmf") 
        @time Hd = compute_diagonal(ci_vector, cluster_ops, clustered_ham)
        # @printf(" %-50s", "Compute <0|H0|0>: ")
        # @time E0 = compute_expectation_value_parallel(ci_vector, cluster_ops, clustered_ham_0)[1]
        # @time Eref = compute_expectation_value_parallel(ci_vector, cluster_ops, clustered_ham)[1]
        # Hd .+= Eref - E0
        @printf(" Now iterate: \n")
        flush(stdout)
        @time e,v = BlockDavidson.eigs(davidson, Adiag=Hd);
    else
        @time e,v = BlockDavidson.eigs(davidson);
    end
    set_vector!(vec_out, v)
    
    clustered_S2 = extract_S2(ci_vector.clusters)
    @printf(" %-50s", "Compute S2 expectation values: ")
    @time s2 = compute_expectation_value_parallel(vec_out, cluster_ops, clustered_S2)
    flush(stdout)
    @printf(" %5s %12s %12s\n", "Root", "Energy", "S2") 
    for r in 1:R
        @printf(" %5s %12.8f %12.8f\n",r, e[r], abs(s2[r]))
    end

    if verbose > 1
        for r in 1:R
            display(vec_out, root=r)
        end
    end

    @printf(" ==================================================================|\n")
    return e, vec_out 
end
#=}}}=#


"""
    tps_ci_matvec(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator) where {T,N,R}

# Compute the action of `clustered_ham` on `ci_vector`. 
"""
function tps_ci_matvec(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator) where {T,N,R}
    #={{{=#

    jobs = []

    bra_idx = 0
    for (fock_bra, configs_bra) in ci_vector.data
        for (config_bra, coeff_bra) in configs_bra
            bra_idx += 1
            push!(jobs, (bra_idx, fock_bra, config_bra, coeff_bra, zeros(T,R)) )
        end
    end

    function do_job(job)
        fock_bra = job[2]
        config_bra = job[3]
        coeff_bra = job[4]
        sig_out = job[5]
    
        for (fock_trans, terms) in clustered_ham
            fock_ket = fock_bra - fock_trans

            haskey(ci_vector.data, fock_ket) || continue
            
            configs_ket = ci_vector[fock_ket]


            for (config_ket, coeff_ket) in configs_ket
                for term in clustered_ham[fock_trans]
                    check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue
    
                    #norm(term.ints)*maximum(abs.(coeff_ket)) > 1e-5 || continue
                    #@btime norm($term.ints)*maximum(abs.($coeff_ket)) > 1e-12 
                    

                    me = contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_ket, config_ket)
                    #if term isa ClusteredTerm4B
                    #    @btime contract_matrix_element($term, $cluster_ops, $fock_bra, $config_bra, $fock_ket, $config_ket)
                    #end
                    @simd for r in 1:R
                        @inbounds sig_out[r] += me * coeff_ket[r]
                    end
                    #@btime $sig_out .+= $me .* $ci_vector[$fock_ket][$config_ket] 
                end

            end

        end
    end

    #for job in jobs
    Threads.@threads :static for job in jobs
        do_job(job)
        #@btime $do_job($job)
    end

    sigv = zeros(size(ci_vector))
    for job in jobs
        #for r in 1:R
        #    sigv[job[1],r] += job[5][r]
        #end
        sigv[job[1],:] .+= job[5]
    end

    return sigv
end
#=}}}=#



function print_tpsci_iter(ci_vector::TPSCIstate{T,N,R}, it, e0, converged) where {T,N,R}
#={{{=#
    if converged 
        @printf("*TPSCI Iter %-3i Dim: %-6i", it, length(ci_vector))
    else
        @printf(" TPSCI Iter %-3i Dim: %-6i", it, length(ci_vector))
    end
    @printf(" E(var): ")
    for i in 1:R
        @printf("%13.8f ", e0[i])
    end
#    @printf(" E(pt2): ")
#    for i in 1:R
#        @printf("%13.8f ", e2[i])
#    end
    println()
end
#=}}}=#

"""
    compute_expectation_value(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator; nbody=4) where {T,N,R}

Compute expectation value of a `ClusteredOperator` (`clustered_ham`) for state `ci_vector`
"""
function compute_expectation_value(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator; nbody=4) where {T,N,R}
    #={{{=#

    out = zeros(T,R)

    for (fock_bra, configs_bra) in ci_vector.data

        for (fock_ket, configs_ket) in ci_vector.data
            fock_trans = fock_bra - fock_ket

            # check if transition is connected by H
            haskey(clustered_ham, fock_trans) || continue

            for (config_bra, coeff_bra) in configs_bra
                for (config_ket, coeff_ket) in configs_ket

                    me = 0.0
                    for term in clustered_ham[fock_trans]

                        length(term.clusters) <= nbody || continue
                        check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue

                        me += contract_matrix_element(term, cluster_ops, 
                                                      fock_bra, config_bra, 
                                                      fock_ket, config_ket)
                    end

                    #out .+= coeff_bra .* coeff_ket .* me
                    for r in 1:R
                        out[r] += coeff_bra[r] * coeff_ket[r] * me
                    end

                end

            end
        end
    end

    return out 
end
#=}}}=#

"""
    function compute_expectation_value_parallel(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator) where {T,N,R}
"""
function compute_expectation_value_parallel(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator) where {T,N,R}
    #={{{=#

    # 
    # This will be were we collect our results
    evals = zeros(T,R)

    jobs = []

    for (fock_bra, configs_bra) in ci_vector.data
        for (config_bra, coeff_bra) in configs_bra
            push!(jobs, (fock_bra, config_bra, coeff_bra, zeros(T,R)) )
        end
    end

    function _add_val!(eval_job, me, coeff_bra, coeff_ket)
        for ri in 1:R
            #for rj in ri:R
            #    @inbounds eval_job[ri,rj] += me * coeff_bra[ri] * coeff_ket[rj] 
            #    #eval_job[rj,ri] = eval_job[ri,rj]
            #end
            @inbounds eval_job[ri] += me * coeff_bra[ri] * coeff_ket[ri] 
        end
    end

    function do_job(job)
        fock_bra = job[1]
        config_bra = job[2]
        coeff_bra = job[3]
        eval_job = job[4]
        ket_idx = 0

        for (fock_ket, configs_ket) in ci_vector.data
            fock_trans = fock_bra - fock_ket

            # check if transition is connected by H
            if haskey(clustered_ham, fock_trans) == false
                ket_idx += length(configs_ket)
                continue
            end

            for (config_ket, coeff_ket) in configs_ket
                #ket_idx += 1
                #ket_idx <= job[1] || continue

                me = 0.0
                for term in clustered_ham[fock_trans]

                    #length(term.clusters) <= 2 || continue
                    check_term(term, fock_bra, config_bra, fock_ket, config_ket) || continue

                    me += contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_ket, config_ket)
                    #if term isa ClusteredTerm4B
                    #    @btime contract_matrix_element($term, $cluster_ops, $fock_bra, $config_bra, $fock_ket, $config_ket)
                    #end
                    #Hrow[ket_idx] += me 
                    #H[job[1],ket_idx] += me 
                end
                #
                # now add the results
                #@inbounds for ri in 1:R
                #    @simd for rj in ri:R
                _add_val!(eval_job, me, coeff_bra, coeff_ket)
                #for ri in 1:R
                #    for rj in ri:R
                #        eval_job[ri,rj] += me * coeff_bra[ri] * coeff_ket[rj] 
                #        #eval_job[rj,ri] = eval_job[ri,rj]
                #    end
                #end
            end
        end
    end

    #for job in jobs
    #Threads.@threads :static for job in jobs
    @qthreads for job in jobs
        do_job(job)
        #@btime $do_job($job)
    end

    for job in jobs
        evals .+= job[4]
    end

    return evals 
end
#=}}}=#

"""
    compute_diagonal(vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham) where {T,N,R}

Form the diagonal of the hamiltonan, `clustered_ham`, in the basis defined by `vector`
"""
function compute_diagonal(vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator) where {T,N,R}
    Hd = zeros(size(vector)[1])
    idx = 0
    zero_trans = TransferConfig([(0,0) for i in 1:N])
    for (fock_bra, configs_bra) in vector.data
        for (config_bra, coeff_bra) in configs_bra
            idx += 1
            for term in clustered_ham[zero_trans]
                try
                    Hd[idx] += contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_bra, config_bra)
                catch
                    display(term)
                    display(fock_bra)
                    display(config_bra)
                    error()
                end

            end
        end
    end
    return Hd
end

"""
    compute_diagonal(vector::TPSCIstate{T,N,R}, cluster_ops, opstring::String) where {T,N,R}

Fast version, used for PT2
"""
function compute_diagonal(vector::TPSCIstate{T,N,R}, cluster_ops, opstring::String) where {T,N,R}
    Hd = zeros(T, size(vector)[1])
    compute_diagonal!(Hd, vector, cluster_ops, opstring)
    return Hd
end


"""
    compute_diagonal!(Hd, vector::TPSCIstate{T,N,R}, cluster_ops, opstring::String) where {T,N,R}


Fast version, used for PT2, overwrites Hd data with diagonal.
"""
function compute_diagonal!(Hd, vector::TPSCIstate{T,N,R}, cluster_ops, opstring::String) where {T,N,R}
    fill!(Hd,0.0)
    idx = 1
    for (fock, configs) in vector.data
        for c in vector.clusters
            mat = []
            try
                mat = diag(cluster_ops[c.idx][opstring][(fock[c.idx],fock[c.idx])])
            catch
                println(c, fock[c.idx])
                error()
            end
            
            idxc = idx + 0
            for (config, _) in configs
                Hd[idxc] += mat[config[c.idx]]
                idxc += 1
            end
        end
        idx += length(configs)
    end
    return Hd
end


"""
    compute_diagonal!(Hd, vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham) where {T,N,R}

Form the diagonal of the hamiltonan, `clustered_ham`, in the basis defined by `vector`
"""
function compute_diagonal!(Hd, vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator) where {T,N,R}
    #={{{=#
    idx = 0
    zero_trans = TransferConfig([(0,0) for i in 1:N])
    for (fock_bra, configs_bra) in vector.data
        for (config_bra, coeff_bra) in configs_bra
            idx += 1
            for term in clustered_ham[zero_trans]
		    try
			    Hd[idx] += contract_matrix_element(term, cluster_ops, fock_bra, config_bra, fock_bra, config_bra)
		    catch
			    display(term)
			    display(fock_bra)
			    display(config_bra)
			    error()
		    end

            end
        end
    end
    return
end
#=}}}=#


"""
    expand_each_fock_space!(s::TPSCIstate{T,N,R}, bases::Vector{ClusterBasis}) where {T,N,R}

For each fock space sector defined, add all possible basis states
- `basis::Vector{ClusterBasis}` 
"""
function expand_each_fock_space!(s::TPSCIstate{T,N,R}, bases::Vector{ClusterBasis{A,T}}) where {T,N,R,A}
    # {{{
    println("\n Make each Fock-Block the full space")
    # create full space for each fock block defined
    for (fblock,configs) in s.data
        #println(fblock)
        dims::Vector{UnitRange{Int16}} = []
        #display(fblock)
        for c in s.clusters
            # get number of vectors for current fock space
            dim = size(bases[c.idx][fblock[c.idx]], 2)
            push!(dims, 1:dim)
        end
        for newconfig in Iterators.product(dims...)
            #display(newconfig)
            #println(typeof(newconfig))
            #
            # this is not ideal - need to find a way to directly create key
            config = ClusterConfig(collect(newconfig))
            s.data[fblock][config] = zeros(SVector{R,T}) 
            #s.data[fblock][[i for i in newconfig]] = 0
        end
    end
end
# }}}

"""
    expand_to_full_space!(s::AbstractState, bases::Vector{ClusterBasis}, na, nb)

Define all possible fock space sectors and add all possible basis states
- `basis::Vector{ClusterBasis}` 
- `na`: Number of alpha electrons total
- `nb`: Number of alpha electrons total
"""
function expand_to_full_space!(s::AbstractState, bases::Vector{ClusterBasis{A,T}}, na, nb) where {A,T}
    # {{{
    println("\n Expand to full space")
    ns = []

    for c in s.clusters
        nsi = []
        for (fspace,basis) in bases[c.idx]
            push!(nsi,fspace)
        end
        push!(ns,nsi)
    end
    for newfock in Iterators.product(ns...)
        nacurr = 0
        nbcurr = 0
        for c in newfock
            nacurr += c[1]
            nbcurr += c[2]
        end
        if (nacurr == na) && (nbcurr == nb)
            config = FockConfig(collect(newfock))
            add_fockconfig!(s,config) 
        end
    end
    expand_each_fock_space!(s,bases)

    return
end
# }}}




"""
    project_out!(v::TPSCIstate, w::TPSCIstate)

Project w out of v 
    |v'> = |v> - |w><w|v>
"""
function project_out!(v::TPSCIstate, w::TPSCIstate)
    for (fock,configs) in w.data 
        if haskey(v.data, fock)
            for (config, coeff) in configs
                if haskey(v.data[fock], config)
                    delete!(v.data[fock], config)
                end
            end
            if length(v[fock]) == 0
                delete!(v.data, fock)
            end
        end
    end
    # I'm not sure why this is necessary
    idx = 0
    for (fock,configs) in v.data
        for (config, coeffs) in v.data[fock]
            idx += 1
        end
    end
end



"""
    hosvd(ci_vector::TPSCIstate{T,N,R}, cluster_ops; hshift=1e-8, truncate=-1) where {T,N,R}

Peform HOSVD aka Tucker Decomposition of TPSCIstate
"""
function hosvd(ci_vector::TPSCIstate{T,N,R}, cluster_ops; hshift=1e-8, truncate=-1) where {T,N,R}
#={{{=#
   
    cluster_rotations = []
    for ci in ci_vector.clusters
        println()
        println(" --------------------------------------------------------")
        println(" Density matrix: Cluster ", ci.idx)
        println()
        println(" Compute BRDM")
        println(" Hshift = ",hshift)
        
        dims = Dict()
        for (fock, mat) in cluster_ops[ci.idx]["H"]
            fock[1] == fock[2] || error("?")
            dims[fock[1]] = size(mat,1)
        end
        
        rdms = build_brdm(ci_vector, ci, dims)
        norm = 0
        entropy = 0
        rotations = Dict{Tuple,Matrix{T}}() 
        for (fspace,rdm) in rdms
            fspace_norm = 0
            fspace_entropy = 0
            @printf(" Diagonalize RDM for Cluster %2i in Fock space: ",ci.idx)
            println(fspace)
            F = eigen(Symmetric(rdm))

            idx = sortperm(F.values, rev=true) 
            n = F.values[idx]
            U = F.vectors[:,idx]


            # Either truncate the unoccupied cluster states, or remix them with a hamiltonian to be unique
            if truncate < 0
                remix = []
                for ni in 1:length(n)
                    if n[ni] < 1e-8
                        push!(remix, ni)
                    end
                end
                U2 = U[:,remix]
                Hlocal = U2' * cluster_ops[ci.idx]["H"][(fspace,fspace)] * U2
                
                F = eigen(Symmetric(Hlocal))
                n2 = F.values
                U2 = U2 * F.vectors
                
                U[:,remix] .= U2[:,:]
            
            else
                keep = []
                for ni in 1:length(n) 
                    if abs(n[ni]) > truncate
                        push!(keep, ni)
                    end
                end
                @printf(" Truncated Tucker space. Starting: %5i Ending: %5i\n" ,length(n), length(keep))
                U = U[:,keep]
            end
        

           
            
            n = diag(U' * rdm * U)
            Elocal = diag(U' * cluster_ops[ci.idx]["H"][(fspace,fspace)] * U)
            
            norm += sum(n)
            fspace_norm = sum(n)
            @printf("                 %4s:    %12s    %12s\n", "","Population","Energy")
            for (ni_idx,ni) in enumerate(n)
                if abs(ni/norm) > 1e-16
                    fspace_entropy -= ni*log(ni/norm)/norm
                    entropy -=  ni*log(ni)
                    @printf("   Rotated State %4i:    %12.8f    %12.8f\n", ni_idx,ni,Elocal[ni_idx])
                end
           end
           @printf("   ----\n")
           @printf("   Entanglement entropy:  %12.8f\n" ,fspace_entropy) 
           @printf("   Norm:                  %12.8f\n" ,fspace_norm) 

           #
           # let's just be careful that our vectors remain orthogonal
           F = svd(U)
           U = F.U * F.Vt
           check_orthogonality(U) 
           rotations[fspace] = U
        end
        @printf(" Final entropy:.... %12.8f\n",entropy)
        @printf(" Final norm:....... %12.8f\n",norm)
        @printf(" --------------------------------------------------------\n")

        flush(stdout) 

        #ci.rotate_basis(rotations)
        #ci.check_basis_orthogonality()
        push!(cluster_rotations, rotations)
    end
    return cluster_rotations
end
#=}}}=#




"""
    build_brdm(ci_vector::TPSCIstate, ci, dims)
    
Build block reduced density matrix for `Cluster`,  `ci`
- `ci_vector::TPSCIstate` = input state
- `ci` = Cluster type for whihch we want the BRDM
- `dims` = list of dimensions for each fock sector
"""
function build_brdm(ci_vector::TPSCIstate, ci, dims)
    # {{{
    rdms = OrderedDict()
    for (fspace, configs) in ci_vector.data
        curr_dim = dims[fspace[ci.idx]]
        rdm = zeros(curr_dim,curr_dim)
        for (configi,coeffi) in configs
            for cj in 1:curr_dim

                configj = [configi...]
                configj[ci.idx] = cj
                configj = ClusterConfig(configj)

                if haskey(configs, configj)
                    rdm[configi[ci.idx],cj] += sum(coeffi.*configs[configj])
                end
            end
        end


        if haskey(rdms, fspace[ci.idx]) 
            rdms[fspace[ci.idx]] += rdm 
        else
            rdms[fspace[ci.idx]] = rdm 
        end

    end
    return rdms
end
# }}}



function dump_tpsci(filename::AbstractString, ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator) where {T,N,R}
    @save filename ci_vector cluster_ops clustered_ham
end

#function load_tpsci(filename::AbstractString) 
#    a = @load filename
#    return eval.(a)
#end


"""
    do_fois_ci(ref::TPSCIstate, cluster_ops, clustered_ham; H0="Hcmf", thresh_foi=1e-6, nbody=4, tol=1e-5, kwargs...)

Perform a CI in the first-order interacting space (FOIS) defined over a CMF
reference. Solves the zeroth-order problem in `ref`, generates the FOIS from up
to `nbody`-body terms acting on the reference (keeping coefficients above
`thresh_foi`), and variationally diagonalizes the Hamiltonian in the combined
reference + FOIS space. Returns the variational energies and the expanded
`TPSCIstate`. `H0` selects the zeroth-order Hamiltonian (default the CMF
Hamiltonian `"Hcmf"`).
"""
function do_fois_ci(ref::TPSCIstate{T,N,R}, cluster_ops, clustered_ham;
                    H0          = "Hcmf",
                    max_iter    = 50,
                    nbody       = 4,
                    thresh_foi  = 1e-6,
                    tol         = 1e-5,
                    thresh_clip = 1e-6,
                    threaded    =false,
                    prescreen   = false,
                    compress    = false,
                    pt          =false,
                    verbose     = true) where {T,N,R}
    @printf("\n-------------------------------------------------------\n")
    @printf(" Do CI in FOIS\n")
    @printf("   H0                      = %-s\n", H0)
    @printf("   thresh_foi              = %-8.1e\n", thresh_foi)
    @printf("   nbody                   = %-i\n", nbody)
    @printf("\n")
    @printf("   Length of Reference     = %-i\n", length(ref))
    @printf("\n-------------------------------------------------------\n")

# 
    # Solve variationally in reference space
    ref_vec = deepcopy(ref)
    @printf(" Solve zeroth-order problem. Dimension = %10i\n", length(ref_vec))
    @time e0, ref_vec = tps_ci_direct(ref_vec, cluster_ops, clustered_ham, conv_thresh=tol)
    

    #
    # Get First order wavefunction
    println()
    println(" Compute FOIS. Reference space dim = ", length(ref_vec))
    # pt1_vec= deepcopy(ref_vec)
    # pt1_vec=matvec(pt1_vec)
    if threaded == true
        pt1_vec = open_matvec_thread(ref_vec, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi, prescreen=prescreen)
    else
        pt1_vec = open_matvec_serial(ref_vec, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi, prescreen=prescreen)
    end
    for i in 1:R
        @printf("Arnab: %12.8f\n", sqrt.(orth_dot(pt1_vec,pt1_vec))[i])
    end
    project_out!(pt1_vec, ref)
    # Compress FOIS
    if compress==true
        norm1 = sqrt.(orth_dot(pt1_vec, pt1_vec))
        dim1 = length(pt1_vec)
        clip!(pt1_vec, thresh=thresh_clip) #does clip! function do the compression? or have to write a compress function.
        norm2 = sqrt.(orth_dot(pt1_vec, pt1_vec))
        dim2 = length(pt1_vec)
        @printf(" %-50s%10i → %-10i (thresh = %8.1e)\n", "FOIS Compressed from: ", dim1, dim2, thresh_foi)
        for i in 1:R
            @printf(" %-50s%10.2e → %-10.2e (thresh = %8.1e)\n", "Norm of |1>: ",norm1[i], norm2[i], thresh_foi)
        end
    end
    for i in 1:R
        @printf(" %-50s%10.6f\n", "Overlap between <1|0>: ", overlap(pt1_vec, ref_vec)[i])
    end

    add!(ref_vec, pt1_vec)
    # Solve for first order wavefunction 
    println(" Compute CI energy in the space = ", length(ref_vec))
   
    eci, ref_vec = tps_ci_direct(ref_vec, cluster_ops, clustered_ham;)
    for i in 1:R
        @printf(" E(Ref)   for %ith state   = %12.8f\n",i, e0[i])
        @printf(" E(CI) tot for %ith state = %12.8f\n",i, eci[i])
    end
    if pt==true
        e_pt2,pt1_vec= compute_pt1_wavefunction(ref_vec, cluster_ops, clustered_ham;  H0=H0,verbose=verbose)  
        for i in 1:R
            @printf(" E(PT2)  for %ith state   = %12.8f\n",i, e_pt2[i])
        end
    end
    return eci, ref_vec 
    # println("debugging")
    # error()
end

"""
do_fois_cepa(ref::TPSCIstate{T,N,R}, cluster_ops, clustered_ham;
                    max_iter=20,
                    cepa_shift="cepa",
                    cepa_mit=30,
                    nbody=4,
                    thresh_foi=1e-6,
                    thresh_clip=1e-5,
                    tol=1e-8,
                    compress=false,
                    compress_type="matvec",
                    verbose=1) where {T,N,R}

Do CEPA in FOIS defined by ref and thresh_foi
    -`ref`: reference state
    -`cluster_ops`: cMF cluster operators
    -`clustered_ham`: cMF clustered hamiltonian
    -`cepa_shift`: type of CEPA calculation
    -`cepa_mit`: maximum number of CEPA iterations
    -`nbody`: number of cluster terms to include
    -`thresh_foi`: threshold for first order interaction space
    -`thresh_clip`: threshold for clipping
    -`tol`: tolerance for convergence
    -`compress`: compress the first order interaction space
    -`compress_type`: type of compression
    -`solver`: `:krylov` (default, matrix-free CG via open_matvec_thread),
               `:minres` (builds H_qq once according to `build_hqq`, then uses
               MINRES; drastically reduces peak memory for large FOIS), or
               `:pcg` (same H_qq as `:minres`, solved with Jacobi-preconditioned
               CG and warm starts; falls back to MINRES for any solve whose
               shifted operator is not positive definite)
    -`build_hqq`: how H is applied on Q — `:sparse`, `:direct`, `:parallel`,
               `:matvec` (nothing stored, entries recomputed per apply), `:fois`
               (applied through `open_matvec_thread`), or `:auto` (default:
               `:fois` for `:krylov`, `:direct` otherwise)
    -`block_roots`: solve all R roots in one pass, sharing one H apply per
               iteration, rather than one solve per root (default `true`). Not
               available for `solver=:krylov`, which is single-RHS. See
               `tpsci_cepa_solve` for the full pathway table.
    -`verbose`: verbosity level

"""



"""
    do_fois_cepa(ref::TPSCIstate, cluster_ops, clustered_ham; cepa_shift="cepa", thresh_foi=1e-6, nbody=4, solver=:krylov, kwargs...)

Apply a CEPA-style correction over the first-order interacting space (FOIS)
defined over a CMF reference. Solves the zeroth-order problem in `ref`, builds
the FOIS from up to `nbody`-body terms (threshold `thresh_foi`), and solves the
dressed CEPA equations. `cepa_shift` selects the correction flavour (default
`"cepa"`) and `solver` the linear solver — `:krylov` (default), `:minres`, or
`:pcg`. Returns the CEPA-corrected energies and the associated wavefunction.
"""
function do_fois_cepa(ref::TPSCIstate{T,N,R}, cluster_ops, clustered_ham;
                        cepa_shift="cepa",
                        cepa_mit=30,
                        nbody=4,
                        thresh_foi=1e-6,
                        thresh_clip=1e-5,
                        tol=1e-8,
                        thresh_sigma=1e-8,
                        cg_maxiter=300,
                        compress=false,
                        compress_type="matvec",
                        solver=:krylov,
                        build_hqq=:auto,
                        block_roots=true,
                        verbose=1) where {T,N,R}
    @printf("\n-------------------------------------------------------\n")
    @printf(" Do CEPA\n")
    @printf("   thresh_foi              = %-8.1e\n", thresh_foi)
    @printf("   nbody                   = %-i\n", nbody)
    @printf("\n")
    @printf("   Length of Reference     = %-i\n", length(ref))
    @printf("   Calculation type        = %s\n", cepa_shift)
    @printf("   Compression type        = %s\n", compress_type)
    @printf("\n-------------------------------------------------------\n")

    # 
    # Solve variationally in reference space
    println()
    ref_vec = deepcopy(ref)
    @printf(" Solve zeroth-order problem. Dimension = %10i\n", length(ref_vec))
    @time e0, ref_vec = tps_ci_direct(ref_vec, cluster_ops, clustered_ham, conv_thresh=tol)

    #
     # Get First order wavefunction
     println()
     println(" Compute FOIS. Reference space dim = ", length(ref_vec))
     pt1_vec = deepcopy(ref_vec)
     pt1_vec=open_matvec_thread(pt1_vec, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi)
    project_out!(pt1_vec, ref)
    # display(pt1_vec)

    # Compress FOIS
    if compress==true
        norm1 = sqrt.(orth_dot(pt1_vec, pt1_vec))
        dim1 = length(pt1_vec)
        clip!(pt1_vec, thresh=thresh_clip)
        norm2 = sqrt.(orth_dot(pt1_vec, pt1_vec))
        dim2 = length(pt1_vec)
        @printf(" %-50s%10i → %-10i (thresh = %8.1e)\n", "FOIS Compressed from: ", dim1, dim2, thresh_foi)
        for i in 1:R
            @printf(" %-50s%10.2e → %-10.2e (thresh = %8.1e)\n", "Norm of |1>: ",norm1[i], norm2[i], thresh_foi)
        end
    end
    for i in 1:R
        @printf(" %-50s%10.6f\n", "Overlap between <1|0>: ", overlap(pt1_vec, ref_vec)[i])
    end
    # 
    
    # Solve CEPA with shared FOIS for all R roots simultaneously
    println()
    println(" Do CEPA: shared FOIS dim = ", length(pt1_vec))
    @time Ec, e_cepa = tpsci_cepa_solve(ref_vec, e0, pt1_vec, cluster_ops, clustered_ham,
                                         cepa_shift, cepa_mit, tol=tol, cg_maxiter=cg_maxiter,
                                         thresh_sigma=thresh_sigma, solver=solver,
                                         build_hqq=build_hqq, block_roots=block_roots, verbose=verbose)

    for i in 1:R
        @printf(" E(cepa) root %i  corr= %12.8f  total= %12.8f\n", i, Ec[i], e_cepa[i])
    end

    return e_cepa, pt1_vec
end


"""
    cepa_pcg_linsolve(Hq_mv, b, Hdiag; eshift, tol=1e-8, maxiter=300, verbose=0, x0=nothing)

Jacobi-preconditioned conjugate gradient for `(H_qq - eshift*I) x = b`.

`Hq_mv` applies H_qq to a dense vector and `Hdiag` is the Q-space Hamiltonian
diagonal, which is root independent — the caller builds it once and reuses it for
every root and macro-iteration. The preconditioner is `M = diag(H_qq) - eshift`,
with near-zero denominators floored exactly as `precondition_sharded` does.

CG only converges for a positive-definite shifted operator. The caller is expected
to have screened that with `minimum(Hdiag)`, but the diagonal test is necessary and
not sufficient: if the iteration meets a direction with `p'Ap <= 0` it stops and
reports `indefinite=true` so the caller can redo that solve with MINRES.

Returns `(x, (iters=, residual=, converged=, indefinite=))`.
"""
function cepa_pcg_linsolve(Hq_mv, b::Vector{T}, Hdiag::Vector{T};
                           eshift,
                           tol=1e-8,
                           maxiter=300,
                           verbose=0,
                           x0=nothing) where {T}
#={{{=#
    n = length(b)
    length(Hdiag) == n ||
        throw(DimensionMismatch("Hdiag has length $(length(Hdiag)), expected $n"))
    shift = T(eshift)

    # M = diag(H_qq) - eshift, floored the same way precondition_sharded floors it
    # so a Q-space state sitting on top of the shift cannot blow the preconditioner up.
    M = similar(Hdiag)
    @inbounds for i in 1:n
        d0 = Hdiag[i] - shift
        fl = sqrt(eps(T)) * max(abs(shift), abs(Hdiag[i]), one(T))
        M[i] = abs(d0) < fl ? copysign(fl, iszero(d0) ? one(T) : d0) : d0
    end

    x  = x0 === nothing ? zeros(T, n) : Vector{T}(copy(x0))
    r  = copy(b)
    Ap = Vector{T}(undef, n)

    # Measure convergence against ||b||, not against the starting residual: a warm
    # start makes the latter small, which would silently tighten `tol` on every
    # macro-iteration and cancel the benefit of warm starting.
    threshold = tol * max(norm(b), one(T))

    # Warm start: solve for the correction, r = b - A*x0.
    if x0 !== nothing
        Ap .= Hq_mv(x) .- shift .* x
        r .-= Ap
    end

    z = r ./ M
    p = copy(z)
    resid = norm(r)
    rz = dot(r, z)
    converged = resid <= threshold
    indefinite = false
    pAp0 = zero(T)
    iters = 0

    for it in 1:maxiter
        converged && break
        iters = it
        Ap .= Hq_mv(p) .- shift .* p
        pAp = dot(p, Ap)

        # A genuinely indefinite operator gives pAp <= 0 and CG cannot proceed. Report
        # it rather than throwing: the caller redoes the solve with MINRES, which is
        # built for the symmetric-indefinite case.
        if pAp <= zero(T)
            verbose > 0 &&
                @printf("   CEPA PCG met an indefinite direction (pAp = %.3e)\n", pAp)
            indefinite = true
            break
        end
        it == 1 && (pAp0 = pAp)
        # Guard only against true underflow of p'Ap. It shrinks like ||r||^2, so a
        # cutoff of eps*pAp0 would stop the iteration once the residual reached
        # sqrt(eps) -- capping CG at ~1e-8 relative no matter what `tol` asked for.
        # alpha = rz/pAp stays well scaled as both shrink together, so let the
        # residual test decide when to stop and only catch the actual floor here.
        if pAp <= eps(T)^2 * pAp0
            verbose > 1 && @printf("   CEPA PCG stalled at fp precision (iter %i)\n", it)
            break
        end

        alpha = rz / pAp
        x .+= alpha .* p
        r .-= alpha .* Ap
        resid = norm(r)
        verbose > 2 && @printf("   CEPA PCG iter %4i residual %12.4e\n", it, resid)
        if resid <= threshold
            converged = true
            break
        end

        z .= r ./ M
        rz_new = dot(r, z)
        beta = rz_new / rz
        # p .= z + beta*p
        @inbounds for i in 1:n
            p[i] = z[i] + beta * p[i]
        end
        rz = rz_new
    end

    return x, (iters=iters, residual=resid, converged=converged, indefinite=indefinite)
end
#=}}}=#


# ─────────────────────────────────────────────────────────────────────────────
# Block (multi-root) CEPA linear solvers.
#
# All R roots share one H_qq apply per iteration, but each root keeps its own
# shift, its own Krylov scalars and its own convergence test — the roots are *not*
# coupled, this is R independent recurrences riding on one batched matvec. True
# block CG/MINRES (coupled search directions) is not available to us anyway: it
# needs one shared matrix, and each root carries a different `eshift`.
#
# Blocks are root-major, `R × dim_q`; see the block-operator notes above.
# ─────────────────────────────────────────────────────────────────────────────

# Y[c,j] += a[c]*X[c,j]
@inline function _block_axpy!(Y::Matrix{T}, a::Vector{T}, X::Matrix{T}) where {T}
    R, n = size(Y)
    @inbounds for j in 1:n
        @simd for c in 1:R
            Y[c, j] += a[c] * X[c, j]
        end
    end
    return Y
end

# out[c] = dot(A[c,:], B[c,:])
@inline function _block_dot!(out::Vector{T}, A::Matrix{T}, B::Matrix{T}) where {T}
    R, n = size(A)
    fill!(out, zero(T))
    @inbounds for j in 1:n
        @simd for c in 1:R
            out[c] += A[c, j] * B[c, j]
        end
    end
    return out
end

# out[c] = norm(A[c,:])
@inline function _block_nrm2!(out::Vector{T}, A::Matrix{T}) where {T}
    _block_dot!(out, A, A)
    @inbounds for c in eachindex(out)
        out[c] = sqrt(abs(out[c]))
    end
    return out
end

# A[c,:] .*= a[c]
@inline function _block_scale!(A::Matrix{T}, a::Vector{T}) where {T}
    R, n = size(A)
    @inbounds for j in 1:n
        @simd for c in 1:R
            A[c, j] *= a[c]
        end
    end
    return A
end

@inline function _block_zero_row!(A::Matrix{T}, c::Int) where {T}
    @inbounds for j in 1:size(A, 2)
        A[c, j] = zero(T)
    end
    return A
end

"""
    _apply_shifted!(Y, op, X, eshifts)

`Y[c,:] = (H_qq - eshifts[c]*I) * X[c,:]` for every root, with one shared H_qq
apply. The shift is diagonal, so per-root shifts cost nothing to batch — which is
why differing `eshift` values are no obstacle to solving all the roots together.
"""
function _apply_shifted!(Y::Matrix{T}, op, X::Matrix{T}, eshifts::Vector{T}) where {T}
    mul_block!(Y, op, X)
    R, n = size(Y)
    @inbounds for j in 1:n
        @simd for c in 1:R
            Y[c, j] -= eshifts[c] * X[c, j]
        end
    end
    return Y
end


"""
    cepa_pcg_block(op, B, Hdiag, eshifts; tol, maxiter, verbose, X0) -> (X, info)

Jacobi-preconditioned CG for `(H_qq - eshifts[c]*I) x_c = b_c`, all roots at once.

`B` and the returned `X` are root-major (`R × dim_q`); `Hdiag` is the Q-space
diagonal, shared by every root; `eshifts[c]` is root `c`'s shift. The
preconditioner is `diag(H_qq) - eshifts[c]`, floored the way `precondition_sharded`
floors it.

A root that meets a `p'Ap <= 0` direction is frozen and flagged in
`info.indefinite`; the caller redoes those roots with MINRES. Roots that converge
early are frozen too, so they stop contributing work — but they stay in the block,
since the matvec cost is shared and compacting it would buy nothing.

`info` carries `iters`, `residual`, `converged` and `indefinite`, one entry per root.
"""
function cepa_pcg_block(op, B::Matrix{T}, Hdiag::Vector{T}, eshifts::Vector{T};
                        tol=1e-8, maxiter::Int=300, verbose=0, X0=nothing) where {T}
#={{{=#
    R, n = size(B)
    length(Hdiag) == n || throw(DimensionMismatch("Hdiag has length $(length(Hdiag)), expected $n"))
    length(eshifts) == R || throw(DimensionMismatch("eshifts has length $(length(eshifts)), expected $R"))

    # M[c,j] = diag(H_qq)[j] - eshifts[c], floored so a Q-space state sitting on top
    # of a shift cannot blow the preconditioner up.
    M = Matrix{T}(undef, R, n)
    @inbounds for j in 1:n
        d = Hdiag[j]
        for c in 1:R
            d0 = d - eshifts[c]
            fl = sqrt(eps(T)) * max(abs(eshifts[c]), abs(d), one(T))
            M[c, j] = abs(d0) < fl ? copysign(fl, iszero(d0) ? one(T) : d0) : d0
        end
    end

    X  = X0 === nothing ? zeros(T, R, n) : Matrix{T}(copy(X0))
    Rr = copy(B)
    AP = Matrix{T}(undef, R, n)
    Z  = Matrix{T}(undef, R, n)

    # Measure against ||b||, not the starting residual: a warm start makes the latter
    # small, which would silently tighten `tol` on every macro-iteration.
    bnorm = zeros(T, R)
    _block_nrm2!(bnorm, B)
    threshold = [tol * max(bnorm[c], one(T)) for c in 1:R]

    if X0 !== nothing
        _apply_shifted!(AP, op, X, eshifts)
        @inbounds for j in 1:n, c in 1:R
            Rr[c, j] -= AP[c, j]
        end
    end

    Z .= Rr ./ M
    P  = copy(Z)
    resid = zeros(T, R); _block_nrm2!(resid, Rr)
    rz    = zeros(T, R); _block_dot!(rz, Rr, Z)
    pAp   = zeros(T, R)
    alpha = zeros(T, R)
    beta  = zeros(T, R)
    pAp0  = zeros(T, R)

    converged  = [resid[c] <= threshold[c] for c in 1:R]
    indefinite = fill(false, R)
    active     = [!converged[c] for c in 1:R]
    iters      = zeros(Int, R)

    for c in 1:R
        active[c] || _block_zero_row!(P, c)
    end

    for it in 1:maxiter
        any(active) || break
        _apply_shifted!(AP, op, P, eshifts)
        _block_dot!(pAp, P, AP)

        for c in 1:R
            active[c] || (alpha[c] = zero(T); continue)
            iters[c] = it
            # A genuinely indefinite operator gives pAp <= 0 and CG cannot proceed.
            if pAp[c] <= zero(T)
                verbose > 0 &&
                    @printf("   CEPA PCG root %i met an indefinite direction (pAp = %.3e)\n", c, pAp[c])
                indefinite[c] = true
                active[c] = false
                alpha[c] = zero(T)
                continue
            end
            it == 1 && (pAp0[c] = pAp[c])
            # Underflow guard only -- see the note in `cepa_pcg_linsolve`: p'Ap
            # shrinks like ||r||^2, so an eps*pAp0 cutoff would cap CG at a relative
            # residual of sqrt(eps) regardless of `tol`.
            if pAp[c] <= eps(T)^2 * pAp0[c]
                verbose > 1 && @printf("   CEPA PCG root %i stalled at fp precision (iter %i)\n", c, it)
                active[c] = false
                alpha[c] = zero(T)
                continue
            end
            alpha[c] = rz[c] / pAp[c]
        end

        _block_axpy!(X, alpha, P)
        @inbounds for j in 1:n
            @simd for c in 1:R
                Rr[c, j] -= alpha[c] * AP[c, j]
            end
        end
        _block_nrm2!(resid, Rr)

        for c in 1:R
            active[c] || continue
            if resid[c] <= threshold[c]
                converged[c] = true
                active[c] = false
            end
        end
        # Freeze the finished roots: a zero search direction keeps them out of every
        # subsequent update while the block keeps its shape.
        for c in 1:R
            active[c] || _block_zero_row!(P, c)
        end
        any(active) || break

        Z .= Rr ./ M
        for c in 1:R
            if active[c]
                rz_new = zero(T)
                @inbounds @simd for j in 1:n
                    rz_new += Rr[c, j] * Z[c, j]
                end
                beta[c] = rz_new / rz[c]
                rz[c] = rz_new
            else
                beta[c] = zero(T)
                _block_zero_row!(Z, c)
            end
        end
        # P .= Z + beta*P
        @inbounds for j in 1:n
            @simd for c in 1:R
                P[c, j] = Z[c, j] + beta[c] * P[c, j]
            end
        end
    end

    return X, (iters=iters, residual=resid, converged=converged, indefinite=indefinite)
end
#=}}}=#


"""
    cepa_minres_block(op, B, eshifts; tol, maxiter, verbose, X0) -> (X, info)

MINRES for `(H_qq - eshifts[c]*I) x_c = b_c`, all roots at once.

A column-wise port of the Paige–Saunders recurrence in `IterativeSolvers.minres`
(Lanczos plus Givens rotations, three-term recurrences for both V and W), with the
per-root scalars carried as length-R vectors so that the R recurrences share a
single batched H_qq apply per iteration. Convergence uses the same
`|r_k| <= reltol*|r_0|` test as the single-root solver.

Unlike CG this needs no definiteness assumption, which is what makes it the right
default for excited roots, where `H_qq - E0` is generally indefinite.

`info` carries `iters`, `residual` and `converged`, one entry per root.
"""
function cepa_minres_block(op, B::Matrix{T}, eshifts::Vector{T};
                           tol=1e-8, maxiter::Int=300, verbose=0, X0=nothing) where {T}
#={{{=#
    R, n = size(B)
    length(eshifts) == R || throw(DimensionMismatch("eshifts has length $(length(eshifts)), expected $R"))

    X      = X0 === nothing ? zeros(T, R, n) : Matrix{T}(copy(X0))
    v_prev = zeros(T, R, n)
    v_curr = copy(B)
    v_next = zeros(T, R, n)
    w_prev = zeros(T, R, n)
    w_curr = zeros(T, R, n)
    w_next = zeros(T, R, n)

    if X0 !== nothing
        # v_next doubles as scratch for A*x0, exactly as IterativeSolvers does.
        _apply_shifted!(v_next, op, X, eshifts)
        @inbounds for j in 1:n, c in 1:R
            v_curr[c, j] -= v_next[c, j]
        end
    end

    resnorm = zeros(T, R); _block_nrm2!(resnorm, v_curr)
    tolerance = [tol * resnorm[c] for c in 1:R]

    H1 = zeros(T, R); H2 = zeros(T, R); H3 = zeros(T, R); H4 = zeros(T, R)
    rhs1 = copy(resnorm); rhs2 = zeros(T, R)
    c_prev = ones(T, R);  s_prev = zeros(T, R)
    c_curr = ones(T, R);  s_curr = zeros(T, R)
    proj   = zeros(T, R)
    invv   = zeros(T, R)

    active = [resnorm[c] > tolerance[c] && resnorm[c] > zero(T) for c in 1:R]
    converged = [!active[c] for c in 1:R]
    iters = zeros(Int, R)

    # Normalize the first Krylov vector; dead rows are zeroed so that the shared
    # matvec can never grow them into an overflow.
    for c in 1:R
        if active[c]
            invv[c] = inv(resnorm[c])
        else
            invv[c] = zero(T)
            _block_zero_row!(v_curr, c)
        end
    end
    _block_scale!(v_curr, invv)

    for it in 1:maxiter
        any(active) || break

        _apply_shifted!(v_next, op, v_curr, eshifts)
        if it > 1
            @inbounds for j in 1:n
                @simd for c in 1:R
                    v_next[c, j] -= H2[c] * v_prev[c, j]
                end
            end
        end

        # Orthogonalize against v_curr
        _block_dot!(proj, v_curr, v_next)
        for c in 1:R
            active[c] || (proj[c] = zero(T))
            H3[c] = proj[c]
        end
        @inbounds for j in 1:n
            @simd for c in 1:R
                v_next[c, j] -= proj[c] * v_curr[c, j]
            end
        end

        _block_nrm2!(H4, v_next)
        for c in 1:R
            if !active[c]
                invv[c] = zero(T)
            elseif iszero(H4[c])
                # Exact Krylov termination: the residual is already zero.
                invv[c] = zero(T)
                converged[c] = true
                active[c] = false
                resnorm[c] = zero(T)
            else
                invv[c] = inv(H4[c])
                iters[c] = it
            end
        end
        _block_scale!(v_next, invv)

        for c in 1:R
            active[c] || continue
            if it > 2
                H1[c] = s_prev[c] * H2[c]
                H2[c] = c_prev[c] * H2[c]
            end
            if it > 1
                tmp   = -s_curr[c] * H2[c] + c_curr[c] * H3[c]
                H2[c] =  c_curr[c] * H2[c] + s_curr[c] * H3[c]
                H3[c] = tmp
            end
            cc, ss, r = LinearAlgebra.givensAlgorithm(H3[c], H4[c])
            H3[c] = r
            rhs2[c] = -ss * rhs1[c]
            rhs1[c] =  cc * rhs1[c]
            # Advance the rotations; the vector swap below is shared by all roots.
            c_prev[c], s_prev[c] = c_curr[c], s_curr[c]
            c_curr[c], s_curr[c] = cc, ss
        end

        # w_next = (v_curr - H2*w_curr - H1*w_prev) / H3
        for c in 1:R
            invv[c] = active[c] ? inv(H3[c]) : zero(T)
        end
        @inbounds for j in 1:n
            @simd for c in 1:R
                w_next[c, j] = (v_curr[c, j]
                                - (it > 1 ? H2[c] * w_curr[c, j] : zero(T))
                                - (it > 2 ? H1[c] * w_prev[c, j] : zero(T))) * invv[c]
            end
        end

        # x += rhs1 * w_next  (rhs1 is zeroed for dead roots so they stay put)
        for c in 1:R
            active[c] || (rhs1[c] = zero(T))
        end
        _block_axpy!(X, rhs1, w_next)

        v_prev, v_curr, v_next = v_curr, v_next, v_prev
        w_prev, w_curr, w_next = w_curr, w_next, w_prev

        for c in 1:R
            active[c] || continue
            rhs1[c] = rhs2[c]
            H2[c]   = H4[c]
            resnorm[c] = abs(rhs2[c])
            if resnorm[c] <= tolerance[c]
                converged[c] = true
                active[c] = false
            end
        end

        # Zero the dead rows so the shared matvec never amplifies stale data.
        for c in 1:R
            active[c] && continue
            _block_zero_row!(v_prev, c); _block_zero_row!(v_curr, c)
            _block_zero_row!(w_prev, c); _block_zero_row!(w_curr, c)
        end
        verbose > 2 && @printf("   CEPA block MINRES iter %4i  max resid %12.4e\n",
                               it, maximum(resnorm))
    end

    return X, (iters=iters, residual=resnorm, converged=converged)
end
#=}}}=#


"""
    tpsci_cepa_solve(ref_vector, e0, cepa_vector, cluster_ops, clustered_ham, cepa_shift, cepa_mit;
                     solver, build_hqq, block_roots, tol, verbose)

Multi-root CEPA solver for TPSCIstate.

# Arguments
- `ref_vector`: pre-solved reference state (R roots)
- `e0`: reference energies for each root (length R, pre-computed by caller)
- `cepa_vector`: shared FOIS — union of all roots' first-order interacting spaces
- `cluster_ops`, `clustered_ham`: operators
- `cepa_shift`: "cepa" (CEPA-0), "acpf", "aqcc", "cisd"
- `cepa_mit`: max CEPA iterations (only relevant for acpf/aqcc)

The amplitude equation for each root I is solved with the shared H_xx:

    (H_xx - (E0[I] + shift[I])) * C_x[I] = -h[I]
    E[I] = E0[I] + C_x[I]' * h[I]

where h[I] = <Q|H|A_I> (coupling vector for root I).

# Pathway
Three independent choices, and the banner printed at the top of the solve reports
which combination was taken.

`solver` — the linear solver:
- `:pcg` — Jacobi-preconditioned CG, preconditioner `diag(H_qq) - eshift`, warm
  started across macro-iterations. Falls back to MINRES whenever the shifted
  operator turns out not to be positive definite.
- `:minres` — no definiteness assumption, which is what the excited roots need.
- `:krylov` — `KrylovKit.linsolve`. Single-RHS, so it cannot be blocked; kept as
  the legacy default.

`build_hqq` — how H is applied on Q:
- `:sparse` — H_qq stored as a sparse matrix, `O(nnz)` to build. Applied through
  [`ThreadedSymSpMV`](@ref), which threads what `SparseArrays` runs on one core.
- `:direct` / `:parallel` — H_qq stored dense, `dim_q^2 * 8` bytes.
- `:matvec` — nothing stored; entries are recomputed every apply
  ([`StructuredHqq`](@ref)). `O(nthreads × R × dim_q)` memory, so this is what is
  left once even the sparse matrix will not fit.
- `:fois` — H applied through `open_matvec_thread` over the full first-order
  interacting space and projected back ([`FoisHqq`](@ref)). Cheapest in memory,
  by far the most expensive per apply.
- `:auto` (default) — `:fois` for `:krylov`, `:direct` otherwise, i.e. what each
  solver did before the other options existed.

`block_roots` — whether the R roots are solved in one pass. The solver is
multiroot either way; this decides whether the roots share the applies or take
turns. Batching does not reduce
flops; it amortises the work every root shares — streaming H_qq from memory,
recomputing its entries, or screening FOIS terms — so the win is large exactly
where that shared work dominates, which is all three of `:sparse`, `:matvec` and
`:fois`. The roots keep independent shifts, Krylov scalars and convergence tests;
only the matvec is shared. Blocked solves use the root-major `R × dim_q` layout
described with the block operators above.

h[I] is always computed matrix-free through `open_matvec_thread`. With
`block_roots`, all R coupling vectors come from one FOIS matvec instead of R —
note that its screening is `maximum(abs, coef_ket)` over the roots, so the blocked
h keeps the union of the roots' surviving terms and is not bitwise identical to
the root-at-a-time result.
"""
function tpsci_cepa_solve(ref_vector::TPSCIstate{T,N,R}, e0::Vector,
                           cepa_vector::TPSCIstate{T,N,R2},
                           cluster_ops, clustered_ham,
                           cepa_shift="cepa",
                           cepa_mit=50;
                           tol=1e-5,
                           cg_maxiter=300,
                           nbody=4,
                           thresh_sigma = 1e-8,
                           solver=:krylov,
                           build_hqq=:auto,
                           block_roots=true,
                           verbose=0) where {T,N,R,R2}

    n_clusters = length(ref_vector.clusters)
    dim_q = length(cepa_vector)

    # ── Pathway selection ────────────────────────────────────────────────────────
    solver in (:krylov, :minres, :pcg) ||
        error("Unknown CEPA solver: $solver. Use :krylov, :minres, or :pcg.")
    build_hqq in (:auto, :fois, :matvec, :sparse, :direct, :parallel) ||
        error("Unknown build_hqq: $build_hqq. Use :auto, :fois, :matvec, :sparse, :direct, or :parallel.")

    # :auto reproduces what each solver used to do on its own.
    storage = build_hqq == :auto ? (solver == :krylov ? :fois : :direct) : build_hqq

    # KrylovKit.linsolve takes one right-hand side, so :krylov cannot be blocked.
    # Everything else can, and there is no case where doing so costs time.
    block = block_roots && R > 1 && solver != :krylov
    if block_roots && R > 1 && solver == :krylov
        @printf(" note: solver=:krylov is single-RHS, so the roots are solved one at a time.\n")
        @printf("       for a blocked matrix-free solve use solver=:pcg or :minres with build_hqq=:fois\n")
    end

    @printf(" CEPA pathway: dim_q=%i  shift=%s  solver=%s  storage=%s  roots=%s\n",
            dim_q, cepa_shift, solver, storage, block ? "block($R)" : "one-at-a-time($R)")
    flush(stdout)

    cepa_work = TPSCIstate(cepa_vector, R=1)   # 1-root Q-space state, defines the ordering
    nrb = block ? R : 1

    # ── Build (or wrap) the Q-space operator ─────────────────────────────────────
    H_qq_stored = nothing   # set only when a matrix is actually materialised
    Hq = if storage == :sparse
        @printf(" Building H_qq (%i × %i) [sparse] — reused by every solve\n", dim_q, dim_q)
        @time H_qq_stored = build_H_qq_sparse(cepa_work, cluster_ops, clustered_ham)
        @printf(" H_qq nnz = %i  (%.3f%% fill, %.2f GiB CSC)\n",
                nnz(H_qq_stored), 100*nnz(H_qq_stored)/dim_q^2,
                (nnz(H_qq_stored)*(sizeof(T)+8) + (dim_q+1)*8)/2^30)
        ThreadedSymSpMV(H_qq_stored)
    elseif storage == :direct
        # Peak memory: 1×dim_q²×8 B — threads write directly into H rows (no scratch)
        @printf(" Building H_qq (%i × %i) [dense, %.2f GiB] — reused by every solve\n",
                dim_q, dim_q, dim_q^2*sizeof(T)/2^30)
        @time H_qq_stored = build_H_qq(cepa_work, cluster_ops, clustered_ham)
        ThreadedSymDenseMV(H_qq_stored)
    elseif storage == :parallel
        # build_full_H_parallel peaks at 2×dim_q²×8 B because of its scratch copies
        @printf(" Building H_qq (%i × %i) [dense/parallel] — reused by every solve\n", dim_q, dim_q)
        @time H_qq_stored = build_full_H_parallel(cepa_work, cepa_work, cluster_ops, clustered_ham, sym=true)
        ThreadedSymDenseMV(H_qq_stored)
    elseif storage == :matvec
        @printf(" H_qq recomputed on every apply [matvec] — nothing stored, %.2f GiB of thread scratch\n",
                Threads.maxthreadid()*nrb*dim_q*sizeof(T)/2^30)
        StructuredHqq(cepa_work, cluster_ops, clustered_ham; nroots=nrb)
    else   # :fois
        @printf(" H applied through the FOIS matvec [fois] — nothing stored\n")
        FoisHqq(cepa_vector, cluster_ops, clustered_ham; nbody=nbody, nroots=nrb)
    end
    flush(stdout)

    # ── Coupling vectors h[c,:] = <Q|H|A_c>, root-major ──────────────────────────
    h = zeros(T, R, dim_q)
    if block
        @printf(" Compute coupling vectors for all %i roots (one FOIS matvec)\n", R)
        sig = open_matvec_thread(ref_vector, cluster_ops, clustered_ham,
                                 nbody=nbody, thresh=thresh_sigma)
        project_to_Q!(h, sig, cepa_vector)
    else
        hi = zeros(T, 1, dim_q)
        for i in 1:R
            @printf(" Compute coupling vector h for root %i\n", i)
            ref_i = extract_chosen_root(ref_vector, i)
            sig_i = open_matvec_thread(ref_i, cluster_ops, clustered_ham,
                                       nbody=nbody, thresh=thresh_sigma)
            project_to_Q!(hi, sig_i, cepa_vector)
            @views h[i, :] .= hi[1, :]
        end
    end

    # ── Jacobi preconditioner data for :pcg ──────────────────────────────────────
    # The Q-space diagonal is root independent: build it once, reuse it for every
    # root and macro-iteration. Only the shift changes.
    Hdiag     = T[]
    hdiag_min = zero(T)
    if solver == :pcg
        if H_qq_stored !== nothing
            Hdiag = Vector{T}(diag(H_qq_stored))
        else
            @printf(" Computing Q-space diagonal for the PCG preconditioner\n")
            @time Hdiag = Vector{T}(compute_diagonal(cepa_work, cluster_ops, clustered_ham))
        end
        hdiag_min = minimum(Hdiag)
    end

    Ec      = zeros(T, R)
    Ec_prev = fill(T(Inf), R)
    rhs     = -h                       # same every macro-iteration; only the shift moves
    Cd      = zeros(T, R, dim_q)       # amplitudes, root-major
    have_prev = false

    for it in 1:cepa_mit
        shifts = zeros(T, R)
        for i in 1:R
            if     cepa_shift == "cepa";  shifts[i] = zero(T)
            elseif cepa_shift == "acpf";  shifts[i] = Ec[i] * 2.0 / n_clusters
            elseif cepa_shift == "aqcc"
                shifts[i] = (1.0 - (n_clusters-3.0)*(n_clusters-2.0) /
                              (n_clusters*(n_clusters-1.0))) * Ec[i]
            elseif cepa_shift == "cisd";  shifts[i] = Ec[i]
            else;  error("Unknown cepa_shift: $cepa_shift")
            end
        end
        eshifts = T[e0[i] + shifts[i] for i in 1:R]

        if block
            @printf(" CEPA Iter %3i  Shifts = %s\n", it,
                    join((@sprintf("%12.8f", s) for s in shifts), " "))
            flush(stdout)
            Cd, info, used = _cepa_block_solve(Hq, rhs, Hdiag, eshifts, solver, hdiag_min;
                                               tol=tol, maxiter=cg_maxiter, verbose=verbose,
                                               X0=have_prev ? Cd : nothing)
            if verbose > 0
                for i in 1:R
                    @printf(" Iter %3i  Root %i  [%s]  nops=%4i  res=%8.2e  E_corr = %16.12f%s\n",
                            it, i, used, info.iters[i], info.residual[i],
                            dot(view(Cd, i, :), view(h, i, :)),
                            info.converged[i] ? "" : "   (not converged)")
                end
            end
        else
            for i in 1:R
                @printf(" CEPA Iter %3i  Root %i  Shift = %12.8f\n", it, i, shifts[i])
                flush(stdout)
                Cd_i = _cepa_single_solve(Hq, Vector{T}(rhs[i, :]), Hdiag, eshifts[i],
                                          solver, hdiag_min, dim_q, i;
                                          tol=tol, maxiter=cg_maxiter, verbose=verbose,
                                          x0=have_prev ? Vector{T}(Cd[i, :]) : nothing,
                                          hvec=view(h, i, :), it=it)
                @views Cd[i, :] .= Cd_i
            end
        end
        have_prev = true

        for i in 1:R
            Ec[i] = dot(view(Cd, i, :), view(h, i, :))
        end

        cepa_shift == "cepa" && break
        maximum(abs.(Ec .- Ec_prev)) < tol && break
        Ec_prev .= Ec
    end

    return Ec, e0 .+ Ec
end


"""
    _cepa_block_solve(op, RHS, Hdiag, eshifts, solver, hdiag_min; ...) -> (X, info, used)

Solve all R amplitude equations together, picking CG or MINRES for the whole block.

CG needs a positive-definite shifted operator and MINRES does not, so the block is
handled as a unit: PCG is used only when *every* root clears the diagonal screen,
and if any root then reports an indefinite direction the whole block is redone with
MINRES. Splitting the block by root would need two matvec passes, and since a pass
costs the same whether one root or all six ride on it, that would give back exactly
what blocking bought.
"""
function _cepa_block_solve(op, RHS::Matrix{T}, Hdiag::Vector{T}, eshifts::Vector{T},
                           solver::Symbol, hdiag_min::T;
                           tol, maxiter, verbose, X0=nothing) where {T}
    if solver == :pcg && all(hdiag_min - e > 0 for e in eshifts)
        X, info = cepa_pcg_block(op, RHS, Hdiag, eshifts;
                                 tol=tol, maxiter=maxiter, verbose=verbose, X0=X0)
        any(info.indefinite) || return X, info, :pcg
        verbose > 0 && @printf("   roots %s hit an indefinite direction; redoing the block with MINRES\n",
                               findall(info.indefinite))
    elseif solver == :pcg && verbose > 0
        @printf("   shifted diagonal is not positive for every root (min %.3e); using MINRES\n",
                hdiag_min - maximum(eshifts))
    end
    X, info = cepa_minres_block(op, RHS, eshifts;
                                tol=tol, maxiter=maxiter, verbose=verbose, X0=X0)
    return X, info, :minres
end


"""
    _cepa_single_solve(op, b, Hdiag, eshift, solver, hdiag_min, dim_q, root; ...) -> Cd

Solve one root's amplitude equation. Same solver selection as the blocked path, one
root at a time, and the only route that can use `KrylovKit`.
"""
function _cepa_single_solve(op, b::Vector{T}, Hdiag::Vector{T}, eshift::T,
                            solver::Symbol, hdiag_min::T, dim_q::Int, root::Int;
                            tol, maxiter, verbose, x0=nothing, hvec, it) where {T}
    Cd_i = nothing
    if solver == :pcg && (hdiag_min - eshift) > 0
        Cd_pcg, history = cepa_pcg_linsolve(op, b, Hdiag;
                                            eshift=eshift, tol=tol, maxiter=maxiter,
                                            verbose=verbose, x0=x0)
        if history.indefinite
            verbose > 0 &&
                @printf("   shifted operator is indefinite; redoing root %i with MINRES\n", root)
        else
            Cd_i = Cd_pcg
            if verbose > 0
                @printf(" Iter %3i  Root %i  [pcg]  nops=%4i  res=%8.2e  E_corr = %16.12f%s\n",
                        it, root, history.iters, history.residual, dot(Cd_i, hvec),
                        history.converged ? "" : "   (not converged)")
            end
        end
    elseif solver == :pcg && verbose > 0
        @printf("   shifted diagonal is not positive (min %.3e); using MINRES\n", hdiag_min - eshift)
    end

    if Cd_i === nothing && solver != :krylov
        # MINRES handles symmetric indefinite (H_qq - eI can be indefinite)
        H_eff = LinearMap{T}(v -> op(v) .- eshift .* v, dim_q; issymmetric=true)
        Cd_i, history = IterativeSolvers.minres(H_eff, b; reltol=tol, maxiter=maxiter, log=true)
        if verbose > 0
            @printf(" Iter %3i  Root %i  [minres]  nops=%4i  res=%8.2e  E_corr = %16.12f\n",
                    it, root, history.iters, history.data[:resnorm][end], dot(Cd_i, hvec))
        end
    elseif Cd_i === nothing
        Afunc = v -> op(v) .- eshift .* v
        Cd_i, info = KrylovKit.linsolve(Afunc, b; tol=tol, maxiter=maxiter,
                                        issymmetric=true, isposdef=true, verbosity=0)
        if verbose > 0
            @printf(" Iter %3i  Root %i  [krylov]  nops=%4i  E_corr = %16.12f\n",
                    it, root, info.numops, dot(Cd_i, hvec))
        end
    end
    return Cd_i
end
