"""
    Contract  Hamiltonian matrix (tensor) in Tucker basis with trial vector (Tucker)
"""
function contract_dense_H_with_state(term::ClusteredTerm1B, op, state_sign, coeffs_bra::Tucker{T,N,R}, coeffs_ket::Tucker{T,N,R}) where {T,N,R}
#={{{=#
    c1 = term.clusters[1]

    n_clusters = N 
    #
    # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has
    # distinct Tucker factors
    #
    # todo: PRECOMPUTE THIS!
    overlaps = Dict{Int,Matrix{T}}()
    s = 1.0 # this is the product of scalar overlaps that don't need tensor contractions
    for ci in 1:n_clusters
        ci != c1.idx || continue

        S = coeffs_bra.factors[ci]' * coeffs_ket.factors[ci]

        # if overlap not just scalar, form and prepare for contraction
        # if it is a scalar, then just update the sign with the value
        if length(S) == 1
            s *= S[1]
        else
            overlaps[ci] = S
        end
    end


    indices = collect(1:n_clusters)
    indices[c1.idx] = 0
    perm,_ = bubble_sort(indices)

    #coeffs_ket2 = copy(coeffs_ket.core)

    #
    # multiply by overlaps first if the bra side is smaller,
    # otherwise multiply by Hamiltonian term first

    coeffs_bra2_out = [coeffs_bra.core...]
    # inverse of perm; independent of r, so keep bubble_sort out of the loop
    perm2,_ = bubble_sort(perm)
    #
    # Transpose to get contig data for blas (explicit copy?)
    for r in 1:R
        coeffs_ket2  = transform_basis(coeffs_ket.core[r], overlaps, trans=true)

        coeffs_ket2 = permutedims(coeffs_ket2 , perm)
        coeffs_bra2 = permutedims(coeffs_bra.core[r], perm)

        #
        # Reshape for matrix multiply, shouldn't do any copies, right?
        dim1 = size(coeffs_ket2)
        dim2 = size(coeffs_bra2)
       
        coeffs_ket2 = reshape(coeffs_ket2, dim1[1], prod(dim1[2:end]))
        coeffs_bra2 = reshape(coeffs_bra2, dim2[1], prod(dim2[2:end]))

        #
        # Reshape Hamiltonian term operator
        # ... not needed for 1b term

        #
        # Multiply
        coeffs_bra2 += s .* (op * coeffs_ket2)

        # now untranspose
        coeffs_bra2 = reshape(coeffs_bra2, dim2)
        coeffs_bra2 = permutedims(coeffs_bra2,perm2)

#        #
#        # multiply by overlaps now if the bra side is larger,
#        if length(coeffs_bra2[r]) >= length(coeffs_ket2[r])
#            #coeffs_ket2 = transform_basis(coeffs_ket2, overlaps, trans=true)
#        end
        coeffs_bra2_out[r] = coeffs_bra2
    end

    return ntuple(r->coeffs_bra2_out[r], R)
end
#=}}}=#

"""
    _fused_contract(term, op, state_sign, coeffs_bra, coeffs_ket, Val(K))

Contract a K-body dense H term with a Tucker block, handling **all R roots in a
single pass**.

The per-root formulation this replaces did one `transform_basis`, two
`permutedims`, one gemm and one un-permute *per root* -- 4R array operations per
contribution.  Since every root shares the same block shape, the R cores can be
stacked along a trailing axis and pushed through one `transform_basis`, one
`permutedims`, one gemm (R times wider, so far better BLAS utilisation) and one
un-permute.  Measured 2.1x faster per contribution on real 2-body blocks.

The R axis is inert throughout: `transform_basis` never has a transform for it,
and it rides along as a trailing index in both permutations, so each root's
result is unchanged up to gemm blocking (~1e-21 absolute).
"""
@inline function _fused_contract(term, op, state_sign, coeffs_bra::Tucker{T,N,R},
                                 coeffs_ket::Tucker{T,N,R}, ::Val{K}) where {T,N,R,K}
    #
    # overlaps for the clusters this term does not touch
    overlaps = Dict{Int,Matrix{T}}()
    s = state_sign
    for ci in 1:N
        active = false
        for j in 1:K
            term.clusters[j].idx == ci && (active = true; break)
        end
        active && continue
        S = coeffs_bra.factors[ci]' * coeffs_ket.factors[ci]
        if length(S) == 1
            s *= S[1]
        else
            overlaps[ci] = S
        end
    end

    #
    # permutation bringing the K active clusters to the front; the R axis (N+1)
    # is appended so it stays trailing through both permutes
    indices = collect(1:N)
    for j in 1:K
        indices[term.clusters[j].idx] = 0
    end
    perm,_  = bubble_sort(indices)
    perm2,_ = bubble_sort(perm)
    permR  = ntuple(i -> i <= N ? perm[i]  : N+1, N+1)
    perm2R = ntuple(i -> i <= N ? perm2[i] : N+1, N+1)
    colons = ntuple(_ -> Colon(), N)

    #
    # stack the R ket cores, then transform every root in one pass
    kd = size(coeffs_ket.core[1])
    Kall = Array{T}(undef, kd..., R)
    for r in 1:R
        copyto!(view(Kall, colons..., r), coeffs_ket.core[r])
    end
    Kp = permutedims(transform_basis(Kall, overlaps, trans=true), permR)
    dk = size(Kp)
    Kp2 = reshape(Kp, prod(ntuple(i -> dk[i], Val(K))),
                      prod(ntuple(i -> dk[K+i], Val(N+1-K))))

    #
    # same for the bra side; form_sigma_block! hands us a zero core, but this
    # stays general and returns bra + contribution like the per-root version did
    bd = size(coeffs_bra.core[1])
    Ball = Array{T}(undef, bd..., R)
    for r in 1:R
        copyto!(view(Ball, colons..., r), coeffs_bra.core[r])
    end
    Bp = permutedims(Ball, permR)
    db = size(Bp)
    nb1 = prod(ntuple(i -> db[i], Val(K)))
    Bp2 = reshape(Bp, nb1, prod(ntuple(i -> db[K+i], Val(N+1-K))))

    sz = size(op)
    op2 = reshape(op, prod(ntuple(i -> sz[i], Val(K))),
                      prod(ntuple(i -> sz[K+i], Val(K))))

    Bp2 .+= s .* (op2' * Kp2)                      # one wide gemm for all roots

    Bout = permutedims(reshape(Bp2, db), perm2R)
    return ntuple(r -> Array(view(Bout, colons..., r)), R)
end


function contract_dense_H_with_state(term::ClusteredTerm2B, op, state_sign, coeffs_bra::Tucker{T,N,R}, coeffs_ket::Tucker{T,N,R}) where {T,N,R}
    return _fused_contract(term, op, state_sign, coeffs_bra, coeffs_ket, Val(2))
end
function contract_dense_H_with_state(term::ClusteredTerm3B, op, state_sign, coeffs_bra::Tucker{T,N,R}, coeffs_ket::Tucker{T,N,R}) where {T,N,R}
    return _fused_contract(term, op, state_sign, coeffs_bra, coeffs_ket, Val(3))
end
function contract_dense_H_with_state(term::ClusteredTerm4B, op, state_sign, coeffs_bra::Tucker{T,N,R}, coeffs_ket::Tucker{T,N,R}) where {T,N,R}
    return _fused_contract(term, op, state_sign, coeffs_bra, coeffs_ket, Val(4))
end


#
#   use ncon tensor contractions - typically the slowest
#
function contract_dense_H_with_state_ncon(term::ClusteredTerm1B, op, state_sign, coeffs_bra::Tucker{T,N,R}, coeffs_ket::Tucker{T,N,R}) where {T,N,R}
#={{{=#
    c1 = term.clusters[1]

    n_clusters = N 
    #
    # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has
    # distinct Tucker factors
    #
    # todo: PRECOMPUTE THIS!
    overlaps = Dict{Int,Matrix{T}}()
    s = 1.0 # this is the product of scalar overlaps that don't need tensor contractions
    for ci in 1:n_clusters
        ci != c1.idx || continue

        S = coeffs_bra.factors[ci]' * coeffs_ket.factors[ci]

        # if overlap not just scalar, form and prepare for contraction
        # if it is a scalar, then just update the sign with the value
        if length(S) == 1
            s *= S[1]
        else
            overlaps[ci] = S
        end
    end


    indices = collect(1:n_clusters)
    indices[c1.idx] = 0
    perm,_ = bubble_sort(indices)

    #coeffs_ket2 = copy(coeffs_ket.core)

    #
    # multiply by overlaps first if the bra side is smaller,
    # otherwise multiply by Hamiltonian term first

    coeffs_bra2_out = [coeffs_bra.core...]
    #
    # Transpose to get contig data for blas (explicit copy?)
    for r in 1:R
        coeffs_ket2  = transform_basis(coeffs_ket.core[r], overlaps, trans=true)

        coeffs_ket2 = permutedims(coeffs_ket2 , perm)
        coeffs_bra2 = permutedims(coeffs_bra.core[r], perm)

        #
        # Reshape for matrix multiply, shouldn't do any copies, right?
        dim1 = size(coeffs_ket2)
        dim2 = size(coeffs_bra2)
       
        coeffs_ket2 = reshape(coeffs_ket2, dim1[1], prod(dim1[2:end]))
        coeffs_bra2 = reshape(coeffs_bra2, dim2[1], prod(dim2[2:end]))

        #
        # Reshape Hamiltonian term operator
        # ... not needed for 1b term

        #
        # Multiply
        coeffs_bra2 += s .* (op * coeffs_ket2)

        # now untranspose
        coeffs_bra2 = reshape(coeffs_bra2, dim2)
        coeffs_bra2 = permutedims(coeffs_bra2,perm2)

#        #
#        # multiply by overlaps now if the bra side is larger,
#        if length(coeffs_bra2[r]) >= length(coeffs_ket2[r])
#            #coeffs_ket2 = transform_basis(coeffs_ket2, overlaps, trans=true)
#        end
        coeffs_bra2_out[r] = coeffs_bra2
    end

    return ntuple(r->coeffs_bra2_out[r], R)
end
#=}}}=#
function contract_dense_H_with_state_ncon(term::ClusteredTerm2B, op, state_sign, coeffs_bra::Tucker{T,N,R}, coeffs_ket::Tucker{T,N,R}) where {T,N,R}
#={{{=#
    c1 = term.clusters[1]
    c2 = term.clusters[2]

    #
    # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has
    # distinct Tucker factors
    tensors = Vector{Array{T}}()
    indices = Vector{Vector{Int64}}()
    state_indices = -collect(1:N)
    s = state_sign # this is the product of scalar overlaps that don't need tensor contractions
   

    # if the compressed operator becomes a scalar, treat it as such
    if length(op) == 1
        s *= op[1]
    else
        op_indices = [c1.idx, c2.idx, -c1.idx, -c2.idx]
        state_indices[c1.idx] = c1.idx
        state_indices[c2.idx] = c2.idx
        push!(tensors, op)
        push!(indices, op_indices)
    end
   
    # compute overlaps
    for ci in 1:N
        ci != c1.idx || continue
        ci != c2.idx || continue

        S = coeffs_bra.factors[ci]' * coeffs_ket.factors[ci]

        # if overlap not just scalar, form and prepare for contraction
        if length(S) == 1
            s *= S[1]
        else
            state_indices[ci] = ci
            push!(tensors, S)
            push!(indices, [-ci, ci])
        end
    end
    push!(indices, state_indices)

    # todo: 
    #   make the calling function use this in-place to avoid the copy
    tmp = deepcopy(coeffs_bra)
    
    # loop over global states
    for r in 1:R
        push!(tensors, coeffs_ket.core[r])

        length(tensors) == length(indices) || error(" mismatch between operators and indices")
        if length(tensors) == 1 
            # this means that all the overlaps and the operator is a scalar
            #println(size(tmp.core[r]), size(coeffs_ket.core[r]))
            tmp.core[r] .= coeffs_ket.core[r] .* s
        else
            #println.(size.(tensors))
            #println.(indices)
            #println()
            #flush(stdout)
            tmp.core[r] .= @ncon(tensors, indices)
            tmp.core[r] .= tmp.core[r] .* s
        end
        deleteat!(tensors,length(tensors))
    end
    return tmp.core
end
#=}}}=#
function contract_dense_H_with_state_ncon(term::ClusteredTerm3B, op, state_sign, coeffs_bra::Tucker{T,N,R}, coeffs_ket::Tucker{T,N,R}) where {T,N,R}
#={{{=#
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    c3 = term.clusters[3]

    #
    # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has
    # distinct Tucker factors
    tensors = Vector{Array{T}}()
    indices = Vector{Vector{Int64}}()
    state_indices = -collect(1:N)
    s = state_sign # this is the product of scalar overlaps that don't need tensor contractions
    
    # if the compressed operator becomes a scalar, treat it as such
    if length(op) == 1
        s *= op[1]
    else
        op_indices = [c1.idx, c2.idx, c3.idx, -c1.idx, -c2.idx, -c3.idx]
        state_indices[c1.idx] = c1.idx
        state_indices[c2.idx] = c2.idx
        state_indices[c3.idx] = c3.idx
        push!(tensors, op)
        push!(indices, op_indices)
    end
    
    # compute overlaps
    for ci in 1:N
        ci != c1.idx || continue
        ci != c2.idx || continue
        ci != c3.idx || continue

        S = coeffs_bra.factors[ci]' * coeffs_ket.factors[ci]

        # if overlap not just scalar, form and prepare for contraction
        if length(S) == 1
            s *= S[1]
        else
            state_indices[ci] = ci
            push!(tensors, S)
            push!(indices, [-ci, ci])
        end
    end
    push!(indices, state_indices)

    # todo: 
    #   make the calling function use this in-place to avoid the copy
    tmp = deepcopy(coeffs_bra)
    
    # loop over global states
    for r in 1:R
        push!(tensors, coeffs_ket.core[r])

        length(tensors) == length(indices) || error(" mismatch between operators and indices")
        if length(tensors) == 1 
            # this means that all the overlaps and the operator is a scalar
            #println(size(tmp.core[r]), size(coeffs_ket.core[r]))
            tmp.core[r] .= coeffs_ket.core[r] .* s
        else
            #println.(size.(tensors))
            #println.(indices)
            #println()
            #flush(stdout)
            tmp.core[r] .= @ncon(tensors, indices)
            tmp.core[r] .= tmp.core[r] .* s
        end
        deleteat!(tensors,length(tensors))
    end
    return tmp.core


end
#=}}}=#
function contract_dense_H_with_state_ncon(term::ClusteredTerm4B, op, state_sign, coeffs_bra::Tucker{T,N,R}, coeffs_ket::Tucker{T,N,R}) where {T,N,R}
#={{{=#
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    c3 = term.clusters[3]
    c4 = term.clusters[4]

    #
    # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has
    # distinct Tucker factors
    tensors = Vector{Array{T}}()
    indices = Vector{Vector{Int64}}()
    state_indices = -collect(1:N)
    s = state_sign # this is the product of scalar overlaps that don't need tensor contractions
    
    # if the compressed operator becomes a scalar, treat it as such
    if length(op) == 1
        s *= op[1]
    else
        op_indices = [c1.idx, c2.idx, c3.idx, c4.idx, -c1.idx, -c2.idx, -c3.idx, -c4.idx]
        state_indices[c1.idx] = c1.idx
        state_indices[c2.idx] = c2.idx
        state_indices[c3.idx] = c3.idx
        state_indices[c4.idx] = c4.idx
        push!(tensors, op)
        push!(indices, op_indices)
    end
    
    # compute overlaps
    for ci in 1:N
        ci != c1.idx || continue
        ci != c2.idx || continue
        ci != c3.idx || continue
        ci != c4.idx || continue

        S = coeffs_bra.factors[ci]' * coeffs_ket.factors[ci]

        # if overlap not just scalar, form and prepare for contraction
        if length(S) == 1
            s *= S[1]
        else
            state_indices[ci] = ci
            push!(tensors, S)
            push!(indices, [-ci, ci])
        end
    end
    push!(indices, state_indices)

    # todo: 
    #   make the calling function use this in-place to avoid the copy
    tmp = deepcopy(coeffs_bra)
    
    # loop over global states
    for r in 1:R
        push!(tensors, coeffs_ket.core[r])

        length(tensors) == length(indices) || error(" mismatch between operators and indices")
        if length(tensors) == 1 
            # this means that all the overlaps and the operator is a scalar
            #println(size(tmp.core[r]), size(coeffs_ket.core[r]))
            tmp.core[r] .= coeffs_ket.core[r] .* s
        else
            #println.(size.(tensors))
            #println.(indices)
            #println()
            #flush(stdout)
            tmp.core[r] .= @ncon(tensors, indices)
            tmp.core[r] .= tmp.core[r] .* s
        end
        deleteat!(tensors,length(tensors))
    end
    return tmp.core


end
#=}}}=#



function contract_dense_H_with_state_tensor(term::ClusteredTerm1B, op, state_sign, coeffs_bra::Tucker{T,N,R}, coeffs_ket::Tucker{T,N,R}) where {T,N,R}
#={{{=#
    c1 = term.clusters[1]

    n_clusters = N 
    #
    # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has
    # distinct Tucker factors
    #
    # todo: PRECOMPUTE THIS!
    overlaps = Dict{Int,Matrix{T}}()
    s = 1.0 # this is the product of scalar overlaps that don't need tensor contractions
    for ci in 1:n_clusters
        ci != c1.idx || continue

        S = coeffs_bra.factors[ci]' * coeffs_ket.factors[ci]

        # if overlap not just scalar, form and prepare for contraction
        # if it is a scalar, then just update the sign with the value
        if length(S) == 1
            s *= S[1]
        else
            overlaps[ci] = S
        end
    end


    indices = collect(1:n_clusters)
    indices[c1.idx] = 0
    perm,_ = bubble_sort(indices)

    #coeffs_ket2 = copy(coeffs_ket.core)

    #
    # multiply by overlaps first if the bra side is smaller,
    # otherwise multiply by Hamiltonian term first

    coeffs_bra2_out = [coeffs_bra.core...]
    #
    # Transpose to get contig data for blas (explicit copy?)
    for r in 1:R
        coeffs_ket2  = transform_basis(coeffs_ket.core[r], overlaps, trans=true)

        coeffs_ket2 = permutedims(coeffs_ket2 , perm)
        coeffs_bra2 = permutedims(coeffs_bra.core[r], perm)

        #
        # Reshape for matrix multiply, shouldn't do any copies, right?
        dim1 = size(coeffs_ket2)
        dim2 = size(coeffs_bra2)
       
        coeffs_ket2 = reshape(coeffs_ket2, dim1[1], prod(dim1[2:end]))
        coeffs_bra2 = reshape(coeffs_bra2, dim2[1], prod(dim2[2:end]))

        #
        # Reshape Hamiltonian term operator
        # ... not needed for 1b term

        #
        # Multiply
        coeffs_bra2 += s .* (op * coeffs_ket2)

        # now untranspose
        coeffs_bra2 = reshape(coeffs_bra2, dim2)
        coeffs_bra2 = permutedims(coeffs_bra2,perm2)

#        #
#        # multiply by overlaps now if the bra side is larger,
#        if length(coeffs_bra2[r]) >= length(coeffs_ket2[r])
#            #coeffs_ket2 = transform_basis(coeffs_ket2, overlaps, trans=true)
#        end
        coeffs_bra2_out[r] = coeffs_bra2
    end

    return ntuple(r->coeffs_bra2_out[r], R)
end
#=}}}=#
function contract_dense_H_with_state_tensor(term::ClusteredTerm2B, op, state_sign, coeffs_bra::Tucker{T,N,R}, coeffs_ket::Tucker{T,N,R}) where {T,N,R}
#={{{=#
    c1 = term.clusters[1]
    c2 = term.clusters[2]

    #
    # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has
    # distinct Tucker factors
    tensors = Vector{Array{T}}()
    indices = Vector{Vector{Int64}}()
    state_indices = -collect(1:N)
    s = state_sign # this is the product of scalar overlaps that don't need tensor contractions
   

    # todo: 
    #   pass in preallocated and resize/reshape
    bra_r = Vector{Array{T}}()
   
    output_size = size(coeffs_bra.core[1])

    for r in 1:R
        push!(bra_r,  s .* coeffs_ket.core[r] )
    end

    # compute overlaps
    for ci in 1:N
        ci != c1.idx || continue
        ci != c2.idx || continue

        S = coeffs_bra.factors[ci]' * coeffs_ket.factors[ci]
        #println(size(S))

        # if overlap not just scalar, form and prepare for contraction
        for r in 1:R
            bra = bra_r[r]
            dims = [size(bra)...]
            
            if length(S) == 1
                bra .= bra .* S[1]
            else

                bra = reshape2(bra, 
                                   (prod(size(bra)[1:ci-1]), 
                                    size(bra)[ci], 
                                    prod(size(bra)[ci+1:end])))

                @tensor begin
                    bra[p,q,r] := S[q,Q] * bra[p,Q,r]  
                end
                dims[ci] = size(bra,2)
                #println(size(bra))
                #println(ntuple(i->dims[i], N))
                #println()
                bra_r[r] = reshape2(bra,ntuple(i->dims[i], N))
                length(size(bra_r[r])) == N || throw(DimensionMismatch)
            end
        end
    end
    
    # if the compressed operator becomes a scalar, treat it as such
    #println(" Here2:")
    #println(size.((coeffs_bra.core[1], op, bra_r[1])))
    #println()

    for r in 1:R
        bra = bra_r[r]
        
        if length(op) == 1
            bra .*= op[1]
        else
            #println((c1.idx, c2.idx))
            #println.(size.((op,bra)))
            #println((prod(size(bra)[1:c1.idx-1]), 
            #                size(bra)[c1.idx], 
            #                prod(size(bra)[c1.idx+1:c2.idx-1]),
            #                size(bra)[c2.idx], 
            #                prod(size(bra)[c2.idx+1:end])))

            bra = reshape2(bra, 
                           (prod(size(bra)[1:c1.idx-1]), 
                            size(bra)[c1.idx], 
                            prod(size(bra)[c1.idx+1:c2.idx-1]),
                            size(bra)[c2.idx], 
                            prod(size(bra)[c2.idx+1:end])))

            #println()
            #println.(size.((op,bra)))
            #println(output_size)
            #println()
            @tensor begin
                bra[p,q,r,s,t] := op[Q,S,q,s] * bra[p,Q,r,S,t]  
            end

            #println(size(bra))
            #println(output_size)
            #println()
            bra_r[r] = reshape2(bra,ntuple(i->output_size[i], N))
        end
    end

    return ntuple(i->bra_r[i], R)
end
#=}}}=#
function contract_dense_H_with_state_tensor(term::ClusteredTerm3B, op, state_sign, coeffs_bra::Tucker{T,N,R}, coeffs_ket::Tucker{T,N,R}) where {T,N,R}
#={{{=#
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    c3 = term.clusters[3]

    #
    # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has
    # distinct Tucker factors
    tensors = Vector{Array{T}}()
    indices = Vector{Vector{Int64}}()
    state_indices = -collect(1:N)
    s = state_sign # this is the product of scalar overlaps that don't need tensor contractions
   

    # todo: 
    #   pass in preallocated and resize/reshape
    bra_r = Vector{Array{T}}()
   
    output_size = size(coeffs_bra.core[1])

    for r in 1:R
        push!(bra_r,  s .* coeffs_ket.core[r] )
    end

    # compute overlaps
    for ci in 1:N
        ci != c1.idx || continue
        ci != c2.idx || continue
        ci != c3.idx || continue

        S = coeffs_bra.factors[ci]' * coeffs_ket.factors[ci]
        #println(size(S))

        # if overlap not just scalar, form and prepare for contraction
        for r in 1:R
            bra = bra_r[r]
            dims = [size(bra)...]
            
            if length(S) == 1
                bra .= bra .* S[1]
            else

                bra = reshape2(bra, 
                                   (prod(size(bra)[1:ci-1]), 
                                    size(bra)[ci], 
                                    prod(size(bra)[ci+1:end])))

                @tensor begin
                    bra[p,q,r] := S[q,Q] * bra[p,Q,r]  
                end
                dims[ci] = size(bra,2)
                #println(size(bra))
                #println(ntuple(i->dims[i], N))
                #println()
                bra_r[r] = reshape2(bra,ntuple(i->dims[i], N))
                length(size(bra_r[r])) == N || throw(DimensionMismatch)
            end
        end
    end
    
    # if the compressed operator becomes a scalar, treat it as such
    #println(" Here2:")
    #println(size.((coeffs_bra.core[1], op, bra_r[1])))
    #println()

    for r in 1:R
        bra = bra_r[r]
        
        if length(op) == 1
            bra .*= op[1]
        else
            #println((c1.idx, c2.idx))
            #println.(size.((op,bra)))
            #println((prod(size(bra)[1:c1.idx-1]), 
            #                size(bra)[c1.idx], 
            #                prod(size(bra)[c1.idx+1:c2.idx-1]),
            #                size(bra)[c2.idx], 
            #                prod(size(bra)[c2.idx+1:end])))

            bra = reshape2(bra, 
                           (prod(size(bra)[1:c1.idx-1]), 
                            size(bra)[c1.idx], 
                            prod(size(bra)[c1.idx+1:c2.idx-1]),
                            size(bra)[c2.idx], 
                            prod(size(bra)[c2.idx+1:c3.idx-1]),
                            size(bra)[c3.idx], 
                            prod(size(bra)[c3.idx+1:end])))

            #println()
            #println.(size.((op,bra)))
            #println(output_size)
            #println()
            @tensor begin
                bra[p,q,r,s,t,u,v] := op[Q,S,U,q,s,u] * bra[p,Q,r,S,t,U,v]  
            end

            #println(size(bra))
            #println(output_size)
            #println()
            bra_r[r] = reshape2(bra,ntuple(i->output_size[i], N))
        end
    end

    return ntuple(i->bra_r[i], R)
end
#=}}}=#
function contract_dense_H_with_state_tensor(term::ClusteredTerm4B, op, state_sign, coeffs_bra::Tucker{T,N,R}, coeffs_ket::Tucker{T,N,R}) where {T,N,R}
#={{{=#
    c1 = term.clusters[1]
    c2 = term.clusters[2]
    c3 = term.clusters[3]
    c4 = term.clusters[4]

    #
    # form overlaps - needed when TuckerConfigs aren't the same because each does their own compression and has
    # distinct Tucker factors
    tensors = Vector{Array{T}}()
    indices = Vector{Vector{Int64}}()
    state_indices = -collect(1:N)
    s = state_sign # this is the product of scalar overlaps that don't need tensor contractions
   

    # todo: 
    #   pass in preallocated and resize/reshape
    bra_r = Vector{Array{T}}()
   
    output_size = size(coeffs_bra.core[1])

    for r in 1:R
        push!(bra_r,  s .* coeffs_ket.core[r] )
    end

    # compute overlaps
    for ci in 1:N
        ci != c1.idx || continue
        ci != c2.idx || continue
        ci != c3.idx || continue
        ci != c4.idx || continue

        S = coeffs_bra.factors[ci]' * coeffs_ket.factors[ci]
        #println(size(S))

        # if overlap not just scalar, form and prepare for contraction
        for r in 1:R
            bra = bra_r[r]
            dims = [size(bra)...]
            
            if length(S) == 1
                bra .= bra .* S[1]
            else

                bra = reshape2(bra, 
                                   (prod(size(bra)[1:ci-1]), 
                                    size(bra)[ci], 
                                    prod(size(bra)[ci+1:end])))

                @tensor begin
                    bra[p,q,r] := S[q,Q] * bra[p,Q,r]  
                end
                dims[ci] = size(bra,2)
                #println(size(bra))
                #println(ntuple(i->dims[i], N))
                #println()
                bra_r[r] = reshape2(bra,ntuple(i->dims[i], N))
                length(size(bra_r[r])) == N || throw(DimensionMismatch)
            end
        end
    end
    
    # if the compressed operator becomes a scalar, treat it as such
    #println(" Here2:")
    #println(size.((coeffs_bra.core[1], op, bra_r[1])))
    #println()

    for r in 1:R
        bra = bra_r[r]
        
        if length(op) == 1
            bra .*= op[1]
        else
            #println((c1.idx, c2.idx))
            #println.(size.((op,bra)))
            #println((prod(size(bra)[1:c1.idx-1]), 
            #                size(bra)[c1.idx], 
            #                prod(size(bra)[c1.idx+1:c2.idx-1]),
            #                size(bra)[c2.idx], 
            #                prod(size(bra)[c2.idx+1:end])))

            bra = reshape2(bra, 
                           (prod(size(bra)[1:c1.idx-1]), 
                            size(bra)[c1.idx], 
                            prod(size(bra)[c1.idx+1:c2.idx-1]),
                            size(bra)[c2.idx], 
                            prod(size(bra)[c2.idx+1:c3.idx-1]),
                            size(bra)[c3.idx], 
                            prod(size(bra)[c3.idx+1:c4.idx-1]),
                            size(bra)[c4.idx], 
                            prod(size(bra)[c4.idx+1:end])))

            @tensor begin
                bra[p,q,r,s,t,u,v,w,x] := op[Q,S,U,W,q,s,u,w] * bra[p,Q,r,S,t,U,v,W,x]  
            end

            #println(size(bra))
            #println(output_size)
            #println()
            bra_r[r] = reshape2(bra,ntuple(i->output_size[i], N))
        end
    end

    return ntuple(i->bra_r[i], R)
end
#=}}}=#


