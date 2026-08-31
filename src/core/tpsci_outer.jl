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
    -`build_hqq`: how the Q-space Hamiltonian is formed for `:minres`/`:pcg` —
               `:sparse`, `:direct`, `:parallel`, or `:matvec` (nothing stored).
               Ignored when `solver=:krylov`.
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
                        compress=false,
                        compress_type="matvec",
                        solver=:krylov,
                        build_hqq=:direct,
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
                                         cepa_shift, cepa_mit, tol=tol, thresh_sigma=thresh_sigma, solver=solver,
                                         build_hqq=build_hqq, verbose=verbose)

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
        # Near the solution p -> 0, so pAp underflows on its own. That is convergence
        # stalling at the floating-point floor, not a breakdown: test it relative to
        # the first iteration rather than against eps.
        if pAp <= eps(T) * pAp0
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


"""
    tpsci_cepa_solve(ref_vector, e0, cepa_vector, cluster_ops, clustered_ham, cepa_shift, cepa_mit; tol, verbose)

Multi-root CEPA solver for TPSCIstate.

# Arguments
- `ref_vector`: pre-solved reference state (R roots)
- `e0`: reference energies for each root (length R, pre-computed by caller)
- `cepa_vector`: shared FOIS — union of all roots' first-order interacting spaces
- `cluster_ops`, `clustered_ham`: operators
- `cepa_shift`: "cepa" (CEPA-0), "acpf", "aqcc", "cisd"
- `cepa_mit`: max CEPA iterations (only relevant for acpf/aqcc)

The amplitude equation for each root I is solved independently with the shared H_xx:
    (H_xx - (E0[I] + shift[I])) * C_x[I] = -h[I]
    E[I] = E0[I] + C_x[I]' * h[I]

where h[I] = <Q|H|A_I> (coupling vector for root I).

`solver` picks how the amplitude equations are solved:
- `:krylov` — fully matrix-free. Every KrylovKit CG matvec re-runs
  `open_matvec_thread` over the whole Q-space, so memory stays O(Q_dim) but each
  iteration costs a full FOIS matvec. `build_hqq` is ignored.
- `:minres` — forms H_qq once according to `build_hqq`, then MINRES. Handles the
  indefinite shifted operator that shows up for the higher roots.
- `:pcg` — same H_qq as `:minres`, solved with Jacobi-preconditioned CG
  (preconditioner `diag(H_qq) - eshift`, warm started across macro-iterations).
  Falls back to MINRES for any solve whose shifted diagonal is not positive.

h[I] is always computed matrix-free: apply H to |A_I> via `open_matvec_thread`,
then project onto Q. That is one cheap P-space matvec per root.
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
                           build_hqq=:direct,
                           verbose=0) where {T,N,R,R2}

    n_clusters = length(ref_vector.clusters)
    dim_q = length(cepa_vector)

    solver in (:krylov, :minres, :pcg) ||
        error("Unknown CEPA solver: $solver. Use :krylov (matrix-free), :minres, or :pcg.")
    build_hqq in (:matvec, :sparse, :direct, :parallel) ||
        error("Unknown build_hqq: $build_hqq. Use :sparse, :direct, :parallel, or :matvec.")

    @printf(" CEPA solver: dim_q=%i, R=%i, shift=%s, solver=%s, build_hqq=%s\n",
            dim_q, R, cepa_shift, solver,
            solver == :krylov ? "n/a (matrix-free)" : string(build_hqq))

    # ── Helper: project a TPSCIstate onto Q-space, return dense vector ─────────
    # Collects only the configs present in cepa_vector (Q-space).
    function project_to_Q(sig::TPSCIstate, root=1)
        v = zeros(T, dim_q)
        idx = 0
        for (fock, configs) in cepa_vector.data
            for (config, _) in configs
                idx += 1
                if haskey(sig, fock) && haskey(sig[fock], config)
                    v[idx] = sig[fock][config][root]
                end
            end
        end
        return v
    end

    # ── h[I] = <Q|H|A_I>: apply H to root I of ref_vector, project to Q ───────
    # Each root requires one open_matvec call (cheap: P-space only).
    h = zeros(T, dim_q, R)
    for i in 1:R
        @printf(" Compute coupling vector h for root %i\n", i)
        ref_i = extract_chosen_root(ref_vector, i)
        sig_i = open_matvec_thread(ref_i, cluster_ops, clustered_ham,
                                   nbody=nbody, thresh=thresh_sigma)
        h[:, i] = project_to_Q(sig_i, 1)
    end

    # ── Build or wrap the Q-space matvec ─────────────────────────────────────────
    cepa_work = TPSCIstate(cepa_vector, R=1)   # reusable 1-root Q-space state

    H_qq_stored = nothing   # set only when the Q-space matrix is actually built

    if solver != :krylov
        # Build H_qq once. This costs ~dim_q² × 8 bytes for the dense builders
        # (e.g. 2.3 GB for dim_q=17093) stored once, replacing per-iteration
        # open_matvec_thread calls that each allocate ~180 GiB.
        @printf(" Q-space H_qq (%i × %i) [build_hqq=%s], reused by every %s solve\n",
                dim_q, dim_q, build_hqq, uppercase(string(solver)))
        if build_hqq == :matvec
            # Peak memory: O(nthreads × dim_q) — no H_qq stored at all.
            # Each iteration recomputes H_qq×v on the fly. Viable for dim_q=262K
            # where both :sparse and open_matvec_thread OOM.
            @printf(" Using matrix-free H_qq matvec (no H_qq stored)\n")
            Hq_mv = v -> matvec_H_qq(cepa_work, cluster_ops, clustered_ham, v)
        elseif build_hqq == :sparse
            # Peak memory: O(nnz) — per-thread COO triplets, no dim_q² allocation
            @time H_qq_stored = build_H_qq_sparse(cepa_work, cluster_ops, clustered_ham)
            # @printf(" H_qq sparsity: nnz=%i  (%.2f%% fill)\n",
            #         nnz(H_qq_stored), 100*nnz(H_qq_stored)/dim_q^2)
            Hq_mv = let A = H_qq_stored; v -> A * v; end
        elseif build_hqq == :direct
            # Peak memory: 1×dim_q²×8 B — threads write directly into H rows (no scratch copies)
            @time H_qq_stored = build_H_qq(cepa_work, cluster_ops, clustered_ham)
            Hq_mv = let A = H_qq_stored; v -> A * v; end
        else
            # build_hqq == :parallel — uses build_full_H_parallel (peak 2×dim_q²×8 B due to scratch)
            @time H_qq_stored = build_full_H_parallel(cepa_work, cepa_work, cluster_ops, clustered_ham, sym=true)
            Hq_mv = let A = H_qq_stored; v -> A * v; end
        end
    else
        function Hq_matvec(v::Vector{T})
            set_vector!(cepa_work, v, root=1)
            sig = open_matvec_thread(cepa_work, cluster_ops, clustered_ham,
                                     nbody=nbody, thresh=0.0)
            return project_to_Q(sig, 1)
        end
        Hq_mv = Hq_matvec
    end

    # ── Jacobi preconditioner data for :pcg ──────────────────────────────────────
    # The Q-space diagonal is root independent, so build it once here and reuse it
    # for every root and macro-iteration. Only the shift changes.
    Hdiag     = T[]
    hdiag_min = zero(T)
    Cd_prev   = zeros(T, 0, 0)
    if solver == :pcg
        if H_qq_stored !== nothing
            Hdiag = Vector{T}(diag(H_qq_stored))
        else
            # build_hqq=:matvec keeps nothing around, so form the diagonal directly.
            @printf(" Computing Q-space diagonal for the PCG preconditioner\n")
            @time Hdiag = Vector{T}(compute_diagonal(cepa_work, cluster_ops, clustered_ham))
        end
        hdiag_min = minimum(Hdiag)
        # Warm-start storage: amplitudes carry over between macro-iterations.
        Cd_prev = zeros(T, dim_q, R)
    end

    Ec      = zeros(T, R)
    Ec_prev = fill(T(Inf), R)

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

        # Solve (H_xx - eshift*I) * Cd = -h[I] for each root
        for i in 1:R
            @printf(" CEPA Iter %3i  Root %i  Shift = %12.8f\n", it, i, shifts[i])
            eshift = e0[i] + shifts[i]
            # CG only converges for a positive-definite shifted operator. The cheapest
            # screen is the diagonal: if any Q-space state sits below the shift, go
            # straight to MINRES for this solve.
            try_pcg = solver == :pcg && (hdiag_min - eshift) > 0
            if solver == :pcg && !try_pcg && verbose > 0
                @printf("   shifted diagonal is not positive (min %.3e); using MINRES\n",
                        hdiag_min - eshift)
            end

            Cd_i = nothing
            if try_pcg
                # Successive macro-iterations move the shift by only a small amount, so
                # the previous amplitudes are an excellent guess and later solves
                # converge in a few CG steps.
                Cd_pcg, history = cepa_pcg_linsolve(Hq_mv, -h[:, i], Hdiag;
                                                    eshift=eshift, tol=tol,
                                                    maxiter=cg_maxiter, verbose=verbose,
                                                    x0=(it > 1 ? view(Cd_prev, :, i) : nothing))
                if history.indefinite
                    # The diagonal screen is necessary but not sufficient: CG found a
                    # direction the shifted operator is not positive on, so throw the
                    # partial solve away and redo this root with MINRES.
                    verbose > 0 &&
                        @printf("   shifted operator is indefinite; redoing root %i with MINRES\n", i)
                else
                    Cd_i = Cd_pcg
                    if verbose > 0
                        @printf(" Iter %3i  Root %i  nops=%4i  res=%8.2e  E_corr = %16.12f%s\n",
                                it, i, history.iters, history.residual, dot(Cd_i, h[:, i]),
                                history.converged ? "" : "   (not converged)")
                    end
                end
            end

            if Cd_i === nothing && solver != :krylov
                # MINRES handles symmetric indefinite (H_qq - eI can be indefinite)
                H_eff = LinearMap{T}(v -> Hq_mv(v) .- eshift .* v, dim_q; issymmetric=true)
                Cd_i, history = IterativeSolvers.minres(H_eff, -h[:, i];
                                                        reltol=tol, maxiter=cg_maxiter,
                                                        log=true)
                if verbose > 0
                    @printf(" Iter %3i  Root %i  nops=%4i  res=%8.2e  E_corr = %16.12f\n",
                            it, i, history.iters, history.data[:resnorm][end], dot(Cd_i, h[:, i]))
                end
            elseif Cd_i === nothing
                Afunc = v -> Hq_mv(v) .- eshift .* v
                Cd_i, info = KrylovKit.linsolve(Afunc, -h[:, i];
                                                tol=tol, maxiter=cg_maxiter,
                                                issymmetric=true, isposdef=true,
                                                verbosity=0)
                if verbose > 0
                    @printf(" Iter %3i  Root %i  nops=%4i  E_corr = %16.12f\n",
                            it, i, info.numops, dot(Cd_i, h[:, i]))
                end
            end

            solver == :pcg && (Cd_prev[:, i] .= Cd_i)
            Ec[i] = dot(Cd_i, h[:, i])
        end

        cepa_shift == "cepa" && break
        maximum(abs.(Ec .- Ec_prev)) < tol && break
        Ec_prev .= Ec
    end

    return Ec, e0 .+ Ec
end
