using LinearMaps
using IterativeSolvers
using .BlockDavidson
#using TensorDecompositions
#using TimerOutputs




"""
    get_map(ci_vector::SPTstate, cluster_ops, clustered_ham)

Get LinearMap with takes a vector and returns action of H on that vector
"""
function get_map(ci_vector::SPTstate{T,N,R}, cluster_ops, clustered_ham; shift = nothing, cache=false) where {T,N,R}
    #={{{=#
    iters = 0
    
    dim = length(ci_vector)
    function mymatvec(v)
        iters += 1

        all(size(ci_vector) .== size(v)) || error(DimensionMismatch)
        set_vector!(ci_vector, v)

        #fold!(ci_vector)
        sig = deepcopy(ci_vector)
        zero!(sig)
        build_sigma!(sig, ci_vector, cluster_ops, clustered_ham, cache=cache)

        #unfold!(ci_vector)

        sigv = get_vector(sig)

        if shift != nothing
            # this is how we do CEPA
            sigv += shift * get_vector(ci_vector)
        end
        flush(stdout)

        return sigv
    end
    return LinearMap(mymatvec, dim, dim; issymmetric=true, ismutating=false, ishermitian=true)
end
#=}}}=#

"""
    function ci_solve(ci_vector_in::SPTstate{T,N,R}, cluster_ops, clustered_ham; 
                         conv_thresh = 1e-5,
                         max_ss_vecs = 12,
                         max_iter    = 40,
                         shift       = nothing,
                         precond     = false,
                         verbose     = 0,
                         solver      = "davidson") where {T,N,R}

Solve for ground state in the space spanned by `ci_vector`'s compression vectors
# Arguments
- `conv_thresh`: residual convergence threshold
- `max_ss_vecs`: max number of subspace vectors
- `max_iter`: Max iterations in solver
- `shift`:  Use a shift? this is for CEPA type 
- `precond`: use preconditioner? Only applied to Davidson and not yet working,
- `verbose`: print level
- `solver`: Which solver to use. Options = ["davidson", "krylovkit"]
"""
function ci_solve(ci_vector_in::SPTstate{T,N,R}, cluster_ops, clustered_ham; 
                         conv_thresh    = 1e-5,
                         max_ss_vecs    = 12,
                         max_iter       = 40,
                         lindep_thresh  = 1e-10,
                         shift          = nothing,
                         precond        = true,
                         verbose        = 0,
                         nbody          = 4,
                         solver         = "davidson") where {T,N,R}
#={{{=#
    @printf(" |== SPT CI ========================================================\n")
    @printf(" %-50s", "Solve CI with # variables: ")
    @printf("%10i\n", length(ci_vector_in))
    vec = deepcopy(ci_vector_in)
    orthonormalize!(vec)
    #flush term cache
    flush_cache(clustered_ham)


    # 
    #   Rotate tucker basis to diagonalize CMF hamiltonian for better preconditioning    
    # Hcmf_diag = SPTstate(deepcopy(vec), R=1) 
    Hcmf_diag = SPTstate(vec, R=1) 
    if precond
        verbose < 2 || @printf(" form_1body_operator_diagonal... \n")
        form_1body_operator_diagonal!(vec, Hcmf_diag, cluster_ops, pseudo_canon=true)
        verbose < 2 || @printf(" done.\n")
    end
    
    #Hmap = get_map(vec, cluster_ops, clustered_ham, cache=true)
    iters = 0
    
    function matvec(v::Matrix{T}) where T
        iters += 1
        #all(size(vec) .== size(v)) || error(DimensionMismatch)
        vec_i = SPTstate(vec, R=size(v,2))
        set_vector!(vec_i, v)

        sig = deepcopy(vec_i)
        zero!(sig)
        build_sigma!(sig, vec_i, cluster_ops, clustered_ham, cache=cache, nbody=nbody)

        return get_vector(sig) 
    end
    function matvec(v::Vector{T}) where T
        iters += 1
        #all(size(vec) .== size(v)) || error(DimensionMismatch)
        vec_i = SPTstate(vec, R=1)
        set_vector!(vec_i, v)

        sig = deepcopy(vec_i)
        zero!(sig)
        build_sigma!(sig, vec_i, cluster_ops, clustered_ham, cache=cache, nbody=nbody)

        return get_vector(sig)[:,1]
    end




    v0 = get_vector(vec)

    cache=true
    if cache
        @printf(" %-50s", "Cache Hamiltonian: ")
        flush(stdout)
        @time cache_hamiltonian(vec, vec, cluster_ops, clustered_ham)
        flush(stdout)
    end

    Hmap = TPSChem.LinOpMat{T}(matvec, length(vec), true)
    
    vguess = get_vector(vec, 1)[:,1]

    e = [] 
    v = [[]]

    if solver == "krylovkit"

        time = @elapsed e, v, info = KrylovKit.eigsolve(Hmap, vguess, R, :SR, 
                                                        verbosity   = verbose, 
                                                        maxiter     = max_iter, 
                                                        #krylovdim   = max_ss_vecs, 
                                                        issymmetric = true, 
                                                        ishermitian = true, 
                                                        eager       = true,
                                                        tol         = conv_thresh)

        @printf(" Number of matvecs performed: %5i\n", info.numops)
        @printf(" Number of subspace restarts: %5i\n", info.numiter)
        if info.converged >= R
            @printf(" CI Converged: %5i roots\n", info.converged)
        end
        println(" Residual Norms")
        for r in 1:R
            @printf(" State %5i %16.1e\n", r, info.normres[r])
        end
        
        println()
        @printf(" %-50s%10.6f seconds\n", "Diagonalization time: ",time)
        set_vector!(vec,hcat(v[1:R]...))

    elseif solver == "davidson"

        #for (ftrans,terms) in clustered_ham
        #    for term in terms
        #        println("nick: ", length(term.cache))
        #    end
        #end

        #cache_hamiltonian(ci_vector, ci_vector, cluster_ops, clustered_ham)

        davidson = Davidson(Hmap,v0=v0,max_iter=max_iter, max_ss_vecs=max_ss_vecs, nroots=R, 
                            tol=conv_thresh, lindep_thresh=lindep_thresh)
        flush(stdout)
        if precond
            time = @elapsed e, v = BlockDavidson.eigs(davidson, Adiag=get_vector(Hcmf_diag))
            @printf(" %-50s%10.6f seconds\n", "Diagonalization time: ", time)
        else
            time = @elapsed e, v = BlockDavidson.eigs(davidson)
            @printf(" %-50s%10.6f seconds\n", "Diagonalization time: ", time)
        end
        #println(" Memory used by cache: ", mem_used_by_cache(clustered_ham))
        set_vector!(vec,v)

    else
        error(" Bad value for `solver`")
    end
    #flush term cache
    flush_cache(clustered_ham)
    
    clustered_S2 = extract_S2(vec.clusters)
    @printf(" %-50s", "Compute <S^2>: ")
    flush(stdout)
    tmp = deepcopy(vec)
    zero!(tmp)
    @time build_sigma_cepa!(tmp, vec, cluster_ops, clustered_S2)
    s2 = orth_dot(tmp,vec)
    flush(stdout)
    @printf(" %5s %12s %12s\n", "Root", "Energy", "S2") 
    for r in 1:R
        @printf(" %5s %12.8f %12.8f\n",r, e[r], abs(s2[r]))
    end

    @printf(" ==================================================================|\n")
    return e,vec
end
#=}}}=#







"""
    tucker_cepa_solve!(ref_vector::SPTstate, cepa_vector::SPTstate, cluster_ops, clustered_ham; tol=1e-5, cache=true)

# Arguments
- `ref_vector`: Input reference state. 
- `cepa_vector`: SPTstate which defines the configurational space defining {X}. This 
should be the first-order interacting space (or some compressed version of it).
- `cluster_ops`
- `clustered_ham`
- `tol`: haven't yet set this up (NYI)
- `cache`: Should we cache the compressed H operators? Speeds up drastically, but uses lots of memory

Compute compressed CEPA.
Since there can be non-zero overlap with a multireference state, we need to generalize.

    HC = SCe

    |Haa + Hax| |1 | = |I   + Sax| |1 | E
    |Hxa + Hxx| |Cx|   |Sxa + I  | |Cx|

    Haa + Hax*Cx = (1 + Sax*Cx)E
    Hxa + HxxCx = SxaE + CxE

The idea for CEPA is to approximate E in the amplitude equation.
CEPA(0): E = Eref

    (Hxx-Eref)*Cx = Sxa*Eref - Hxa

Ax=b

After solving, the Energy can be obtained as:
    
    E = (Eref + Hax*Cx) / (1 + Sax*Cx)
"""
function tucker_cepa_solve(ref_vector::SPTstate{T,N,R}, cepa_vector::SPTstate, cluster_ops, clustered_ham,
                           cepa_shift = "cepa",
                           cepa_mit   = 50;
                           tol        = 1e-6,
                           cache      = true,
                           max_iter   = 30,
                           verbose    = false,
                           solver     = :minres) where {T,N,R}
#={{{=#

    sig = deepcopy(ref_vector)
    zero!(sig)
    build_sigma_cepa!(sig, ref_vector, cluster_ops, clustered_ham, cache=false)
    e0 = nonorth_dot(ref_vector, sig)
    length(e0) == 1 || error("Only one state at a time please", e0)
    e0 = e0[1]
    @printf(" Reference Energy: %12.8f\n", e0)

    n_clusters = length(cepa_vector.clusters)

    x_vector = deepcopy(cepa_vector)

    # h[:,r] = <X|H|A_r>  in the Tucker core basis of the Q-space
    b = deepcopy(x_vector)
    zero!(b)
    build_sigma_cepa!(b, ref_vector, cluster_ops, clustered_ham, cache=false)
    h = get_vector(b)   # dim_x × R

    # Sx[:,r] = <X|A_r>  —  Tucker factor overlap between Q-space and reference
    # Needed because P- and Q-space Tucker factors are non-orthogonal.
    # Formula: for each (fock,tconfig) block,
    #   Sx.core[r] = transform_basis(ref.core[r], [ref.factors[i]' * x.factors[i] for i])
    Sx = deepcopy(x_vector)
    zero!(Sx)
    for (fock, tconfigs) in Sx
        for (tconfig, tuck) in tconfigs
            haskey(ref_vector, fock)           || continue
            haskey(ref_vector[fock], tconfig)  || continue
            ref_tuck = ref_vector[fock][tconfig]
            overlaps = [ref_tuck.factors[i]' * tuck.factors[i] for i in 1:N]
            for r in 1:R
                Sx[fock][tconfig].core[r] .= transform_basis(ref_tuck.core[r], overlaps)
            end
        end
    end
    sx = get_vector(Sx)   # dim_x × R;  sx[:,r] = <X|A_r>

    @printf(" Norm of Sx (raw): %12.8e\n", norm(sx))

    # Build Q-space mask: P-space Tucker configs (shared with ref_vector) must be
    # projected out of the CEPA amplitude space.  Without this, for any (fock,tconfig)
    # block shared between cepa_vector and ref_vector:
    #   H_PP * C_ref = e0 * C_ref  →  rhs_P = sx_P*e0 - h_P = 0
    #   (H_PP - e0) is singular along C_ref  →  trivial solution x_P = -sx_P
    #   resulting in <A|X>Cx = -1 and unphysical (above-reference) energies.
    p_mask = ones(T, length(x_vector))
    let idx = 1
        for (fock, tconfigs) in x_vector
            for (tconfig, tuck) in tconfigs
                dim1 = prod(size(tuck))
                if haskey(ref_vector, fock) && haskey(ref_vector[fock], tconfig)
                    p_mask[idx:idx+dim1-1] .= zero(T)
                end
                idx += dim1
            end
        end
    end
    # Apply mask: zero P-space entries so CEPA lives entirely in Q-space
    h  .*= p_mask
    sx .*= p_mask
    @printf(" Norm of Sx (Q-space): %12.8e\n", norm(sx))

    # Build cache once — Tucker factors don't change across CEPA iterations
    if cache
        flush_cache(clustered_ham)
        @printf(" %-50s", "Cache zeroth-order Hamiltonian: ")
        @time cache_hamiltonian(x_vector, x_vector, cluster_ops, clustered_ham)
    end

    Ec    = zeros(T, R)
    Ecepa = zeros(T, R)

    for it in 1:cepa_mit
        @printf(" CEPA cycle: %i\n", it)
        Ec_old = copy(Ec)   # save before updating this iteration

        for r in 1:R
            if     cepa_shift == "cepa";  shift = zero(T)
            elseif cepa_shift == "acpf";  shift = Ec[r] * 2.0 / n_clusters
            elseif cepa_shift == "aqcc"
                shift = (1.0 - (n_clusters-3.0)*(n_clusters-2.0) /
                               (n_clusters*(n_clusters-1.0))) * Ec[r]
            elseif cepa_shift == "cisd";  shift = Ec[r]
            else;  error("NYI: cepa_shift not available: $cepa_shift")
            end

            eshift = e0 + shift
            # Correct CEPA RHS: (H_xx - eshift) Cx = Sxa*eshift - Hxa
            rhs = sx[:, r] .* eshift .- h[:, r]

            # matvec: (H_xx - eshift) v, restricted to Q-space (P-space projected out)
            function mymatvec(v)
                v_q = v .* p_mask   # project input to Q-space
                xr = SPTstate(x_vector, R=1)
                xl = SPTstate(x_vector, R=1)
                set_vector!(xr, vec(v_q), root=1)
                zero!(xl)
                build_sigma_cepa!(xl, xr, cluster_ops, clustered_ham, cache=cache)
                tmp = deepcopy(xr)
                scale!(tmp, -eshift)
                orth_add!(xl, tmp)
                return vec(get_vector(xl)) .* p_mask   # project output to Q-space
            end

            x0 = vec(get_vector(x_vector, r)) .* p_mask   # start in Q-space
            @printf(" Root %i  eshift = %12.8f  |rhs| = %12.8f\n",
                    r, eshift, norm(rhs))

            if solver == :krylovkit
                time_elapsed = @elapsed begin
                    x_sol, info = KrylovKit.linsolve(mymatvec, rhs, x0;
                                                     tol         = tol,
                                                     maxiter     = max_iter,
                                                     issymmetric = true,
                                                     isposdef    = false,
                                                     verbosity   = 0)
                end
                numops   = info.numops
                convflag = info.converged == 1
                @printf(" %-50s%10.6f seconds  nops=%i\n",
                        "Time to solve CEPA (KrylovKit): ", time_elapsed, numops)

            elseif solver == :minres
                warm_rhs = rhs .- mymatvec(x0)
                n    = length(rhs)
                A_op = LinearOperator(Float64, n, n, true, true,
                                      (y, v) -> (y .= mymatvec(v)))
                time_elapsed = @elapsed begin
                    dx_sol, stats = Krylov.minres(A_op, warm_rhs;
                                                  atol  = tol,
                                                  rtol  = tol,
                                                  itmax = max_iter)
                end
                x_sol    = dx_sol .+ x0
                numops   = stats.niter
                convflag = stats.solved
                @printf(" %-50s%10.6f seconds  nops=%i\n",
                        "Time to solve CEPA (MINRES): ", time_elapsed, numops)

            else
                error("Unknown solver: $solver. Choose :krylovkit or :minres")
            end

            if !convflag
                @warn " Root $r did not converge (solver=:$solver)"
            end

            x_sol = x_sol .* p_mask   # enforce Q-space (remove floating-point P-space leakage)
            set_vector!(x_vector, x_sol, root=r)

            # E = (e0 + <A|H|X>Cx) / (1 + <A|X>Cx)
            ecorr    = dot(x_sol, h[:, r])
            SxC      = dot(x_sol, sx[:, r])
            Ecepa[r] = (e0 + ecorr) / (1.0 + SxC)
            Ec[r]    = Ecepa[r] - e0              # correlation energy for shift
            @printf(" Root %i  E_corr = %18.12f  E_total = %18.12f  <A|X>Cx = %10.3e  |X| = %10.3e\n",
                    r, Ec[r], Ecepa[r], SxC, norm(x_sol))
        end

        @printf("Iter: %4d  ", it)
        for r in 1:R
            @printf("  Root%i: %18.12f", r, Ecepa[r])
        end
        println()

        cepa_shift == "cepa" && break   # CEPA-0: zero shift → one linear solve is exact

        # convergence check for iterative shifts (ACPF/AQCC)
        if maximum(abs.(Ec .- Ec_old)) < tol
            @printf(" Converged %s\n", cepa_shift)
            break
        end
    end

    flush_cache(clustered_ham)
    return Ecepa, x_vector
end#=}}}=#
 
"""
    tucker_cepa_solve!(ref_vector::SPTstate, cepa_vector::SPTstate, cluster_ops, clustered_ham; tol=1e-5, cache=true)

# Arguments
- `ref_vector`: Input reference state. 
- `cepa_vector`: SPTstate which defines the configurational space defining {X}. This 
should be the first-order interacting space (or some compressed version of it).
- `cluster_ops`
- `clustered_ham`
- `tol`: haven't yet set this up (NYI)
- `cache`: Should we cache the compressed H operators? Speeds up drastically, but uses lots of memory

Compute compressed CEPA.
Since there can be non-zero overlap with a multireference state, we need to generalize.

HC = SCe

|Haa + Hax| |1 | = |I   + Sax| |1 | E
|Hxa + Hxx| |Cx|   |Sxa + I  | |Cx|

Haa + Hax*Cx = (1 + Sax*Cx)E
Hxa + HxxCx = SxaE + CxE

The idea for CEPA is to approximate E in the amplitude equation.
CEPA(0): E = Eref

(Hxx-Eref)*Cx = Sxa*Eref - Hxa

Ax=b

After solving, the Energy can be obtained as:
E = (Eref + Hax*Cx) / (1 + Sax*Cx)
"""
function tucker_cepa_solve2(ref_vector::SPTstate, cepa_vector::SPTstate, cluster_ops, clustered_ham; tol=1e-5, cache=true, max_iter=30, verbose=false,nbody=4, do_pt2=false)
#={{{=#
    sig = deepcopy(ref_vector)
    zero!(sig)
    build_sigma!(sig, ref_vector, cluster_ops, clustered_ham, cache=false)
    e0 = nonorth_dot(ref_vector, sig)
    length(e0) == 1 || error("Only one state at a time please", e0)
    e0 = e0[1]
    @printf(" Reference Energy: %12.8f\n",e0)

    e0_1b = 0.0
    if do_pt2
        sig = deepcopy(ref_vector)
        zero!(sig)
        build_sigma!(sig, ref_vector, cluster_ops, clustered_ham, nbody=nbody)
        e0_1b = nonorth_dot(ref_vector, sig)
    end


    x_vector = deepcopy(cepa_vector)
    a_vector = deepcopy(ref_vector)


#    project_out!(x_vector, a_vector)
#    #
#    # Project out reference space
#    for (fock,tconfigs) in x_vector 
#        for (tconfig, tuck) in tconfigs
#            if haskey(ref_vector, fock)
#                if haskey(ref_vector[fock], tconfig)
#                    ref_tuck = ref_vector[fock][tconfig]
#
#                    ovlp = nonorth_dot(tuck, ref_tuck) / nonorth_dot(ref_tuck, ref_tuck)
#                    tmp = scale(ref_tuck, -1.0 * ovlp)
#                    x_vector[fock][tconfig] = nonorth_add(tuck, tmp, thresh=1e-16)
#                end
#            end
#        end
#    end
    @printf(" Overlap between <1|0>:          %8.1e\n", nonorth_dot(x_vector, ref_vector, verbose=0))

    b = deepcopy(x_vector)
    zero!(b)
    build_sigma!(b, ref_vector, cluster_ops, clustered_ham, cache=false)
    bv = -get_vector(b)

    
    #
    # Get Overlap <X|A>C(A)
    Sx = deepcopy(x_vector)
    zero!(Sx)
    for (fock,tconfigs) in Sx 
        for (tconfig, tuck) in tconfigs
            if haskey(ref_vector, fock)
                if haskey(ref_vector[fock], tconfig)
                    ref_tuck = ref_vector[fock][tconfig]
                    # Cr(i,j,k...) Ur(Ii) Ur(Jj) ...
                    # Ux(Ii') Ux(Jj') ...
                    #
                    # Cr(i,j,k...) S(ii') S(jj')...
                    overlaps = []
                    for i in 1:length(Sx.clusters)
                        push!(overlaps, ref_tuck.factors[i]' * tuck.factors[i])
                    end
                    Sx[fock][tconfig].core .= transform_basis(ref_tuck.core, overlaps)
                end
            end
        end
    end

    bv .= bv .+ get_vector(Sx)*e0

    @printf(" Norm of Sx overlap: %12.8f\n", orth_dot(Sx,Sx))
    @printf(" Norm of b         : %12.8f\n", sum(bv.*bv))

    #
    # Currently, we need to solve each root separately, this should be fixed
    # by writing our own CG solver
    function mymatvec(v)
        xr = SPTstate(x_vector, R=1)
        xl = SPTstate(x_vector, R=1)
        length(xr) == length(v) || throw(DimensionMismatch())
        set_vector!(xr, vec(v), root=1)
        zero!(xl)
        build_sigma!(xl, xr, cluster_ops, clustered_ham, cache=cache)
        tmp = deepcopy(xr)
        if do_pt2
            scale!(tmp, -e0_1b)
        else
            scale!(tmp, -e0)
        end
        orth_add!(xl, tmp)
        return get_vector(xl)
    end

    flush_cache(clustered_ham)
    println(" Start CEPA iterations with dimension = ", length(x_vector))
    x0 = vec(get_vector(x_vector))
    x_sol, info = KrylovKit.linsolve(mymatvec, vec(bv), x0;
                                      tol=tol, maxiter=max_iter,
                                      issymmetric=true, isposdef=true,
                                      verbosity=0)
    @printf(" nops=%i\n", info.numops)
    flush_cache(clustered_ham)

    set_vector!(x_vector, x_sol)


    SxC = orth_dot(Sx,x_vector)
    @printf(" <A|X>C(X) = %12.3e\n", SxC)

    sig = deepcopy(ref_vector)
    zero!(sig)
    @time build_sigma!(sig,x_vector, cluster_ops, clustered_ham)
    ecorr = nonorth_dot(sig,ref_vector)
    @printf(" Cepa: %12.8f\n", ecorr)
    length(ecorr) == 1 || error(" Dimension Error", ecorr)
    ecorr = ecorr[1]
    @printf(" <1|1> = %12.8f\n", orth_dot(x_vector,x_vector))

    @printf(" E(CEPA) = %12.8f\n", (e0 + ecorr)/(1+SxC))

    #x, info = linsolve(Hmap,zeros(size(v0)))
    return (ecorr+e0)/(1+SxC), x_vector 
end#=}}}=#



"""
    define_foi_space(v::SPTstate, clustered_ham; nbody=2)
Compute the first-order interacting space as defined by clustered_ham

#Arguments
- `v::SPTstate`: input state
- `clustered_ham`: Hamiltonian
- `nbody`: allows one to limit (max 4body) terms in the Hamiltonian considered

#Returns
- `foi::OrderedDict{FockConfig,Vector{TuckerConfig}}`

"""
function define_foi_space(cts::T, clustered_ham; nbody=2) where T<:Union{BSstate, SPTstate}
    println(" Define the FOI space for SPTstate. nbody = ", nbody)#={{{=#

    foi_space = OrderedDict{FockConfig,Vector{TuckerConfig}}()

    for (fock, tconfigs) in cts

        for (fock_trans, terms) in clustered_ham

            #
            # new fock sector configuration
            new_fock = fock + fock_trans


            #
            # check that each cluster doesn't have too many/few electrons
            ok = true
            for ci in cts.clusters
                if new_fock[ci.idx][1] > length(ci) || new_fock[ci.idx][2] > length(ci)
                    ok = false
                end
                if new_fock[ci.idx][1] < 0 || new_fock[ci.idx][2] < 0
                    ok = false
                end
            end
            ok == true || continue



            #
            # find the cluster state index ranges (TuckerConfigs) reached by Hamiltonian
            for (tconfig, coeffs) in tconfigs
                for term in terms
                    new_tconfig = deepcopy(tconfig)

                    length(term.clusters) <= nbody || continue

                    new_tconfigs = []
                    tmp = [] # list of lists of index ranges, the cartesian product is the set needed
                    #
                    # for current term, expand index ranges for active clusters
                    for cidx in 1:length(term.clusters)
                        ci = term.clusters[cidx]
                        new_tconfig[ci.idx] = cts.q_spaces[ci.idx][new_fock[ci.idx]]
                        tmp2 = []
                        if haskey(cts.p_spaces[ci.idx], new_fock[ci.idx])
                            push!(tmp2, cts.p_spaces[ci.idx][new_fock[ci.idx]])
                        end
                        if haskey(cts.q_spaces[ci.idx], new_fock[ci.idx])
                            push!(tmp2, cts.q_spaces[ci.idx][new_fock[ci.idx]])
                        end
                        push!(tmp, tmp2)
                    end

                    for prod in Iterators.product(tmp...)
                        new_tconfig = deepcopy(tconfig)
                        for cidx in 1:length(term.clusters)
                            ci = term.clusters[cidx]
                            new_tconfig[ci.idx] = prod[cidx]
                        end
                        push!(new_tconfigs, new_tconfig)
                    end

                    if haskey(foi_space, new_fock)
                        foi_space[new_fock] = unique((foi_space[new_fock]..., new_tconfigs...))
                    else
                        foi_space[new_fock] = new_tconfigs
                    end


                end
            end
        end
    end
    return foi_space
#=}}}=#
end


"""
    build_compressed_1st_order_state(ψ::SPTstate{T,N,R}, cluster_ops, clustered_ham; 
    thresh=1e-7, 
    max_number=nothing, 
    nbody=4, 
    prescreen=false,
    compress_twice=true
    )  where {T,N,R}

TBW
"""
function build_compressed_1st_order_state(ψ::SPTstate{T,N,R}, cluster_ops, clustered_ham; 
    thresh=1e-7, 
    max_number=nothing, 
    nbody=4, 
    prescreen=false,
    compress_iter=false,
    compress_twice=true
    )  where {T,N,R}

    println(" In build_compressed_1st_order_state")
    σ = SPTstate(ψ.clusters, ψ.p_spaces, ψ.q_spaces, T=T, R=R)
    clusters = ψ.clusters
    jobs = Dict{FockConfig{N},Vector{Tuple}}()


    # 
    # define batches (FockConfigs present in resolvant)
    @printf(" %-50s", "Setup threaded jobs: ")
    @time for (fock_ψ, configs_ψ) in ψ.data
        for (ftrans, terms) in clustered_ham
            fock_σ = ftrans + fock_ψ

            #
            # check to make sure this fock config doesn't have negative or too many electrons in any cluster
            all(f[1] >= 0 for f in fock_σ) || continue 
            all(f[2] >= 0 for f in fock_σ) || continue 
            all(f[1] <= length(clusters[fi]) for (fi,f) in enumerate(fock_σ)) || continue 
            all(f[2] <= length(clusters[fi]) for (fi,f) in enumerate(fock_σ)) || continue 
          
            # 
            # Check to make sure we don't create states that we have already discarded
            found = true
            for c in ψ.clusters
                if haskey(cluster_ops[c.idx]["H"], (fock_σ[c.idx], fock_σ[c.idx])) == false
                    found = false
                    continue
                end
            end
            found == true || continue

            job_input = (terms, fock_ψ, configs_ψ)
            if haskey(jobs, fock_σ)
                push!(jobs[fock_σ], job_input)
            else
                jobs[fock_σ] = [job_input]
            end
            
        end
    end

    jobs_vec = []
    for (fock_σ, job) in jobs
        push!(jobs_vec, (fock_σ, job))
    end
    @printf(" %-50s%10i\n", "Number of tasks: ", length(jobs_vec))

    
    scr_v = Vector{Vector{Vector{T}} }()
    jobs_out = Vector{SPTstate{T,N,R}}()
    for tid in 1:Threads.maxthreadid()

        # Initialize each thread with it's own SPTstate
        push!(jobs_out, SPTstate(ψ.clusters, ψ.p_spaces, ψ.q_spaces, T=T, R=R))
        push!(scr_v, Vector{Vector{Float64}}([zeros(T, 1000) for i in 1:N]))

    end

    blas_num_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    
    @printf(" %-50s", "Compute matrix-vector: ")
    @time @Threads.threads :static for (fock_σ, jobs) in jobs_vec
        tid = Threads.threadid()
        _build_compressed_1st_order_state_job(fock_σ, jobs, jobs_out[tid],
                cluster_ops, nbody, thresh, max_number, prescreen, compress_twice,
                scr_v[tid])
    end
    flush(stdout)

    @printf(" %-50s", "Now collect thread results : ")
    flush(stdout)
    @time for threadid in 1:Threads.maxthreadid()
        for (fock, configs) in jobs_out[threadid].data
            haskey(σ, fock) == false || error(" why me")
            σ[fock] = configs
        end
    end

    # Reset BLAS num_threads
    BLAS.set_num_threads(blas_num_threads)
    if compress_iter==true
        @printf(" Compressing final σ vector:\n")
        σ = compress_iteratively(σ, thresh)
    end
    return σ

end

function _build_compressed_1st_order_state_job(fock_σ, jobs, σ::SPTstate{T,N,R}, 
    cluster_ops, nbody, thresh, max_number, prescreen, compress_twice,
    scr_v) where {T,N,R}

    add_fockconfig!(σ, fock_σ)

    data = OrderedDict{TuckerConfig{N},Vector{Tucker{T,N,R}}}()

    for jobi in jobs

        terms, fock_ψ, tconfigs_ψ = jobi

        for term in terms

            length(term.clusters) <= nbody || continue

            for (tconfig_ψ, tuck_ψ) in tconfigs_ψ
                #
                # find the σ TuckerConfigs reached by applying current Hamiltonian term to tconfig_ψ.
                #
                # For example:
                #
                #   [(p'q), I, I, (r's), I ] * |P,Q,P,Q,P>  --> |X, Q, P, X, P>  where X = {P,Q}
                #
                #   This this term, will couple to 4 distinct tucker blocks (assuming each of the active clusters
                #   have both non-zero P and Q spaces within the current fock sector, "fock_σ".
                #
                # We will loop over all these destination TuckerConfig's by creating the cartesian product of their
                # available spaces, this list of which we will keep in "available".
                #

                available = [] # list of lists of index ranges, the cartesian product is the set needed
                #
                # for current term, expand index ranges for active clusters
                for ci in term.clusters
                    tmp = []
                    if haskey(σ.p_spaces[ci.idx], fock_σ[ci.idx])
                        push!(tmp, σ.p_spaces[ci.idx][fock_σ[ci.idx]])
                    end
                    if haskey(σ.q_spaces[ci.idx], fock_σ[ci.idx])
                        push!(tmp, σ.q_spaces[ci.idx][fock_σ[ci.idx]])
                    end
                    push!(available, tmp)
                end


                #
                # Now loop over cartesian product of available subspaces (those in X above) and
                # create the target TuckerConfig and then evaluate the associated terms
                for prod in Iterators.product(available...)
                    tconfig_σ = [tconfig_ψ.config...]
                    for cidx in 1:length(term.clusters)
                        ci = term.clusters[cidx]
                        tconfig_σ[ci.idx] = prod[cidx]
                    end
                    tconfig_σ = TuckerConfig(tconfig_σ)

                    #
                    # the `term` has now coupled our ket TuckerConfig, to a sig TuckerConfig
                    # let's compute the matrix element block, then compress, then add it to any existing compressed
                    # coefficient tensor for that sig TuckerConfig.
                    #
                    # Both the Compression and addition takes a fair amount of work.


                    check_term(term, fock_σ, tconfig_σ, fock_ψ, tconfig_ψ) || continue


                    if prescreen
                        bound = calc_bound(term, cluster_ops,
                            fock_σ, tconfig_σ,
                            fock_ψ, tconfig_ψ, tuck_ψ,
                            prescreen=thresh)
                        bound == true || continue
                    end

                    tuck_σ = form_sigma_block_expand(term, cluster_ops,
                        fock_σ, tconfig_σ,
                        fock_ψ, tconfig_ψ, tuck_ψ,
                        max_number=max_number,
                        prescreen=thresh)
                    
                    length(tuck_σ) > 0 || continue
                    # norm(tuck_σ) > thresh || continue

                    #compress new addition
                    tuck_σ = compress(tuck_σ, thresh=thresh)

                    length(tuck_σ) > 0 || continue

                    #add to current sigma vector
                    if haskey(data, tconfig_σ)
                        push!(data[tconfig_σ], tuck_σ)
                    else
                        data[tconfig_σ] = [tuck_σ]
                    end
                end
            end
        end
    end

    # 
    # Add results together to get final FOIS for this job
    for (tconfig, tucks) in data
        if compress_twice
            σ[fock_σ][tconfig] = compress(nonorth_add(tucks, scr_v), thresh=thresh)
        else
            σ[fock_σ][tconfig] = nonorth_add(tucks, scr_v)
        end
    end

end
    





"""
    build_compressed_1st_order_state_old(ket_cts::SPTstate{T,N,R}, cluster_ops, clustered_ham; 
        thresh=1e-7, 
        max_number=nothing, 
        nbody=4, 
        compress_twice=true) where {T,N,R}
        thresh=1e-7, 
        max_number=nothing, 
        nbody=4) where {T,N}
Apply the Hamiltonian to `v` expanding into the uncompressed space.
This is done only partially, where each term is recompressed after being computed.
Lots of overhead probably from compression, but never completely uncompresses.

#Arguments
- `cts::SPTstate`: input state
- `cluster_ops`:
- `clustered_ham`: Hamiltonian
- `thresh`: Threshold for each HOSVD 
- `max_number`: max number of tucker factors kept in each HOSVD
- `nbody`: allows one to limit (max 4body) terms in the Hamiltonian considered
- `compress_twice`: Should we recompress after adding the tuckers together

#Returns
- `v1::SPTstate`

"""
function build_compressed_1st_order_state_old(ket_cts::SPTstate{T,N,R}, cluster_ops, clustered_ham; 
        thresh=1e-7, 
        max_number=nothing, 
        nbody=4, 
        compress_twice=true, 
        prescreen=false) where {T,N,R}
    #
    # Initialize data for our output sigma, which we will convert to a
    sig_cts = SPTstate(ket_cts.clusters, OrderedDict{FockConfig{N},OrderedDict{TuckerConfig{N},Tucker{T,N,R}} }(),  ket_cts.p_spaces, ket_cts.q_spaces)

    data = OrderedDict{FockConfig{N}, OrderedDict{TuckerConfig{N}, Vector{Tucker{T,N,R}} } }()
    blas_num_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    lk = ReentrantLock()

    #
    #   2body:
    #       term: H(IK,I'K') = h(pq) G1(pII') G3(qKK')     
    #       ket: C(I'J'K')  = c(i'j'k') U1(I'i') U2(J'j') U3(K'k')
    #
    #       sigma: Σ(IJ'K) = h(pq) X1(pIi') U2(J'j') X3(qKk') c(i'j'k')    diagonal in j'
    #           
    #           sigma is quadratic in cluster dimension. We can reduce that sometimes by 
    #           compressing X
    #
    #       X1(pIi') = x1(pii') V1(Ii)   where V1(Ii) are the left singular vectors of X1(I,pi') 
    #                                    such that when dim(p)*dim(i') < dim(I) we get exact reduction
    #       X3(qKk') = x3(qkk') V3(Kk)   
    #                                   
    #       Σ(IJ'K) = σ(ij'k) V1(Ii) U2(J'j') V3(Kk)
    #
    #       at this point, Σ has the form of an hosvd with σ as teh core tensor
    #
    #       σ(ij'k) =  h(pq) x1(pii') x3(qkk') c(i'j'k')
    #
    #
    nscr = 10
    scr = Vector{Vector{Vector{Float64}} }()
    for tid in 1:Threads.maxthreadid()
        tmp = Vector{Vector{Float64}}() 
        [push!(tmp, zeros(Float64,10000)) for i in 1:nscr]
        push!(scr, tmp)
    end
       
    #for (fock_trans, terms) in clustered_ham
    keys_to_loop = [keys(clustered_ham.trans)...]
        
    @printf(" %-50s%10i\n", "Number of tasks: ", length(keys_to_loop))
    @printf(" %-50s\n", "Compute tasks: ")
    stats = @timed Threads.@threads :static for fock_trans in keys_to_loop
        for (ket_fock, ket_tconfigs) in ket_cts
            terms = clustered_ham[fock_trans]

            #
            # new fock sector configuration
            sig_fock = ket_fock + fock_trans

            #
            # check that each cluster doesn't have too many/few electrons
            ok = true
            for ci in ket_cts.clusters
                if sig_fock[ci.idx][1] > length(ci) || sig_fock[ci.idx][2] > length(ci)
                    ok = false
                end
                if sig_fock[ci.idx][1] < 0 || sig_fock[ci.idx][2] < 0
                    ok = false
                end
            end
            ok == true || continue

            for term in terms

                #
                # only proceed if current term acts on no more than our requested max number of clusters
                length(term.clusters) <= nbody || continue
                for (ket_tconfig, ket_tuck) in ket_tconfigs

                    #
                    # find the sig TuckerConfigs reached by applying current Hamiltonian term to ket_tconfig.
                    #
                    # For example:
                    #
                    #   [(p'q), I, I, (r's), I ] * |P,Q,P,Q,P>  --> |X, Q, P, X, P>  where X = {P,Q}
                    #
                    #   This this term, will couple to 4 distinct tucker blocks (assuming each of the active clusters
                    #   have both non-zero P and Q spaces within the current fock sector, "sig_fock".
                    #
                    # We will loop over all these destination TuckerConfig's by creating the cartesian product of their
                    # available spaces, this list of which we will keep in "available".
                    #

                    available = [] # list of lists of index ranges, the cartesian product is the set needed
                    #
                    # for current term, expand index ranges for active clusters
                    for ci in term.clusters
                        tmp = []
                        if haskey(ket_cts.p_spaces[ci.idx], sig_fock[ci.idx])
                            push!(tmp, ket_cts.p_spaces[ci.idx][sig_fock[ci.idx]])
                        end
                        if haskey(ket_cts.q_spaces[ci.idx], sig_fock[ci.idx])
                            push!(tmp, ket_cts.q_spaces[ci.idx][sig_fock[ci.idx]])
                        end
                        push!(available, tmp)
                    end


                    #
                    # Now loop over cartesian product of available subspaces (those in X above) and
                    # create the target TuckerConfig and then evaluate the associated terms
                    for prod in Iterators.product(available...)
                        sig_tconfig = [ket_tconfig.config...]
                        for cidx in 1:length(term.clusters)
                            ci = term.clusters[cidx]
                            sig_tconfig[ci.idx] = prod[cidx]
                        end
                        sig_tconfig = TuckerConfig(sig_tconfig)

                        #
                        # the `term` has now coupled our ket TuckerConfig, to a sig TuckerConfig
                        # let's compute the matrix element block, then compress, then add it to any existing compressed
                        # coefficient tensor for that sig TuckerConfig.
                        #
                        # Both the Compression and addition takes a fair amount of work.

                        check_term(term, sig_fock, sig_tconfig, ket_fock, ket_tconfig) || continue

                        if prescreen
                            bound = calc_bound(term, cluster_ops,
                                            sig_fock, sig_tconfig,
                                            ket_fock, ket_tconfig, ket_tuck,
                                            prescreen=thresh)
                            bound == true || continue
                        end
                        

                        sig_tuck = form_sigma_block_expand(term, cluster_ops,
                                                           sig_fock, sig_tconfig,
                                                           ket_fock, ket_tconfig, ket_tuck,
                                                           max_number=max_number,
                                                           prescreen=thresh)

                        #if (term isa ClusteredTerm2B) && false
                        #    @btime del = form_sigma_block_expand2($term, $cluster_ops,
                        #                                        $sig_fock, $sig_tconfig,
                        #                                        $ket_fock, $ket_tconfig, $ket_tuck,
                        #                                        $scr[Threads.threadid()],
                        #                                        max_number=$max_number,
                        #                                        prescreen=$thresh)
                        #    #del = form_sigma_block_expand2(term, cluster_ops,
                        #    #                                    sig_fock, sig_tconfig,
                        #    #                                    ket_fock, ket_tconfig, ket_tuck,
                        #    #                                    scr[Threads.threadid()],
                        #    #                                    max_number=max_number,
                        #    #                                    prescreen=thresh)
                        #end

                        if length(sig_tuck) == 0
                            continue
                        end
                        if norm(sig_tuck) < thresh 
                            continue
                        end
                       
                        sig_tuck = compress(sig_tuck, thresh=thresh)

    
                        #sig_tuck = compress(sig_tuck, thresh=1e-16, max_number=max_number)

                        length(sig_tuck) > 0 || continue


                        begin
                            lock(lk)
                            try
                                if haskey(data, sig_fock)
                                    if haskey(data[sig_fock], sig_tconfig)
                                        #
                                        # In this case, our sigma vector already has a compressed coefficient tensor.
                                        # Consequently, we need to add these two together

                                        push!(data[sig_fock][sig_tconfig], sig_tuck)
                                        #sig_tuck = add([sig_tuck, sig_cts[sig_fock][sig_tconfig]])
                                        ##sig_tuck = compress(sig_tuck, thresh=thresh, max_number=max_number)
                                        #sig_cts[sig_fock][sig_tconfig] = sig_tuck

                                    else
                                        data[sig_fock][sig_tconfig] = [sig_tuck]
                                        #sig_cts[sig_fock][sig_tconfig] = sig_tuck
                                    end
                                else
                                    #sig_cts[sig_fock] = OrderedDict(sig_tconfig => sig_tuck)
                                    data[sig_fock] = OrderedDict(sig_tconfig => [sig_tuck])
                                end
                            finally
                                unlock(lk)
                            end
                        end

                    end

                end
            end
        end
    end
    BLAS.set_num_threads(blas_num_threads)

    @printf(" %-50s%10.6f seconds %10.2e Gb GC: %7.1e sec\n", "Time building compressed vector: ", stats.time, stats.bytes/1e9, stats.gctime)

    # return sig_cts, data, compress_twice, thresh, N
    _add_results!(sig_cts, data, compress_twice, thresh, N)
    # @profilehtml _add_results!(sig_cts, data, compress_twice, thresh, N)

    return sig_cts
end
   
function _add_results!(sig_cts, data, compress_twice, thresh, N)
    scr = Vector{Vector{Float64}}([Vector{Float64}([]) for i in 1:N]);

    flush(stdout)
    stats = @timed for (fock,tconfigs) in data
        for (tconfig, tuck) in tconfigs
            if haskey(sig_cts, fock)
                if compress_twice
                    sig_cts[fock][tconfig] = compress(nonorth_add(tuck, scr), thresh=thresh)

                    # sig_cts[fock][tconfig] = compress(nonorth_add(tuck), thresh=thresh)
                    # println("start")
                    # @time sig_cts[fock][tconfig] = compress(nonorth_add(tuck, scr), thresh=thresh)
                    # flush(stdout)
                    # println(length(sig_cts[fock][tconfig]))
                    # println(fock)
                    # println(tconfig)
                    # flush(stdout)
                    # println("finish")
                    # dim1 = length(sig_cts[fock][tconfig])
                    # sig_cts[fock][tconfig] = compress(sig_cts[fock][tconfig], thresh=thresh)
                    # dim2 = length(sig_cts[fock][tconfig])
                else
                    sig_cts[fock][tconfig] = nonorth_add(tuck, scr)
                end
            else
                if compress_twice
                    sig_cts[fock] = OrderedDict(tconfig => compress(nonorth_add(tuck, scr)))
                else
                    sig_cts[fock] = OrderedDict(tconfig => nonorth_add(tuck, scr))
                end
                # dim1 = length(sig_cts[fock][tconfig])
                # sig_cts[fock][tconfig] = compress(sig_cts[fock][tconfig], thresh=thresh)
                # dim2 = length(sig_cts[fock][tconfig])
            end
        end
    end
    @printf(" %-50s%10.6f seconds %10.2e Gb GC: %7.1e sec\n", "Add results together: ", stats.time, stats.bytes/1e9, stats.gctime)

    flush(stdout)


    return sig_cts

end

    
function do_fois_ci(ref::SPTstate{T,N,R}, cluster_ops, clustered_ham;
            H0          = "Hcmf",
            max_iter    = 50,
            nbody       = 4,
            thresh_foi  = 1e-6,
            thresh_pt   = 1e-5,
            tol         = 1e-5,
            pt          = false,
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
    @time e0, ref_vec = ci_solve(ref_vec, cluster_ops, clustered_ham, conv_thresh=tol)

    #
    # Get First order wavefunction
    println()
    println(" Compute FOIS. Reference space dim = ", length(ref_vec))
    @time pt1_vec  = build_compressed_1st_order_state(ref_vec, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi)
    for i in 1:R
        @printf(" Nick: %12.8f\n", sqrt.(orth_dot(pt1_vec,pt1_vec))[i])
    end
    project_out!(pt1_vec, ref)

    # 
    # Compress FOIS
    norm1 = sqrt.(orth_dot(pt1_vec, pt1_vec))
    dim1 = length(pt1_vec)
    pt1_vec = compress(pt1_vec, thresh=thresh_foi)
    norm2 = sqrt.(orth_dot(pt1_vec, pt1_vec))
    dim2 = length(pt1_vec)
    @printf(" %-50s%10i → %-10i (thresh = %8.1e)\n", "FOIS Compressed from: ", dim1, dim2, thresh_foi)
    for i in 1:R
        @printf(" %-50s%10.2e → %-10.2e (thresh = %8.1e)\n", "Norm of |1>: ",norm1[i], norm2[i], thresh_foi)
    end
    for i in 1:R
        @printf(" %-50s%10.6f\n", "Overlap between <1|0>: ", nonorth_dot(pt1_vec, ref_vec, verbose=0)[i])
    end

    nonorth_add!(ref_vec, pt1_vec)
    # 
    # Solve for first order wavefunction 
    println(" Compute CI energy in the space = ", length(ref_vec))
    if pt==true
        pt1_vec, e_pt2= hylleraas_compressed_mp2(pt1_vec, ref_vec, cluster_ops, clustered_ham; tol=tol, max_iter=max_iter, H0=H0)
        nonorth_add!(ref_vec, pt1_vec)
        ref_vec = compress(ref_vec, thresh=thresh_pt)
    end
    eci, ref_vec = ci_solve(ref_vec, cluster_ops, clustered_ham;)
    for i in 1:R
        @printf(" E(Ref)   for %ith state   = %12.8f\n",i, e0[i])
        @printf(" E(CI) tot for %ith state = %12.8f\n",i, eci[i])
    end
    return eci, ref_vec 
end
 
"""
    do_fois_cepa(ref::SPTstate{T,N,R}, cluster_ops, clustered_ham;
                 max_iter=20,
                 cepa_shift="cepa",
                 cepa_mit=30,
                 nbody=4,
                 thresh_foi=1e-6,
                 tol=1e-5,
                 compress_type="matvec",
                 prescreen=false,
                 verbose=true) where {T,N,R}

Perform Coupled Electron Pair Approximation (CEPA) calculations.

# Arguments
            - `ref::SPTstate{T,N,R}`: The reference `SPTstate` object representing the wavefunction.
            - `cluster_ops`: Operators related to the cluster.
            - `clustered_ham`: Clustered Hamiltonian.
            - `max_iter::Int`: Maximum number of iterations for the CEPA solver. Default is 20.
            - `cepa_shift::String`: Type of CEPA calculation. Default is "cepa".
            - `cepa_mit::Int`: Maximum number of iterations for the CEPA method. Default is 30.
            - `nbody::Int`: Number of bodies for the first-order interaction space. Default is 4.
            - `thresh_foi::Float64`: Threshold for the first-order interaction space. Default is 1e-6.
            - `tol::Float64`: Convergence threshold. Default is 1e-5.
            - `compress_type::String`: Type of compression for the first-order interaction space. Default is "matvec".
            - `prescreen::Bool`: Whether to prescreen interactions. Default is false.
            - `verbose::Bool`: Whether to print verbose output. Default is true.

# Returns
            - `Vector{Float64}`: A vector containing the CEPA correlation energies.
"""


function do_fois_cepa(ref::SPTstate{T,N,R}, cluster_ops, clustered_ham;
    max_iter     = 20,
    cepa_shift   = "cepa",
    cepa_mit     = 30,
    nbody        = 4,
    thresh_foi   = 1e-6,
    tol          = 1e-5,
    compress_type = "matvec",
    prescreen    = false,
    compute_S2   = false,
    verbose      = true,
    solver       = :minres) where {T,N,R}   
 
    @printf("\n-------------------------------------------------------\n")
    @printf(" Do CEPA\n")
    @printf("   thresh_foi              = %-8.1e\n", thresh_foi)
    @printf("   nbody                   = %-i\n", nbody)
    @printf("\n")
    @printf("   Length of Reference     = %-i\n", length(ref))
    @printf("   Calculation type        = %s\n", cepa_shift)
    @printf("   Compression type        = %s\n", compress_type)
    @printf("   Solver                  = %s\n", solver)
    @printf("\n-------------------------------------------------------\n")
 
    println()
    ref_vec = deepcopy(ref)
    @printf(" Solve zeroth-order problem. Dimension = %10i\n", length(ref_vec))
    @time e0, ref_vec = ci_solve(ref_vec, cluster_ops, clustered_ham, conv_thresh=tol)
 
    println()
    println(" Compute FOIS. Reference space dim = ", length(ref_vec))
    @time pt1_vec = build_compressed_1st_order_state(ref_vec, cluster_ops, clustered_ham,
                                                      nbody=nbody, thresh=thresh_foi,
                                                      prescreen=prescreen)
    project_out!(pt1_vec, ref)
 
    if compress_type == "pt_vec"
        println()
        println(" Compute PT vector. Reference space dim = ", length(ref_vec))
        pt1_vec, e_pt2 = hylleraas_compressed_mp2(pt1_vec, ref_vec, cluster_ops, clustered_ham;
                                                   tol=tol, do_pt=true)
    end
 
    dim1    = length(pt1_vec)
    pt1_vec = compress(pt1_vec, thresh=thresh_foi)
    dim2    = length(pt1_vec)
    @printf(" %-50s%10i → %-10i (thresh = %8.1e)\n", "FOIS Compressed from: ", dim1, dim2, thresh_foi)
    @printf(" %-50s", "Overlap between <1|0>: ")
    ovlp = nonorth_dot(pt1_vec, ref_vec, verbose=0)
    [@printf("%10.6f", ovlp[r]) for r in 1:R]
    println()
 
    println()
    cepa_vec   = deepcopy(pt1_vec)
    e_cepa_vec = []
 
    println(" Do CEPA: Dim = ", length(cepa_vec))
    cepa_vec_f = deepcopy(ref_vec)
 
    for i in 1:R
        ref_vec_i  = TPSChem.SPTstate(ref_vec, i)
        cepa_vec_i = TPSChem.SPTstate(cepa_vec, i)
        zero!(cepa_vec_i)
        println(" Do CEPA: Dim = ", length(cepa_vec_i))
 
        @time e_cepa, x_cepa = tucker_cepa_solve(
            ref_vec_i, cepa_vec_i, cluster_ops, clustered_ham,
            cepa_shift, cepa_mit;
            tol      = tol,
            max_iter = max_iter,
            verbose  = verbose,
            solver   = solver)    # ← passed through
 
        @printf(" E(cepa) corr =                 %12.8f\n", e_cepa[1])
        @printf(" X(cepa) norm =                 %12.8f\n", sqrt(orth_dot(x_cepa, x_cepa)[1]))
        flush(stdout)
        if compute_S2==true
            clustered_S2 = extract_S2(x_cepa.clusters)
            @printf(" %-50s", "Compute <S^2>: ")
            flush(stdout)
            tmp = deepcopy(x_cepa)
            zero!(tmp)
            @time build_sigma_cepa!(tmp, x_cepa, cluster_ops, clustered_S2)
            s2 = orth_dot(tmp, x_cepa)
            flush(stdout)
            @printf(" %5s %12s %12s\n", "Root", "Energy", "S2")
            @printf("%5i %12.8f %12.8f\n", i, e_cepa[1], s2[1])
        end
        push!(e_cepa_vec, e_cepa[1])
    end
 
    return e_cepa_vec
end
"""
    do_fois_cepa(ref::SPTstate{T,N,1}, cluster_ops, clustered_ham;
                 max_iter=20,
                 cepa_shift="cepa",
                 cepa_mit=30,
                 nbody=4,
                 thresh_foi=1e-6,
                 tol=1e-5,
                 compress_type="matvec",
                 prescreen=false,
                 verbose=true) where {T,N}

Perform Coupled Electron Pair Approximation (CEPA) calculations.

# Arguments
        - `ref::SPTstate{T,N,1}`: The reference `SPTstate` object representing the wavefunction with only one root.
        - `cluster_ops`: Operators related to the cluster.
        - `clustered_ham`: Clustered Hamiltonian.
        - `max_iter::Int`: Maximum number of iterations for the CEPA solver. Default is 20.
        - `cepa_shift::String`: Type of CEPA calculation. Default is "cepa".
        - `cepa_mit::Int`: Maximum number of iterations for the CEPA method. Default is 30.
        - `nbody::Int`: Number of bodies for the first-order interaction space. Default is 4.
        - `thresh_foi::Float64`: Threshold for the first-order interaction space. Default is 1e-6.
        - `tol::Float64`: Convergence threshold. Default is 1e-5.
        - `compress_type::String`: Type of compression for the first-order interaction space. Default is "matvec".
        - `prescreen::Bool`: Whether to prescreen interactions. Default is false.
        - `verbose::Bool`: Whether to print verbose output. Default is true.

# Returns
        - `Tuple{Float64, SPTstate{T,N,1}}`: The CEPA correlation energy and the updated wavefunction.
"""

function do_fois_cepa(ref::SPTstate{T,N,1}, cluster_ops, clustered_ham;
                        max_iter=20,
                        cepa_shift="cepa",
                        cepa_mit=30,
                        nbody=4,
                        thresh_foi=1e-6,
                        tol=1e-5,
                        compress_type="matvec",
                        prescreen=false,
                        verbose=true) where {T,N}
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
    @time e0, ref_vec = ci_solve(ref_vec, cluster_ops, clustered_ham, conv_thresh=tol)

    #
    # Get First order wavefunction
    println()
    println(" Compute FOIS. Reference space dim = ", length(ref_vec))
    @time pt1_vec = build_compressed_1st_order_state(ref_vec, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi, prescreen=prescreen)

    project_out!(pt1_vec, ref)

    if compress_type == "pt_vec"
        println()
        println(" Compute PT vector. Reference space dim = ", length(ref_vec))
        pt1_vec, e_pt2 = hylleraas_compressed_mp2(pt1_vec, ref_vec, cluster_ops, clustered_ham; tol=tol, do_pt=true)
    end


    # 
    # Compress FOIS
    # norm1 = orth_dot(pt1_vec, pt1_vec)
    dim1 = length(pt1_vec)
    pt1_vec = compress(pt1_vec, thresh=thresh_foi)
    # norm2 = orth_dot(pt1_vec, pt1_vec)
    dim2 = length(pt1_vec)
    @printf(" %-50s%10i → %-10i (thresh = %8.1e)\n", "FOIS Compressed from: ", dim1, dim2, thresh_foi)
    #@printf(" %-50s%10.2e → %-10.2e (thresh = %8.1e)\n", "Norm of |1>: ",norm1, norm2, thresh_foi)
    @printf(" %-50s", "Overlap between <1|0>: ")
    ovlp = nonorth_dot(pt1_vec, ref_vec, verbose=0)
    [@printf("%10.6f", ovlp[1])]
    println()

    # 
    # Solve CEPA 
    println()
    cepa_vec = deepcopy(pt1_vec)
    zero!(cepa_vec)
    println(" Do CEPA: Dim = ", length(cepa_vec))
    @time e_cepa, x_cepa = tucker_cepa_solve(ref_vec, cepa_vec, cluster_ops, clustered_ham, cepa_shift, cepa_mit, tol=tol, max_iter=max_iter, verbose=verbose)

    @printf(" E(cepa) corr =                 %12.8f\n", e_cepa[1])
    @printf(" X(cepa) norm =                 %12.8f\n", sqrt(orth_dot(x_cepa, x_cepa)[1]))
    nonorth_add!(x_cepa, ref_vec)
    orthonormalize!(x_cepa)
    return e_cepa, x_cepa
end


"""
    project_out!(v::SPTstate, w::SPTstate; thresh=1e-16)

Project w out of v 
|v'> = |v> - |w><w|v>
"""
function project_out!(v::SPTstate{T,N,Rv}, w::SPTstate{T,N,Rw}; thresh=1e-16) where {T,N,Rv,Rw}

    #S = nonorth_overlap(w,v)
    #wtmp = deepcopy(w)
    #set_vector!(wtmp, -1.0*get_vector(w)*S)
    #nonorth_add!(v,wtmp)

    for rw in 1:Rw
        for (fock,tconfigs) in v 
            for (tconfig, tuck) in tconfigs
                if haskey(w, fock)
                    if haskey(w[fock], tconfig)
                        w_tuck = w[fock][tconfig]

                        ww = sum(w_tuck.core[rw] .* w_tuck.core[rw])
                        for rv in 1:Rv
                            ovlp = nonorth_dot(tuck, w_tuck, rv, rw) / ww
                            tmp = scale(w_tuck, -1.0 * ovlp)
                            v[fock][tconfig] = nonorth_add(tuck, tmp, thresh=thresh)
                        end
                    end
                end
            end
        end
    end
end
