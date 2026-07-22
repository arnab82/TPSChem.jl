



"""
    QCBase.compute_energy(ints::InCoreInts{T}, rdm1s::Dict{Integer,RDM1{T}}, rdm2s::Dict{Integer,RDM2{T}}, clusters::Vector{MOCluster}; verbose=0) where T

Compute the energy of a cluster-wise product state (CMF),
specified by a list of 1 and 2 particle rdms local to each cluster.
This method uses the full system integrals.

- `ints::InCoreInts`: integrals for full system
- `rdm1s`: dictionary (`ci.idx => RDM1`) of 1rdms from each cluster
- `rdm2s`: dictionary (`ci.idx => RDM2`) of 2rdms from each cluster
- `clusters::Vector{MOCluster}`: vector of cluster objects

return the total CMF energy
"""
function QCBase.compute_energy(ints::InCoreInts{T}, rdm1s::Dict{Integer,RDM1{T}}, rdm2s::Dict{Integer,RDM2{T}}, clusters::Vector{MOCluster}; verbose=0) where T
    e1 = zeros((length(clusters),1))
    e2 = zeros((length(clusters),length(clusters)))
    for ci in clusters
        noi = n_orb(ints)
        ints_i = subset(ints, ci.orb_list)

        e1[ci.idx] = compute_energy(ints_i, rdm1s[ci.idx], rdm2s[ci.idx])
    end
    for ci in clusters
        for cj in clusters
            if ci.idx >= cj.idx
                continue
            end
            v_pqrs = ints.h2[ci.orb_list, ci.orb_list, cj.orb_list, cj.orb_list]
            v_psrq = ints.h2[ci.orb_list, cj.orb_list, cj.orb_list, ci.orb_list]
            tmp = 0


            @tensor begin
                tmp  = v_pqrs[p,q,r,s] * rdm1s[ci.idx].a[p,q] * rdm1s[cj.idx].a[r,s]
                tmp -= v_psrq[p,s,r,q] * rdm1s[ci.idx].a[p,q] * rdm1s[cj.idx].a[r,s]

                tmp += v_pqrs[p,q,r,s] * rdm1s[ci.idx].b[p,q] * rdm1s[cj.idx].b[r,s]
                tmp -= v_psrq[p,s,r,q] * rdm1s[ci.idx].b[p,q] * rdm1s[cj.idx].b[r,s]

                tmp += v_pqrs[p,q,r,s] * rdm1s[ci.idx].a[p,q] * rdm1s[cj.idx].b[r,s]

                tmp += v_pqrs[p,q,r,s] * rdm1s[ci.idx].b[p,q] * rdm1s[cj.idx].a[r,s]
            end


            e2[ci.idx, cj.idx] = tmp
        end
    end
    if verbose>1
        for ei = 1:length(e1)
            @printf(" MOCluster %3i E =%12.8f\n",ei,e1[ei])
        end
    end
    return ints.h0 + sum(e1) + sum(e2)
end


"""
    cmf_ci_iteration(ints::InCoreInts{T}, clusters::Vector{MOCluster}, in_rdm1::RDM1{T}, fspace; 
                          use_pyscf = false, 
                          verbose   = 1, 
                          sequential= false, 
                          spin_avg  = true, 
                          tol_ci    = 1e-8,
                          maxiter_ci    = 100
    ) where T

Perform single CMF-CI iteration, returning new energy, and density
"""
function cmf_ci_iteration(ints::InCoreInts{T}, clusters::Vector{MOCluster}, in_rdm1::RDM1{T}, fspace;
                          use_pyscf = false,
                          use_nosym = false,
                          verbose   = 1,
                          sequential= false,
                          spin_avg  = true,
                          tol_ci    = 1e-8,
                          maxiter_ci    = 100
    ) where T
    rdm1 = deepcopy(in_rdm1)
    rdm1_dict = Dict{Integer,RDM1{T}}()
    rdm2_dict = Dict{Integer,RDM2{T}}()
    
    for ci in clusters
        flush(stdout)

        ansatz = FCIAnsatz(length(ci), fspace[ci.idx][1],fspace[ci.idx][2])
        verbose < 2 || display(ansatz)
        ints_i = subset(ints, ci, rdm1)

        d1a = rdm1.a[ci.orb_list, ci.orb_list]
        d1b = rdm1.b[ci.orb_list, ci.orb_list]

        na = fspace[ci.idx][1]
        nb = fspace[ci.idx][2]

        no = length(ci)

        d1 = RDM1(no)
        d2 = RDM2(no)

        e = 0.0
        if ansatz.dim == 1
       
            da = zeros(T, no, no)
            db = zeros(T, no, no)
            if ansatz.na == no
                da = Matrix(1.0I, no, no)
            end
            if ansatz.nb == no
                db = Matrix(1.0I, no, no)
            end
            
            if spin_avg
                da = .5*(da + db)
                db .= da
            end

            d1 = RDM1(da,db)
            d2 = RDM2(d1)
            
            e = compute_energy(ints_i, d1)
            verbose < 2 || @printf(" Slater Det Energy: %12.8f\n", e)
        else
            #
            # run PYSCF FCI
            #e, d1a,d1b, d2 = pyscf_fci(ints_i,fspace[ci.idx][1],fspace[ci.idx][2], verbose=verbose)
          
            if use_pyscf
                # implemented in ext/TPSChemPyCallExt.jl; requires PyCall + pyscf
                e, d1, d2 = pyscf_fci_rdm12s(ints_i, na, nb, tol_ci, maxiter_ci; use_nosym=use_nosym)
            else
                solver = SolverSettings(verbose=1, tol=tol_ci, maxiter=maxiter_ci)
                solution = solve(ints_i, ansatz, solver)
                d1a, d1b, d2aa, d2bb, d2ab = compute_1rdm_2rdm(solution)

                verbose < 2 || display(solution)

                #            if spin_avg
                #                v = solution.vectors[:,1]
                #                v = reshape(v, (ansatz.dima, ansatz.dimb))
                #                v = Matrix(v')
                #                v = reshape(v, (ansatz.dima * ansatz.dimb, 1))
                #        
                #                ansatz_flipped = FCIAnsatz(length(ci), fspace[ci.idx][2],fspace[ci.idx][1])
                #                solution_flipped = Solution(ansatz_flipped, solution.energies, v)
                #                #solution_flipped = solve(ints_i, ansatz_flipped, solver)
                #                _d1a, _d1b, _d2aa, _d2bb, _d2ab = compute_1rdm_2rdm(solution_flipped)
                #               
                #                d1a = (d1a + _d1a) * .5
                #                d1b = (d1b + _d1b) * .5
                #                d2aa = (d2aa + _d2aa) * .5
                #                d2ab = (d2ab + _d2ab) * .5
                #                d2bb = (d2bb + _d2bb) * .5
                #            end

                d1 = RDM1(d1a, d1b)
                d2 = RDM2(d2aa, d2ab, d2bb)
            end

        end

        if spin_avg
            d1.a .= .5*(d1.a + d1.b)
            d1.b .= d1.a
            d2.aa .= .5*(d2.aa + d2.bb)
            d2.bb .= d2.aa
            ab = reshape(d2.ab, (no*no, no*no))
            ab .= (ab .+ ab').*.5
            ab = reshape(ab, (no, no, no, no))
            d2.ab .= ab
        end
        
        rdm1_dict[ci.idx] = d1
        rdm2_dict[ci.idx] = d2

        if sequential==true
            rdm1.a[ci.orb_list,ci.orb_list] = d1.a
            rdm1.b[ci.orb_list,ci.orb_list] = d1.b
        end
    end
    e_curr = compute_energy(ints, rdm1_dict, rdm2_dict, clusters, verbose=verbose)
    
    if verbose > 0
        @printf(" CMF-CI Curr: Elec %12.8f Total %12.8f\n", e_curr-ints.h0, e_curr)
    end

    return e_curr, rdm1_dict, rdm2_dict
end



"""
    cmf_ci(ints, clusters, fspace, in_rdm1::RDM1; 
                maxiter_ci  = 100, 
                maxiter_d1  = 20, 
                tol_d1      = 1e-6, 
                tol_ci      = 1e-8, 
                verbose     = 1,
                sequential  = false)

Optimize the 1RDM for CMF-CI

#Arguments
- `ints::InCoreInts`: integrals for full system
- `clusters::Vector{MOCluster}`: vector of cluster objects
- `fspace::Vector{Vector{Integer}}`: vector of particle number occupations for each cluster specifying the sectors of fock space 
- `in_rdm1`: initial guess for 1particle density matrix
- `tol_d1`: Convergence threshold for change in density 
- `tol_ci`: Convergence threshold for the cluster CI problems 
- `maxiter_d1`: Max number of iterations for density optimization
- `maxiter_ci`: Max number of iterations for CI diagonalization
- `sequential`: Use the density matrix of the previous cluster in a cMF iteration to form effective integrals. Improves comvergence, may depend on cluster orderings   
- `verbose`: Printing level 
    
# Returns
- `e`: Energy
- `rdm1_dict`: Dictionary of 1RDMs; cluster index --> RDM1
- `rdm2_dict`: Dictionary of 2RDMs; cluster index --> RDM2
"""
function cmf_ci(ints, clusters, fspace, in_rdm1::RDM1;
                maxiter_ci  = 100,
                maxiter_d1  = 20,
                tol_d1      = 1e-6,
                tol_ci      = 1e-8,
                verbose     = 1,
                use_pyscf   = true,
                use_nosym   = false,
                sequential  = false)
    rdm1 = deepcopy(in_rdm1)
    energies = []
    e_prev = 0

    rdm1_dict = 0
    rdm2_dict = 0
    rdm1_dict = Dict{Integer,Array}()
    rdm2_dict = Dict{Integer,Array}()
    # rdm2_dict = Dict{Integer, Array}()
    for iter = 1:maxiter_d1
        if verbose > 1
            println()
            println(" ------------------------------------------ ")
            println(" CMF CI Iter: ", iter)
            println(" ------------------------------------------ ")
        end
        e_curr, rdm1_dict, rdm2_dict = cmf_ci_iteration(ints, clusters, rdm1, fspace,
                                                        maxiter_ci  = maxiter_ci,
                                                        tol_ci      = tol_ci,
                                                        verbose     = verbose,
                                                        use_pyscf   = use_pyscf,
                                                        use_nosym   = use_nosym,
                                                        sequential  = sequential
                                                       )
        rdm1_curr = assemble_full_rdm(clusters, rdm1_dict)

        append!(energies,e_curr)
        error = (rdm1_curr.a+rdm1_curr.b) - (rdm1.a+rdm1.b)
        d_err = norm(error)
        e_err = e_curr-e_prev
        if verbose>1
            @printf(" CMF-CI Energy: %12.8f | Change: RDM: %6.1e Energy %6.1e\n\n", e_curr, d_err, e_err)
        end
        e_prev = e_curr*1
        rdm1 = rdm1_curr
        if d_err < tol_d1 
            if verbose>1
                @printf("*CMF-CI: Elec %12.8f Total %12.8f\n", e_curr-ints.h0, e_curr)
            end
            break
        end
    end
    if verbose>0
        println(" Energy per Iteration:")
        for i in energies
            @printf(" Elec: %12.8f Total: %12.8f\n", i-ints.h0, i)
        end
    end
    return e_prev, rdm1_dict, rdm2_dict
end










"""
    cmf_oo(ints::InCoreInts{T}, clusters::Vector{MOCluster}, fspace, dguess::RDM1{T}; 
                max_iter_oo=100, 
                max_iter_ci=100, 
                gconv=1e-6, 
                verbose=0, 
                method="bfgs", 
                alpha=nothing,
                sequential=false) where T

Do CMF with orbital optimization

#Arguments
- `ints::InCoreInts`: integrals for full system
- `clusters::Vector{MOCluster}`: vector of cluster objects
- `fspace::Vector{Vector{Integer}}`: vector of particle number occupations for each cluster specifying the sectors of fock space 
- `dguess_a`: initial guess for 1particle density matrix
- `max_iter_oo`: Max iter for the orbital optimization iterations 
- `max_iter_ci`: Max iter for the cmf iteration for the cluster states 
- `gconv`: Convergence threshold for change in gradient of energy 
- `sequential`: If true use the density matrix of the previous cluster in a cMF iteration to form effective integrals. Improves comvergence, may depend on cluster orderings   
- `verbose`: Printing level 
- `method`: optimization method
"""
function cmf_oo(ints::InCoreInts{T}, clusters::Vector{MOCluster}, fspace, dguess::RDM1{T}; 
                max_iter_oo=100, 
                max_iter_ci=100, 
                gconv=1e-6, 
                verbose=0, 
                method="bfgs", 
                alpha=nothing,
                use_pyscf=true,
                sequential=false) where T
    norb = size(ints.h1)[1]

    #   
    #   Initialize optimization data
    #
    e_prev = 0
    e_tmp  = 0
    e_curr = 0

    g_prev = 0
    g_tmp  = 0
    g_curr = 0
  
    d1_prev = deepcopy(dguess) 
    d1_tmp  = deepcopy(dguess) 
    d1_curr = deepcopy(dguess) 

    iter = 0
    kappa = zeros(norb*(norb-1)÷2)

    #
    #   Define Objective function (energy)
    #
    function f(k)
        K = unpack_gradient(k, norb)
        U = exp(K)
        ints_tmp = orbital_rotation(ints,U)
        e, rdm1_dict, _ = cmf_ci(ints_tmp, clusters, fspace, orbital_rotation(d1_curr, U), 
                                         tol_d1=gconv/10.0, 
                                         verbose=0, 
                                         sequential=sequential,
                                         use_pyscf=use_pyscf)
        
        d1_tmp = assemble_full_rdm(clusters, rdm1_dict)
        d1_tmp = orbital_rotation(d1_tmp, U')
        e_tmp = e
        
        return e
    end

    #
    #   Define Gradient function
    #
    function g(kappa)
        norb = n_orb(ints)
        K = unpack_gradient(kappa, norb)
        U = exp(K)
       
        ints_tmp = orbital_rotation(ints,U)
        e, rdm1_dict, rdm2_dict = cmf_ci(ints_tmp, clusters, fspace, orbital_rotation(d1_curr,U), 
                                         tol_d1=gconv/10.0, verbose=verbose)

        gd1, gd2 = assemble_full_rdm(clusters, rdm1_dict, rdm2_dict)
        gout = build_orbital_gradient(ints_tmp, gd1, gd2)
        g_tmp = norm(gout)
        return gout
    end

    #   
    #   Define Callback for logging and checking for convergence
    #
    function callback(k)
       
        # reset initial RDM guess for each cmf_ci
        d1_curr  = deepcopy(d1_tmp) 
        e_curr   = e_tmp
        g_curr   = g_tmp

        iter += 1
        if (g_curr < gconv) 
            @printf("*ooCMF Iter: %4i Total= %16.12f Active= %16.12f G= %12.2e\n", iter, e_curr, e_curr-ints.h0, g_curr)
            return true 
        else
            @printf(" ooCMF Iter: %4i Total= %16.12f Active= %16.12f G= %12.2e\n", iter, e_curr, e_curr-ints.h0, g_curr)
            return false 
        end
    end

    if (method=="bfgs") || (method=="cg")
        optmethod = BFGS()
        if method=="cg"
            optmethod = ConjugateGradient()
        end

        options = Optim.Options(
                                callback = callback, 
                                g_tol=gconv,
                                iterations=max_iter_oo,
                               )

        res = optimize(f, g, kappa, optmethod, options; inplace = false )
        summary(res)
        e = Optim.minimum(res)
        display(res)
        @printf("*ooCMF %12.8f \n", e)

        kappa = Optim.minimizer(res)
        K = unpack_gradient(kappa, norb)
        U = exp(K)
        d1 = orbital_rotation(d1_curr, U)
        return e, U, d1

    elseif method=="gd"
        res = do_gd(f, g, callback, kappa, gconv, max_iter_oo, method)
    elseif method=="diis"
        res = do_diis(f, g, callback, kappa, gconv, max_iter_oo, method)
    end

end



function do_gd(f, g, callback, kappa, gconv,max_iter, method)
    throw("Not yet implemented")
end


function do_diis(f,g,callback,kappa, gconv,max_iter, method)
    throw("Not yet implemented")
end


"""
    unpack_gradient(kappa,norb)
"""
function unpack_gradient(kappa,norb)
    length(kappa) == norb*(norb-1)÷2 || throw(DimensionMismatch)
    K = zeros(norb,norb)
    ind = 1
    for i in 1:norb
        for j in i+1:norb
            K[i,j] = kappa[ind]
            K[j,i] = -kappa[ind]
            ind += 1
        end
    end
    return K
end
"""
    pack_gradient(K,norb)
"""
function pack_gradient(K,norb)
    length(K) == norb*norb || throw(DimensionMismatch)
    kout = zeros(norb*(norb-1)÷2)
    ind = 1
    for i in 1:norb
        for j in i+1:norb
            kout[ind] = K[i,j]
            ind += 1
        end
    end
    return kout
end



"""
    assemble_full_rdm(clusters::Vector{MOCluster}, rdm1s::Dict{Integer, RDM1{T}}) where T

Return spin summed 1 and 2 RDMs
"""
function assemble_full_rdm(clusters::Vector{MOCluster}, rdm1s::Dict{Integer, RDM1{T}}) where T
    norb = sum([length(i) for i in clusters])
    d1 = RDM1(norb)

    for ci in clusters
        d1.a[ci.orb_list, ci.orb_list] .= rdm1s[ci.idx].a
        d1.b[ci.orb_list, ci.orb_list] .= rdm1s[ci.idx].b
    end
    
    return d1
end

"""
    assemble_full_rdm(clusters::Vector{MOCluster}, rdm1s::Dict{Integer, Array}, rdm2s::Dict{Integer, Array})

Return full system 1 and 2 RDMs
"""
function assemble_full_rdm(clusters::Vector{MOCluster}, rdm1s::Dict{Integer, RDM1{T}}, rdm2s::Dict{Integer, RDM2{T}}) where T
    norb = sum([length(i) for i in clusters])

    rdm1 = RDM1(norb)
    rdm2 = RDM2(norb)
    
    for ci in clusters
        rdm1.a[ci.orb_list, ci.orb_list] .= rdm1s[ci.idx].a
        rdm1.b[ci.orb_list, ci.orb_list] .= rdm1s[ci.idx].b
    end
   
    rdm2 = RDM2(rdm1)
    for ci in clusters
        rdm2.aa[ci.orb_list, ci.orb_list, ci.orb_list, ci.orb_list] .= rdm2s[ci.idx].aa
        rdm2.ab[ci.orb_list, ci.orb_list, ci.orb_list, ci.orb_list] .= rdm2s[ci.idx].ab
        rdm2.bb[ci.orb_list, ci.orb_list, ci.orb_list, ci.orb_list] .= rdm2s[ci.idx].bb
    end
    return rdm1, rdm2
end



"""
    orbital_objective_function(ints, clusters, kappa, fspace, da, db; 
                                    ci_conv     = 1e-9,
                                    sequential  = false,
                                    verbose     = 1)
Objective function to minimize in OO-CMF
"""
function orbital_objective_function(ints, clusters, kappa, fspace, rdm::RDM1; 
                                    ci_conv     = 1e-9,
                                    sequential  = false,
                                    verbose     = 0)

    norb = n_orb(ints)
    K = unpack_gradient(kappa, norb)
    U = exp(K)
    ints2 = orbital_rotation(ints,U)
    d1 = orbital_rotation(rdm,U)
    e, rdm1_dict, rdm2_dict = cmf_ci(ints2, clusters, fspace, d1,
        tol_ci=ci_conv,
        verbose=verbose,
        sequential=sequential)
    return e
end

"""
    orbital_gradient_numerical(ints, clusters, kappa, fspace, da, db; 
                                    gconv = 1e-8, 
                                    verbose = 1,
                                    stepsize = 1e-6)
Compute orbital gradient with finite difference
"""
function orbital_gradient_numerical(ints, clusters, kappa, fspace, d::RDM1; 
                                    ci_conv = 1e-10, 
                                    verbose = 0,
                                    stepsize = 1e-6)
    grad = zeros(size(kappa))
    for (ii,i) in enumerate(kappa)
        
        #ii == 2 || continue
    
        k1 = deepcopy(kappa)
        k1[ii] += stepsize
        e1 = orbital_objective_function(ints, clusters, k1, fspace, d, ci_conv=ci_conv, verbose=verbose) 
        
        k2 = deepcopy(kappa)
        k2[ii] -= stepsize
        e2 = orbital_objective_function(ints, clusters, k2, fspace, d, ci_conv=ci_conv, verbose=verbose) 
        
        grad[ii] = (e1-e2)/(2*stepsize)
        #println(e1)
    end
    return grad
end

"""
    cmf_oo_gd( ints_in::InCoreInts{T}, clusters::Vector{MOCluster}, fspace, dguess::RDM1{T}; 
                    maxiter_oo      = 100, 
                    maxiter_ci      = 100, 
                    maxiter_d1      = 100, 
                    tol_oo          = 1e-6, 
                    tol_d1          = 1e-7, 
                    tol_ci          = 1e-8, 
                    verbose         = 0, 
                    alpha           = .1,
                    zero_intra_rots = true,
                    sequential      = false) where T

Do CMF with orbital optimization

# Arguments

- `ints::InCoreInts`: integrals for full system
- `clusters::Vector{MOCluster}`: vector of cluster objects
- `fspace::Vector{Vector{Integer}}`: vector of particle number occupations for each cluster specifying the sectors of fock space 
- `dguess`: initial guess for 1particle density matrix
- `maxiter_oo`: Max iter for the orbital optimization iterations 
- `maxiter_d1`: Max iter for the cmf iteration for the 1RDM 
- `maxiter_ci`: Max iter for the CI diagonalization of the cluster states 
- `tol_oo`: Convergence threshold for change in orbital gradient 
- `tol_ci`: Convergence threshold for the cluster CI problems 
- `tol_d1`: Convergence threshold for the CMF 1RDM 
- `sequential`: If true use the density matrix of the previous cluster in a cMF iteration to form effective integrals. Improves comvergence, may depend on cluster orderings   
- `verbose`: Printing level 

# Returns

- `e`: Energy
- `U::Matrix`: Orbital rotation matrix from input to output orbitals
- `d1::RDM1`: Optimized 1RDM in the optimized orbital basis
"""
function cmf_oo_gd( ints_in::InCoreInts{T}, clusters::Vector{MOCluster}, fspace, dguess::RDM1{T}; 
                    maxiter_oo      = 100, 
                    maxiter_ci      = 100, 
                    maxiter_d1      = 100, 
                    tol_oo          = 1e-6, 
                    tol_d1          = 1e-7, 
                    tol_ci          = 1e-8, 
                    verbose         = 0, 
                    alpha           = .1,
                    zero_intra_rots = true,
                    sequential      = false
    ) where T
    #={{{=#
    ints = deepcopy(ints_in)
    norb = n_orb(ints)
    d1   = deepcopy(dguess) 
    U    = Matrix(1.0I, norb, norb)
    e    = 0.0

    function step!(ints, d1, k)
        norb = n_orb(ints)
        K = unpack_gradient(k, norb)
        if zero_intra_rots
            # Remove intracluster rotations
            for ci in clusters
                K[ci.orb_list, ci.orb_list] .= 0
            end
        end
       
        Ui = exp(K)
        
        tmp = orbital_rotation(ints,Ui)
        ints.h1 .= tmp.h1
        ints.h2 .= tmp.h2

        tmp = orbital_rotation(d1,Ui)
        d1.a .= tmp.a
        d1.b .= tmp.b

        e, rdm1_dict, rdm2_dict = cmf_ci(ints, clusters, fspace, d1, 
                                         maxiter_d1 = maxiter_d1, 
                                         maxiter_ci = maxiter_ci, 
                                         tol_d1     = tol_d1, 
                                         tol_ci     = tol_ci, 
                                         verbose    = 0, 
                                         sequential = sequential)
        
        gd1, gd2 = assemble_full_rdm(clusters, rdm1_dict, rdm2_dict)
        g = build_orbital_gradient(ints, gd1, gd2)
        return e, g, Ui, gd1
    end


    # Compute initial gradient
    converged = false
    step_i = zeros(norb*(norb-1)÷2) 
    for i in 1:maxiter_oo
        ei, gi, Ui, d1 = step!(ints, d1, step_i)
        step_i = -alpha*gi
        e = ei
        U = U*Ui

        converged = norm(gi) < tol_oo 
        if converged
            @printf("*Step: %4i E: %16.12f G: %4.1e\n", i, ei, norm(gi)) 
            break
        else
            @printf(" Step: %4i E: %16.12f G: %4.1e\n", i, ei, norm(gi)) 
        end
    end

    return e, U, d1 
#=}}}=#
end

"""
    cmf_oo_diis( ints_in::InCoreInts{T}, clusters::Vector{MOCluster}, fspace, dguess::RDM1{T}; 
                    maxiter_oo      = 100, 
                    maxiter_ci      = 100, 
                    maxiter_d1      = 100, 
                    tol_oo          = 1e-6, 
                    tol_d1          = 1e-7, 
                    tol_ci          = 1e-8, 
                    verbose         = 0, 
                    max_ss_size     = 8, 
                    diis_start      = 1,
                    alpha           = .1,
                    zero_intra_rots = true 
                    sequential      = false
                    ) where T

Do CMF with orbital optimization using DIIS

# Arguments

- `ints::InCoreInts`: integrals for full system
- `clusters::Vector{MOCluster}`: vector of cluster objects
- `fspace::Vector{Vector{Integer}}`: vector of particle number occupations for each cluster specifying the sectors of fock space 
- `dguess`: initial guess for 1particle density matrix
- `maxiter_oo`: Max iter for the orbital optimization iterations 
- `maxiter_d1`: Max iter for the cmf iteration for the 1RDM 
- `maxiter_ci`: Max iter for the CI diagonalization of the cluster states 
- `tol_oo`: Convergence threshold for change in orbital gradient 
- `tol_ci`: Convergence threshold for the cluster CI problems 
- `tol_d1`: Convergence threshold for the CMF 1RDM 
- `sequential`: If true use the density matrix of the previous cluster in a cMF iteration to form effective integrals. Improves comvergence, may depend on cluster orderings   
- `verbose`: Printing level 
- `max_ss_size`: Max number of DIIS vectors
- `diis_start`: When to start doing DIIS extrapolations
- `alpha`: New vector added to ss is k=-alpha*g where g is the orbital gradient. This should be improved with Hessian.
- `zero_intra_rots`: Should we zero out rotations within a cluster? Helps with FCI solvers which should have zero gradients.

# Returns

- `e`: Energy
- `U::Matrix`: Orbital rotation matrix from input to output orbitals
- `d1::RDM1`: Optimized 1RDM in the optimized orbital basis
"""
function cmf_oo_diis(ints_in::InCoreInts{T}, clusters::Vector{MOCluster}, fspace, dguess::RDM1{T}; 
                    maxiter_oo      = 100, 
                    maxiter_ci      = 100, 
                    maxiter_d1      = 100, 
                    tol_oo          = 1e-6, 
                    tol_d1          = 1e-7, 
                    tol_ci          = 1e-8, 
                    verbose         = 0, 
                    max_ss_size     = 8, 
                    diis_start      = 1,
                    alpha           = .1,
                    zero_intra_rots = true,
                    use_pyscf      = true,
                    sequential      = false
    ) where T
    #={{{=#
    println(" Solve OO-CMF with DIIS")
    ints = deepcopy(ints_in)
    norb = n_orb(ints)
    e    = 0.0
    U    = zeros(T, norb, norb)
    d1   = deepcopy(dguess) 
    norb2 = norb*(norb-1)÷2

    nss   = 0
    k_ss  = zeros(T,norb2,0)
    g_ss  = zeros(T,norb2,0)
    converged = false
    condB = 0.0

    function step!(k)
        K = unpack_gradient(k, norb)
        Ui = exp(K)
        
        ints_i = orbital_rotation(ints,Ui)
        d1_i = orbital_rotation(d1,Ui)

        e_i, rdm1_dict, rdm2_dict = cmf_ci(ints_i, clusters, fspace, d1_i, 
                                         maxiter_d1 = maxiter_d1, 
                                         maxiter_ci = maxiter_ci, 
                                         tol_d1     = tol_d1, 
                                         tol_ci     = tol_ci, 
                                         verbose    = 0, 
                                         use_pyscf  = use_pyscf,
                                         sequential = sequential)
        d1_i, d2_i = assemble_full_rdm(clusters, rdm1_dict, rdm2_dict)
        
        d1_i = orbital_rotation(d1_i, Ui')
        d2_i = orbital_rotation(d2_i, Ui')
        g_i = build_orbital_gradient(ints, d1_i, d2_i)
        #g_i = build_orbital_gradient(ints_i, d1_i, d2_i)
       
        if zero_intra_rots
            g_i = unpack_gradient(g_i, norb)
            for ci in clusters
                g_i[ci.orb_list, ci.orb_list] .= 0
            end
            g_i = pack_gradient(g_i, norb)
        end
        e = e_i
        U = Ui
        return e_i, g_i, d1_i
    end
    
    # First step
    e_i, g_i, d1_i = step!(zeros(norb2))
    k_i = zeros(norb2)
        
    #g_i = reshape(g_i, (norb2,1))
    #k_i = reshape(k_i, (norb2,1))
    #g_ss = hcat(g_ss, g_i)
    #k_ss = hcat(k_ss, k_i)
    nss = size(g_ss,2)

    for i in 1:maxiter_oo
    
        k_i = k_i - alpha*g_i
       
        if nss < max_ss_size
            nss += 1
            g_ss = hcat(g_ss, g_i)
            k_ss = hcat(k_ss, k_i)
        else
            g_ss[:,1:end-1] .= g_ss[:,2:end]
            k_ss[:,1:end-1] .= k_ss[:,2:end]
            
            g_ss[:,nss] .= g_i
            k_ss[:,nss] .= k_i
        end

        # Check for linear dependence
        #tmp = 1*g_ss
        #
        #for i in 1:nss
        #    tmp[:,i] ./= norm(tmp[:,i])
        #end
        #nvecs = 0
        #for si in svdvals(tmp)
        #    println(si)
        #    if si > 1e-12
        #        nvecs += 1
        #    end
        #end

            
        @assert nss == size(g_ss,2)
        
        if i >= diis_start 
            steptype = "diis"
        
            B = zeros(T,nss+1,nss+1)
            B[1:nss, 1:nss] .= g_ss'*g_ss
            B[nss+1, :] .= -1 
            B[:, nss+1] .= -1 
            B[nss+1, nss+1] = 0 

            # Normalize
            Bmax  = max.(abs.(B[1:nss, 1:nss])...)
            B[1:nss, 1:nss] ./= Bmax

            verbose < 2 || println("B")
            verbose < 2 || display(B)

            b = zeros(T,nss+1)
            b[nss+1] = -1
            #println("b")
            #display(b)

            x = pinv(B)*b
            verbose < 2 || println("x")
            verbose < 2 || display(x)
           
            # new step:
            k_i = k_ss * x[1:nss]
            g_i = g_ss * x[1:nss]
            #display(norm(g_i))
            verbose < 2 || println("k")
            verbose < 2 || display(k_i)
            verbose < 2 || println("g (approx)")
            verbose < 2 || display(norm(g_i))

            verbose < 2 || println(" k")
            verbose < 2 || display(k_ss)
            verbose < 2 || println(" g")
            verbose < 2 || display(g_ss)
        end
       
        if zero_intra_rots
            # Remove intracluster rotations
            k_i = unpack_gradient(k_i, norb)
            for ci in clusters
                k_i[ci.orb_list, ci.orb_list] .= 0
            end
            k_i = pack_gradient(k_i, norb)
        end
       
        # 
        # Compute energy and gradient
        #
        e_i, g_i, d1_i = step!(k_i)
        g_i = reshape(g_i, (norb2,1))
        k_i = reshape(k_i, (norb2,1))
        d_i = d1_i
        # take gradient to be error vector
            
        
        if norm(g_i) < tol_oo 
            @printf("*ooCMF Iter: %4i Total= %16.12f G= %12.2e #SS: %4s\n", i, e_i, norm(g_i), nss)
            break
        end

        #g_ss = hcat(g_ss, g_i)
        #k_ss = hcat(k_ss, k_i)
        #nss = size(g_ss,2)

       

        @printf(" ooCMF Iter: %4i Total= %16.12f G= %12.2e #SS: %4s\n", i, e_i, norm(g_i), nss)
    end

    U = exp(unpack_gradient(k_i,norb))
    d1 = orbital_rotation(d1_i, U)
    return e, U, d1
#=}}}=#
end

function convert_pairs(original_list, rearranged_pairs)
    index_mapping = Dict{Int,Int}()
    for (original_idx, orbital) in enumerate(original_list)
        index_mapping[original_idx] = orbital
    end
    reverted_pairs = [(index_mapping[pair[1]], index_mapping[pair[2]]) for pair in rearranged_pairs]
    return reverted_pairs
end

function projection_vector(ansatze::Vector{<:Ansatz}, clusters, norb)
    n_dim = norb * (norb - 1) ÷ 2
    tmp_mat = Matrix(1I, n_dim, n_dim)
    clusters_new = [[orb for orb in cluster.orb_list] for cluster in clusters]
    invar = Vector{Tuple{Int,Int}}()
    count = 0
    for cluster in ansatze
        count += 1
        tmp = ActiveSpaceSolvers.invariant_orbital_rotations(cluster)
        tmp_global = convert_pairs(clusters_new[count], tmp)
        append!(invar, tmp_global)
    end
    # RASCI inter-cluster pairs commented out (RASCI not supported)
    # for (index_i, i) in enumerate(ansatze)
    #     if typeof(i) == RASCIAnsatz
    #         ...
    #     end
    # end
    fci = ActiveSpaceSolvers.FCIAnsatz(norb, 0, 0)
    full_list = ActiveSpaceSolvers.invariant_orbital_rotations(fci)
    keep_list = []
    for (a, b) in enumerate(full_list)
        if b in invar
            continue
        else
            push!(keep_list, a)
        end
    end
    proj_vec = tmp_mat[:, keep_list]
    return proj_vec
end

function cmf_ci_iteration(ints::InCoreInts{T}, clusters::Vector{MOCluster}, in_rdm1::RDM1{T}, fspace, ansatze::Vector{<:Ansatz};
                          use_pyscf = true,
                          use_nosym = false,
                          verbose   = 1,
                          sequential= false,
                          spin_avg  = true,
                          tol_ci    = 1e-8,
                          maxiter_ci    = 100
    ) where T
    rdm1 = deepcopy(in_rdm1)
    rdm1_dict = Dict{Integer,RDM1{T}}()
    rdm2_dict = Dict{Integer,RDM2{T}}()

    for i in 1:length(clusters)
        ci = clusters[i]
        flush(stdout)

        ansatz = ansatze[i]
        verbose < 2 || display(ansatz)
        ints_i = subset(ints, ci, rdm1)

        d1a = rdm1.a[ci.orb_list, ci.orb_list]
        d1b = rdm1.b[ci.orb_list, ci.orb_list]

        na = ansatz.na
        nb = ansatz.nb

        no = length(ci)

        d1 = RDM1(no)
        d2 = RDM2(no)

        e = 0.0
        if ansatz.dim == 1
            da = zeros(T, no, no)
            db = zeros(T, no, no)
            if ansatz.na == no
                da = Matrix(1.0I, no, no)
            end
            if ansatz.nb == no
                db = Matrix(1.0I, no, no)
            end
            d1 = RDM1(da,db)
            d2 = RDM2(d1)
            e = compute_energy(ints_i, d1)
            verbose < 2 || @printf(" Slater Det Energy: %12.8f\n", e)
        else
            if use_pyscf && typeof(ansatz) == FCIAnsatz
                # implemented in ext/TPSChemPyCallExt.jl; requires PyCall + pyscf
                e, d1, d2 = pyscf_fci_rdm12s(ints_i, na, nb, tol_ci, maxiter_ci; use_nosym=use_nosym)
            else
                solver = SolverSettings(verbose=1, tol=tol_ci, maxiter=maxiter_ci)
                solution = ActiveSpaceSolvers.solve(ints_i, ansatz, solver)
                d1a, d1b, d2aa, d2bb, d2ab = compute_1rdm_2rdm(solution)
                verbose < 2 || display(solution)
                d1 = RDM1(d1a, d1b)
                d2 = RDM2(d2aa, d2ab, d2bb)
            end
        end

        if spin_avg
            d1.a .= .5*(d1.a + d1.b)
            d1.b .= d1.a
            d2.aa .= .5*(d2.aa + d2.bb)
            d2.bb .= d2.aa
            ab = reshape(d2.ab, (no*no, no*no))
            ab .= (ab .+ ab').*.5
            ab = reshape(ab, (no, no, no, no))
            d2.ab .= ab
        end

        rdm1_dict[i] = d1
        rdm2_dict[i] = d2

        if sequential==true
            rdm1.a[ci.orb_list,ci.orb_list] = d1.a
            rdm1.b[ci.orb_list,ci.orb_list] = d1.b
        end
    end
    e_curr = compute_energy(ints, rdm1_dict, rdm2_dict, clusters, verbose=verbose)

    if verbose > 0
        @printf(" CMF-CI Curr: Elec %12.8f Total %12.8f\n", e_curr-ints.h0, e_curr)
    end

    return e_curr, rdm1_dict, rdm2_dict
end

function cmf_ci(ints, clusters, fspace, ansatze::Vector{<:Ansatz}, in_rdm1::RDM1;
                maxiter_ci  = 100,
                maxiter_d1  = 20,
                tol_d1      = 1e-6,
                tol_ci      = 1e-8,
                verbose     = 1,
                use_pyscf = true,
                use_nosym = false,
                sequential  = false)
    rdm1 = deepcopy(in_rdm1)
    energies = []
    e_prev = 0

    rdm1_dict = 0
    rdm2_dict = 0
    rdm1_dict = Dict{Integer,Array}()
    rdm2_dict = Dict{Integer,Array}()
    for iter = 1:maxiter_d1
        if verbose > 1
            println()
            println(" ------------------------------------------ ")
            println(" CMF CI Iter: ", iter)
            println(" ------------------------------------------ ")
        end
        e_curr, rdm1_dict, rdm2_dict = cmf_ci_iteration(ints, clusters, rdm1, fspace, ansatze,
                                                        maxiter_ci  = maxiter_ci,
                                                        tol_ci      = tol_ci,
                                                        verbose     = verbose,
                                                        use_pyscf   = use_pyscf,
                                                        use_nosym   = use_nosym,
                                                        sequential  = sequential
                                                       )
        rdm1_curr = assemble_full_rdm(clusters, rdm1_dict)
        append!(energies,e_curr)
        error = (rdm1_curr.a+rdm1_curr.b) - (rdm1.a+rdm1.b)
        d_err = norm(error)
        e_err = e_curr-e_prev
        if verbose>1
            @printf(" CMF-CI Energy: %12.8f | Change: RDM: %6.1e Energy %6.1e\n\n", e_curr, d_err, e_err)
        end
        e_prev = e_curr*1
        rdm1 = rdm1_curr
        if d_err < tol_d1
            if verbose>1
                @printf("*CMF-CI: Elec %12.8f Total %12.8f\n", e_curr-ints.h0, e_curr)
            end
            break
        end
    end
    if verbose>0
        println(" Energy per Iteration:")
        for i in energies
            @printf(" Elec: %12.8f Total: %12.8f\n", i-ints.h0, i)
        end
    end
    return e_prev, rdm1_dict, rdm2_dict
end

function cmf_oo(ints::InCoreInts{T}, clusters::Vector{MOCluster}, fspace, ansatze::Vector{<:Ansatz}, dguess::RDM1{T};
                max_iter_oo=100,
                max_iter_ci=100,
                gconv=1e-6,
                tol_d1=1e-7,
                tol_ci=1e-8,
                verbose=0,
                method="bfgs",
                alpha=nothing,
                use_pyscf=true,
                sequential=false) where T
    norb = size(ints.h1)[1]

    e_prev = 0
    e_tmp  = 0
    e_curr = 0

    g_prev = 0
    g_tmp  = 0
    g_curr = 0

    d1_prev = deepcopy(dguess)
    d1_tmp  = deepcopy(dguess)
    d1_curr = deepcopy(dguess)

    iter = 0
    kappa = zeros(norb*(norb-1)÷2)

    function f(k)
        K = unpack_gradient(k, norb)
        U = exp(K)
        ints_tmp = orbital_rotation(ints,U)
        e, rdm1_dict, _ = cmf_ci(ints_tmp, clusters, fspace, ansatze, orbital_rotation(d1_curr, U),
                                         tol_d1=tol_d1,
                                         tol_ci=tol_ci,
                                         verbose=0,
                                         use_pyscf=use_pyscf,
                                         sequential=sequential)
        d1_tmp = assemble_full_rdm(clusters, rdm1_dict)
        d1_tmp = orbital_rotation(d1_tmp, U')
        e_tmp = e
        return e
    end

    function g(kappa)
        norb = n_orb(ints)
        K = unpack_gradient(kappa, norb)
        U = exp(K)
        ints_tmp = orbital_rotation(ints,U)
        e, rdm1_dict, rdm2_dict = cmf_ci(ints_tmp, clusters, fspace, ansatze, orbital_rotation(d1_curr,U),
                                         tol_d1=tol_d1,tol_ci=tol_ci, verbose=verbose,use_pyscf=use_pyscf)
        gd1, gd2 = assemble_full_rdm(clusters, rdm1_dict, rdm2_dict)
        gout = build_orbital_gradient(ints_tmp, gd1, gd2)
        g_tmp = norm(gout)
        return gout
    end

    function callback(k)
        d1_curr  = deepcopy(d1_tmp)
        e_curr   = e_tmp
        g_curr   = g_tmp
        iter += 1
        if (g_curr < gconv)
            @printf("*ooCMF Iter: %4i Total= %16.12f Active= %16.12f G= %12.2e\n", iter, e_curr, e_curr-ints.h0, g_curr)
            return true
        else
            @printf(" ooCMF Iter: %4i Total= %16.12f Active= %16.12f G= %12.2e\n", iter, e_curr, e_curr-ints.h0, g_curr)
            return false
        end
    end

    if (method=="bfgs") || (method=="cg")
        optmethod = BFGS()
        if method=="cg"
            optmethod = ConjugateGradient()
        end
        options = Optim.Options(
                                callback = callback,
                                g_tol=gconv,
                                iterations=max_iter_oo,
                               )
        res = optimize(f, g, kappa, optmethod, options; inplace = false)
        summary(res)
        e = Optim.minimum(res)
        display(res)
        @printf("*ooCMF %12.8f \n", e)
        kappa = Optim.minimizer(res)
        K = unpack_gradient(kappa, norb)
        U = exp(K)
        d1 = orbital_rotation(d1_curr, U)
        return e, U, d1
    end
end

function cmf_oo_gd(ints_in::InCoreInts{T}, clusters::Vector{MOCluster}, fspace, ansatze::Vector{<:Ansatz}, dguess::RDM1{T};
                    maxiter_oo      = 100,
                    maxiter_ci      = 100,
                    maxiter_d1      = 100,
                    tol_oo          = 1e-6,
                    tol_d1          = 1e-7,
                    tol_ci          = 1e-8,
                    verbose         = 0,
                    alpha           = .1,
                    use_pyscf=true,
                    zero_intra_rots = true,
                    sequential      = false
    ) where T
    #={{{=#
    ints = deepcopy(ints_in)
    norb = n_orb(ints)
    d1   = deepcopy(dguess)
    U    = Matrix(1.0I, norb, norb)
    e    = 0.0

    function step!(ints, d1, k)
        norb = n_orb(ints)
        K = unpack_gradient(k, norb)
        if zero_intra_rots
            for ci in clusters
                K[ci.orb_list, ci.orb_list] .= 0
            end
        end
        Ui = exp(K)
        tmp = orbital_rotation(ints,Ui)
        ints.h1 .= tmp.h1
        ints.h2 .= tmp.h2
        tmp = orbital_rotation(d1,Ui)
        d1.a .= tmp.a
        d1.b .= tmp.b
        e_i, rdm1_dict, rdm2_dict = cmf_ci(ints, clusters, fspace, ansatze, d1,
                                         maxiter_d1 = maxiter_d1,
                                         maxiter_ci = maxiter_ci,
                                         tol_d1     = tol_d1,
                                         tol_ci     = tol_ci,
                                         verbose    = 0,
                                         use_pyscf=use_pyscf,
                                         sequential = sequential)
        gd1, gd2 = assemble_full_rdm(clusters, rdm1_dict, rdm2_dict)
        g = build_orbital_gradient(ints, gd1, gd2)
        return e_i, g, Ui, gd1
    end

    converged = false
    step_i = zeros(norb*(norb-1)÷2)
    for i in 1:maxiter_oo
        ei, gi, Ui, d1 = step!(ints, d1, step_i)
        step_i = -alpha*gi
        e = ei
        U = U*Ui
        converged = norm(gi) < tol_oo
        if converged
            @printf("*Step: %4i E: %16.15f G: %4.3e\n", i, ei, norm(gi))
            break
        else
            @printf(" Step: %4i E: %16.15f G: %4.3e\n", i, ei, norm(gi))
        end
    end

    return e, U, d1
    #=}}}=#
end

function cmf_oo_diis(ints_in::InCoreInts{T}, clusters::Vector{MOCluster}, fspace, ansatze::Vector{<:Ansatz}, dguess::RDM1{T};
                    maxiter_oo      = 100,
                    maxiter_ci      = 100,
                    maxiter_d1      = 100,
                    tol_oo          = 1e-6,
                    tol_d1          = 1e-7,
                    tol_ci          = 1e-8,
                    verbose         = 0,
                    max_ss_size     = 8,
                    diis_start      = 1,
                    alpha           = .1,
                    step_trust_region=0.95,
                    use_pyscf       = true,
                    zero_intra_rots = true,
                    orb_hessian     = true,
                    sequential      = false,
                    trust_region    = false
    ) where T
    #={{{=#
    println(" Solve OO-CMF with DIIS")
    ints = deepcopy(ints_in)
    norb = n_orb(ints)
    e    = 0.0
    U    = zeros(T, norb, norb)
    d1   = deepcopy(dguess)
    norb2 = norb*(norb-1)÷2

    nss   = 0
    k_ss  = zeros(T,norb2,0)
    g_ss  = zeros(T,norb2,0)
    converged = false
    condB = 0.0

    function step!(k)
        K = unpack_gradient(k, norb)
        Ui = exp(K)
        ints_i = orbital_rotation(ints,Ui)
        d1_i = orbital_rotation(d1,Ui)
        e_i, rdm1_dict, rdm2_dict = cmf_ci(ints_i, clusters, fspace, ansatze, d1_i,
                                         maxiter_d1 = maxiter_d1,
                                         maxiter_ci = maxiter_ci,
                                         tol_d1     = tol_d1,
                                         tol_ci     = tol_ci,
                                         use_pyscf=use_pyscf,
                                         verbose    = verbose,
                                         sequential = sequential
                                         )
        d1_i, d2_i = assemble_full_rdm(clusters, rdm1_dict, rdm2_dict)
        d1_i = orbital_rotation(d1_i, Ui')
        d2_i = orbital_rotation(d2_i, Ui')
        g_i = build_orbital_gradient(ints, d1_i, d2_i)
        if orb_hessian
            h = RDM.build_orbital_hessian(ints,d1_i,d2_i)
        else
            h = nothing
        end
        if verbose == 1
            display(unpack_gradient(g_i, norb))
        end
        e = e_i
        U = Ui
        return e_i, g_i, d1_i, h
    end

    # First step
    e_i, g_i, d1_i, h_i = step!(zeros(norb2))
    k_i = zeros(norb2)
    nss = size(g_ss,2)

    if zero_intra_rots
        proj_vec = projection_vector(ansatze, clusters, norb)
    end

    for i in 1:maxiter_oo
        if zero_intra_rots && orb_hessian
            tmp_step = (pinv(proj_vec'*h_i*proj_vec))*(proj_vec'*g_i)
            step_i = proj_vec*tmp_step
        elseif orb_hessian && zero_intra_rots==false
            step_i = pinv(h_i)*g_i
        elseif orb_hessian==false && zero_intra_rots
            tmp_step = proj_vec'*g_i
            g_i = proj_vec*tmp_step
            step_i = alpha*g_i
        else
            step_i = alpha*g_i
        end
        if trust_region==true
            if norm(step_i) > step_trust_region
                step_i = step_i*step_trust_region/norm(step_i)
            end
        end
        k_i = k_i - step_i

        if nss < max_ss_size
            nss += 1
            g_ss = hcat(g_ss, step_i)
            k_ss = hcat(k_ss, k_i)
        else
            g_ss[:,1:end-1] .= g_ss[:,2:end]
            k_ss[:,1:end-1] .= k_ss[:,2:end]
            g_ss[:,nss] .= step_i
            k_ss[:,nss] .= k_i
        end

        if i >= diis_start
            steptype = "diis"
            B = zeros(T,nss+1,nss+1)
            B[1:nss, 1:nss] .= g_ss'*g_ss
            B[nss+1, :] .= -1
            B[:, nss+1] .= -1
            B[nss+1, nss+1] = 0
            Bmax  = max.(abs.(B[1:nss, 1:nss])...)
            B[1:nss, 1:nss] ./= Bmax
            verbose < 2 || println("B")
            verbose < 2 || display(B)
            b = zeros(T,nss+1)
            b[nss+1] = -1
            x = pinv(B)*b
            verbose < 2 || println("x")
            verbose < 2 || display(x)
            k_i = k_ss * x[1:nss]
            g_i = g_ss * x[1:nss]
            verbose < 2 || println("k")
            verbose < 2 || display(k_i)
            verbose < 2 || println("g (approx)")
            verbose < 2 || display(norm(g_i))
            verbose < 2 || println(" k")
            verbose < 2 || display(k_ss)
            verbose < 2 || println(" g")
            verbose < 2 || display(g_ss)
        end

        e_i, g_i, d1_i, h_i = step!(k_i)
        g_i = reshape(g_i, (norb2,1))
        k_i = reshape(k_i, (norb2,1))
        d_i = d1_i

        if norm(g_i) < tol_oo
            @printf("*ooCMF Iter: %4i Total= %16.15f G= %12.3e step_size= %12.3e #SS: %4s\n", i, e_i, norm(g_i),norm(k_i), nss)
            break
        end

        @printf("*ooCMF Iter: %4i Total= %16.15f G= %12.3e step_size= %12.3e #SS: %4s\n", i, e_i, norm(g_i),norm(k_i), nss)
    end

    U = exp(unpack_gradient(k_i,norb))
    d1 = orbital_rotation(d1_i, U)
    return e, U, d1
    #=}}}=#
end

"""
    cmf_oo_newton(ints, clusters, fspace, ansatze, dguess::RDM1; kwargs...)
    cmf_oo_newton(ints, clusters, fspace, dguess::RDM1; kwargs...)

Cluster mean-field (CMF) orbital optimization using a Newton solver that uses
both the (analytically computed) orbital gradient and Hessian. Alternates
cluster CI solves with Newton steps on the orbital-rotation parameters until the
orbital gradient (`tol_oo`), 1-RDM (`tol_d1`), and CI (`tol_ci`) are converged.
Pass `zero_intra_rots=true` to freeze intra-cluster rotations. Returns
`(e_cmf, U, d1)`: the CMF energy, the orbital-rotation unitary, and the
converged 1-RDM.
"""
function cmf_oo_newton(ints_in::InCoreInts{T}, clusters::Vector{MOCluster}, fspace, ansatze::Vector{<:Ansatz}, dguess::RDM1{T};
                    maxiter_oo      = 100,
                    maxiter_ci      = 100,
                    maxiter_d1      = 100,
                    tol_oo          = 1e-6,
                    tol_d1          = 1e-7,
                    tol_ci          = 1e-8,
                    verbose         = 0,
                    step_trust_region=0.95,
                    use_pyscf=true,
                    use_nosym=false,
                    zero_intra_rots =true,
                    sequential      = false,
                    trust_region=false
    ) where T
    #={{{=#
    println(" Solve OO-CMF with newton")
    ints = deepcopy(ints_in)
    norb = n_orb(ints)
    d1   = deepcopy(dguess)
    U    = Matrix(1.0I, norb, norb)
    e    = 0.0
    if zero_intra_rots
        proj_vec = projection_vector(ansatze, clusters, norb)
    end
    function step!(ints, d1, k)
        norb = n_orb(ints)
        K = unpack_gradient(k, norb)
        Ui = exp(K)
        tmp = orbital_rotation(ints,Ui)
        ints.h1 .= tmp.h1
        ints.h2 .= tmp.h2
        tmp = orbital_rotation(d1,Ui)
        d1.a .= tmp.a
        d1.b .= tmp.b
        e, rdm1_dict, rdm2_dict = cmf_ci(ints, clusters, fspace, ansatze, d1,
                                         maxiter_d1 = maxiter_d1,
                                         maxiter_ci = maxiter_ci,
                                         tol_d1     = tol_d1,
                                         tol_ci     = tol_ci,
                                         verbose    = verbose,
                                         use_pyscf=use_pyscf,
                                         use_nosym=use_nosym,
                                         sequential = sequential
                                         )
        gd1, gd2 = assemble_full_rdm(clusters, rdm1_dict, rdm2_dict)
        g_i = build_orbital_gradient(ints, gd1, gd2)
        packed_hessian = RDM.build_orbital_hessian(ints, gd1, gd2)
        return e, g_i, Ui, gd1, packed_hessian
    end

    converged = false
    step_i = zeros(norb*(norb-1)÷2)
    for i in 1:maxiter_oo
        ei, g, Ui, d1, h_i = step!(ints, d1, step_i)
        g_i = g
        if zero_intra_rots
            tmp_step = (pinv(proj_vec'*h_i*proj_vec))*(proj_vec'*g_i)
            step_i = -proj_vec*tmp_step
        else
            step_i = -(pinv(h_i)*(g_i))
        end
        if trust_region==true
            if norm(step_i) > step_trust_region
                step_i = step_i*step_trust_region/norm(step_i)
            end
        end
        e = ei
        U = U*Ui
        converged = norm(g_i) < tol_oo
        if converged
            @printf(" Step: %4i E: %16.12f G: %12.2e   \n", i, ei, norm(g_i))
            break
        else
            @printf(" Step: %4i E: %16.12f G: %12.2e step_size: %12.2e \n", i, ei, norm(g_i), norm(step_i))
        end
    end
    return e, U, d1
    #=}}}=#
end

function cmf_oo_newton(ints_in::InCoreInts{T}, clusters::Vector{MOCluster}, fspace, dguess::RDM1{T};
                    maxiter_oo      = 100,
                    maxiter_ci      = 100,
                    maxiter_d1      = 100,
                    tol_oo          = 1e-6,
                    tol_d1          = 1e-7,
                    tol_ci          = 1e-8,
                    verbose         = 0,
                    step_trust_region=0.95,
                    use_pyscf=true,
                    use_nosym=false,
                    sequential      = false,
                    trust_region=false,
    ) where T
    #={{{=#
    println(" Solve OO-CMF with newton")
    ints = deepcopy(ints_in)
    norb = n_orb(ints)
    d1   = deepcopy(dguess)
    U    = Matrix(1.0I, norb, norb)
    e    = 0.0
    function step!(ints, d1, k)
        norb = n_orb(ints)
        K = unpack_gradient(k, norb)
        Ui = exp(K)
        tmp = orbital_rotation(ints,Ui)
        ints.h1 .= tmp.h1
        ints.h2 .= tmp.h2
        tmp = orbital_rotation(d1,Ui)
        d1.a .= tmp.a
        d1.b .= tmp.b
        e, rdm1_dict, rdm2_dict = cmf_ci(ints, clusters, fspace, d1,
                                         maxiter_d1 = maxiter_d1,
                                         maxiter_ci = maxiter_ci,
                                         tol_d1     = tol_d1,
                                         tol_ci     = tol_ci,
                                         verbose    = verbose,
                                         use_pyscf=use_pyscf,
                                         use_nosym=use_nosym,
                                         sequential = sequential)
        gd1, gd2 = assemble_full_rdm(clusters, rdm1_dict, rdm2_dict)
        g_i = build_orbital_gradient(ints, gd1, gd2)
        packed_hessian = RDM.build_orbital_hessian(ints, gd1, gd2)
        return e, g_i, Ui, gd1, packed_hessian
    end

    converged = false
    step_i = zeros(norb*(norb-1)÷2)
    for i in 1:maxiter_oo
        ei, g, Ui, d1, h_i = step!(ints, d1, step_i)
        g_i = g
        step_i = -(pinv(h_i)*(g_i))
        if trust_region==true
            if norm(step_i) > step_trust_region
                step_i = step_i*step_trust_region/norm(step_i)
            end
        end
        e = ei
        U = U*Ui
        converged = norm(g_i) < tol_oo
        if converged
            @printf(" Step: %4i E: %16.12f G: %12.2e   \n", i, ei, norm(g_i))
            break
        else
            @printf(" Step: %4i E: %16.12f G: %12.2e step_size: %12.2e \n", i, ei, norm(g_i), norm(step_i))
        end
    end
    return e, U, d1
    #=}}}=#
end
