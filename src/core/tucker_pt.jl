using IterativeSolvers

 """
    hylleraas_compressed_mp2a(sig_in::SPTstate{T,N,R}, ref::SPTstate{T,N,R}, cluster_ops, clustered_ham;
    H0="Hcmf",
    tol=1e-6,
    nbody=4,
    max_iter=100,
    verbose=1,
    thresh=1e-8) where {T,N,R}


- `H0`: ["H", "Hcmf"] 

Compute compressed PT2.
Since there can be non-zero overlap with a multireference state, we need to generalize.

HC = SCe

|Haa + Hax| |Ca| = |I   + Sax| |Ca| E
|Hxa + Hxx| |Cx|   |Sxa + I  | |Cx|

Haa*Ca + Hax*Cx = Ca*E + Sax*Cx*E
Hxa*Ca + Hxx*Cx = Sxa*Ca*E + Cx*E

(Hxx - Es)*Cx = Sxa*Ca*E - Hxa*Ca 

Perturbation partitioning: 
        H = F + <0|H-F|0> + λ(H - F - <0|H-F|0>)

Keeping only 1st order terms: 
    (Hxx^0 - E^0)*Cx^1 = Sxa*Ca^0*E^1 - Hxa^1*Ca^0 

Partitioning ensures zero 1st order energy correction
    E^1 = 0

    (Hxx^0 - E^0)*Cx^1 = - Hxa^1*Ca^0 


    (Fxx - <0|F|0>)*Cx = - <x|H - F - <0|V|0> |a> Ca

Leading to a linear system for `Cx`
    Ax=b

After solving, the energy can be obtained as:

    Ca'*Haa*Ca + Ca'*Hax*Cx = E + Ca'*Sax*Cx*E

    E = (Eref + Ca'*Hax*Cx) / (1 + Ca'*Sax*Cx)


"""




"""
    compute_pt1_wavefunction(σ_in::SPTstate{T,N,R}, ψ0::SPTstate{T,N,R}, cluster_ops, clustered_ham;
    H0="Hcmf",
    nbody=4,
    verbose=1) where {T,N,R}

H0 = F + <0|H - F|0>
V  = H - F - <0|H - F|0>

ψ1 = <X|H - F - E0 + F0|0> / (<0|F|0> - Fx)
   = <X|H|0> - <X|F|0> - (E0-F0)<X|0> / Dx

E2 = <0|V|X>ψ1

The local Tucker factors are first canonicalized to avoid having to solve the Hylleraas functional
"""
function compute_pt1_wavefunction(σ_in::SPTstate{T,N,R}, ψ0::SPTstate{T,N,R}, cluster_ops, clustered_ham, clustered_ham_0, E0, F0;
    H0="Hcmf",
    verbose=1) where {T,N,R}
    
    #
    # Copy data
    verbose < 2 || @printf(" copy sigma... \n")
    flush(stdout)
    σ = deepcopy(σ_in)
    zero!(σ)
    verbose < 2 || @printf(" done.\n")
    flush(stdout)


    #
    # Rotate the local tucker factors to diagonalize the zeroth order hamiltonians and build F diagonal
    verbose < 2 || @printf(" form_1body_operator_diagonal... \n")
    flush(stdout)
    Fdiag = SPTstate(σ, R=1)
    form_1body_operator_diagonal!(σ, Fdiag, cluster_ops, pseudo_canon=true)
    verbose < 2 || @printf(" done.\n")
    flush(stdout)
    
    
    #
    # Get Overlap <X|A>C(A)
    verbose < 2 || @printf(" get_overlap... \n")
    flush(stdout)
    Sx = TPSChem.project_into_new_basis(ψ0, σ)

    verbose < 2 || @printf(" done.\n")
    flush(stdout)


    # 
    # We need the numerator: <X|V|0> = <X|H|0> - <X|F|0> - (E0 - F0)<X|0>
    # 
    # Build exact Hamiltonian within FOIS defined by `sig`: <X|H|0>
    verbose < 2 || @printf("   Norm of σ before: \n")
    verbose < 2 || println(orth_dot(σ,σ))
    verbose < 2 || @printf(" build_sigma!... \n")
    flush(stdout)
    verbose < 1 || @printf(" %-50s%10i\n", "Length of input      FOIS: ", length(σ))
    time = @elapsed alloc = @allocated build_sigma!(σ, ψ0, cluster_ops, clustered_ham, verbose=verbose)
    verbose < 1 || @printf(" %-50s%10.6f seconds %10.2e Gb\n", "Compute <X|H|0>: ", time, alloc/1e9)
    verbose < 2 || @printf(" done.\n")
    verbose < 2 || @printf("   Norm of σ after: \n")
    verbose < 2 || println(orth_dot(σ,σ))
    flush(stdout)

    # subtract <X|F|0>
    verbose < 2 || @printf(" build_sigma!... \n")
    flush(stdout)
    XF0 = deepcopy(σ)
    zero!(XF0)
    time = @elapsed alloc = @allocated build_sigma!(XF0, ψ0, cluster_ops, clustered_ham_0)
    verbose < 1 || @printf(" %-50s%10.6f seconds %10.2e Gb\n", "Compute <X|F|0>: ", time, alloc/1e9)
    verbose < 2 || @printf(" done.\n")
    flush(stdout)

    verbose < 2 || @printf(" compute correction... \n")
    flush(stdout)

    #
    # σ ← <X|H|0> - <X|F|0> - (E0-F0)<X|0>
    #
    # XF0 and Sx were both built from σ, so all three carry the same blocks and
    # the same Tucker factors.  That makes this a pure core-wise update, exactly
    # equal to `σ - XF0 - scale(Sx, E0 .- F0)` but without allocating three more
    # copies of the FOIS -- the peak that limits multi-root runs.
    dE = E0 .- F0
    for (fock, configs) in σ
        for (config, tuck) in configs
            xf0 = XF0[fock][config].core
            sx  = Sx[fock][config].core
            for r in 1:R
                @. tuck.core[r] -= xf0[r] + dE[r] * sx[r]
            end
        end
    end
    empty!(XF0.data)   # release the <X|F|0> and overlap copies before forming ψ1
    empty!(Sx.data)

    #
    # ψ1 = σ / (F0 - Fdiag) and ecorr = <σ|ψ1>, formed in one pass.
    #
    # σ is not needed once ψ1 exists, so divide in place and reuse its storage
    # rather than taking a further deepcopy and round-tripping every root
    # through a dense get_vector/set_vector pair.
    ψ1 = σ
    ecorr = zeros(T, R)
    for (fock, configs) in ψ1
        for (config, tuck) in configs
            fdiag = Fdiag[fock][config].core[1]
            for r in 1:R
                c = tuck.core[r]
                acc = zero(T)
                # the 1e-12 shift just protects against exact/accidental zeros
                @inbounds for i in eachindex(c)
                    num  = c[i]
                    c[i] = num / (F0[r] - fdiag[i] + 1e-12)
                    acc += num * c[i]
                end
                ecorr[r] += acc
            end
        end
    end

    E2 = zeros(R)
    for r in 1:R
        E2[r] = E0[r] + ecorr[r]
    end
    
    verbose < 2 || @printf(" done.\n")
    flush(stdout)
    
    return ψ1, E2, ecorr

end

"""
    compute_pt1_wavefunction(σ_in::SPTstate{T,N,R}, ψ0::SPTstate{T,N,R}, cluster_ops, clustered_ham;
    H0="Hcmf",
    nbody=4,
    verbose=1) where {T,N,R}

TBW
"""
function compute_pt1_wavefunction(σ_in::SPTstate{T,N,R}, ψ0::SPTstate{T,N,R}, cluster_ops, clustered_ham;
    H0="Hcmf",
    verbose=1) where {T,N,R}

    verbose < 1 || println()
    verbose < 1 || println(" |...................................SPT-PT2............................................")
    flush(stdout)
    #
    # Copy data
    verbose < 2 || @printf(" copy sigma... \n")
    flush(stdout)
    σ = deepcopy(σ_in)
    zero!(σ)
    verbose < 2 || @printf("done.\n")
    flush(stdout)


    #
    # Extract zeroth-order hamiltonian
    verbose < 2 || @printf(" extract_1body_operator... \n")
    flush(stdout)
    clustered_ham_0 = extract_1body_operator(clustered_ham, op_string=H0)
    verbose < 2 || @printf("done.\n")
    flush(stdout)
    

    # 
    # get E0 = <0|H|0>
    verbose < 2 || @printf(" compute_expectation_value... \n")
    flush(stdout)
    E0 = compute_expectation_value(ψ0, cluster_ops, clustered_ham)
    verbose < 2 || @printf("done.\n")
    flush(stdout)


    # 
    # get F0 = <0|F|0>
    verbose < 2 || @printf(" compute_expectation_value... \n")
    flush(stdout)
    F0 = compute_expectation_value(ψ0, cluster_ops, clustered_ham_0)
    verbose < 2 || @printf("done.\n")
    flush(stdout)


    if verbose > 1
        @printf(" %5s %12s %12s\n", "Root", "<0|H|0>", "<0|F|0>")
        for r in 1:R
            @printf(" %5s %12.8f %12.8f\n", r, E0[r], F0[r])
        end
    end

    verbose < 2 || @printf(" compute_pt1_wavefunction... \n")
    flush(stdout)
    ψ1, E2, ecorr = compute_pt1_wavefunction(σ, ψ0, cluster_ops, clustered_ham, clustered_ham_0,  E0, F0, H0=H0, verbose=verbose)
    verbose < 2 || @printf("done.\n")
    flush(stdout)

    verbose < 1 || for r in 1:R
        E2[r] = E0[r] + ecorr[r]
        @printf(" State %3i: %-35s%14.8f\n", r, "E(PT2) corr: ", ecorr[r])
    end

    verbose < 1 || @printf(" %5s %12s %12s\n", "Root", "E(0)", "E(2)")
    verbose < 1 || for r in 1:R
        @printf(" %5s %12.8f %12.8f\n", r, E0[r], E2[r])
    end
    verbose < 1 || println(" ......................................................................................|")

    return ψ1, E2, ecorr 

end



"""
    compute_pt1_wavefunction(ψ0::SPTstate{T,N,R}, cluster_ops, clustered_ham;
    H0 = "Hcmf",
    nbody = 4,
    thresh_foi = 1e-7
    verbose = 1,
    max_number = nothing) where {T,N,R}

TBW
"""
function compute_pt1_wavefunction(ψ0::SPTstate{T,N,R}, cluster_ops, clustered_ham;
    H0 = "Hcmf",
    nbody = 4,
    thresh_foi = 1e-7,
    verbose = 1,
    max_number = nothing,
    matvec = 1) where {T,N,R}

    #
    # Build target FOIS
    matvec_function = build_compressed_1st_order_state
    if matvec == 2
        matvec_function = build_compressed_1st_order_state_old
    end

    time = @elapsed alloc = @allocated σ = matvec_function(ψ0, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi, max_number=max_number)
    verbose < 1 || @printf(" %-50s%10.6f seconds %10.2e Gb\n", "Compute Compressed FOIS: ", time, alloc/1e9)

    return compute_pt1_wavefunction(σ, ψ0, cluster_ops, clustered_ham, H0=H0, verbose=verbose)
end


"""
    form_1body_operator_diagonal!(sig::SPTstate{T,N,R}, Fdiag::SPTstate{T,N,1}, cluster_ops; H0="Hcmf", pseudo_canon=false) where {T,N,R}

TBW
"""
function form_1body_operator_diagonal!(sig::SPTstate{T,N,R}, Fdiag::SPTstate{T,N,1}, cluster_ops; H0="Hcmf", pseudo_canon=false) where {T,N,R}
    # Fdiag = SPTstate(v, R=1)
    clusters = sig.clusters

    zero!(Fdiag)
    for (fock, tconfigs) in sig
        for (tconfig, tuck) in tconfigs
    
            rotations = Vector{Matrix{T}}([])

            #
            # Initialize energy list of lists
            energies = Vector{Vector{T}}([])
            for ci in 1:N
                push!(energies, [])
            end

            # 
            # Rotate tucker fractors to diagonalize each cluster hamiltonian
            for ci in clusters
                Ui = tuck.factors[ci.idx]
                if size(Ui, 2) > 1

                    # build the local "fock" operator in the current tucker basis
                    Hi = cluster_ops[ci.idx][H0][(fock[ci.idx], fock[ci.idx])][tconfig[ci.idx], tconfig[ci.idx]]
                    Hi = Ui' * Hi * Ui

                    if pseudo_canon
                        # diagonalize and rotate tucker factors
                        F = eigen(Symmetric(Hi))
                        # sig[fock][tconfig].factors[ci.idx] .= Ui * F.vectors
                        # Fdiag[fock][tconfig].factors[ci.idx] .= sig[fock][tconfig].factors[ci.idx]
                        Fdiag[fock][tconfig].factors[ci.idx] .= sig[fock][tconfig].factors[ci.idx] * F.vectors
                       
                        push!(rotations, F.vectors)
                        energies[ci.idx] = F.values
                    else
                        # just take diagonal
                        energies[ci.idx] = diag(Hi) 
                    end
                elseif size(Ui, 2) == 1
                    Hi = cluster_ops[ci.idx][H0][(fock[ci.idx], fock[ci.idx])][tconfig[ci.idx], tconfig[ci.idx]]
                    e = Ui' * Hi * Ui

                    length(e) == 1 || throw(DimensionMismatch)

                    push!(rotations, ones(T,1,1))
                    energies[ci.idx] = [e[1]]
                end
            end

            #
            # Add the local energies together to form zeroth order energies for each element
            fcore = Fdiag[fock][tconfig].core
            length(fcore) == 1 || throw(DimensionMismatch)
            for i in CartesianIndices(fcore[1])
                for ci in 1:N
                    fcore[1][i] += energies[ci][i[ci]]
                end
            end
   
            # 
            # Rotate the original vector into this basis
            if length(rotations)==N
                transform_basis!(sig[fock][tconfig], rotations)
            else
                continue
            end
        end
    end

end


function hylleraas_compressed_mp2(sig_in::SPTstate{T,N,R}, ref::SPTstate{T,N,R}, cluster_ops, clustered_ham;
    H0="Hcmf",
    tol=1e-6,
    nbody=4,
    max_iter=100,
    verbose=1,
    thresh=1e-8) where {T,N,R}
    
    #
    # Extract zeroth-order hamiltonian
    clustered_ham_0 = extract_1body_operator(clustered_ham, op_string=H0)

    # 
    # Build exact Hamiltonian within FOIS defined by `sig_in`: <X|H|0>
    sig = deepcopy(sig_in)
    verbose < 1 || @printf(" %-50s%10i\n", "Length of input      FOIS: ", length(sig_in))
    #@printf(" %-50s%10i\n", "Length of compressed FOIS: ", length(sig))
    #project_out!(sig, ref)
    zero!(sig)

    time = @elapsed alloc = @allocated build_sigma!(sig, ref, cluster_ops, clustered_ham, verbose=verbose)
    verbose < 1 || @printf(" %-50s%10.6f seconds %10.2e Gb\n", "Compute <X|V|0>: ", time, alloc/1e9)


    e2 = zeros(T, R)

    # 
    # get E_ref = <0|H|0>
    e_ref = compute_expectation_value(ref, cluster_ops, clustered_ham)

    # 
    # get E0 = <0|H0|0>
    e0 = compute_expectation_value(ref, cluster_ops, clustered_ham_0)


    if verbose > 0
        @printf(" %5s %12s %12s\n", "Root", "<0|H|0>", "<0|F|0>")
        for r in 1:R
            @printf(" %5s %12.8f %12.8f\n", r, e_ref[r], e0[r])
        end
    end



    #
    # Get Overlap <X|A>C(A)
    Sx = deepcopy(sig)
    zero!(Sx)
    for (fock, tconfigs) in Sx
        if haskey(ref, fock)
            for (tconfig, tuck) in tconfigs
                if haskey(ref[fock], tconfig)
                    ref_tuck = ref[fock][tconfig]
                    # Cr(i,j,k...) Ur(Ii) Ur(Jj) ...
                    # Ux(Ii') Ux(Jj') ...
                    #
                    # Cr(i,j,k...) S(ii') S(jj')...
                    overlaps = Vector{Matrix{T}}()
                    for i in 1:N
                        push!(overlaps, ref_tuck.factors[i]' * tuck.factors[i])
                    end
                    for r in 1:R
                        Sx[fock][tconfig].core[r] .= transform_basis(ref_tuck.core[r], overlaps)
                    end
                end
            end
        end
    end


    # 
    # Build b
    #   b = <X|F|a>Ca - <X|H|a>Ca + <X|a>Ca (<0|H|0> - <0|F|0>)
    
    # b += - <X|H|0>
    b = -get_vector(sig)

    # b += <X|F|0>
    tmp = deepcopy(sig)
    zero!(tmp)
    time = @elapsed alloc = @allocated build_sigma!(tmp, ref, cluster_ops, clustered_ham_0)
    verbose < 1 || @printf(" %-50s%10.6f seconds %10.2e Gb\n", "Compute <X|F|0>: ", time, alloc/1e9)
    b .+= get_vector(tmp)

    # b += Sx * (<0|H-F|0>)
    #   taken care off in loop over states


    #@printf(" Norm of b         : %18.12f\n", sum(b.*b))
    flush_cache(clustered_ham_0)
    time = @elapsed alloc = @allocated cache_hamiltonian(sig, sig, cluster_ops, clustered_ham_0)
    psi1 = deepcopy(sig)
    verbose < 1 || @printf(" %-50s%10.6f seconds %10.2e Gb\n", "Cache zeroth-order Hamiltonian: ", time, alloc/1e9)


    #  (Fxx + Eref - <0|F|0> - Es)*Cxs = Sxa*Cas*Eref - Hxa
    #
    # Currently, we need to solve each root separately, this should be fixed
    # by writing our own CG solver
    for r in 1:R

        function mymatvec(x)

            xr = SPTstate(sig, R=1)
            xl = SPTstate(sig, R=1)

            #display(size(xr))
            #display(size(x))
            length(xr) .== length(x) || throw(DimensionMismatch)
            set_vector!(xr, x, root=1)
            zero!(xl)
            build_sigma!(xl, xr, cluster_ops, clustered_ham_0, cache=true)

            # subtract off -E0|1>
            #

            scale!(xr, -e0[1])
            #scale!(xr,-e0[r])  # pretty sure this should be uncommented - but it diverges, not sure why
            orth_add!(xl, xr)
            flush(stdout)

            return get_vector(xl)
        end
        
        # b += Sx * (<0|H-F|0>)
        br = b[:, r] .+ get_vector(Sx)[:, r] .* (e_ref[r] - e0[r])


        dim = length(br)
        Axx = LinearMap(mymatvec, dim, dim)


        #@time cache_hamiltonian(sig, sig, cluster_ops, clustered_ham_0, nbody=1)

        #todo:  setting initial value to zero only makes sense when our reference space is projected out. 
        #       if it's not, then we want to add the reference state components |guess> += |ref><ref|guess>
        #
        x_vector = zeros(T, dim)
        # x_vector = get_vector(Sx)[:,r] + get_vector(sig)[:, r] * 0.1
        # x_vector = get_vector(sig)[:, r] * 0.1
        time = @elapsed alloc = @allocated x, solver = IterativeSolvers.cg!(x_vector, Axx, br, log=true, maxiter=max_iter, verbose=true, abstol=tol)
        verbose < 1 || @printf(" %-50s%10.6f seconds %10.2e Gb\n", "Time to solve for PT1 with conjugate gradient: ", time, alloc/1e9)

        set_vector!(psi1, x_vector, root=r)
    end

    flush_cache(clustered_ham_0)

    SxC = orth_dot(Sx, psi1)
    #@printf(" %-50s%10.2f\n", "<A|X>C(X): ", SxC)
    #@printf(" <A|X>C(X) = %12.8f\n", SxC)

    tmp = deepcopy(ref)
    zero!(tmp)
    time = @elapsed alloc = @allocated build_sigma!(tmp, psi1, cluster_ops, clustered_ham)
    verbose < 1 || @printf(" %-50s%10.6f seconds %10.2e Gb\n", "Compute <0|H|1>: ", time, alloc/1e9)

    ecorr = nonorth_dot(tmp, ref)
    
    for r in 1:R
        SS = orth_dot(Sx, Sx)
        @printf(" SxC[r] %12.8f SxSx %12.8f\n", SxC[r], SS[r])
    end
    
    e_pt2 = zeros(T, R)
    for r in 1:R
        e_pt2[r] = (e_ref[r] + ecorr[r]) / (1 + SxC[r])
        @printf(" State %3i: %-35s%14.8f\n", r, "E(PT2) corr: ", e_pt2[r] - e_ref[r])
    end
    for r in 1:R
        @printf(" State %3i: %-35s%14.8f\n", r, "E(PT2): ", e_pt2[r])
    end

    return psi1, e_pt2

end





"""
    function do_fois_pt2(ref::SPTstate, cluster_ops, clustered_ham;
            H0          = "Hcmf",
            max_iter    = 50,
            nbody       = 4,
            thresh_foi  = 1e-6,
            tol         = 1e-5,
            opt_ref     = true,
            verbose     = true)

Do PT2
"""
function do_fois_pt2(ref::SPTstate{T,N,R}, cluster_ops, clustered_ham;
            H0          = "Hcmf",
            max_iter    = 50,
            nbody       = 4,
            thresh_foi  = 1e-6,
            tol         = 1e-5,
            opt_ref     = true,
            verbose     = true) where {T,N,R}
    @printf(" |== Solve for SPT PT1 Wavefunction ================================\n")
    println(" H0          : ", H0          ) 
    println(" max_iter    : ", max_iter    ) 
    println(" nbody       : ", nbody       ) 
    println(" thresh_foi  : ", thresh_foi  ) 
    println(" tol         : ", tol         ) 
    println(" opt_ref     : ", opt_ref     ) 
    println(" verbose     : ", verbose     ) 
    @printf("\n")
    @printf(" %-50s", "Length of Reference: ")
    @printf("%10i\n", length(ref))

    # 
    # Solve variationally in reference space
    ref_vec = deepcopy(ref)
    
    if opt_ref 
        @printf(" %-50s\n", "Solve zeroth-order problem: ")
        time = @elapsed e0, ref_vec = ci_solve(ref_vec, cluster_ops, clustered_ham, conv_thresh=tol)
        @printf(" %-50s%10.6f seconds\n", "Diagonalization time: ",time)
    end

    #
    # Get First order wavefunction
    println()
    @printf(" %-50s\n", "Compute compressed FOIS: ")
    time = @elapsed pt1_vec  = build_compressed_1st_order_state(ref_vec, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh_foi)
    @printf(" %-50s%10.6f seconds\n", "Time spent building compressed FOIS: ",time)
    # display(orth_overlap(pt1_vec, pt1_vec))
    #display(eigen(get_vector(pt1_vec)'*get_vector(pt1_vec)))
    project_out!(pt1_vec, ref)
    
    # 
    # Compress FOIS
    norm1 = sqrt.(orth_dot(pt1_vec, pt1_vec))
    dim1 = length(pt1_vec)
    pt1_vec = compress(pt1_vec, thresh=thresh_foi)
    norm2 = sqrt.(orth_dot(pt1_vec, pt1_vec))
    dim2 = length(pt1_vec)
    @printf(" %-50s%10i → %-10i (thresh = %8.1e)\n", "FOIS Compressed from: ", dim1, dim2, thresh_foi)
    #@printf(" %-50s%10.2e → %-10.2e (thresh = %8.1e)\n", "Norm of |1>: ",norm1, norm2, thresh_foi)
    @printf(" %-50s", "Overlap between <1|0>: ")
    ovlp = nonorth_dot(pt1_vec, ref_vec, verbose=0)
    [@printf("%10.6f ", ovlp[r]) for r in 1:R]
    println()

    # 
    # Solve for first order wavefunction 
    @printf(" %-50s%10i\n", "Compute PT vector. Reference space dim: ", length(ref_vec))
    # display(orth_overlap(pt1_vec, pt1_vec))
    pt1_vec, e_pt2= hylleraas_compressed_mp2(pt1_vec, ref_vec, cluster_ops, clustered_ham)
    # pt1_vec, e_pt2= hylleraas_compressed_mp2(pt1_vec, ref_vec, cluster_ops, clustered_ham; tol=tol, max_iter=max_iter, H0=H0)
    #@printf(" E(Ref)      = %12.8f\n", e0[1])
    #@printf(" E(PT2) tot  = %12.8f\n", e_pt2)
    @printf(" ==================================================================|\n")
    return e_pt2, pt1_vec 
end


"""
    compute_pt2_energy(ref::SPTstate{T,N,R}, cluster_ops, clustered_ham;
                            H0          = "Hcmf",
                            nbody       = 4,
                            thresh_foi  = 1e-6,
                            max_number  = nothing,
                            opt_ref     = true,
                            ci_tol      = 1e-6,
                            verbose     = true) where {T,N,R}

Directly compute the PT2 energy, in parallel, with each thread handling a single `FockConfig` at a time.
"""
function compute_pt2_energy(ref::SPTstate{T,N,R}, cluster_ops, clustered_ham;
                            H0          = "Hcmf",
                            nbody       = 4,
                            thresh_foi  = 1e-6,
                            max_number  = nothing,
                            opt_ref     = true,
                            ci_tol      = 1e-6,
                            verbose     = 1,
                            prescreen   = false,
                            compress_twice = false) where {T,N,R}
    println()
    println(" |...................................SPT-PT2............................................")
    verbose < 1 || println(" H0          : ", H0          ) 
    verbose < 1 || println(" nbody       : ", nbody       ) 
    verbose < 1 || println(" thresh_foi  : ", thresh_foi  ) 
    verbose < 1 || println(" max_number  : ", max_number  ) 
    verbose < 1 || println(" opt_ref     : ", opt_ref     ) 
    verbose < 1 || println(" ci_tol      : ", ci_tol       ) 
    verbose < 1 || println(" verbose     : ", verbose     ) 
    verbose < 1 || @printf("\n")
    verbose < 1 || @printf(" %-50s", "Length of Reference: ")
    verbose < 1 || @printf("%10i\n", length(ref))
    
    lk = ReentrantLock()

    # 
    # Solve variationally in reference space
    ref_vec = deepcopy(ref)
    clusters = ref_vec.clusters
   
    E0 = zeros(T,R)

    if opt_ref 
        @printf(" %-50s\n", "Solve zeroth-order problem: ")
        time = @elapsed E0, ref_vec = ci_solve(ref_vec, cluster_ops, clustered_ham, conv_thresh=ci_tol)
        @printf(" %-50s%10.6f seconds\n", "Diagonalization time: ",time)
    else
        @printf(" %-50s", "Compute zeroth-order energy: ")
        flush(stdout)
        @time E0 = compute_expectation_value(ref_vec, cluster_ops, clustered_ham)
    end

    # 
    # get E0 = <0|H0|0>
    clustered_ham_0 = extract_1body_operator(clustered_ham, op_string = H0)
    @printf(" %-50s", "Compute <0|H0|0>: ")
    @time F0 = compute_expectation_value(ref_vec, cluster_ops, clustered_ham_0)
    
    if verbose > 0 
        @printf(" %5s %12s %12s\n", "Root", "<0|H|0>", "<0|F|0>")
        for r in 1:R
            @printf(" %5s %12.8f %12.8f\n",r, E0[r], F0[r])
        end
    end

    # 
    # define batches (FockConfigs present in resolvant)
    jobs = Dict{FockConfig{N},Vector{Tuple}}()
    for (fock_ket, configs_ket) in ref_vec.data
        for (ftrans, terms) in clustered_ham
            fock_x = ftrans + fock_ket

            #
            # check to make sure this fock config doesn't have negative or too many electrons in any cluster
            all(f[1] >= 0 for f in fock_x) || continue 
            all(f[2] >= 0 for f in fock_x) || continue 
            all(f[1] <= length(clusters[fi]) for (fi,f) in enumerate(fock_x)) || continue 
            all(f[2] <= length(clusters[fi]) for (fi,f) in enumerate(fock_x)) || continue 
           
            job_input = (terms, fock_ket, configs_ket)
            if haskey(jobs, fock_x)
                push!(jobs[fock_x], job_input)
            else
                jobs[fock_x] = [job_input]
            end
        end
    end


    jobs_vec = []
    for (fock_x, job) in jobs
        push!(jobs_vec, (fock_x, job))
    end

    println(" Number of jobs:    ", length(jobs_vec))
    println(" Number of threads: ", Threads.nthreads())
    BLAS.set_num_threads(1)
    flush(stdout)
    
    #ham_0s = Vector{ClusteredOperator}()
    #for t in Threads.nthreads() 
    #    push!(ham_0s, extract_1body_operator(clustered_ham, op_string = H0) )
    #end


    e2_thread = Vector{Vector{Float64}}()
    for tid in 1:Threads.maxthreadid()
        push!(e2_thread, zeros(T,R))
    end

    #tmp = ceil(length(jobs_vec)/100)
    tmp = Int(round(length(jobs_vec)/100))
    if tmp == 0
        tmp += 1
    end
    verbose < 2 || println(" |----------------------------------------------------------------------------------------------------|")
    verbose < 2 || println(" |0%                                                                                              100%|")
    verbose < 2 || print(" |")
    #@profilehtml @Threads.threads :static for job in jobs_vec
    nprinted = 0
    alloc = @allocated t = @elapsed begin
        
        @Threads.threads :static for (jobi,job) in collect(enumerate(jobs_vec))
        #for (jobi,job) in collect(enumerate(jobs_vec))
            fock_sig = job[1]
            tid = Threads.threadid()
            e2_thread[tid] .+= _pt2_job(fock_sig, job[2], ref_vec, cluster_ops, clustered_ham, clustered_ham_0, 
                          nbody, verbose, thresh_foi, max_number, E0, F0, prescreen, compress_twice)
            if verbose > 1
                if  jobi%tmp == 0
                    begin
                        lock(lk)
                        try
                            print("-")
                            nprinted += 1
                            flush(stdout)
                        finally
                            unlock(lk)
                        end
                    end
                end
            end
        end
    end
    flush(stdout)
    verbose < 2 || for i in nprinted+1:100
        print("-")
    end
    verbose < 2 || println("|")
    flush(stdout)
  
    @printf(" %-48s%10.1f s Allocated: %10.1e GB\n", "Time spent computing E2: ",t,alloc*1e-9)
    ecorr = sum(e2_thread) 
   
    E2 = zeros(R)
    for r in 1:R
        E2[r] = E0[r] + ecorr[r]
        @printf(" State %3i: %-35s%14.8f\n", r, "E(PT2) corr: ", ecorr[r])
    end

    @printf(" %5s %12s %12s\n", "Root", "E(0)", "E(2)")
    for r in 1:R
        @printf(" %5s %12.8f %12.8f\n", r, E0[r], E2[r])
    end
    println(" ......................................................................................|")
    
    return E2 
end


function _pt2_job(sig_fock, job, ket::SPTstate{T,N,R}, cluster_ops, clustered_ham, clustered_ham_0,
    nbody, verbose, thresh, max_number, E0, F0, prescreen, compress_twice) where {T,N,R}

    sig = SPTstate(ket.clusters, ket.p_spaces, ket.q_spaces, T=T, R=R)
    add_fockconfig!(sig, sig_fock)

    data = OrderedDict{TuckerConfig{N},Vector{Tucker{T,N,R}}}()

    for jobi in job

        terms, ket_fock, ket_tconfigs = jobi

        for term in terms

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
                    if haskey(ket.p_spaces[ci.idx], sig_fock[ci.idx])
                        push!(tmp, ket.p_spaces[ci.idx][sig_fock[ci.idx]])
                    end
                    if haskey(ket.q_spaces[ci.idx], sig_fock[ci.idx])
                        push!(tmp, ket.q_spaces[ci.idx][sig_fock[ci.idx]])
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
                    #                    if term isa ClusteredTerm2B && false 
                    #                                    
                    #                        #@profilehtml for ii in 1:3
                    #                        #    form_sigma_block_expand(term, cluster_ops,
                    #                        #                               sig_fock, sig_tconfig,
                    #                        #                               ket_fock, ket_tconfig, ket_tuck,
                    #                        #                               max_number=max_number,
                    #                        #                               prescreen=thresh)
                    #                        #end
                    #                        @btime form_sigma_block_expand($term, $cluster_ops,
                    #                                                       $sig_fock, $sig_tconfig,
                    #                                                       $ket_fock, $ket_tconfig, $ket_tuck,
                    #                                                       max_number=$max_number,
                    #                                                       prescreen=$thresh)
                    #                        error("stop")
                    #                    end


                    if length(sig_tuck) == 0
                        continue
                    end
                    if norm(sig_tuck) < thresh
                        continue
                    end


                    #compress new addition
                    sig_tuck = compress(sig_tuck, thresh=thresh)

                    length(sig_tuck) > 0 || continue

                    #add to current sigma vector
                    if haskey(sig[sig_fock], sig_tconfig)

                        if haskey(data, sig_tconfig)
                            push!(data[sig_tconfig], sig_tuck)
                        else
                            data[sig_tconfig] = [sig[sig_fock][sig_tconfig], sig_tuck]
                        end

                        #compress result
                        #sig[sig_fock][sig_tconfig] = compress(sig[sig_fock][sig_tconfig], thresh=thresh)
                    else
                        sig[sig_fock][sig_tconfig] = sig_tuck
                    end

                end
            end
        end
    end

    # 
    # Add results together to get final FOIS for this job
    for (tconfig, tucks) in data
        if compress_twice
            sig[sig_fock][tconfig] = compress(nonorth_add(tucks), thresh=thresh)
        else
            sig[sig_fock][tconfig] = nonorth_add(tucks)
        end
    end

    # Compute PT2 energy for this job
    v_pt, e_pt, ecorr = compute_pt1_wavefunction(sig, ket, cluster_ops, clustered_ham, clustered_ham_0, E0, F0, verbose=0)

    return ecorr 
end

"""
    compute_pt2_energy2(ref::SPTstate{T,N,R}, cluster_ops, clustered_ham;
                            H0          = "Hcmf",
                            nbody       = 4,
                            thresh_foi  = 1e-6,
                            max_number  = nothing,
                            opt_ref     = true,
                            ci_tol      = 1e-6,
                            verbose     = true) where {T,N,R}

Directly compute the PT2 energy, in parallel, with each thread handling a single `FockConfig` at a time.
"""
function compute_pt2_energy2(ref::SPTstate{T,N,R}, cluster_ops, clustered_ham;
                            H0          = "Hcmf",
                            nbody       = 4,
                            thresh_foi  = 1e-6,
                            max_number  = nothing,
                            opt_ref     = true,
                            ci_tol      = 1e-6,
                            verbose     = 1,
                            prescreen   = false,
                            compress_twice = false) where {T,N,R}
    println()
    println(" |...................................SPT-PT2............................................")
    verbose < 1 || println(" H0          : ", H0          ) 
    verbose < 1 || println(" nbody       : ", nbody       ) 
    verbose < 1 || println(" thresh_foi  : ", thresh_foi  ) 
    verbose < 1 || println(" max_number  : ", max_number  ) 
    verbose < 1 || println(" opt_ref     : ", opt_ref     ) 
    verbose < 1 || println(" ci_tol      : ", ci_tol       ) 
    verbose < 1 || println(" verbose     : ", verbose     ) 
    verbose < 1 || @printf("\n")
    verbose < 1 || @printf(" %-50s", "Length of Reference: ")
    verbose < 1 || @printf("%10i\n", length(ref))
    
    lk = ReentrantLock()

    # 
    # Solve variationally in reference space
    ref_vec = deepcopy(ref)
    clusters = ref_vec.clusters
   
    E0 = zeros(T,R)

    if opt_ref 
        @printf(" %-50s\n", "Solve zeroth-order problem: ")
        time = @elapsed E0, ref_vec = ci_solve(ref_vec, cluster_ops, clustered_ham, conv_thresh=ci_tol)
        @printf(" %-50s%10.6f seconds\n", "Diagonalization time: ",time)
    else
        @printf(" %-50s", "Compute zeroth-order energy: ")
        flush(stdout)
        @time E0 = compute_expectation_value(ref_vec, cluster_ops, clustered_ham)
    end

    # 
    # get E0 = <0|H0|0>
    clustered_ham_0 = extract_1body_operator(clustered_ham, op_string = H0)
    @printf(" %-50s", "Compute <0|H0|0>: ")
    @time F0 = compute_expectation_value(ref_vec, cluster_ops, clustered_ham_0)
    
    if verbose > 0 
        @printf(" %5s %12s %12s\n", "Root", "<0|H|0>", "<0|F|0>")
        for r in 1:R
            @printf(" %5s %12.8f %12.8f\n",r, E0[r], F0[r])
        end
    end

    # 
    # define batches (FockConfigs present in resolvant)
    jobs = Dict{FockConfig{N},Vector{Tuple}}()
    for (fock_ket, configs_ket) in ref_vec.data
        for (ftrans, terms) in clustered_ham
            fock_x = ftrans + fock_ket

            #
            # check to make sure this fock config doesn't have negative or too many electrons in any cluster
            all(f[1] >= 0 for f in fock_x) || continue 
            all(f[2] >= 0 for f in fock_x) || continue 
            all(f[1] <= length(clusters[fi]) for (fi,f) in enumerate(fock_x)) || continue 
            all(f[2] <= length(clusters[fi]) for (fi,f) in enumerate(fock_x)) || continue 
           
            job_input = (terms, fock_ket, configs_ket)
            if haskey(jobs, fock_x)
                push!(jobs[fock_x], job_input)
            else
                jobs[fock_x] = [job_input]
            end
        end
    end


    jobs_vec = []
    for (fock_x, job) in jobs
        push!(jobs_vec, (fock_x, job))
    end

    println(" Number of jobs:    ", length(jobs_vec))
    println(" Number of threads: ", Threads.nthreads())
    BLAS.set_num_threads(1)
    flush(stdout)
    
    #ham_0s = Vector{ClusteredOperator}()
    #for t in Threads.nthreads() 
    #    push!(ham_0s, extract_1body_operator(clustered_ham, op_string = H0) )
    #end


    e2_thread = Vector{Vector{Float64}}()
    for tid in 1:Threads.maxthreadid()
        push!(e2_thread, zeros(T,R))
    end

    #tmp = ceil(length(jobs_vec)/100)
    tmp = Int(round(length(jobs_vec)/100))
    if tmp == 0
        tmp += 1
    end
    verbose < 2 || println(" |----------------------------------------------------------------------------------------------------|")
    verbose < 2 || println(" |0%                                                                                              100%|")
    verbose < 2 || print(" |")
    #@profilehtml @Threads.threads :static for job in jobs_vec
    nprinted = 0
    alloc = @allocated t = @elapsed begin
        
        @Threads.threads :static for (jobi,job) in collect(enumerate(jobs_vec))
        #for (jobi,job) in collect(enumerate(jobs_vec))
            fock_sig = job[1]
            tid = Threads.threadid()
            e2_thread[tid] .+= _pt2_job2(fock_sig, job[2], ref_vec, cluster_ops, clustered_ham, clustered_ham_0, 
                          nbody, verbose, thresh_foi, max_number, E0, F0, prescreen, compress_twice)
            if verbose > 1
                if  jobi%tmp == 0
                    begin
                        lock(lk)
                        try
                            print("-")
                            nprinted += 1
                            flush(stdout)
                        finally
                            unlock(lk)
                        end
                    end
                end
            end
        end
    end
    flush(stdout)
    verbose < 2 || for i in nprinted+1:100
        print("-")
    end
    verbose < 2 || println("|")
    flush(stdout)
  
    @printf(" %-48s%10.1f s Allocated: %10.1e GB\n", "Time spent computing E2: ",t,alloc*1e-9)
    ecorr = sum(e2_thread) 
   
    E2 = zeros(R)
    for r in 1:R
        E2[r] = E0[r] + ecorr[r]
        @printf(" State %3i: %-35s%14.8f\n", r, "E(PT2) corr: ", ecorr[r])
    end

    @printf(" %5s %12s %12s\n", "Root", "E(0)", "E(2)")
    for r in 1:R
        @printf(" %5s %12.8f %12.8f\n", r, E0[r], E2[r])
    end
    println(" ......................................................................................|")
    
    return E2 
end


function _pt2_job2(sig_fock, job, ket::SPTstate{T,N,R}, cluster_ops, clustered_ham, clustered_ham_0,
    nbody, verbose, thresh, max_number, E0, F0, prescreen, compress_twice) where {T,N,R}

    tconfigs_to_process = Dict{TuckerConfig{N}, Vector{Vector{Any}}}()

    ecorr = Vector{T}([0.0 for i in 1:R])

    for jobi in job

        terms, ket_fock, ket_tconfigs = jobi

        for term in terms

            length(term.clusters) <= nbody || continue

            for (ket_tconfig, ket_tuck) in ket_tconfigs

                available = [] 
                for ci in term.clusters
                    tmp = []
                    if haskey(ket.p_spaces[ci.idx], sig_fock[ci.idx])
                        push!(tmp, ket.p_spaces[ci.idx][sig_fock[ci.idx]])
                    end
                    if haskey(ket.q_spaces[ci.idx], sig_fock[ci.idx])
                        push!(tmp, ket.q_spaces[ci.idx][sig_fock[ci.idx]])
                    end
                    push!(available, tmp)
                end
                for prod in Iterators.product(available...)
                    sig_tconfig = [ket_tconfig.config...]
                    for cidx in 1:length(term.clusters)
                        ci = term.clusters[cidx]
                        sig_tconfig[ci.idx] = prod[cidx]
                    end
                    sig_tconfig = TuckerConfig(sig_tconfig)
                    # println(sig_tconfig)
                    # println([term, ket_fock, ket_tconfig])
                    if haskey(tconfigs_to_process, sig_tconfig)
                        push!(tconfigs_to_process[sig_tconfig], [term, ket_fock, ket_tconfig])
                    else
                        tconfigs_to_process[sig_tconfig] = [[term, ket_fock, ket_tconfig]]
                    end
                end
            end
        end
    end
    for (sig_tconfig, terms_to_process) in tconfigs_to_process
        curr_tuck = Vector{Any}()    # reset per sig_tconfig: Tucker dims vary between blocks
        for (term, ket_fock, ket_tconfig) in terms_to_process
            ket_tuck = ket[ket_fock][ket_tconfig]
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
            #compress new addition
            sig_tuck = compress(sig_tuck, thresh=thresh)
            
            # println(sig_tuck)
            if length(curr_tuck) == 0
                curr_tuck = sig_tuck
            else
                curr_tuck=nonorth_add([curr_tuck, sig_tuck])
            
            end
            ##curr_tuck=nonorth_add([curr_tuck, sig_tuck])
        end

        if length(curr_tuck) == 0
            continue
        end
        if norm(curr_tuck) < thresh
            continue
        end


        #compress new addition
        curr_tuck = compress(curr_tuck, thresh=thresh)
        if compress_twice
            curr_tuck = compress(curr_tuck, thresh=thresh)
        end
       
        sig = SPTstate(ket.clusters, ket.p_spaces, ket.q_spaces, T=T, R=R)
        add_fockconfig!(sig, sig_fock)
        sig[sig_fock][sig_tconfig] = curr_tuck

        # Compute PT2 energy for this job
        _, _, ecorr_i = compute_pt1_wavefunction(sig, ket, cluster_ops, clustered_ham, clustered_ham_0, E0, F0, verbose=0)
        # println(ecorr_i)
        ecorr += ecorr_i
    end


    return ecorr
end


# =============================================================================
# NEW PT2: block-by-block processing with Tucker rotation
#
# Produces exactly the same E2 as _pt2_job (the reference kernel), for every
# threshold and option combination -- see the parity test in test/test_spt.jl.
# It gets there with a much smaller footprint:
#
#   1. Metadata-only Phase 1: groups (term, ket_fock, ket_tconfig) by sig_tconfig
#      without storing any Tucker data (same as _pt2_job2 Phase 1).
#
#   2. Collect-then-add Phase 2: for each sig_tconfig, calls form_sigma_block_expand
#      once per contributing (term, ket_fock, ket_tconfig), collects into a Vector,
#      then spans them ONCE — no O(K²) iterative SVD merging.  This selects the
#      retained Tucker basis, exactly as _pt2_job does.
#
#   3. Only the factors of that basis are kept (nonorth_basis): the numerator
#      <X|H|0> is then rebuilt in it term by term with form_sigma_block!, which
#      is what build_sigma! does in the reference path.  It must be rebuilt
#      rather than assembled from the Phase-2 expansions -- those are each
#      truncated in their own Tucker basis, and the discarded pieces are not
#      orthogonal to the retained basis, so reusing them biases E2 by O(thresh).
#
#   4. Pseudo-canonical rotation V per cluster, from diagonalising the projected
#      Fock matrix F_proj = U' * F_cluster * U.  Gives the PT denominators
#      without materialising a Fdiag state.
#
#   5. Lightweight F0 and overlap: single-block contractions against ψ0 reusing
#      the calling thread's scratch, instead of the general threaded matvec.
#
#   6. Inline ecorr: computes E² contribution without materialising ψ₁.
#
# Memory vs _pt2_job: never holds the full FOIS SPTstate; peak per thread ≈
#   metadata + one Tucker block at a time (instead of 4-6× full FOIS).
# =============================================================================

"""
    _pt2_job_blockwise(sig_fock, job, ket, cluster_ops, clustered_ham, clustered_ham_0,
                       nbody, thresh, max_number, E0, F0, prescreen, H0) -> ecorr

Compute the PT2 energy correction for one Fock sector using the Tucker-rotation
approach. See module comment above for algorithm details.
"""
@inline function _pt2_available_spaces(ket::SPTstate, sig_fock, ci)
    pspace = ket.p_spaces[ci.idx]
    qspace = ket.q_spaces[ci.idx]
    p = haskey(pspace, sig_fock[ci.idx]) ? pspace[sig_fock[ci.idx]] : nothing
    q = haskey(qspace, sig_fock[ci.idx]) ? qspace[sig_fock[ci.idx]] : nothing
    p === nothing && return q === nothing ? () : (q,)
    q === nothing && return (p,)
    return (p, q)
end

@inline function _pt2_target_tconfig(ket_tconfig::TuckerConfig{N}, clusters,
                                     prod) where {N}
    config = ntuple(Val(N)) do i
        @inbounds for j in eachindex(clusters)
            clusters[j].idx == i && return prod[j]
        end
        return ket_tconfig.config[i]
    end
    return TuckerConfig{N}(config)
end

function _spt_pt2_job_weight(job_item, ref::SPTstate, nbody)
    sig_fock, jobs = job_item
    weight = 0.0
    for (terms, _, configs_ket) in jobs
        ket_size = sum(length(tuck) for (_, tuck) in configs_ket; init=0)
        for term in terms
            length(term.clusters) <= nbody || continue
            ntargets = 1
            for ci in term.clusters
                npq = Int(haskey(ref.p_spaces[ci.idx], sig_fock[ci.idx])) +
                      Int(haskey(ref.q_spaces[ci.idx], sig_fock[ci.idx]))
                ntargets *= npq
            end
            ntargets == 0 && continue
            # Contraction cost grows with the ket Tucker rank, the number of P/Q
            # destinations, and the integral tensor carried by the term.
            weight += float(max(ket_size, 1)) * ntargets *
                      max(length(term.ints), 1)
        end
    end
    return max(weight, 1.0)
end

function _spt_pt2_thread_buckets(jobs_chunk, ref, nbody, nt)
    buckets = [similar(jobs_chunk, 0) for _ in 1:nt]
    loads = zeros(Float64, nt)
    weighted_jobs = [(job, _spt_pt2_job_weight(job, ref, nbody))
                     for job in jobs_chunk]
    sort!(weighted_jobs; by=last, rev=true)
    for (job, weight) in weighted_jobs
        slot = argmin(loads)
        push!(buckets[slot], job)
        loads[slot] += weight
    end
    return buckets
end

"""
    _pt2_accumulate_block!(bra_tuck, sig_fock, sig_tconfig, contributions,
                           cluster_ops, scratch)

Accumulate `Σ_k <X|term_k|ket_k>` into `bra_tuck.core`, where `X` is the block
`(sig_fock, sig_tconfig)` spanned by `bra_tuck.factors` and `contributions` is a
list of `(term, ket_fock, ket_tconfig, ket_tuck)` tuples that have already been
checked with `check_term`.

`form_sigma_block!` returns `bra core + contribution`, so the bra it is handed
must have a zero core; the returned cores are accumulated separately.  This is
the same term-by-term construction `build_sigma!` performs, restricted to a
single destination block.
"""
function _pt2_accumulate_block!(bra_tuck::Tucker{T,N,R}, sig_fock, sig_tconfig,
                                contributions, cluster_ops, scratch) where {T,N,R}
    zero_cores = ntuple(r -> zeros(T, size(bra_tuck.core[r])), R)
    bra_basis  = Tucker{T,N,R}(zero_cores, bra_tuck.factors)
    for (term, ket_fock, ket_tconfig, ket_tuck) in contributions
        add!(bra_tuck.core,
             form_sigma_block!(term, cluster_ops, sig_fock, sig_tconfig,
                               ket_fock, ket_tconfig, bra_basis, ket_tuck,
                               scratch))
    end
    return bra_tuck
end


function _pt2_build_f0_block!(bra_tuck::Tucker{T,N,R}, sig_fock,
                              sig_tconfig, ket::SPTstate{T,N,R}, cluster_ops,
                              clustered_ham_0, scratch) where {T,N,R}
    # form_sigma_block! treats the bra core as an initial value.  Keep an
    # immutable zero-core basis object for each contribution and accumulate the
    # returned cores separately in bra_tuck.
    zero_cores = ntuple(r -> zeros(T, size(bra_tuck.core[r])), R)
    bra_basis = Tucker{T,N,R}(zero_cores, bra_tuck.factors)
    for (ket_fock, ket_tconfigs) in ket
        ftrans = sig_fock - ket_fock
        haskey(clustered_ham_0, ftrans) || continue
        for (ket_tconfig, ket_tuck) in ket_tconfigs
            for term in clustered_ham_0[ftrans]
                check_term(term, sig_fock, sig_tconfig, ket_fock,
                           ket_tconfig) || continue
                contribution = form_sigma_block!(
                    term, cluster_ops, sig_fock, sig_tconfig, ket_fock,
                    ket_tconfig, bra_basis, ket_tuck, scratch)
                add!(bra_tuck.core, contribution)
            end
        end
    end
    return bra_tuck
end

function _pt2_overlap_cores(ket::SPTstate{T,N,R}, sig_fock, sig_tconfig,
                            factors) where {T,N,R}
    haskey(ket, sig_fock) || return nothing
    haskey(ket[sig_fock], sig_tconfig) || return nothing
    ref_tuck = ket[sig_fock][sig_tconfig]
    overlaps = ntuple(i -> ref_tuck.factors[i]' * factors[i], N)
    return ntuple(r -> transform_basis(ref_tuck.core[r], overlaps), R)
end

function _pt2_job_blockwise!(ecorr::Vector{T}, scratch, sig_fock, job,
                        ket::SPTstate{T,N,R}, cluster_ops,
                        clustered_ham, clustered_ham_0::ClusteredOperator{N},
                        nbody, thresh, max_number, E0::Vector{T}, F0::Vector{T},
                        prescreen::Bool, H0::String;
                        compress_twice::Bool=false) where {T,N,R}

    # ------------------------------------------------------------------
    # Phase 1: group (term, ket_fock, ket_tconfig) metadata by sig_tconfig.
    # No Tucker data stored — just lightweight tuples.
    #
    # Note there is deliberately no `nbody` filter here.  `nbody` selects which
    # terms may *generate* FOIS blocks, but the PT2 numerator in each retained
    # block is the exact <X|H|0> over the whole Hamiltonian -- that is what the
    # reference path does, since its `build_sigma!` call takes the default
    # nbody=4 (all terms, 4-body being the highest) regardless of the `nbody`
    # passed to compute_pt2_energy.  The filter is applied in phase 2a, where
    # the basis is chosen.
    # ------------------------------------------------------------------
    Contribution = Tuple{ClusteredTerm{T},FockConfig{N},TuckerConfig{N},Tucker{T,N,R}}
    tconfigs_to_process = Dict{TuckerConfig{N},Vector{Contribution}}()

    for jobi in job
        terms, ket_fock, ket_tconfigs = jobi
        for term in terms
            available = map(ci -> _pt2_available_spaces(ket, sig_fock, ci), term.clusters)
            any(isempty, available) && continue
            for (ket_tconfig, ket_tuck) in ket_tconfigs
                for prod in Iterators.product(available...)
                    sig_tconfig = _pt2_target_tconfig(ket_tconfig, term.clusters, prod)
                    check_term(term, sig_fock, sig_tconfig, ket_fock,
                               ket_tconfig) || continue
                    entry = (term, ket_fock, ket_tconfig, ket_tuck)
                    push!(get!(() -> Contribution[], tconfigs_to_process,
                               sig_tconfig), entry)
                end
            end
        end
    end

    # ------------------------------------------------------------------
    # Phase 2: per sig_tconfig — build H Tucker, rotate, compute ecorr
    # ------------------------------------------------------------------
    clusters = ket.clusters

    for (sig_tconfig, contributions) in tconfigs_to_process

        # -- 2a. Build the retained Tucker basis exactly as _pt2_job does:
        #        compress every term contribution, then combine once.  Only the
        #        compressed blocks are kept; each raw expansion is released as
        #        soon as it has been compressed.
        basis_tucks = Tucker{T,N,R}[]
        for (term, ket_fock, ket_tconfig, ket_tuck) in contributions

            # `nbody` restricts which terms may generate a FOIS block; the
            # numerator built in 2e still uses every term.
            length(term.clusters) <= nbody || continue

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
            length(sig_tuck) == 0 && continue
            norm(sig_tuck)   <  thresh && continue

            # _pt2_job compresses each term contribution before contributions to
            # the same destination block are combined.  Compressing only after
            # the sum selects a different retained basis (the operations do not
            # commute), so keep the reference order.
            sig_tuck = compress(sig_tuck, thresh=thresh)
            length(sig_tuck) == 0 && continue

            push!(basis_tucks, sig_tuck)
        end

        isempty(basis_tucks) && continue

        # -- 2b. The retained basis for this block.  Only the Tucker *factors*
        #        are needed from here on — every core is rebuilt exactly below —
        #        so in the common case take the factors-only path and skip
        #        accumulating a core that would immediately be discarded.
        basis_factors = if length(basis_tucks) == 1
            # _pt2_job stores a lone contribution directly, without the extra
            # compression, so do the same here.
            collect(only(basis_tucks).factors)
        elseif compress_twice
            # compress() truncates on the core, so this branch has to build it.
            collect(compress(nonorth_add(basis_tucks), thresh=thresh).factors)
        else
            nonorth_basis(basis_tucks)
        end
        empty!(basis_tucks)
        core_dims = ntuple(i -> size(basis_factors[i], 2), N)
        any(iszero, core_dims) && continue

        # -- 2c. Pseudo-canonical rotations V: diagonalise U'*F*U for each cluster.
        #        V_rot[ci.idx] is the ki×ki orthogonal rotation matrix.
        #        Fdiag_core[i1,...,iN] = Σ_ci λ_ci[i_ci]  (sum of cluster eigenvalues)
        V_rot      = Vector{Matrix{T}}(undef, N)
        Fdiag_core = zeros(T, core_dims)

        for ci in clusters
            Ui = basis_factors[ci.idx]
            ki = size(Ui, 2)
            F_ci   = cluster_ops[ci.idx][H0][(sig_fock[ci.idx], sig_fock[ci.idx])][sig_tconfig[ci.idx], sig_tconfig[ci.idx]]
            F_proj = Ui' * F_ci * Ui

            if ki > 1
                eig_res       = eigen(Symmetric(F_proj))
                V_rot[ci.idx] = eig_res.vectors
                λ_shape = ntuple(d -> d == ci.idx ? ki : 1, N)
                Fdiag_core .+= reshape(eig_res.values, λ_shape...)
            else
                V_rot[ci.idx] = ones(T, 1, 1)
                Fdiag_core   .+= F_proj[1, 1]
            end
        end

        # -- 2d. Canonical Tucker factors for this block.
        U_canon = ntuple(i -> basis_factors[i] * V_rot[i], N)

        # -- 2e. <X|H|0> in the canonical basis.
        #
        #        This has to be the *exact* matrix element, not the sum of the
        #        truncated expansions that selected the basis above.  Projection
        #        onto the retained basis is linear, so Σ_k P(H_k|0>) is equal to
        #        P(Σ_k H_k|0>) — but only for untruncated H_k|0>.
        #        form_sigma_block_expand discards singular vectors below
        #        `prescreen` in each contribution's own Tucker basis, and those
        #        discarded pieces are not orthogonal to the retained basis, so
        #        reusing them here leaks a truncation error into the numerator
        #        that the standard kernel does not have.  Rebuilding with
        #        form_sigma_block! costs one small dense contraction per
        #        contribution and reproduces build_sigma! term for term.
        H_block = Tucker{T,N,R}(ntuple(_ -> zeros(T, core_dims), R), U_canon)
        _pt2_accumulate_block!(H_block, sig_fock, sig_tconfig, contributions,
                               cluster_ops, scratch)
        H_cores_rot = H_block.core

        # -- 2f. Overlap Sx = <ψ0 | X_canonical>.  Compute just this block;
        #        constructing and deep-copying a temporary SPTstate per block is
        #        unnecessary and dominates allocations for many small sectors.
        Sx_cores = _pt2_overlap_cores(ket, sig_fock, sig_tconfig, U_canon)

        # -- 2g. <X|F|0> in the canonical basis — contract this one block
        #        serially using the scratch buffers owned by the outer PT2
        #        thread.  Calling the general threaded build_sigma! here
        #        allocated ten fresh buffers and scheduling/output containers
        #        for every FOIS block.
        F0_block = Tucker{T,N,R}(ntuple(_ -> zeros(T, core_dims), R), U_canon)
        _pt2_build_f0_block!(F0_block, sig_fock, sig_tconfig, ket, cluster_ops,
                             clustered_ham_0, scratch)

        # -- 2h. Compute ecorr inline (no ψ₁ materialisation, no deepcopy) --
        #        ecorr[r] += Σ_i  num[r,i]² / (F0[r] − Fdiag_core[i] + ε)
        #        num[r,i] = H[r,i] − F0[r,i] − (E0[r]−F0[r])·Sx[r,i]
        Fv    = vec(Fdiag_core)
        nFOIS = length(Fv)

        # Read Sx cores (may be zero if ψ0 doesn't overlap with sig_f0)
        has_Sx = Sx_cores !== nothing

        for r in 1:R
            H_r  = vec(H_cores_rot[r])
            F0_r = vec(F0_block.core[r])
            Sx_r = has_Sx ? vec(Sx_cores[r]) : nothing

            e_corr_r = zero(T)
            dE       = E0[r] - F0[r]
            @inbounds for i in 1:nFOIS
                sx_i  = has_Sx ? Sx_r[i] : zero(T)
                num_i = H_r[i] - F0_r[i] - dE * sx_i
                e_corr_r += num_i * num_i / (F0[r] - Fv[i] + 1e-12)
            end
            ecorr[r] += e_corr_r
        end
    end

    return ecorr
end

function _pt2_job_blockwise(sig_fock, job, ket::SPTstate{T,N,R}, cluster_ops,
                        clustered_ham, clustered_ham_0::ClusteredOperator{N},
                        nbody, thresh, max_number, E0::Vector{T}, F0::Vector{T},
                        prescreen::Bool, H0::String;
                        compress_twice::Bool=false) where {T,N,R}
    scratch = [zeros(T, 1000) for _ in 1:10]
    return _pt2_job_blockwise!(zeros(T, R), scratch, sig_fock, job, ket, cluster_ops,
                               clustered_ham, clustered_ham_0, nbody, thresh,
                               max_number, E0, F0, prescreen, H0,
                               compress_twice=compress_twice)
end


"""
    compute_pt2_energy_blockwise(ref::SPTstate{T,N,R}, cluster_ops, clustered_ham;
                                  H0="Hcmf", nbody=4, thresh_foi=1e-6,
                                  max_number=nothing, opt_ref=true, ci_tol=1e-6,
                                  verbose=1, prescreen=false,
                                  compress_twice=false) -> E2

Compute PT2 energy using the block-by-block Tucker-rotation approach
(`_pt2_job_blockwise`). Agrees with `compute_pt2_energy` to machine precision
for every `thresh_foi`, `prescreen` and `compress_twice` setting -- it selects
the same retained FOIS basis and rebuilds the same exact numerator in it -- but
never materialises a full FOIS `SPTstate` per thread.
"""
function compute_pt2_energy_blockwise(ref::SPTstate{T,N,R}, cluster_ops, clustered_ham;
                                  H0         = "Hcmf",
                                  nbody      = 4,
                                  thresh_foi = 1e-6,
                                  max_number = nothing,
                                  opt_ref    = true,
                                  ci_tol     = 1e-6,
                                  verbose    = 1,
                                  prescreen  = false,
                                  compress_twice = false) where {T,N,R}
    println()
    println(" |...................................SPT-PT2 (fast).......................................")
    verbose < 1 || println(" H0          : ", H0        )
    verbose < 1 || println(" nbody       : ", nbody     )
    verbose < 1 || println(" thresh_foi  : ", thresh_foi)
    verbose < 1 || println(" max_number  : ", max_number)
    verbose < 1 || println(" opt_ref     : ", opt_ref   )
    verbose < 1 || println(" ci_tol      : ", ci_tol    )
    verbose < 1 || println(" verbose     : ", verbose   )
    verbose < 1 || @printf("\n")
    verbose < 1 || @printf(" %-50s%10i\n", "Length of Reference: ", length(ref))

    lk = ReentrantLock()

    # Optionally re-solve / compute reference energy
    ref_vec = deepcopy(ref)
    E0 = zeros(T, R)
    if opt_ref
        @printf(" %-50s\n", "Solve zeroth-order problem: ")
        time_ci = @elapsed E0, ref_vec = ci_solve(ref_vec, cluster_ops, clustered_ham, conv_thresh=ci_tol)
        @printf(" %-50s%10.6f seconds\n", "Diagonalization time: ", time_ci)
    else
        @printf(" %-50s", "Compute zeroth-order energy: ")
        flush(stdout)
        @time E0 = compute_expectation_value(ref_vec, cluster_ops, clustered_ham)
    end

    # Extract 1-body zeroth-order Hamiltonian
    clustered_ham_0 = extract_1body_operator(clustered_ham, op_string=H0)
    @printf(" %-50s", "Compute <0|H0|0>: ")
    @time F0 = compute_expectation_value(ref_vec, cluster_ops, clustered_ham_0)

    if verbose > 0
        @printf(" %5s %12s %12s\n", "Root", "<0|H|0>", "<0|F|0>")
        for r in 1:R
            @printf(" %5s %12.8f %12.8f\n", r, E0[r], F0[r])
        end
    end

    jobs_vec = _spt_make_fock_jobs(ref_vec, cluster_ops, clustered_ham;
                                   require_cluster_space=false)
    println(" Number of jobs:    ", length(jobs_vec))
    println(" Number of threads: ", Threads.nthreads())
    BLAS.set_num_threads(1)
    flush(stdout)

    nt = Threads.nthreads(:default)
    e2_thread = [zeros(T, R) for _ in 1:nt]
    pt2_scratch = [[zeros(T, 1000) for _ in 1:10]
                   for _ in 1:nt]
    thread_jobs = _spt_pt2_thread_buckets(jobs_vec, ref_vec, nbody, nt)

    tmp = max(1, Int(round(length(jobs_vec) / 100)))
    verbose < 2 || println(" |----------------------------------------------------------------------------------------------------|")
    verbose < 2 || println(" |0%                                                                                              100%|")
    verbose < 2 || print(" |")

    nprinted = 0
    njobs_done = 0
    alloc = @allocated t = @elapsed begin
        @Threads.threads :static for slot in 1:nt
            for (fock_sig, job) in thread_jobs[slot]
                _pt2_job_blockwise!(e2_thread[slot], pt2_scratch[slot], fock_sig,
                                    job, ref_vec, cluster_ops, clustered_ham,
                                    clustered_ham_0, nbody, thresh_foi,
                                    max_number, E0, F0, prescreen, H0,
                                    compress_twice=compress_twice)
                if verbose > 1
                    lock(lk)
                    try
                        njobs_done += 1
                        if njobs_done % tmp == 0
                            print("-"); nprinted += 1; flush(stdout)
                        end
                    finally
                        unlock(lk)
                    end
                end
            end
        end
    end
    flush(stdout)
    verbose < 2 || for _ in nprinted+1:100; print("-"); end
    verbose < 2 || println("|")
    flush(stdout)

    @printf(" %-48s%10.1f s Allocated: %10.1e GB\n", "Time spent computing E2: ", t, alloc*1e-9)
    ecorr = sum(e2_thread)

    E2 = zeros(R)
    for r in 1:R
        E2[r] = E0[r] + ecorr[r]
        @printf(" State %3i: %-35s%14.8f\n", r, "E(PT2) corr: ", ecorr[r])
    end
    @printf(" %5s %12s %12s\n", "Root", "E(0)", "E(2)")
    for r in 1:R
        @printf(" %5s %12.8f %12.8f\n", r, E0[r], E2[r])
    end
    println(" ......................................................................................|")

    return E2
end
