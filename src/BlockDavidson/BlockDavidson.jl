module BlockDavidson
using LinearAlgebra
using Printf
using InteractiveUtils

export LinOpMat 
export Davidson
export eigs

mutable struct Davidson{T}
    op 
    dim::Int
    nroots::Int
    max_iter::Int
    max_ss_vecs::Int
    tol::Float64
    converged::Bool
    status::Vector{Bool}        # converged status of each root 
    iter_curr::Int
    vec_curr::Array{T,2}
    sig_curr::Array{T,2}
    vec_prev::Array{T,2}
    sig_prev::Array{T,2}
    ritz_v::Array{T,2}
    ritz_e::Vector{T}
    resid::Vector{T}
    lindep::Float64
    lindep_thresh::Float64
end

"""
    Davidson(op; 
    max_iter=100, 
    max_ss_vecs=8, 
    tol=1e-8, 
    nroots=1, 
    v0=nothing, 
    lindep_thresh=1e-10, 
    T=Float64)

TBW
"""
function Davidson(op; 
    max_iter=100, 
    max_ss_vecs=8, 
    tol=1e-8, 
    nroots=1, 
    v0=nothing, 
    lindep_thresh=1e-10, 
    T=Float64)

    size(op)[1] == size(op)[2] || throw(DimensionMismatch)
    dim = size(op)[1]
    if v0===nothing
        F = qr(rand(T, dim,nroots))
        v0 = Matrix(F.Q)
    else
        size(v0,1) == size(op,1) || throw(DimensionMismatch)
        size(v0,2) == nroots || throw(DimensionMismatch)
    end
    #display(v0'*v0)
    return Davidson{T}(op, 
                    dim, 
                    nroots, 
                    max_iter,
                    max_ss_vecs*nroots,
                    tol,
                    false, 
                    [false for i in 1:nroots],
                    0,
                    v0,
                    v0,
                    zeros(T,dim,0),
                    zeros(T,dim,0),
                    zeros(T,nroots,nroots),
                    zeros(T,nroots),
                    zeros(T,nroots),
                    1.0,
                    lindep_thresh)
end

mutable struct LinOpMat{T} <: AbstractMatrix{T} 
    matvec
    dim::Int
    sym::Bool
end

Base.size(lop::LinOpMat{T}) where {T} = return (lop.dim,lop.dim)
Base.:(*)(lop::LinOpMat{T}, v::AbstractVector{T}) where {T} = return lop.matvec(v)
Base.:(*)(lop::LinOpMat{T}, v::AbstractMatrix{T}) where {T} = return lop.matvec(v)
issymmetric(lop::LinOpMat{T}) where {T} = return lop.sym
    



function print_iter(solver::Davidson{T}) where T<:Real
    @printf(" Iter: %3i SS: %-4i", solver.iter_curr, size(solver.vec_prev)[2])
    @printf(" E: ")
    for i in 1:solver.nroots
        if solver.status[i]
            @printf("%13.8f* ", solver.ritz_e[i])
        else
            @printf("%13.8f  ", solver.ritz_e[i])
        end
    end
    @printf(" R: ")
    for i in 1:solver.nroots
        if solver.status[i]
            @printf("%5.1e* ", solver.resid[i])
        else
            @printf("%5.1e  ", solver.resid[i])
        end
    end
    @printf(" LinDep: ")
    @printf("%5.1e* ", solver.lindep)
    println("")
    flush(stdout)
end

function print_iter(solver::Davidson{T}) where T<:Complex
    @printf(" Iter: %3i SS: %-4i", solver.iter_curr, size(solver.vec_prev)[2])
    @printf(" E: ")
    for i in 1:solver.nroots
        if solver.status[i]
            @printf("%13.8f %13.8fim* ", real(solver.ritz_e[i]), imag(solver.ritz_e[i]))
        else
            @printf("%13.8f %13.8fim  ", real(solver.ritz_e[i]), imag(solver.ritz_e[i]))
        end
    end
    @printf(" R: ")
    for i in 1:solver.nroots
        if solver.status[i]
            @printf("%5.1e* ", abs(solver.resid[i]))
        else
            @printf("%5.1e  ", abs(solver.resid[i]))
        end
    end
    @printf(" LinDep: ")
    @printf("%5.1e* ", abs(solver.lindep))
    println("")
    flush(stdout)
end


"""
    _apply_diagonal_precond!(res_s::Vector{T}, Adiag::Vector{T}, denom::T) where {T}

TBW
"""
function apply_diagonal_precond!(res_s::Vector{T}, Adiag::Vector{T}, denom::T) where {T}
    dim = length(Adiag)
    length(res_s) == length(Adiag) || throw(DimensionMismatch)
    # res_s .= -100 .* res_s ./ (Adiag .- denom)
    
    # @inbounds @simd 
    for i in 1:dim
        res_s[i] = res_s[i] / (denom - Adiag[i])
    end
end

function iteration(solver::Davidson; Adiag=nothing, iprint=0, precond_start_thresh=1.0)

    # 
    # project out v_prev from v_curr
    solver.vec_curr = solver.vec_curr - solver.vec_prev * (solver.vec_prev' * solver.vec_curr)
    solver.vec_curr = Matrix(qr(solver.vec_curr).Q) 
    
    #
    # perform σ_curr = A*v_curr 
    solver.sig_curr = Matrix(solver.op * solver.vec_curr)
   
    #
    # add these new vectors to previous quantities
    solver.sig_prev = hcat(solver.sig_prev, solver.sig_curr)
    solver.vec_prev = hcat(solver.vec_prev, solver.vec_curr)
   
    #
    # Check orthogonality
    ss_metric = solver.vec_prev'*solver.vec_prev
    solver.lindep = abs(1.0 - det(ss_metric))

    #
    # form H in current subspace
    Hss = solver.vec_prev' * solver.sig_prev
    F = eigen(Hss)
    idx = sortperm([(real(i), imag(i)) for i in F.values])

    ss_size = min(size(solver.vec_prev,2), solver.max_ss_vecs)
    
    idx = idx[1:ss_size]

    solver.ritz_e = F.values[idx]
    solver.ritz_v = F.vectors[:,idx]
    
    
    solver.sig_prev = solver.sig_prev * solver.ritz_v 
    solver.vec_prev = solver.vec_prev * solver.ritz_v
    Hss = solver.ritz_v' * Hss * solver.ritz_v

    res = deepcopy(solver.sig_prev[:,1:solver.nroots])
    for s in 1:solver.nroots
        res[:,s] .-= solver.vec_prev[:,s] * Hss[s,s]
    end

    
    

    #solver.statusconv = [false for i in 1:solver.nroots]
    for s in 1:solver.nroots
        solver.resid[s] = norm(res[:,s])
        if norm(res[:,s]) <= solver.tol
            solver.status[s] = true
            continue
        else
            solver.status[s] = false 
        end
        if Adiag !== nothing && solver.status[s] == false && solver.resid[s] < precond_start_thresh
        # if Adiag != nothing && solver.status[s] == false
            # (a `tmp = deepcopy(res)` used to sit here, assigned and never read.
            #  It copied the whole dim x nroots residual once per root per
            #  iteration -- 1.3 GB of garbage per iteration at dim 1e7, R=4.)
            for i in 1:length(Adiag)
                res[i, s] = res[i, s] / (solver.ritz_e[s] - Adiag[i] + 1e-12)
                # if abs(solver.ritz_e[s] - Adiag[i]) < solver.tol
                #     res[i,s] = 0
                # else
                #     res[i,s] = res[i,s] / (solver.ritz_e[s] - Adiag[i])
                # end
            end
        end
    end


    res = res - solver.vec_prev * (solver.vec_prev' * res)
    return Matrix(qr(res).Q) 
end

"""
    eigs(solver::Davidson; Adiag=nothing, iprint=0, precond_start_thresh=1e-1)

TBW
"""
function eigs(solver::Davidson; Adiag=nothing, iprint=0, precond_start_thresh=1e-1)

    for iter = 1:solver.max_iter
        #@btime $solver.vec_curr = $iteration($solver)
        solver.vec_curr = iteration(solver, Adiag=Adiag, iprint=iprint, precond_start_thresh=precond_start_thresh)
        solver.iter_curr = iter
        print_iter(solver)
        if all(solver.status)
            solver.converged = true
            break
        end
        if solver.lindep > solver.lindep_thresh && iter < solver.max_iter
            println(" Linear dependency detected. Restarting.")
            flush(stdout)
            F = qr(solver.vec_prev[:,1:solver.nroots])
            solver.vec_curr = Matrix(F.Q)
            solver.sig_curr = Matrix(F.Q)
            solver.vec_prev = zeros(eltype(solver.vec_prev), solver.dim, 0)
            solver.sig_prev = zeros(eltype(solver.sig_prev), solver.dim, 0)
            solver.ritz_v = zeros(eltype(solver.ritz_v), solver.nroots, solver.nroots)
            solver.ritz_e = zeros(eltype(solver.ritz_e), solver.nroots)
            solver.resid  = zeros(eltype(solver.resid),  solver.nroots)
        end
    end
    return solver.ritz_e[1:solver.nroots], solver.vec_prev[:,1:solver.nroots]
    #return solver.fritz_e[1:solver.nroots], solver.vec_prev*solver.ritz_v[:,1:solver.nroots]
end

end
