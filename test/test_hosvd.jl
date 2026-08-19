using TPSChem
using Printf
using Test
using Random
using LinearAlgebra 

@testset "hosvd" begin
    atoms = []
    clusters = []
    na = 0
    nb = 0
    init_fspace = []
    
    function generate_H_ring(n,radius)
        theta = 2*pi/n

        atoms = []
        for i in 0:n-1
            push!(atoms,Atom(i+1,"H",[radius*cos(theta*i), radius*sin(theta*i), 0]))
        end
        return atoms
    end

    #
    # Test basic Tucker stuff
    Random.seed!(2);
    A = rand(4,6,3,3,5)
    #tuck = TPSChem.Tucker(A, thresh=20, verbose=1)
    tuck = TPSChem.Tucker((A,), thresh=10, verbose=1)
    
    display(size.(tuck.core))
    display(size.(tuck.factors))
    B = TPSChem.recompose(tuck)
    println()
    println(TPSChem.dims_small(tuck))
    println(TPSChem.dims_large(tuck))
    @test all(TPSChem.dims_small(tuck) .== [1, 1, 1, 1, 1])
    @test all(TPSChem.dims_large(tuck) .== [4, 6, 3, 3, 5])
     
    A = rand(4,6,3,3,5)
    tuck = TPSChem.Tucker(A, thresh=-1, verbose=1)
    B = TPSChem.recompose(tuck)
    @test isapprox(abs.(A), abs.(B[1]), atol=1e-12)

    A = rand(4,6,3,3,5)*.1
    B = rand(4,6,3,3,5)*.1
    C = A+B


    #tuckA = TPSChem.Tucker(A, thresh=-1, verbose=1, max_number=2)
    #tuckB = TPSChem.Tucker(B, thresh=-1, verbose=1, max_number=2)
    #tuckC = TPSChem.Tucker(C, thresh=-1, verbose=1, max_number=2)
    tuckA = TPSChem.Tucker(A, thresh=-1, verbose=0)
    tuckB = TPSChem.Tucker(B, thresh=-1, verbose=0)
    tuckC = TPSChem.Tucker(C, thresh=-1, verbose=0)

    tuckD = TPSChem.Tucker(tuckC, R=3)
    TPSChem.randomize!(tuckD)

    tuckDc = TPSChem.compress(tuckD)

    # test Tucker addition
    test = TPSChem.nonorth_add(tuckA, tuckB)
    @test isapprox(TPSChem.orth_dot(tuckC,tuckC), TPSChem.orth_dot(test,test), atol=1e-12)
    @test isapprox(TPSChem.nonorth_dot(tuckC,tuckC), TPSChem.nonorth_dot(test,test), atol=1e-12)


    #
    # Now test basis transformation
    A = rand(4,6,3,3,5)*.1
    
    trans1 = Dict{Int,Matrix{Float64}}() 
    trans1[2] = rand(6,5)
    trans1[4] = rand(3,2)

    trans2 = Vector{Matrix{Float64}}([])
    for i = 1:5
        if haskey(trans1, i)
            push!(trans2, trans1[i])
        else
            push!(trans2, Matrix(1.0I,size(A,i),size(A,i)))
        end
    end

    display((length(A), size(A)))
    
    A1 = TPSChem.transform_basis(A, trans1)
    display((length(A1), size(A1)))
    
    A2 = TPSChem.transform_basis(A, trans2)
    display((length(A1), size(A2)))
    
    #A2 = TPSChem.tucker_recompose(A, trans2)
    #display((length(A2), size(A2)))
    
  

    scr = Vector{Vector{Float64}}([Vector{Float64}([]) for i in 1:ndims(A)])
    # A3 = TPSChem.transform_basis(A, trans2, scr)
    # display((length(A3), size(A3)))

    # @test norm(A3-A2) < 1e-16
    
   
    # # Now test transpose
    # trans2 = [Matrix(t') for t in trans2]
    # println("old")
    # @timev A2 = TPSChem.transform_basis(A, trans2, trans=true)
    # println("new")
    # @timev A3 = TPSChem.transform_basis(A, trans2, scr, trans=true)
    # # println("newnew")
    # # @timev A4 = TPSChem.transform_basis(A, trans2, scr[1], scr[2], trans=true)
    
    # println(norm(A2-A3))
    # # println(norm(A2-A4))
    # @test norm(A3-A2) < 1e-16
    # # @test norm(A4-A2) < 1e-16

    tuckA = TPSChem.Tucker(rand(2,3,4,5,6,8))
    tuckB = TPSChem.Tucker(rand(2,3,4,5,6,8))
    tuckA = TPSChem.compress(tuckA)
    tuckB = TPSChem.compress(tuckB)
    @timev tuckC = TPSChem.nonorth_add([tuckA, tuckB, tuckB, tuckA])
    
    scr = Vector{Vector{Float64}}([Vector{Float64}([]) for i in 1:ndims(tuckA)])
    @timev tuckD = TPSChem.nonorth_add([tuckA, tuckB, tuckB, tuckA], scr)
    println(norm(tuckC))
    println(norm(tuckD))
    @test abs(norm(tuckC) - norm(tuckD)) < 1e-12

    #
    # transform_basis: the Dict method (partial mode list) must agree with the
    # NTuple method (all modes) when the missing modes are filled with identities,
    # for both trans=false and trans=true.
    dims = (3, 4, 2, 5)
    v = rand(dims...)
    for modes in ([1], [2, 4], [1, 2, 3, 4])
        d = Dict{Int,Matrix{Float64}}(i => Matrix(qr(rand(dims[i], dims[i])).Q)
                                      for i in modes)
        full = ntuple(i -> get(d, i, Matrix{Float64}(I, dims[i], dims[i])), 4)
        for trans in (false, true)
            @test isapprox(TPSChem.transform_basis(v, d, trans=trans),
                           TPSChem.transform_basis(v, full, trans=trans), atol=1e-12)
        end
    end

    # A Tucker block may legitimately have a zero-length mode; transform_basis
    # must handle it rather than dividing by the leading dimension.
    for zdims in ((0, 3, 2, 4), (3, 0, 2, 4), (3, 4, 2, 0))
        vz = zeros(zdims...)
        dz = Dict{Int,Matrix{Float64}}(3 => Matrix(qr(rand(2, 2)).Q))
        for trans in (false, true)
            outz = TPSChem.transform_basis(vz, dz, trans=trans)
            @test size(outz) == zdims
        end
    end
end

