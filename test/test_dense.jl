using TPSChem
using Printf
using Test
using LinearAlgebra
using HDF5
using Random

using PyCall
pydir = joinpath(dirname(dirname(pathof(TPSChem))), "tools", "python")
pushfirst!(PyVector(pyimport("sys")."path"), pydir)
ENV["PYTHON"] = Sys.which("python")

#@testset "Clusters" begin
#function run()
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
    tuck = TPSChem.Tucker(A, thresh=20, verbose=1)
    
    display(size(tuck.core))
    display(size.(tuck.factors))
    B = TPSChem.recompose(tuck)
    println()
    println(TPSChem.dims_large(tuck))
    @test all(TPSChem.dims_small(tuck) .== [4, 1, 3, 3, 1])
    @test all(TPSChem.dims_large(tuck) .== [4, 6, 3, 3, 5])
     
    A = rand(4,6,3,3,5)
    tuck = TPSChem.Tucker(A, thresh=-1, verbose=1)
    B = TPSChem.recompose(tuck)
    @test isapprox(abs.(A), abs.(B), atol=1e-12)

    A = rand(4,6,3,3,5)*.1
    B = rand(4,6,3,3,5)*.1
    C = A+B

    #tuckA = TPSChem.Tucker(A, thresh=-1, verbose=1, max_number=2)
    #tuckB = TPSChem.Tucker(B, thresh=-1, verbose=1, max_number=2)
    #tuckC = TPSChem.Tucker(C, thresh=-1, verbose=1, max_number=2)
    tuckA = TPSChem.Tucker(A, thresh=-1, verbose=0)
    tuckB = TPSChem.Tucker(B, thresh=-1, verbose=0)
    tuckC = TPSChem.Tucker(C, thresh=-1, verbose=0)

    # test Tucker addition
    test = tuckA + tuckB
    @test isapprox(TPSChem.dot(tuckC,tuckC), TPSChem.dot(test,test), atol=1e-12)


    #
    # Now test basis transformation
    A = rand(4,6,3,3,5)*.1
    
    trans1 = Dict{Int,Matrix{Float64}}() 
    trans1[2] = rand(6,5)
    trans1[4] = rand(3,2)

    trans2 = []
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
    
    
    @test isapprox(A1, A2, atol=1e-12)

    if false 
        r = 1
        a = 0
        push!(atoms,Atom(1,"H", [0,.0*a,0*r]))
        push!(atoms,Atom(2,"H", [0,.0*a,1*r]))
        push!(atoms,Atom(3,"H", [0,.1*a,2*r]))
        push!(atoms,Atom(4,"H", [0,.1*a,3*r]))
        push!(atoms,Atom(5,"H", [0,.2*a,4*r]))
        push!(atoms,Atom(6,"H", [0,.2*a,5*r]))
        push!(atoms,Atom(7,"H", [0,.3*a,6*r]))
        push!(atoms,Atom(8,"H", [0,.3*a,7*r]))
        push!(atoms,Atom(9,"H", [0,.4*a,8*r]))
        push!(atoms,Atom(10,"H",[0,.4*a,9*r]))
        push!(atoms,Atom(11,"H",[0,.5*a,10*r]))
        push!(atoms,Atom(12,"H",[0,.5*a,11*r]))
        push!(atoms,Atom(13,"H",[0,.6*a,12*r]))
        push!(atoms,Atom(14,"H",[0,.6*a,13*r]))
        push!(atoms,Atom(15,"H",[0,.7*a,14*r]))
        push!(atoms,Atom(16,"H",[0,.7*a,15*r]))
        push!(atoms,Atom(17,"H",[0,.8*a,16*r]))
        push!(atoms,Atom(18,"H",[0,.8*a,17*r]))
    

        clusters    = [(1:2),(3:4),(5:6),(7:8),(9:10),(11:12)]
        init_fspace = [(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)]
        na = 6
        nb = 6
        clusters    = [(1:2),(3:4),(5:6),(7:8)]
        init_fspace = [(1,1),(1,1),(1,1),(1,1)]
        na = 4
        nb = 4
        clusters    = [(1:6),(7:12)]
        init_fspace = [(3,3),(3,3)]
        na = 6
        nb = 6
        clusters    = [(1:6),(7:12),(13:18)]
        init_fspace = [(3,3),(3,3),(3,3)]
        na = 6
        nb = 6
    elseif false 
        push!(atoms,Atom(1,"H",[-1.30,0,0.00]))
        push!(atoms,Atom(2,"H",[-1.30,0,1.00]))
        push!(atoms,Atom(3,"H",[ 0.00,0,0.00]))
        push!(atoms,Atom(4,"H",[ 0.00,0,1.00]))
        push!(atoms,Atom(5,"H",[ 1.33,0,0.00]))
        push!(atoms,Atom(6,"H",[ 1.30,0,1.00]))

        clusters    = [(1:2),(3:4),(5:6)]
        init_fspace = [(1,1),(1,1),(1,1)]
        na = 3
        nb = 3
    elseif true
       
        rad = 3
        
        atoms = generate_H_ring(10,rad)
        clusters    = [(1:2),(3:4),(5:6),(7:8),(9:10)]
        init_fspace = [(1,1),(1,1),(1,1),(1,1),(1,1)]
        clusters    = [(1:4),(5:6),(7:8),(9:10)]
        init_fspace = [(2,2),(1,1),(1,1),(1,1)]
        clusters    = [(1:4),(5:8),(9:10)]
        init_fspace = [(2,2),(2,2),(1,1)]
        na = 5
        nb = 5
        
        
        atoms = generate_H_ring(8,rad)
        clusters    = [(1:2),(3:4),(5:6),(7:8)]
        init_fspace = [(1,1),(1,1),(1,1),(1,1)]
        clusters    = [(1:4),(5:6),(7:8)]
        init_fspace = [(2,2),(1,1),(1,1)]
        clusters    = [(1:4),(5:8)]
        init_fspace = [(2,2),(2,2)]
        na = 4
        nb = 4
        
        
        atoms = generate_H_ring(12,rad)
        clusters    = [(1:2),(3:4),(5:6),(7:8),(9:10),(11:12)]
        init_fspace = [(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)]
        clusters    = [(1:4),(5:8),(9:12)]
        init_fspace = [(2,2),(2,2),(2,2)]
        clusters    = [(1:4),(5:8),(9:10),(11:12)]
        init_fspace = [(2,2),(2,2),(1,1),(1,1)]
        clusters    = [(1:6),(7:12)]
        init_fspace = [(3,3),(3,3)]
        na = 6
        nb = 6
        
    end

    basis = "6-31g"
    basis = "sto-3g"
    mol     = Molecule(0,1,atoms,basis)
   
   
    # get integrals
    mf = TPSChem.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = TPSChem.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    #e_fci, d1_fci, d2_fci = TPSChem.pyscf_fci(ints, na, nb, conv_tol=1e-10,max_cycle=100, nroots=2)
	
    #run fci with pyscf
    if false 
        pyscf = pyimport("pyscf")
        fci = pyimport("pyscf.fci")
        mp = pyimport("pyscf.mp")
        mp2 = mp.MP2(mf)
        cisolver = pyscf.fci.direct_spin1.FCI()
        cisolver.max_cycle = 100 
        cisolver.conv_tol = 1e-10 
        nelec = na + nb
        norb = size(ints.h1)[1]
        e_fci, ci = cisolver.kernel(ints.h1, ints.h2, norb , nelec, ecore=0, nroots = 1, verbose=100)
        e_fci = min(e_fci...)
        @printf(" FCI Energy: %12.8f\n", e_fci)
    end
   
    # localize orbitals
    C = mf.mo_coeff
    Cl = TPSChem.localize(mf.mo_coeff,"lowdin",mf)
    TPSChem.pyscf_write_molden(mol,Cl,filename="lowdin.molden")
    S = TPSChem.get_ovlp(mf)
    U =  C' * S * Cl
    println(" Rotate Integrals")
    flush(stdout)
    ints = TPSChem.orbital_rotation(ints,U)
    println(" done.")
    flush(stdout)
   
    #
    # define clusters
    clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    #e_cmf, U, Da, Db  = TPSChem.cmf_oo(ints, clusters, init_fspace, rdm1, 
    #                                   max_iter_oo=40, verbose=0, gconv=1e-6, method="gd", alpha=1e-1)
    #ints = TPSChem.orbital_rotation(ints,U)
    
    d1 = RDM1(n_orb(ints))
    e_cmf, U, d1  = TPSChem.cmf_oo(ints, clusters, init_fspace, d1, 
                                   max_iter_oo=40, verbose=0, gconv=1e-6, method="bfgs")
    TPSChem.pyscf_write_molden(mol,Cl*U,filename="cmf.molden")
    ints = TPSChem.orbital_rotation(ints,U)

    e_ref = e_cmf - ints.h0

    max_roots = 100
    # build Hamiltonian, cluster_basis and cluster ops
    #display(Da)
    #cluster_bases = TPSChem.compute_cluster_eigenbasis(ints, clusters, verbose=2, max_roots=max_roots)
    cluster_bases = TPSChem.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=max_roots, 
                                                       init_fspace=init_fspace, rdm1a=d1.a, rdm1b=d1.b)
    clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
    cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints);


    
    p_spaces = Vector{TPSChem.ClusterSubspace}()
    q_spaces = Vector{TPSChem.ClusterSubspace}()
  
    # define p spaces
    for ci in clusters
        tss = TPSChem.ClusterSubspace(ci)
        tss[init_fspace[ci.idx]] = 1:1
        push!(p_spaces, tss)
    end
    
    # define q spaces
    for tssp in p_spaces 
        tss = TPSChem.get_ortho_compliment(tssp, cluster_bases[tssp.cluster.idx])
        push!(q_spaces, tss)
    end

    println(" ================= Cluster P Spaces ===================")
    display.(p_spaces)
    println(" ================= Cluster Q Spaces ===================")
    display.(q_spaces)


    nroots = 1
    ci_vector = TPSChem.BSstate(clusters, p_spaces, q_spaces, na, nb)
    ref_vector = deepcopy(ci_vector)

    # for FOI space 
    foi_space = TPSChem.define_foi_space(ref_vector, clustered_ham, nbody=2) 
    ci_vector = TPSChem.BSstate(clusters, p_spaces, q_spaces, foi_space)
    
    #
    # initialize with eye
    TPSChem.set_vector!(ref_vector, Matrix(1.0I, length(ref_vector),nroots))
    TPSChem.add!(ci_vector, ref_vector)
    
    #TPSChem.print_fock_occupations(ci_vector)

    cts_ref  = TPSChem.SPTstate(ref_vector, thresh=-1);
  
    cts = TPSChem.SPTstate(ci_vector, thresh=1e-5)
    #display(cts_fois)
    if false  
        TPSChem.scale!(ci_vector, 1.0/sqrt(TPSChem.dot(ci_vector, ci_vector)[1]))
        println(" Length of CI Vector: ", length(ci_vector))
        @time e_nb2, ci_vector = TPSChem.tucker_ci_solve(ci_vector, cluster_ops, clustered_ham)
        @printf(" E(CI):   Electronic %16.12f Total %16.12f\n", e_nb2[1], e_nb2[1]+ints.h0)
        TPSChem.print_fock_occupations(ci_vector)

        println(" Now compress and resolve")
        cts = TPSChem.SPTstate(ci_vector, thresh=1e-5)
        #TPSChem.compress!(cts,thresh=1e-5)
        TPSChem.normalize!(cts)
        display(length(cts))
        @time e_cts, cts = TPSChem.tucker_ci_solve(cts, cluster_ops, clustered_ham, tol=1e-5)
        @printf(" E(cCI):  Electronic %16.12f Total %16.12f\n", e_cts[1], e_cts[1]+ints.h0)
        TPSChem.display(cts)
        #TPSChem.print_fock_occupations(cts)
    end
    
    if true    
        @time e_ref, cts_ref = TPSChem.tucker_ci_solve(cts_ref, cluster_ops, clustered_ham, tol=1e-5)
       
        cts_var = deepcopy(cts_ref)
        e_var = 0.0
        e_pt2 = 0.0
        #display(abs.(cluster_ops[1]["H"][((2,2),(2,2))]) - abs.(cluster_ops[2]["H"][((2,2),(2,2))]))
       
        @time e_var, v_var = TPSChem.solve_for_compressed_space(cts_ref, cluster_ops, clustered_ham, nbody=4, thresh_var=1e-4, thresh_foi=1e-6, tol_ci=1e-5, tol_tucker=1e-5)
#        for i in 1:10
#            #@profilehtml e_var, e_pt2, cts_var = TPSChem.iterate_pt2!(cts_var, cluster_ops, clustered_ham, nbody=4, thresh=1e-7, tol=1e-5, do_pt=true)
#            @time e_var, e_pt2, cts_var = TPSChem.iterate_pt2!(cts_var, cluster_ops, clustered_ham, nbody=4, thresh=1e-7, tol=1e-5, do_pt=true, method="ci", ratio=1e3)
#            @printf(" E(Ref)      = %12.8f = %12.8f\n", e_ref[1], e_ref[1] + ints.h0 )
#            @printf(" E(PT2) tot  = %12.8f = %12.8f\n", e_ref[1]+e_pt2, e_ref[1]+e_pt2 + ints.h0 )
#            @printf(" E(var) tot  = %12.8f = %12.8f\n", e_var[1], e_var[1] + ints.h0 )
#      
#            if abs(e_ref[1] - e_var[1]) < 1e-8
#                println("*Converged")
#                break
#            end
#            cts_ref = cts_var
#            e_ref = e_var
#        end
#        return cts_var
    end
    
    if false 
        e_ref, ref_vector = TPSChem.tucker_ci_solve(ref_vector, cluster_ops, clustered_ham)
        println(" Reference State:" )
        TPSChem.print_fock_occupations(ref_vector)

        @time e_cepa, x_cepa = TPSChem.tucker_cepa_solve!(ref_vector, ci_vector, cluster_ops, clustered_ham)
        @printf(" E(CEPA): Electronic %16.12f Total %16.12f\n", e_cepa, e_cepa+ints.h0)
        println(e_cepa)
        TPSChem.print_fock_occupations(ci_vector)
    end
    
    function do_work(cts, cluster_ops, clustered_ham; nbody=3, thresh=1e-3, nfoi=3, n_iter=3)
        for iter in 1:n_iter
            for n in 1:nfoi
                cts  = TPSChem.open_sigma(cts, cluster_ops, clustered_ham, nbody=nbody, thresh=thresh)
            end
            TPSChem.normalize!(cts)
            display(length(cts))
            @time e_cts, v_cts = TPSChem.tucker_ci_solve!(cts, cluster_ops, clustered_ham, tol=1e-3)
            @printf(" E(cCI):  Electronic %16.12f Total %16.12f\n", e_cts[1], e_cts[1]+ints.h0)
        end
    end

    if false 

        #
        #
        cts  = TPSChem.open_sigma(cts_ref, cluster_ops, clustered_ham, nbody=4, thresh=1e-4)
        TPSChem.normalize!(cts)
        display(length(cts))
        @time TPSChem.hylleraas_compressed_mp2!(cts, cluster_ops, clustered_ham, tol=1e-6, thresh=1e-6)
    end
   

    
    if false 
        cts = cts_fois

            
        @time e_ref, v_ref = TPSChem.tucker_ci_solve!(cts_ref, cluster_ops, clustered_ham, tol=1e-12)
        tmp = deepcopy(cts_ref)
        TPSChem.scale!(tmp, -e_ref[1])

        cts_sig  = TPSChem.open_sigma(cts_ref, cluster_ops, clustered_ham, nbody=4, thresh=1e-4)
        @printf(" Should be E0  : %12.8f\n", TPSChem.nonorth_dot(cts_sig, cts_ref))
        TPSChem.nonorth_add!(cts_sig, tmp) 
        @printf(" Should be zero: %12.8f\n", TPSChem.nonorth_dot(cts_sig, cts_ref))
        display(cts_sig)
        display(TPSChem.nonorth_dot(cts_sig, cts_ref))

        cts = cts_sig
        TPSChem.normalize!(cts)
        display(length(cts))
        @time e_cts, v_cts = TPSChem.tucker_ci_solve!(cts, cluster_ops, clustered_ham, tol=1e-3)
        @printf(" E(cCI):  Electronic %16.12f Total %16.12f\n", e_cts[1], e_cts[1]+ints.h0)
        display(cts)
        
        #@time do_work(cts, cluster_ops, clustered_ham, nbody=4, thresh=1e-2, nfoi=3, n_iter=10)
        #display(cts)
    end
    #end
#end

#run()
