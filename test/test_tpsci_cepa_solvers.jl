using TPSChem
using TPSChem.QCBase
using TPSChem.RDM
using Test
using JLD2
using LinearAlgebra
using Printf

# The single-node CEPA amplitude equations can be solved four ways: matrix-free
# KrylovKit CG (`:krylov`), MINRES over a stored/structured H_qq (`:minres`), and
# Jacobi-preconditioned CG over the same H_qq (`:pcg`, with a MINRES fallback for
# indefinite shifts). They target the same linear system, so they must agree —
# and an unrecognised `solver` must fail loudly instead of silently dropping to
# the (orders-of-magnitude slower) matrix-free path.
@testset "tpsci cepa solvers" begin
    @load "_testdata_cmf_h12.jld2"

    max_roots = 4
    nroots = 2
    ref_fock = FockConfig(init_fspace)

    cluster_bases = TPSChem.compute_cluster_eigenbasis(ints, clusters;
        verbose=0, max_roots=max_roots, init_fspace=init_fspace,
        rdm1a=d1.a, rdm1b=d1.b)

    clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
    cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints)
    TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)

    ci_vector = TPSChem.TPSCIstate(clusters, ref_fock, R=nroots, T=Float64)
    ci_vector[ref_fock][ClusterConfig([1, 1, 1, 1, 1])] = [1.0, 0.0]
    ci_vector[ref_fock][ClusterConfig([2, 1, 1, 1, 1])] = [0.0, 1.0]

    common = (thresh_foi=1e-4, nbody=4, tol=1e-10, thresh_sigma=1e-8, verbose=0)

    run_cepa(; kwargs...) = first(TPSChem.do_fois_cepa(deepcopy(ci_vector), cluster_ops,
                                                      clustered_ham; common..., kwargs...))

    # CEPA-0: single shift, so every solver sees the same linear system.
    e_minres    = run_cepa(solver=:minres, build_hqq=:sparse)
    e_pcg       = run_cepa(solver=:pcg,    build_hqq=:sparse)
    e_pcg_dense = run_cepa(solver=:pcg,    build_hqq=:direct)
    e_pcg_mf    = run_cepa(solver=:pcg,    build_hqq=:matvec)
    e_krylov    = run_cepa(solver=:krylov)

    @test length(e_minres) == nroots
    @test isapprox(e_pcg,       e_minres, atol=1e-8)
    @test isapprox(e_pcg_dense, e_minres, atol=1e-8)
    @test isapprox(e_pcg_mf,    e_minres, atol=1e-8)
    @test isapprox(e_krylov,    e_minres, atol=1e-8)

    # ACPF drives the shift iteratively, which is what exercises the PCG warm start.
    # It also exercises the MINRES hand-off: the excited-root shift wanders far enough
    # here that (H_qq - eshift) turns indefinite even though its diagonal still clears
    # the shift, so CG hits a p'Ap <= 0 direction and has to give the solve up. Root 1
    # converges to a fixed shift and is what the two solvers must agree on; root 2's
    # ACPF iteration converges for neither, so only require that it stays finite.
    e_acpf_minres = run_cepa(solver=:minres, build_hqq=:sparse, cepa_shift="acpf")
    e_acpf_pcg    = run_cepa(solver=:pcg,    build_hqq=:sparse, cepa_shift="acpf")
    @test isapprox(e_acpf_pcg[1], e_acpf_minres[1], atol=1e-8)
    @test all(isfinite, e_acpf_pcg)

    # A typo in `solver` used to fall through to the matrix-free branch, silently
    # ignoring `build_hqq` and turning a minutes-long run into a days-long one.
    @test_throws ErrorException run_cepa(solver=:pcg_typo, build_hqq=:sparse)
    @test_throws ErrorException run_cepa(solver=:pcg, build_hqq=:sprase)
end
