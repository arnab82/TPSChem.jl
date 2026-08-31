include("common.jl")

# TPS-CEPA with the Q-space Hamiltonian built ONCE as distributed block-sparse
# blocks and reused across all roots, CEPA macro-iterations, and MINRES/CG steps.
# This is the stored-H analogue of cepa_sharded_minres_driver.jl: same result,
# far fewer cluster-operator contractions, because H_QQ is not rebuilt per matvec.

workers_ = init_multinode_workers!()
data = load_problem_data()
clustered_ham = get_clustered_ham(data)

nbody        = env_int("TPSCHEM_NBODY", 4)
thresh_foi   = env_float("TPSCHEM_THRESH_FOI", 1e-6)
thresh_sigma = env_float("TPSCHEM_THRESH_SIGMA", 1e-8)
thresh_clip  = env_float("TPSCHEM_THRESH_CLIP", 1e-5)
tol          = env_float("TPSCHEM_CEPA_TOL", 1e-8)
cepa_mit     = env_int("TPSCHEM_CEPA_MIT", 30)
cg_maxiter   = env_int("TPSCHEM_MINRES_MAXITER", 300)
compress     = env_bool("TPSCHEM_COMPRESS_Q", false)
blas_threads = env_int("TPSCHEM_BLAS_THREADS", 1)
h_storage    = Symbol(get(ENV, "TPSCHEM_H_STORAGE", "auto"))   # auto | blocks | matrixfree
max_mem_H    = env_float("TPSCHEM_MAX_MEM_H", 50.0)            # GB budget for stored H_Q

@printf("\n================ TPS-CEPA stored-H example ================\n")
@printf("Input file:      %s\n", ENV["TPSCHEM_INPUT_JLD2"])
@printf("nbody:           %i\n", nbody)
@printf("thresh_foi:      %.3e\n", thresh_foi)
@printf("h_storage:       %s\n", h_storage)
@printf("max_mem_H (GB):  %.1f\n", max_mem_H)
@printf("MINRES maxiter:  %i\n", cg_maxiter)
flush(stdout)

dops = build_distributed_cluster_ops(data, workers_)
ref = get_reference_state(data; key_default="ref_vec")
e0 = get_e0(data)

if e0 === nothing
    error("""
          This stored-H CEPA driver expects a pre-solved reference and reference
          energies. Put `ref_vec` and `e0` in TPSCHEM_INPUT_JLD2, or set
          TPSCHEM_REF_KEY and TPSCHEM_E0='E1,E2,...'.
          (Alternatively, drop e0 and set reference_solver=:sharded_davidson in
          do_tps_sharded_cepa to solve the reference across nodes too.)
          """)
end

e_cepa, qspace = TPSChem.do_tps_sharded_cepa(
    ref, dops, clustered_ham;
    e0=e0,
    cepa_shift=get(ENV, "TPSCHEM_CEPA_SHIFT", "cepa"),
    cepa_mit=cepa_mit,
    nbody=nbody,
    thresh_foi=thresh_foi,
    thresh_clip=thresh_clip,
    tol=tol,
    thresh_sigma=thresh_sigma,
    compress=compress,
    solver=Symbol(get(ENV, "TPSCHEM_SOLVER", "pcg")),
    block_roots=lowercase(get(ENV, "TPSCHEM_BLOCK_ROOTS", "true")) in ("1","true","yes","on"),
    linsolve_tol=parse(Float64, get(ENV, "TPSCHEM_LINSOLVE_TOL", "1e-6")),
    h_storage=h_storage,       # :blocks reuses the once-built H_Q every iteration
    max_mem_H=max_mem_H,
    cg_maxiter=cg_maxiter,
    workers=workers_,
    threaded_worker=true,
    blas_threads=blas_threads,
    verbose=env_int("TPSCHEM_VERBOSE", 1))

@printf("CEPA total energies: %s\n", e_cepa)
@printf("Sharded Q summary:   %s\n", TPSChem.sharded_state_summary(qspace))

if haskey(ENV, "TPSCHEM_OUTPUT_JLD2")
    JLD2.jldsave(ENV["TPSCHEM_OUTPUT_JLD2"]; e_cepa=e_cepa)
end

TPSChem.destroy!(qspace)
TPSChem.destroy!(dops)
