include("common.jl")

workers_ = init_multinode_workers!()
data = load_problem_data()
clustered_ham = get_clustered_ham(data)

nbody = env_int("TPSCHEM_NBODY", 4)
thresh_foi = env_float("TPSCHEM_THRESH_FOI", 1e-6)
thresh_sigma = env_float("TPSCHEM_THRESH_SIGMA", 1e-8)
thresh_clip = env_float("TPSCHEM_THRESH_CLIP", 1e-5)
tol = env_float("TPSCHEM_CEPA_TOL", 1e-8)
cepa_mit = env_int("TPSCHEM_CEPA_MIT", 30)
cg_maxiter = env_int("TPSCHEM_MINRES_MAXITER", 300)
compress = env_bool("TPSCHEM_COMPRESS_Q", false)
blas_threads = env_int("TPSCHEM_BLAS_THREADS", 1)

@printf("\n================ Sharded TPSCI CEPA/MINRES example ================\n")
@printf("Input file:      %s\n", ENV["TPSCHEM_INPUT_JLD2"])
@printf("nbody:           %i\n", nbody)
@printf("thresh_foi:      %.3e\n", thresh_foi)
@printf("thresh_sigma:    %.3e\n", thresh_sigma)
@printf("tol:             %.3e\n", tol)
@printf("MINRES maxiter:  %i\n", cg_maxiter)
flush(stdout)

dops = build_distributed_cluster_ops(data, workers_)
ref = get_reference_state(data; key_default="ref_vec")
e0 = get_e0(data)

if e0 === nothing
    error("""
          This distributed-cluster-ops CEPA driver expects a pre-solved reference
          and reference energies. Put `ref_vec` and `e0` in TPSCHEM_INPUT_JLD2,
          or set TPSCHEM_REF_KEY and TPSCHEM_E0='E1,E2,...'.
          """)
end

e_cepa, qspace = TPSChem.do_fois_cepa_sharded(
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
    solver=:minres,
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
