include("common.jl")

workers_ = init_multinode_workers!()
data = load_problem_data()
clustered_ham = get_clustered_ham(data)

nbody = env_int("TPSCHEM_NBODY", 4)
thresh_foi = env_float("TPSCHEM_THRESH_FOI", 1e-8)
prescreen = env_bool("TPSCHEM_PRESCREEN", false)
blas_threads = env_int("TPSCHEM_BLAS_THREADS", 1)

@printf("\n================ Sharded TPSCI PT1/FOIS example ================\n")
@printf("Input file:      %s\n", ENV["TPSCHEM_INPUT_JLD2"])
@printf("nbody:           %i\n", nbody)
@printf("thresh_foi:      %.3e\n", thresh_foi)
@printf("prescreen:       %s\n", prescreen)
flush(stdout)

dops = build_distributed_cluster_ops(data, workers_)
ref = get_reference_state(data; key_default="ref_vec")
dref = TPSChem.distribute_tpsci_state(ref; workers=workers_, strategy=:balanced,
                                      blas_threads=blas_threads)

e2, pt1 = TPSChem.compute_pt1_wavefunction_sharded(
    dref, dops, clustered_ham;
    nbody=nbody,
    thresh_foi=thresh_foi,
    prescreen=prescreen,
    workers=workers_,
    threaded_worker=true,
    blas_threads=blas_threads)

@printf("Sharded PT1 E2: %s\n", e2)
@printf("Reference summary: %s\n", TPSChem.sharded_state_summary(dref))
@printf("PT1/Q summary:    %s\n", TPSChem.sharded_state_summary(pt1))

if haskey(ENV, "TPSCHEM_OUTPUT_JLD2")
    JLD2.jldsave(ENV["TPSCHEM_OUTPUT_JLD2"]; e2=e2)
end

TPSChem.destroy!(pt1)
TPSChem.destroy!(dref)
TPSChem.destroy!(dops)
