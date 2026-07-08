include("common.jl")

# Never-gather selected TPSCI. The variational vector lives across workers as a
# `DistributedTPSCIstate` for the ENTIRE selected-CI loop: the diagonalization
# (sharded Davidson), the PT1 selection, clipping, and the variational-space
# growth all operate on sharded states, so no full CI vector is ever
# materialized on the master or on any single node. This is the solver for the
# case where even the variational CI vector exceeds a single node's memory and
# must never be gathered between selected-CI iterations.

workers_ = init_multinode_workers!()
data = load_problem_data()
clustered_ham = get_clustered_ham(data)

nroots        = env_int("TPSCHEM_NROOTS", 1)
thresh_cipsi  = env_float("TPSCHEM_THRESH_CIPSI", 1e-2)   # PT1 selection threshold
thresh_foi    = env_float("TPSCHEM_THRESH_FOI", 1e-6)     # FOIS screening threshold
thresh_asci   = env_bool("TPSCHEM_USE_THRESH_ASCI", false) ?
                    env_float("TPSCHEM_THRESH_ASCI", 1e-2) : nothing
thresh_var    = env_bool("TPSCHEM_USE_THRESH_VAR", false) ?
                    env_float("TPSCHEM_THRESH_VAR", 1e-3) : nothing
max_iter      = env_int("TPSCHEM_MAX_ITER", 10)
conv_thresh   = env_float("TPSCHEM_CONV_THRESH", 1e-4)    # outer selected-CI convergence
nbody         = env_int("TPSCHEM_NBODY", 4)
ci_conv       = env_float("TPSCHEM_CI_CONV", 1e-8)        # inner Davidson convergence
ci_max_iter   = env_int("TPSCHEM_CI_MAX_ITER", 100)
ci_max_ss_vecs = env_int("TPSCHEM_MAX_SS_VECS", 4)        # small: each subspace vec is another full vector
blas_threads  = env_int("TPSCHEM_BLAS_THREADS", 1)
h_storage     = Symbol(get(ENV, "TPSCHEM_H_STORAGE", "auto"))   # auto | blocks | matrixfree
max_mem_H     = env_float("TPSCHEM_MAX_MEM_H", 50.0)     # GB budget for stored block-sparse H
compute_s2    = env_bool("TPSCHEM_COMPUTE_S2", true)

@printf("\n============ Never-gather Sharded Selected TPSCI example ============\n")
@printf("Input file:      %s\n", ENV["TPSCHEM_INPUT_JLD2"])
@printf("nroots:          %i\n", nroots)
@printf("thresh_cipsi:    %.2e\n", thresh_cipsi)
@printf("thresh_foi:      %.2e\n", thresh_foi)
@printf("thresh_asci:     %s\n", thresh_asci === nothing ? "off" : string(thresh_asci))
@printf("thresh_var:      %s\n", thresh_var === nothing ? "off" : string(thresh_var))
@printf("max_iter:        %i\n", max_iter)
@printf("conv_thresh:     %.2e\n", conv_thresh)
@printf("nbody:           %i\n", nbody)
@printf("max_ss_vecs:     %i\n", ci_max_ss_vecs)
@printf("h_storage:       %s\n", h_storage)
@printf("max_mem_H (GB):  %.1f\n", max_mem_H)
flush(stdout)

# Shard the cluster operators too, so nothing scales with node count on the master.
dops = build_distributed_cluster_ops(data, workers_)

# The (small) starting reference. `tpsci_ci_sharded` distributes it once and
# never gathers it again. Point TPSCHEM_REF_KEY at a stored reference, or supply
# `ref_vec`/`ci_vector` in the JLD2; otherwise a CMF-like single-config reference
# built from `init_fspace` is used.
ref = get_reference_state(data; key_default="ref_vec")

e0, vec_var = TPSChem.tpsci_ci_sharded(
    ref, dops, clustered_ham;
    thresh_cipsi=thresh_cipsi,
    thresh_foi=thresh_foi,
    thresh_asci=thresh_asci,
    thresh_var=thresh_var,
    max_iter=max_iter,
    conv_thresh=conv_thresh,
    nbody=nbody,
    ci_conv=ci_conv,
    ci_max_iter=ci_max_iter,
    ci_max_ss_vecs=ci_max_ss_vecs,
    h_storage=h_storage,
    max_mem_H=max_mem_H,
    compute_s2=compute_s2,
    workers=workers_,
    threaded_worker=true,
    blas_threads=blas_threads,
    verbose=1)

@printf("\nFinal variational energies (dim=%i):\n", length(vec_var))
for r in 1:nroots
    @printf("  root %3i   E = %18.10f\n", r, e0[r])
end
@printf("Solution summary:  %s\n", TPSChem.sharded_state_summary(vec_var))
flush(stdout)

if haskey(ENV, "TPSCHEM_OUTPUT_JLD2")
    # Only the small eigenvalue vector is saved by default; the eigenvector stays
    # sharded. Gather explicitly with collect_tpsci_state if you truly need it on
    # the master (do NOT do this if the vector exceeds node memory).
    JLD2.jldsave(ENV["TPSCHEM_OUTPUT_JLD2"]; e0=e0)
end

TPSChem.destroy!(vec_var)
TPSChem.destroy!(dops)

# ===========================================================================
# ALTERNATIVE CALL OPTIONS for `tpsci_ci_sharded`
# ---------------------------------------------------------------------------
# The block above is the env-driven default. Below are the distinct ways you
# would call the never-gather selected-CI loop. They are commented out; copy the
# one that matches your case. `cluster_ops` may be either a plain local
# `Vector{ClusterOps}` (replicated on every worker) or a `DistributedClusterOps`
# (sharded) — the loop accepts both.
# ===========================================================================
#
# --- 1. SINGLE NODE, auto tier (small-to-medium; simplest) -------------------
#     The whole loop stays sharded on one worker; :auto stores the block-sparse
#     H if it fits max_mem_H, else runs matrix-free. Launch: `julia -p 1 ...`.
#
#   ws = workers()                  # a single worker pid
#   e0, v = TPSChem.tpsci_ci_sharded(ref, cluster_ops, clustered_ham;
#               thresh_cipsi=1e-3, thresh_foi=1e-5, conv_thresh=1e-6,
#               ci_conv=1e-8, ci_max_ss_vecs=8,   # bigger ss ok when vector is small
#               nbody=4, h_storage=:auto, max_mem_H=50.0, workers=ws, verbose=1)
#
# --- 2. MULTINODE, force stored block-sparse H, sharded cluster ops ----------
#     For a >node-memory vector and H spread across nodes; cluster ops sharded.
#     Keep ci_max_ss_vecs small — each subspace vector is another full vector.
#
#   dops = TPSChem.compute_cluster_ops_distributed(cluster_bases, ints; workers=ws)
#   TPSChem.add_cmf_operators_distributed!(dops, cluster_bases, ints, d1.a, d1.b)
#   e0, v = TPSChem.tpsci_ci_sharded(ref, dops, clustered_ham;
#               thresh_cipsi=1e-3, thresh_foi=1e-5, conv_thresh=1e-6,
#               ci_conv=1e-8, ci_max_ss_vecs=4, nbody=4,
#               h_storage=:blocks, workers=ws, blas_threads=1, verbose=1)
#
# --- 3. MULTINODE, matrix-free (minimal memory; even sparse H is too big) -----
#     Store nothing; re-contract every matvec. Slowest per iteration but only
#     vector-scale memory anywhere.
#
#   e0, v = TPSChem.tpsci_ci_sharded(ref, dops, clustered_ham;
#               thresh_cipsi=1e-3, thresh_foi=1e-5, conv_thresh=1e-6,
#               ci_conv=1e-8, ci_max_ss_vecs=4, nbody=4,
#               h_storage=:matrixfree, workers=ws, verbose=1)
#
# --- 4. Config-level ASCI clipping (safe) ------------------------------------
#     thresh_asci/thresh_var below the smallest whole-sector weight prune only
#     individual configs and are safe. Avoid clipping so hard that an entire
#     Fock sector leaves the search vector while it remains variational — that
#     can trip the sharded add! ownership guard (documented limitation).
#
#   e0, v = TPSChem.tpsci_ci_sharded(ref, dops, clustered_ham;
#               thresh_cipsi=1e-3, thresh_foi=1e-5, thresh_asci=1e-2,
#               conv_thresh=1e-6, workers=ws, verbose=1)
#
# NOTE: `incremental` sigma updates and the `thresh_spin` S² space extension are
# NOT supported here — use `tpsci_ci_multinode` if you need them.
# ===========================================================================
