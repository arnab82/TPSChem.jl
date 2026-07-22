include("common.jl")

# Across-node variational diagonalization: solve the TPSCI eigenproblem in the
# space of a `DistributedTPSCIstate` whose subspace vectors live across nodes and
# are never gathered onto the master. This is the solver for the case where the
# variational CI vector exceeds a single node's memory.

workers_ = init_multinode_workers!()
data = load_problem_data()
clustered_ham = get_clustered_ham(data)

nroots       = env_int("TPSCHEM_NROOTS", 1)
ci_conv      = env_float("TPSCHEM_CI_CONV", 1e-6)
max_ss_vecs  = env_int("TPSCHEM_MAX_SS_VECS", 4)          # small: minimizes live memory
max_iter     = env_int("TPSCHEM_CI_MAX_ITER", 200)
blas_threads = env_int("TPSCHEM_BLAS_THREADS", 1)
h_storage    = Symbol(get(ENV, "TPSCHEM_H_STORAGE", "auto"))   # auto | blocks | matrixfree
max_mem_H    = env_float("TPSCHEM_MAX_MEM_H", 50.0)            # GB budget for stored H

@printf("\n================ Sharded TPSCI Davidson example ================\n")
@printf("Input file:      %s\n", ENV["TPSCHEM_INPUT_JLD2"])
@printf("nroots:          %i\n", nroots)
@printf("max_ss_vecs:     %i\n", max_ss_vecs)
@printf("h_storage:       %s\n", h_storage)
@printf("max_mem_H (GB):  %.1f\n", max_mem_H)
flush(stdout)

# Shard the cluster operators too, so nothing scales with node count on the master.
dops = build_distributed_cluster_ops(data, workers_)

# The variational space + initial guess. Point TPSCHEM_REF_KEY at a stored
# variational vector, or supply `ci_vector`/`ref_vec` in the JLD2.
guess = get_reference_state(data; key_default="ci_vector")
dguess = TPSChem.distribute_tpsci_state(guess; workers=workers_, strategy=:balanced,
                                        blas_threads=blas_threads)

# Report the block-sparse H size so you can see which storage tier :auto picks.
rep = TPSChem.sharded_H_memory_report(dguess, clustered_ham)
@printf("Variational dim:   %i  (nfocks=%i)\n", rep.dim, rep.nfocks)
@printf("Block-sparse H:    nnz=%.3e   mem=%.3f GB\n", rep.nnz, rep.gb)
@printf("Guess summary:     %s\n", TPSChem.sharded_state_summary(dguess))
flush(stdout)

eigvals, ci_out = TPSChem.tps_ci_davidson_sharded(
    dguess, dops, clustered_ham;
    nroots=nroots,
    conv_thresh=ci_conv,
    max_ss_vecs=max_ss_vecs,
    max_iter=max_iter,
    h_storage=h_storage,
    max_mem_H=max_mem_H,
    workers=workers_,
    threaded_worker=true,
    blas_threads=blas_threads,
    verbose=1)

@printf("\nConverged eigenvalues:\n")
for r in 1:nroots
    @printf("  root %3i   E = %18.10f\n", r, eigvals[r])
end
@printf("Solution summary:  %s\n", TPSChem.sharded_state_summary(ci_out))

if haskey(ENV, "TPSCHEM_OUTPUT_JLD2")
    # Only the small eigenvalue vector is saved by default; the eigenvectors stay
    # sharded. Gather explicitly with collect_tpsci_state if you truly need them
    # on the master (do not do this if the vector exceeds node memory).
    JLD2.jldsave(ENV["TPSCHEM_OUTPUT_JLD2"]; eigvals=eigvals)
end

TPSChem.destroy!(ci_out)
TPSChem.destroy!(dguess)
TPSChem.destroy!(dops)

# ===========================================================================
# ALTERNATIVE CALL OPTIONS for `tps_ci_davidson_sharded`
# ---------------------------------------------------------------------------
# The block above is the env-driven default. Below are the distinct ways you
# would call the solver for different problem sizes. They are commented out;
# copy the one that matches your case. `cluster_ops` may be either a plain local
# `Vector{ClusterOps}` (replicated on every worker) or a `DistributedClusterOps`
# (sharded) — the solver accepts both.
# ===========================================================================
#
# --- 0. Always measure the block-sparse H first (cheap, metadata only) -------
#     Tells you whether stored H (:blocks) fits your memory, i.e. 1 node vs many.
#
#   rep = TPSChem.sharded_H_memory_report(dvar, clustered_ham)
#   @printf("dim=%i nfocks=%i nnz=%.3e  H=%.3f GB\n", rep.dim, rep.nfocks, rep.nnz, rep.gb)
#
# --- 1. SINGLE NODE, stored block-sparse H (recommended for ~400K vectors) ---
#     The CI vector is small (~3 MB/root at 400K); only the block-sparse H is
#     large. If it fits one node, run with ONE worker and h_storage=:blocks to
#     get `tps_ci_direct`-like per-iteration speed without a dense (dim^2) H.
#     Launch julia with a single worker:  `julia -p 1 --project ...`
#
#   ws  = workers()                 # a single worker pid
#   dv  = TPSChem.distribute_tpsci_state(guess; workers=ws, strategy=:balanced)
#   e, v = TPSChem.tps_ci_davidson_sharded(dv, cluster_ops, clustered_ham;
#              nroots=4, conv_thresh=1e-6, max_ss_vecs=8,   # bigger ss ok: vector is tiny
#              h_storage=:blocks, workers=ws, verbose=1)
#
# --- 2. SINGLE NODE, matrix-free (store nothing; minimal memory) -------------
#     Use when even the block-sparse H is inconvenient, or as a quick check.
#     Slower per iteration (re-contracts every matvec) but trivial memory.
#
#   e, v = TPSChem.tps_ci_davidson_sharded(dv, cluster_ops, clustered_ham;
#              nroots=4, conv_thresh=1e-6, max_ss_vecs=8,
#              h_storage=:matrixfree, workers=ws, verbose=1)
#
# --- 3. MULTINODE, auto tier by memory budget (the general case) -------------
#     :auto builds stored H if it fits `max_mem_H` GB (aggregate), else falls
#     back to matrix-free. Use a small max_ss_vecs when the VECTOR is large.
#
#   ws  = workers()                 # one worker per node (see common.jl)
#   dv  = TPSChem.distribute_tpsci_state(guess; workers=ws, strategy=:balanced)
#   e, v = TPSChem.tps_ci_davidson_sharded(dv, cluster_ops, clustered_ham;
#              nroots=4, conv_thresh=1e-6, max_ss_vecs=4,   # small: each subspace vec is another full vector
#              h_storage=:auto, max_mem_H=200.0, workers=ws, verbose=1)
#
# --- 4. MULTINODE, force stored block-sparse H, sharded cluster ops ----------
#     For a >node-memory H spread across nodes; cluster ops sharded too.
#
#   dops = TPSChem.compute_cluster_ops_distributed(cluster_bases, ints; workers=ws)
#   TPSChem.add_cmf_operators_distributed!(dops, cluster_bases, ints, d1.a, d1.b)
#   e, v = TPSChem.tps_ci_davidson_sharded(dv, dops, clustered_ham;
#              nroots=4, conv_thresh=1e-6, max_ss_vecs=4,
#              h_storage=:blocks, workers=ws, blas_threads=1, verbose=1)
#
# --- 5. Build the block-sparse H once and reuse the handle across calls ------
#     Useful if you diagonalize the same fixed space more than once.
#
#   bh   = TPSChem.build_block_h_sharded(dv, cluster_ops, clustered_ham; workers=ws)
#   # ... apply H*x directly if needed: TPSChem.apply_sharded_H(bh, x)
#   TPSChem.destroy!(bh)
#
# --- 6. Inside the selected-CI loop -----------------------------------------
#     Same solver, driven by the multinode selected-CI driver. NOTE: only the
#     diagonalization is sharded here; the PT/selection steps still gather
#     vec_var on the master (fine when the vector fits a node, as at ~400K).
#
#   e0, vout = TPSChem.tpsci_ci_multinode(ci_vector, cluster_ops, clustered_ham;
#                  thresh_cipsi=1e-3, thresh_foi=1e-5, conv_thresh=1e-6,
#                  ci_conv=1e-6, ci_max_ss_vecs=4, nbody=4,
#                  use_sharded=true, h_storage=:auto, max_mem_H=200.0,
#                  workers=ws)
# ===========================================================================
