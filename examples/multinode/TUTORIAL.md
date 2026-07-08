# How TPSCI, TPS-CEPA, and SPT Are Computed Across Nodes — Step by Step

This document follows the three production workflows through the multinode
code, step by step: for each stage it says **where it runs** (master or
workers), **what is sent over the network**, **what each worker computes**,
and **how the pieces are combined**. Read §1 once for the small amount of
shared machinery; then each walkthrough (§2 TPSCI, §3 TPS-CEPA, §4 SPT) is
self-contained.

Notation used throughout:

- `M` = the master Julia process (`myid()==1`); `W1, W2, …` = worker
  processes, one per node.
- "ship X → Wk" means X is serialized and sent with `remotecall_fetch`.
- All worker fan-outs run **concurrently** (`@sync for pid … @async
  remotecall_fetch`), so a stage's wall time is the slowest worker, not the
  sum.

If you have never run a distributed or multinode job before, read §0 first —
it builds the mental model and gets you a working run before the deep dive.

---

## 0. New to distributed / multinode runs? Start here

**Why bother.** Some calculations produce a CI vector (the list of coefficients
we solve for) that is larger than the RAM of any single computer — hundreds of
GB or more. You cannot hold it on one machine, so you spread it across several
machines ("nodes") and have them cooperate. That is all "multinode" means here.

**Process vs. thread (the one distinction to internalize).** Two different
kinds of parallelism are used together:

- A **thread** shares memory with its siblings inside *one* program. Threads
  are how each node uses its many CPU cores on the data it already has. No
  copying needed — they all see the same arrays.
- A **process** (here a Julia "worker") has its *own separate memory*, even if
  it happens to run on the same physical machine. Two processes cannot see each
  other's arrays; to share data one must **send it over a connection**
  (serialize → socket → deserialize). This is what lets workers live on
  *different* nodes.

So: **one worker process per node, many threads inside each worker.** Threads
handle on-node core parallelism for free; processes handle across-node scaling
at the cost of having to explicitly move any data that must cross between them.
(No MPI is involved — this uses Julia's built-in `Distributed` standard
library.)

**Master vs. worker.** One process is the **master** (`M`): it coordinates, holds
only small *metadata* (who owns what, sizes, energies), and never touches the
bulk data. The other processes are **workers** (`W1, W2, …`): each holds a
*slice* of the big data and does the heavy math on its slice.

**The core trick: shard, don't gather.** The CI vector is split into pieces
("**shards**") by **Fock sector** (a Fock sector is one block of the vector —
one way of distributing the electrons among the orbital clusters). Each shard
lives on exactly one worker and *stays there*. Work is organized so that a
worker mostly needs its own shard; when it briefly needs a piece of someone
else's, it fetches just that piece. Crucially, the full vector is **never
reassembled** ("gathered") onto any single process — because it wouldn't fit.
"Never-gather" is the whole point: metadata and small numbers move freely, but
the 500-GB object never does.

Everything in §1–§5 is just the careful bookkeeping that makes "keep each shard
in place, move only small things" actually work for each calculation.

### Run your first one (local, two workers on your laptop)

You don't need a cluster to try it — you can put both "nodes" on one machine as
two worker processes. Start Julia with two workers and run the across-node
Davidson driver on a small problem:

```bash
export TPSCHEM_INPUT_JLD2=/path/to/small_problem.jld2   # needs ints, clusters, d1, init_fspace, cluster_bases
export TPSCHEM_NROOTS=1
export TPSCHEM_H_STORAGE=auto
julia -p 2 --project examples/multinode/tpsci_sharded_davidson_driver.jl
```

`-p 2` launches two worker processes; the driver's `init_multinode_workers!`
(in `common.jl`) sets them up. **What success looks like:** it prints the master
and worker pids and their thread counts, a "Block-sparse H" size line, a few
Davidson iterations, then `Converged eigenvalues`. On a real cluster you
instead submit one of the `run_*_4nodes.slurm` scripts (which builds a node
list and calls the same driver) — see `README.md` for the per-driver env vars.

### Minimum vocabulary

| Term | Meaning here |
| --- | --- |
| node | one physical machine in the cluster |
| process / worker | a Julia program with its own private memory; one per node |
| master (`M`) | the coordinating process; holds only metadata |
| thread | a unit of parallelism *inside* one process, sharing its memory |
| shard | one piece of a distributed object, living on one worker |
| Fock sector | the block used to split the CI vector across workers |
| gather | reassemble a full distributed object on one process (avoided in production) |
| FOIS | first-order interacting space — the configurations one step "outside" the current space, searched for what to add next |
| PT1 vector | the perturbative estimate of those outside coefficients; large ones get selected |
| `destroy!` | free a distributed object on *every* worker — required, or memory leaks there |

With that in hand, §1 describes the shared machinery and §2–§4 trace each
workflow. If a run breaks, jump to §5 — most multinode failures are harness
issues with known fixes, not physics.

---

## 1. The shared machinery (read once)

### 1.1 Workers

One Julia process per node (`Distributed` stdlib; no MPI), threads inside each
worker for on-node parallelism. Locally: `addprocs(2;
exeflags="--project=$(Base.active_project())")`. On SLURM:
`init_multinode_workers!()` (`common.jl`) starts one worker per hostname in
`TPSCHEM_MACHINE_FILE`. Every entry point first calls
`ensure_tpsci_multinode_workers!`, which on each worker restores the default
`LOAD_PATH`, activates + instantiates the project, and loads TPSChem.

### 1.2 The two distributed containers

**`DistributedTPSCIstate`** — a TPSCI vector split by **Fock sector**. Master
keeps only metadata (`owners: FockConfig→pid`, per-sector lengths/offsets);
each worker holds an ordinary local `TPSCIstate` with exactly its owned
sectors, stored in a worker-global registry `_TPSCI_SHARDED_STATES[id]` under
the handle's `gensym` id. Coefficients never live on the master.
`destroy!(state)` deletes the payload on every worker — **mandatory** when
done, or it leaks there.

**`DistributedClusterOps`** — cluster operator tables split by **cluster
index**; each cluster's tables are built and stored only on its owner. A
worker needing tables for clusters it does not own fetches just those
(`_materialize_cluster_ops_for_indices`).

Ownership assignment (`strategy=`): `:balanced` = greedy least-loaded
(weighted by config counts), **preserving existing owners** when a space
grows; `:hash` = deterministic `hash(key) mod nworkers`.

### 1.3 The five communication shapes

Everything below reduces to five moves:

| # | Move | Cost on the wire |
|---|------|------------------|
| 1 | ship a small local object (reference state, thresholds) to all workers | once per call |
| 2 | worker fetches a ket Fock block from its owner | the only bulk transfer in sharded matvecs |
| 3 | worker returns per-root scalars / small matrices (energies, overlaps, RDM partials); master **sums** | k×k numbers |
| 4 | worker returns `{fock: length}` only; master rebuilds a metadata handle | tiny; coefficients stay put |
| 5 | debug-only full gather (`collect_tpsci_state`) | avoid in production |

---

## 2. TPSCI multinode, step by step

The selected-CI loop is: *diagonalize in the current space → build the PT1
vector in the first-order interacting space (FOIS) → keep PT configs with
|c⁽¹⁾| > thresh_cipsi → add them to the space → repeat*. There are two
multinode versions; the steps below cover both, flagging where they differ.

- `tpsci_ci_multinode` — **replicated**: every worker caches the full CI
  vector; the job loops are split. Memory win: no dense H, split scratch.
- `tpsci_ci_sharded` — **never-gather**: the CI vector is a
  `DistributedTPSCIstate` for the whole loop; no process ever holds all of it.

### Step 0 — setup (both, once)

1. M: `ensure_tpsci_multinode_workers!` (§1.1).
2. M: orthonormalize the small starting reference.
3. Sharded only: `distribute_tpsci_state` — M assigns each Fock sector of the
   reference to a worker (`:balanced`) and ships each worker *its sectors
   only* (move 1, but partitioned). From here on the vector never returns to
   M.

### Step 1 — diagonalize H in the current space

**Replicated path** (`tps_ci_davidson_distributed`): M keeps the dense
coefficient vector `v` (dim × R). Each Davidson iteration:

1. M splits the *rows* of σ = H·v (one job per bra configuration) round-robin
   over workers. The full `v`, cluster ops, and Hamiltonian were cached on
   every worker once at call start (move 1).
2. Wk, for each assigned bra config: loops connected Fock transfers, and for
   every ket config in its cached copy evaluates
   `check_term → contract_matrix_element(term, …, bra, ket) · v[ket]`,
   accumulating its rows of σ.
3. Wk → M: `(rows, values)`; M assembles the full σ (move 3/4 hybrid) and
   runs the standard dense Davidson update on it.

**Sharded path** (`tps_ci_davidson_sharded`): subspace vectors `V` and sigmas
`HV` are lists of R=1 sharded states. Per iteration:

1. σ = H·v via `tps_ci_matvec_sharded`:
   - M sends each worker the list of *its own* output sectors (move 1, tiny).
   - Wk **prefetches** every connected ket Fock block it does not own from
     that block's owner (move 2) — this is the only bulk traffic — then
     threads over its owned (fock_bra, config_bra) pairs computing
     `Σ_ket Σ_terms contract_matrix_element·v[ket]` into its local σ block.
   - Wk → M: `{fock: length}` (move 4). σ stays distributed.
   With `h_storage=:blocks` (Tier A) the contraction is replaced by dense
   block-GEMV against Hamiltonian sub-blocks built **once** before the loop
   (see §3 Step 3, same machinery).
2. Small projected matrix: `Hss[i,j] = overlap(V[i], HV[j])` — each worker
   returns one number per pair (move 3); M holds only the k×k `Hss`.
3. M: `eigen(Symmetric(Hss))` → Ritz values θ and coefficients y (tiny).
4. Ritz vectors/residuals: `x = Σ y[i]·V[i]`, `r = HV·y − θx` — pure
   worker-local `add_scaled!` passes; only the residual norms (numbers) reach
   M.
5. Preconditioning `t = r/(θ − Hdiag)` — worker-local, using the sharded
   diagonal computed once at call start (each worker builds the
   `zero-transfer` diagonal for its own configs; nothing shipped).
6. Restart when the subspace hits `max_ss_vecs·nroots`: collapse to the Ritz
   vectors, **re-orthonormalize them (2-pass MGS) and rebuild `Hss` from real
   ⟨V|H|V⟩ elements** — this re-orthonormalization is what keeps repeated
   restarts numerically stable (skipping it caused ghost eigenvalues below
   the true ground state on near-degenerate roots; fixed 2026-07-07).
7. Every transient sharded vector is `destroy!`ed each iteration.

### Step 2 — zeroth-order energies for the PT denominators

`Efock[r] = ⟨r|Hcmf|r⟩`. Hcmf is diagonal in the cluster eigenbasis, so each
worker sums `c²·⟨config|Hcmf|config⟩` over its own configs and returns R
numbers; M adds them (move 3). (`compute_expectation_value_sharded_h0`;
replicated path computes the same thing from its cached copy.)

### Step 3 — PT1 vector in the FOIS

(`compute_pt1_wavefunction_distributed` / `compute_pt1_wavefunction_sharded`)

1. M enumerates the FOIS **output sectors**: every `fock_ket + ftrans` for
   Hamiltonian transfers `ftrans`, filtered to valid electron counts and
   available cluster bases (metadata only, no coefficients touched).
2. M assigns each output sector to a worker. **Sharded:** sectors already in
   the variational state keep their owner; new sectors go to the least-loaded
   worker (`_tpsci_sharded_output_chunks`). This owner-preservation is what
   makes Step 5's merge safe. **Replicated:** plain round-robin.
3. Wk, per assigned output sector: builds the σ block
   `⟨FOIS-config|H|0⟩` with the screened open-ended contraction
   (`_open_matvec_thread_job`, threads inside the worker; terms screened by
   `thresh_foi`). Sharded: ket blocks fetched as in the matvec (move 2).
4. `project_out!`: remove configs already in the variational space —
   worker-local deletions (each worker owns both the σ sector and the
   matching variational sector, by construction of step 2).
5. Denominators + PT2 energy: worker-local
   `c¹ = ⟨X|H|0⟩ / (E0 − ⟨X|Hcmf|X⟩)`; each worker returns its contribution
   to `e2[r] = Σ ⟨X|H|0⟩·c¹` (move 3). The PT1 vector stays distributed
   (sharded) or is assembled on M (replicated).

### Step 4 — selection

`clip!(vec_pt, thresh=thresh_cipsi)`: drop PT configs with small |c⁽¹⁾|.
Worker-local deletions; M refreshes sector lengths (move 4).

### Step 5 — grow the variational space

`zero!(vec_pt); add!(vec_var, vec_pt)` — the selected configurations enter
the space **with zero coefficients** (the next diagonalization determines
them). Sharded: each worker merges the PT sectors it owns into its
variational chunk; a sector new to the space simply appears on the worker
that produced it — the **ownership-reconciling merge**. The `add!` guard
verifies shared sectors have identical owners (guaranteed by step 3.2) and
errors loudly otherwise.

### Step 6 — converged?

M compares `e0` with the previous iteration (numbers only). If not converged,
go to Step 1 — the sharded path re-enters it **without any gather**; the
replicated `use_sharded=true` path of `tpsci_ci_multinode` instead calls
`collect_tpsci_state` here, which is exactly the limitation
`tpsci_ci_sharded` removes.

**Verification**: `tpsci_ci_sharded` reproduces single-node `tpsci_ci`
energies and the same selected space on h8 (test
`never-gather tpsci_ci_sharded vs single-node tpsci_ci`).

---

## 3. TPS-CEPA multinode, step by step

(`do_tps_sharded_cepa`, `tps_cepa_sharded.jl`.) CEPA solves the linear system
`(H_QQ − E0 − shift)·x = −h` in the Q space (the FOIS of the reference), with
a self-consistent shift (cepa/acpf/aqcc/cisd), per root. The expensive object
is `H_QQ`; the design decision is to **build it once and reuse it across all
roots × shift-iterations × solver steps**.

### Step 1 — reference solve

Get `(e0, ref)`: either passed in, or solved by the sharded Davidson (§2 Step
1), `tps_ci_direct` (single-node dense), or the replicated distributed
Davidson.

### Step 2 — build the sharded Q space

1. M distributes the (small) reference (move 1, partitioned).
2. `open_matvec_sharded(ref)` — exactly §2 Step 3.1–3.3: each worker builds
   its assigned FOIS sectors of H·ref, screened by `thresh_foi`. Result: the
   Q basis as a `DistributedTPSCIstate`.
3. `project_out!(Q, ref)` — worker-local removal of reference configs.
   Optional `clip!`.

### Step 3 — build H_QQ once, distributed (`h_storage=:blocks`)

(`build_block_h_sharded`; `:auto` first estimates
`nnz = Σ n_bra·n_ket` over connected sector pairs from metadata alone and
falls back to matrix-free if it exceeds `max_mem_H`.)

1. M sends each worker the list of Q sectors it owns (row ownership).
2. Wk, for each owned `fock_bra` and each connected `fock_ket`: fetches the
   ket sector's config list (move 2, cached), then fills the dense block
   `H[bra_configs × ket_configs]` element-by-element with
   `contract_matrix_element`. Blocks are stored worker-locally
   (`_TPSCI_SHARDED_BLOCK_H`), never shipped.
3. Wk → M: byte counts only. H_QQ now exists as distributed block-sparse
   rows, owned by the same workers that own the Q vector's rows.

### Step 4 — per root r: coupling vector

`h = ⟨Q|H|ref_r⟩`: one matrix-free `open_matvec_sharded` of the single-root
reference (cheap, rectangular, once per root), then
`restrict_to_basis_sharded` maps it onto the Q layout (zero-filling configs
the screened build missed; ownership follows Q).

### Step 5 — shift macro-iterations

For each shift update (`shift = f(E_corr)` per the CEPA variant):

Solve `(H_QQ − (e0+shift))x = −h` with **sharded MINRES** (default; CG
available). Each MINRES iteration costs exactly:

1. one operator apply `H_QQ·v`: worker-local block-GEMV over the stored
   blocks (`mul!`), ket blocks fetched/cached (move 2) — with Tier A **no
   contraction is recomputed, ever**;
2. one `add_scaled!` for the shift (worker-local);
3. two to three `overlap`/`norm` reductions → scalars to M (move 3);
4. Givens rotations on 4 numbers, on M;
5. worker-local axpys to update the iterates; all scratch `destroy!`ed.

Then `E_corr = ⟨x|h⟩` (one reduction), update the shift, repeat until
`|ΔE_corr| < tol`. The matrix-free names (`do_fois_cepa_sharded`,
`tpsci_cepa_solve_sharded`) run the same solver with the apply replaced by a
full re-contraction — same answers (tested to 1e-7), many times the work.

### Step 6 — cleanup

`destroy!` the block H (`finally`-guarded) and all per-root temporaries.
Returns `(E_corr per root, e0 .+ E_corr)` and the sharded Q vector.

---

## 4. SPT (`subspace_product_tucker`) multinode, step by step

SPT states are Tucker-compressed. There are **two** multinode paths; which you
want depends on whether the *compressed* state fits a single node:

- **Path A — compute-distributed (`subspace_product_tucker_multinode`).** The
  state stays on the master; only the expensive *job loops* (FOIS, sigma builds,
  expectation values, PT2 sums) are scattered to workers. Simple and fast, but
  the master still holds the whole FOIS / PT1 / variational state, and the
  variational solve is a node-local `ci_solve` — so it only works when that
  compressed state fits one node. Use it for **speed** when memory is not the
  bottleneck.
- **Path B — never-gather (`subspace_product_tucker_sharded`).** The variational
  vector, the FOIS, and the PT1 all stay `DistributedSPTstate`s across the
  workers for the whole loop; nothing is ever gathered onto the master. Use it
  when the **compressed SPT state is itself larger than a node** — the case
  Path A cannot handle. This mirrors the never-gather TPSCI loop of §2.

> Do not assume "Tucker-compressed ⇒ small". Compressed SPT states can be large;
> when they overflow a node, Path A's master-side FOIS/PT1 and its node-local
> `ci_solve` (which holds `~2·max_ss_vecs·R` full vectors) are the ceiling —
> that is exactly what Path B removes.

### Path A — compute-distributed (state on the master)

Per SPT iteration:

### Step 1 — compress + reference energy

1. M: `compress(ref, thresh_var)`, `orthonormalize!` — local (Tucker SVDs on
   the compressed cores).
2. `e0 = ⟨0|H|0⟩` via `compute_expectation_value_distributed`:
   - M splits the destination Fock sectors of σ = H|0⟩ across workers by
     block size (`_spt_fock_chunks_by_length`).
   - M ships each worker *its slice* of the σ basis plus the full (small,
     compressed) ket state (move 1).
   - Wk: materializes only the cluster-op tables its terms need, runs the
     standard local `build_sigma!` on its slice (Tucker-blockwise
     contractions).
   - Wk → M: its σ slice; M merges the disjoint Fock blocks and takes
     `orth_dot(σ, ψ)` locally.
3. Same machinery for `⟨S²⟩`.

### Step 2 — FOIS construction (the big one)

`build_compressed_1st_order_state_distributed`:

1. M enumerates jobs: one per FOIS output sector, each carrying the list of
   (terms, ket sector, ket Tucker blocks) that feed it
   (`_spt_make_fock_jobs`), and splits them across workers **weighted by
   estimated cost** `Σ n_terms·n_ket_blocks` (`_spt_job_chunks`).
2. Wk: threads over its jobs; each job runs the unchanged single-node kernel
   `_build_compressed_1st_order_state_job` (screened Tucker contractions,
   compression at `thresh_foi`).
3. Wk → M: its output sectors; M merges (each sector produced exactly once —
   duplicates are an error, and checked).

### Step 3 — PT1 / PT2

`compute_pt1_wavefunction_distributed` (SPT flavor):

1. M: builds the H0 diagonal on the FOIS basis (local), and the overlap
   projection `Sx` (local).
2. `⟨X|H|0⟩` and `⟨X|Hcmf|0⟩`: two distributed sigma builds into the fixed
   FOIS basis (`build_sigma_distributed`, same scatter/merge as Step 1.2).
3. M: assembles `σ − XF0 − Sx(E0−F0)`, applies denominators, takes
   `ecorr = orth_dot(σ, ψ₁)` — all local on the compressed objects.

(For a PT2 *energy only*, `compute_pt2_energy_distributed` skips storing any
global σ: workers run `_pt2_job`/`_pt2_job_blockwise` per FOIS sector and
return only per-root energy partials, which M sums — move 3. The σ·σ variance
`compute_spt_sigma_norm_blockwise_distributed` is the same shape, returning
`⟨σ|σ⟩` partials; note `σ² = ⟨0|H²|0⟩`, so `variance = σ² − E0²` exactly.)

### Step 4 — grow and solve variationally (master-local)

M: `nonorth_add!(var_vec, pt1_vec)` → `compress` → `orthonormalize!` →
`ci_solve` (the local Tucker CI). This is Path A's key assumption — and its
limit: the whole compressed variational state must fit one node, and the local
`ci_solve` Davidson holds `~2·max_ss_vecs·R` full copies of it. When that
overflows a node, switch to Path B, whose `spt_ci_davidson_sharded` keeps those
subspace vectors sharded. If cluster ops were sharded (`DistributedClusterOps`),
Path A gathers them for this one step (with a warning); Path B does not.

### Step 5 — converged?

Compare `e_var` with the previous iteration; loop to Step 1.

**So the Path A division of labor is:** distributed = FOIS jobs, sigma builds,
expectation values, PT2/σ² sums (all "scatter jobs, sum/merge results");
local = compression, basis projection, variational Tucker CI.

### Path B — never-gather sharded SPT

`subspace_product_tucker_sharded` reuses the §2 sharded machinery. The state is a
`DistributedSPTstate` — the SPT analogue of `DistributedTPSCIstate`: Tucker
blocks split by Fock sector, the master holding only metadata. Ownership is
`:hash`, so any two states sharing a Fock sector place it on the **same** worker;
that makes the per-block ops (compress, `nonorth_add!`, `project_into_new_basis`)
fully worker-local — only the matvec and FOIS fetch across Fock sectors (move 2).

Each single-node step becomes a sharded one:

1. **compress / orthonormalize** — `compress_sharded` (per block, no comms);
   `orthonormalize!` (R×R Gram via sharded `overlap`, then a worker-local
   rotation of the cores).
2. **e0 = ⟨0|H|0⟩ and ⟨S²⟩** — `spt_expectation_sharded`: apply the operator with
   the never-gather Tucker matvec (`build_sigma_sharded`), then take the sharded
   overlap diagonal. Only R numbers reach M (move 3).
3. **FOIS** — `build_compressed_1st_order_state_sharded`: M assigns each output
   Fock sector to a worker (reference sectors pinned to their current owner); the
   worker fetches the reference ket blocks it needs (move 2) and runs the
   *unchanged* local kernel `_build_compressed_1st_order_state_job`. The FOIS is
   born sharded — never assembled on the master.
4. **PT1 / PT2** — `compute_pt1_wavefunction_sharded`: rotate the FOIS to its
   pseudo-canonical basis (shard-local, giving the diagonal `Fdiag`), form `Sx`,
   then `⟨X|H|0⟩` and `⟨X|F|0⟩` with the never-gather matvec, and assemble
   `σ − XF0 − Sx(E0−F0)`, the denominators, and `ecorr` from shard-local core
   arithmetic + sharded reductions. No full FOIS/PT1 copy on the master.
5. **grow + solve** — `nonorth_add_sharded!` grows the variational space in place;
   `compress_sharded` + `orthonormalize!`; `project_into_new_basis_sharded`
   carries the previous solution as the guess; then the **distributed Tucker CI
   solver** `spt_ci_davidson_sharded` diagonalizes in the fixed Tucker basis with
   subspace vectors that are themselves sharded. It shares the *same* block
   Davidson core as the sharded TPSCI solver, via the `AbstractShardedState`
   supertype — so the ghost-eigenvalue-safe restart (§2 Step 1.6) is inherited.

Everything crossing the network is a k×k reduction or a single block fetch;
validated against single-node to machine precision (Davidson 8.9e-15, PT2
1.8e-15, full loop to 10 digits). Not supported here (use Path A / `spt_multinode`):
the S² spin extension (`thresh_spin`) — the sharded driver errors on it. A stored
block-sparse Tucker-H (Tier A) is future work; the sharded matvec is matrix-free.

**Energy-only shortcuts (memory-safe under either path):** if you only need the
PT2 *energy* or the σ² variance, `compute_pt2_energy_distributed` and
`compute_spt_sigma_norm_blockwise_distributed` are blockwise — a worker builds one
FOIS sector, extracts its scalar, frees it, and returns only per-root numbers, so
no global σ is ever stored anywhere (even under Path A).

Concrete end-to-end script: `run_spt_multinode_cr2_morokuma.jl` (and the TPSCI
sibling `run_tpsci_multinode_cr2_morokuma.jl`).

---

## 5. What is identical to single-node, and what to check when things break

The physics kernels — `contract_matrix_element`, `_open_matvec_thread_job`,
`_build_compressed_1st_order_state_job`, `_pt2_job*`, `build_sigma!`,
denominators — are **the same functions** the single-node code calls. The
multinode layer only decides *who* runs them and combines partials over disjoint
work. That is why every distributed routine is tested against its single-node
sibling (RDMs/σ²/expectations to ~1e-12; solver energies to ~1e-7–1e-14; the
sharded SPT solver/PT2 to ~1e-15), and why multinode failures are almost always
harness issues:

| Symptom | Likely cause |
|---|---|
| `Package Pkg not found` on worker | spawned under `Pkg.test` sandbox; fixed in `ensure_tpsci_multinode_workers!` (restores `LOAD_PATH`); spawn test workers with `--project` |
| `requires matching Fock ownership` | mixing sharded states from independent `distribute_tpsci_state` / `distribute_spt_state` calls; derive related states from each other, or use `:hash` so ownership is a stable function of the Fock sector |
| `No sharded TPSCI state cached` / `No sharded SPT state cached` | use-after-`destroy!`, or wrong worker list |
| worker memory grows | leaked sharded states — audit `destroy!` |
| Davidson energies dive below the ground state | pre-2026-07-07 build without the restart re-orthonormalization (shared by the TPSCI and SPT sharded solvers via `AbstractShardedState`) |

Memory rules: `destroy!` every sharded state and block-H you create; the
solvers model this (per-iteration frees, `finally` blocks). Use
`sharded_state_summary` / `sharded_spt_summary` / `cluster_ops_summary` to audit
per-worker sizes. `collect_tpsci_state` / `collect_spt_state` are debug-only.

When to use what:

- **Single node** while everything fits — fastest, zero serialization.
- **Replicated / compute-distributed multinode** (`tpsci_ci_multinode`,
  `subspace_product_tucker_multinode`) — when dense H or core count is the
  blocker but the vector/state still fits a node.
- **Sharded / never-gather** (`tpsci_ci_sharded`, `do_tps_sharded_cepa`,
  `subspace_product_tucker_sharded`) — when the CI / Q / SPT vector itself
  outgrows a node.
- **Energy-only** (`compute_pt2_energy_distributed`,
  `compute_spt_sigma_norm_blockwise_distributed`) — memory-safe at any size when
  you only need the PT2 energy or σ² variance.
