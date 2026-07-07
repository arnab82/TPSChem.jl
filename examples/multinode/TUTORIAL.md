# TPSCI Multinode: How It Works

A complete tour of the distributed backend in TPSChem.jl: how workers are
defined, how a calculation is divided across nodes, how partial results are
accumulated, what packages and algorithms are used, and how the multinode path
differs from the single-node path. Written against the code as reviewed and
tested on 2026-07-07 (all multinode testsets passing: 17/17 properties/RDMs/
variance, 18/18 sharded Davidson, 6/6 stored-H CEPA).

Source files (≈5100 lines):

| File | Contents |
|---|---|
| `src/core/tpsci_multinode.jl` | distributed data structures, sharded vector algebra, matvecs, replicated backend, worker setup |
| `src/core/tpsci_sharded_davidson.jl` | across-node block Davidson, stored block-sparse H |
| `src/core/tps_cepa_sharded.jl` | stored-H TPS-CEPA, sharded MINRES/CG |
| `src/core/spt_multinode.jl` | distributed SPT: FOIS, sigma, expectation values, PT2 |
| `src/core/spt_variance_multinode.jl` | distributed SPT σ·σ (variance) |
| `src/core/tpsci_property_multinode.jl` | distributed 1-/2-RDMs and one-electron properties |

---

## 1. Execution model: one worker per node, threads inside

TPSChem uses **two nested levels of parallelism**, and no MPI:

```text
┌─────────── master process (myid()=1) ───────────┐
│  metadata, small k×k matrices, orchestration     │
└────┬──────────────────┬──────────────────┬───────┘
     │ Distributed       │                  │   (TCP; remotecall_fetch)
┌────▼─────┐       ┌─────▼────┐       ┌─────▼────┐
│ worker 2 │       │ worker 3 │       │ worker 4 │   ← one per NODE
│ (node A) │       │ (node B) │       │ (node C) │     owns a memory slice
│ ┌──────┐ │       │ ┌──────┐ │       │ ┌──────┐ │
│ │thread│…│       │ │thread│…│       │ │thread│…│   ← Threads.@threads
│ └──────┘ │       │ └──────┘ │       │ └──────┘ │     + threaded BLAS
└──────────┘       └──────────┘       └──────────┘
```

- **Across nodes**: Julia's `Distributed` stdlib. Each worker is a full Julia
  process, typically one per node, owning a disjoint slice of the data.
  Communication is explicit message passing (`remotecall_fetch`).
- **Within a node**: Julia threads (`Threads.@threads :static`) over the
  worker's job slice, plus multithreaded BLAS (`BLAS.set_num_threads`).

This is why every distributed entry point takes `workers=` (which processes)
plus `threaded_worker=` and `blas_threads=` (on-node parallelism). The design
rule: *one memory-owning worker per node; never many single-threaded workers
per node* — that would replicate the large read-only objects once per process.

### Packages used, and why

| Package | Role |
|---|---|
| `Distributed` (stdlib) | worker processes, `remotecall_fetch`, `@sync`/`@async` fan-out, serialization of arguments/results |
| `Base.Threads` | on-node loop parallelism inside worker kernels |
| `LinearAlgebra`/BLAS | dense block GEMV/GEMM in stored-H paths; `BLAS.set_num_threads` per worker |
| `Pkg` (stdlib) | activating/instantiating the project on each worker at setup |
| `OrderedCollections` | deterministic ownership and length maps in the distributed handles |
| `JLD2` | problem input / result output in the cluster drivers |

No MPI, no `SharedArrays`, no GPU.

---

## 2. Defining workers

### Locally (tests, laptop)

```julia
using Distributed
addprocs(2; exeflags="--project=$(Base.active_project())")
```

The `exeflags` matter: under `Pkg.test()`, bare `addprocs` workers inherit a
restricted load path in which even `import Pkg` fails (see §10).

### On a cluster (SLURM)

`examples/multinode/common.jl :: init_multinode_workers!()` reads
`TPSCHEM_MACHINE_FILE` (one hostname per line) and starts one worker per host:

```julia
addprocs(hosts; exeflags="--project=$(project_root) --threads=$(threads)",
         env=worker_env)   # forwards JULIA_DEPOT_PATH, OMP/MKL/OPENBLAS threads
```

### Making workers ready

Every distributed routine funnels through
`ensure_tpsci_multinode_workers!(workers=…)`, which idempotently prepares each
worker:

```julia
copy!(LOAD_PATH, ["@", "@v#.#", "@stdlib"])  # restore default load path (Pkg.test sandbox omits @stdlib)
import Pkg
Pkg.activate(project_root; io=devnull)
Pkg.instantiate(; io=devnull)                # ensure dependencies are resolved on this node
using TPSChem
```

After this, any worker can execute any TPSChem function shipped to it.

---

## 3. Two multinode backends — know which one you are using

This is the most important thing to understand about the code. There are two
distinct distributed backends with different memory behavior:

### Backend A: replicated (job-distributed)

The full `TPSCIstate`/`SPTstate` and cluster ops are **serialized to every
worker** (or cached there once per call); only the *work loop* is split.
Memory win: no dense Hamiltonian, and FOIS/matvec *scratch* is split. The CI
vector itself still must fit on every node.

Entry points: `open_matvec_distributed`, `tps_ci_matvec_distributed`,
`tps_ci_davidson_distributed`, `compute_pt1_wavefunction_distributed`,
`tpsci_ci_multinode`, all the property/RDM `compute_*_distributed` routines,
and the SPT routines in `spt_multinode.jl` / `spt_variance_multinode.jl`.

Worker-side cache: the per-call `Ref`s `_TPSCI_MULTINODE_CI`,
`_TPSCI_MULTINODE_CLUSTER_OPS`, `_TPSCI_MULTINODE_CLUSTERED_HAM` (populated by
`_tpsci_multinode_cache_problem!` so the big read-only objects ship **once**
per call, not once per job).

### Backend B: sharded (data-distributed)

The vector itself is split: no process ever holds all of it. This is the path
for CI vectors larger than one node's RAM.

Entry points: everything taking/returning a `DistributedTPSCIstate` —
`distribute_tpsci_state`, `open_matvec_sharded`, `tps_ci_matvec_sharded`,
`tps_ci_davidson_sharded`, `compute_pt1_wavefunction_sharded`,
`do_tps_sharded_cepa`, plus the sharded vector algebra (§5).

The SPT results and the property/RDM outputs are *gathered* on the master
(they are small: RDMs are `norb⁴`-ish, energies are scalars) — only the CI/Q
vectors ever need sharding.

---

## 4. The distributed data structures

### `DistributedTPSCIstate{T,N,R}` — CI vector sharded by Fock sector

A TPSCI vector is a dict `FockConfig → (ClusterConfig → coefficients)`. Each
**Fock sector** is assigned to exactly one owning worker:

```julia
mutable struct DistributedTPSCIstate{T,N,R}
    id            :: Symbol                        # handle; payload key on workers
    clusters      :: Vector{MOCluster}
    workers       :: Vector{Int}
    owners        :: OrderedDict{FockConfig,Int}   # fock sector → owning pid
    lengths       :: OrderedDict{FockConfig,Int}   # #configs per sector
    offsets       :: OrderedDict{FockConfig,Int}   # global offsets (deterministic)
    local_lengths :: OrderedDict{Int,Int}          # #configs owned per pid
    total_length  :: Int
end
```

**The master holds only this metadata — never coefficients.** Each worker
keeps an ordinary local `TPSCIstate` containing exactly its owned sectors, in
a worker-global registry:

```julia
const _TPSCI_SHARDED_STATES = Dict{Symbol,Any}()   # id → local TPSCIstate (on each worker)
```

Every sharded operation creates result states under a fresh `gensym` id and
returns a new metadata handle; `destroy!(state)` deletes the payload on every
worker. **You must call `destroy!` on every sharded state you are done with**
— worker-side payloads are not garbage-collected when the master handle goes
out of scope (§9).

### `DistributedClusterOps{T}` — operator tables sharded by cluster

Cluster operator tables (local TDMs, `H`, `Hcmf`, optionally 2-RDM `Ppqsr`)
are sharded by **cluster index**, matching the natural independence of
`compute_cluster_ops`: each cluster's tables are built *only on its owner*
(`compute_cluster_ops_distributed`, `add_cmf_operators_distributed!`,
`add_spinfree_2rdm_ops_distributed!`, `compute_cluster_ops_2rdm_distributed`).

When a worker needs tables for clusters it does not own,
`_materialize_cluster_ops_for_indices(ops, needed)` fetches **only the needed
clusters** from their owners (kernels first compute `needed` from the actual
terms they will touch — see `_tpsci_needed_cluster_indices`). A small
per-worker cache (`_TPSCI_SHARDED_CLUSTER_H0_DIAGS`) additionally memoizes the
`Hcmf` diagonals used by preconditioners and PT denominators, so those are
fetched once, not per call.

### Ownership assignment (`strategy=`)

- `:balanced` (default) — greedy: each sector/cluster goes to the currently
  least-loaded worker, weighted by configuration count (vectors) or basis-size
  work estimate (cluster ops). For output spaces that extend an input space
  (`_tpsci_sharded_output_chunks`), **sectors already owned keep their owner**
  and only new sectors are placed — so selected-CI growth never reshuffles the
  existing layout.
- `:hash` — `owner = pids[hash(key) mod nworkers + 1]`: deterministic and
  reproducible across runs/objects, at the cost of balance.

---

## 5. Communication patterns: scatter, gather, and sharded algebra

### The universal scatter idiom

Everything uses explicit fan-out — no `pmap`, no `@distributed`:

```julia
results = Dict{Int,Any}()
@sync for pid in pids
    @async begin
        results[pid] = Distributed.remotecall_fetch(worker_kernel, pid,
                                                    chunks[pid], args...)
    end
end
```

All workers run their chunk **concurrently**; the master blocks until all
return. (The `@async` tasks are cooperatively scheduled on the master's
thread, so the `results[pid] = …` writes cannot race.)

### Three gather shapes

1. **Sum of partials** — energies, RDMs, σ², overlaps. Work units are
   disjoint, so summing per-worker partials reconstructs the serial answer
   exactly (verified to ~1e-12 in the tests):
   `gamma_aa .+= results[pid].gamma_1`.
2. **Block assembly** — each worker returns whole blocks it owns; the master
   writes them into slots (2-RDM root-pair blocks) or, for sharded outputs,
   collects only **per-sector lengths** and rebuilds a metadata handle
   (`_tpsci_sharded_metadata_from_lengths`); coefficients never move.
3. **Never gathered** — the sharded solvers. Only `R×R` `overlap` matrices
   (typically `1×1` scalars) cross the wire; all vector arithmetic is
   worker-local.

### The sharded vector algebra

These make a `DistributedTPSCIstate` behave like a vector without gathering.
Each is a one-remote-call-per-worker operation on owned blocks:

| Operation | Returns to master | Notes |
|---|---|---|
| `overlap(s1,s2)`, `norm(s)` | `R×R` / `R` numbers | reduction |
| `scale!`, `zero!`, `mult!` (root rotation), `add_scaled!(d,α,s)`, `linear_combination!(d,α,s,β)` | nothing | in-place on owned blocks; ownership must match (guarded by loud errors) |
| `add!`, `project_out!`, `clip!` | refreshed metadata | may change sector lengths → `_tpsci_sharded_refresh_metadata!` |
| `orthonormalize!` | — | Löwdin: one `overlap` + one broadcast `mult!` |
| `copy_sharded_state`, `similar_sharded_state` | new handle | worker-local copy/zero-like |
| `extract_root_sharded`, `combine_roots_sharded` | new handle | R↔1 conversions, ownership-preserving |
| `restrict_to_basis_sharded(src, basis)` | new handle | project onto `basis`' layout, zero-fill; used for Q-space |
| `collect_tpsci_state`, `collect_cluster_ops` | full object | **debug/small cases only** |

---

## 6. Walk-through 1: a distributed property (replicated backend)

`compute_1rdm_distributed(bra, ket, cluster_ops)`:

1. `ensure_tpsci_multinode_workers!()` → workers ready.
2. Split the `R1×R2` **transition root pairs** across workers
   (`_tpsci_property_root_pair_chunks`; balanced or hash).
3. Scatter: each worker materializes only the cluster ops it needs, then runs
   the **unchanged single-node kernel** (`compute_1rdm` /
   `compute_1rdm_threaded`) per root pair.
4. Gather: master sums `norb×norb×R1×R2` partials.

The 2-RDM, spin-flip 1-RDM, one-electron property, SPT PT2/σ² routines follow
the same recipe with different work units (root pairs vs. FOIS Fock-sector
jobs) and kernels. **The physics kernels are identical to single-node** —
`contract_matrix_element`, `form_sigma_block_expand`, `_pt2_job*` — the
distributed layer only decides *who* runs them and adds the partials.

### Aside: SPT σ² and the variance

`compute_spt_sigma_norm_blockwise[_distributed]` builds `σ = H|0⟩`
block-by-block over **all** reachable destinations — including the reference's
own Fock sector and p-space blocks (the FOIS jobs are built with
`require_cluster_space=false` and destination spaces include both p and q).
So `σ² = ⟨0|H²|0⟩` (within nbody/thresh truncation), and the driver's

```julia
variance = sigma2 .- E0.^2      # spt_variance_driver.jl
```

is the textbook `⟨H²⟩−⟨H⟩²`. (Verified numerically: for the he4 CMF reference,
`sigma2 − E0² ≥ 0` and small, as a near-eigenstate's variance should be.)

---

## 7. Walk-through 2: the sharded matvec (data-distributed backend)

`tps_ci_matvec_sharded(v, cluster_ops, clustered_ham)` computes `σ = H v`
inside the current variational space, with `v` and `σ` both sharded:

```text
master:  chunks[pid] = Fock sectors owned by pid          (row ownership)
         @sync remotecall_fetch(matvec_chunk!, pid, …)    (scatter)

worker pid, for each owned output sector fock_bra:
  1. prefetch ket cache: for every fock_ket connected to fock_bra by some
     Hamiltonian transfer, fetch that sector's configs
        - from local storage if pid owns it,
        - else remotecall_fetch from its owner    ← the ONLY bulk communication
  2. (DistributedClusterOps) materialize only the cluster ops the touched
     terms need
  3. Threads.@threads over owned (fock_bra, config_bra):
        σ[bra] = Σ_ket Σ_terms check_term ? contract_matrix_element(...)·v[ket]
     (the ket cache is fully prefetched BEFORE threading → read-only during
      the parallel loop → no locks needed)
  4. store σ chunk under the output id; return {fock: length} only

master:  rebuild metadata handle from lengths              (gather shape 2)
```

`open_matvec_sharded` is the same pattern for the *rectangular* map into the
FOIS (output sectors = all Hamiltonian-reachable sectors, assigned by
`_tpsci_sharded_output_chunks` with owner-preservation), used for PT1
(`compute_pt1_wavefunction_sharded`) and CEPA coupling vectors.

Key property: network traffic per matvec = (connected ket sectors fetched
cross-worker) + (a lengths dict). Coefficients of the *output* never move.

---

## 8. Walk-through 3: the across-node solvers

### Sharded Davidson (`tps_ci_davidson_sharded`)

Diagonalizes H in the space of a sharded vector; **no full vector ever exists
anywhere**. Subspace basis `V` and sigma `HV` are lists of R=1 sharded states.

```text
Hdiag = compute_diagonal_sharded(...)          # worker-local zero-transfer diagonal
Hop   = :blocks ? stored block-sparse H : matrix-free (tps_ci_matvec_sharded)
seeds = extract_root_sharded.(guess, 1:nroots), MGS-orthonormalized

iterate:
  Hss[i,j] = overlap(V[i], HV[j])              # k×k on master  ← only wire traffic
  eigen(Symmetric(Hss)) → (θ, y)               # tiny, on master
  Ritz x_s = Σ_i y[i,s]·V[i]                   # sharded add_scaled!
  residual r_s = HV·y_s − θ_s x_s;  |r_s| ≤ tol ∀s → done
  precondition: t = r/(θ − Hdiag)              # worker-local, scaled sign-preserving floor
  if subspace would exceed max_ss_vecs·nroots (default 4·nroots):
      collapse to Ritz vectors, RE-ORTHONORMALIZE them (2-pass MGS), and
      REBUILD Hss from actual ⟨x|H|x⟩          # ← stability-critical
  MGS new directions against V; drop if norm ≤ lindep_thresh; append
  destroy! every transient sharded vector each iteration
```

Two Hamiltonian tiers, chosen by `h_storage`:

- **Tier A `:blocks`** — `build_block_h_sharded` computes each connected
  `(fock_bra, fock_ket)` pair as a dense sub-block via
  `contract_matrix_element`, stored on the **row-owner** worker
  (`_TPSCI_SHARDED_BLOCK_H`). Every matvec is then worker-local block-GEMV
  (`mul!`) over cached ket blocks — `tps_ci_direct`-like per-iteration speed,
  paid once. Requires the Davidson subspace vectors to share the space's
  config layout (they do: all are built by `similar/copy/extract` from it).
- **Tier B `:matrixfree`** — every matvec re-contracts cluster operators.
  Minimal memory, slower per iteration.
- `:auto` — `sharded_H_memory_report` estimates `nnz(H) = Σ n_bra·n_ket` over
  connected sector pairs (metadata-only, O(nfocks²)) and picks Tier A iff the
  estimate fits `max_mem_H` GB.

**Numerical stability (fixed 2026-07-07):** the restart originally reused the
Ritz vectors as-is with `Hss = diag(θ)`. Round-off accumulated across
restarts; near-degenerate roots then fed in-span corrections whose normalized
round-off corrupted `Hss`, producing ghost eigenvalues collapsing *below* the
true ground state. The fix re-orthonormalizes at restart and rebuilds `Hss`
from real matrix elements, plus a scaled floor in the preconditioner
(`√eps·max(|θ|,|Hdiag|,1)`, sign-preserving) instead of a fixed `+1e-12`.
Verified: 20/20 random-guess seeds converge to ~1e-14 of `tps_ci_direct` in
both tiers (previously ~40% diverged).

### Stored-H TPS-CEPA (`do_tps_sharded_cepa`)

Solves `(H_QQ − eshift)·x = −h` over the sharded Q space (FOIS of the
reference), per root, with macro-iterations updating the shift
(cepa/acpf/aqcc/cisd):

1. Reference solve (or take `e0`): `:sharded_davidson` (default), `:direct`,
   or `:distributed_davidson`.
2. Q space: `open_matvec_sharded(ref)` → `project_out!(ref)` → optional clip.
3. **`H_QQ` built once** (`h_storage=:blocks|:auto`) and reused across *all*
   roots × macro-iterations × solver steps — this is the entire point; the
   matrix-free path re-contracts every MINRES/CG iteration. (Measured ~4×
   faster already on a 79-config toy Q space.)
4. Coupling vector `h = ⟨Q|H|ref⟩`: matrix-free `open_matvec_sharded` once per
   root, then `restrict_to_basis_sharded` onto the Q layout.
5. Linear solver: hand-rolled **sharded MINRES** (default; symmetric
   indefinite, Givens rotations) or **CG**, both operating purely on the
   sharded algebra of §5 with careful `destroy!` of all scratch.
6. `E_corr = ⟨x|h⟩`; iterate shift to self-consistency.

The old names `do_fois_cepa_sharded` / `tpsci_cepa_solve_sharded` still work —
they are thin wrappers calling the same code with `h_storage=:matrixfree`.

### The outer selected-CI loop (`tpsci_ci_multinode`)

Replicated-backend selected CI: matrix-free distributed Davidson (or, with
`use_sharded=true`, the sharded Davidson for the diagonalization step),
distributed PT1, clip, add, repeat. **Honest caveat (documented in the code):**
with `use_sharded=true` the PT/selection steps still run on the replicated
backend, so `vec_var` is re-gathered on the master between iterations
(`collect_tpsci_state`). This distributes the diagonalization working set —
the dominant memory cost — but a fully never-gather selected-CI loop still
needs sharded selection + an ownership-reconciling merge (future work).

---

## 9. Memory management rules

1. **`destroy!` every sharded state and `ShardedBlockH` you create.** Worker
   payloads live in global registries keyed by `gensym` ids; dropping the
   master handle leaks the payload on every worker. The solvers model the
   discipline (explicit `destroy!` per iteration; `finally` blocks around
   stored H). At production scale one leaked subspace vector can be hundreds
   of GB.
2. Cluster ops and the clustered Hamiltonian are cached per call on workers
   (`_tpsci_sharded_cache_operator_problem!`) — repeated matvecs in one solve
   do not re-ship them.
3. `collect_tpsci_state` / `collect_cluster_ops` are debug helpers; anything
   using them (including `tpsci_ci_multinode`'s inter-iteration gather) caps
   the problem size at one node's memory for that step.
4. Diagnostics: `sharded_state_summary(state)` and `cluster_ops_summary(ops)`
   report per-worker sizes.

---

## 10. Single-node vs. multinode

| | Single node | Multinode replicated (A) | Multinode sharded (B) |
|---|---|---|---|
| CI vector | one `TPSCIstate` in RAM | full copy cached per worker | Fock sectors split; nobody holds it all |
| Cluster ops | one `Vector{ClusterOps}` | shipped/cached per call | sharded by cluster, fetched on demand |
| Hamiltonian | dense H (`tps_ci_direct`) or matrix-free (`tps_ci_davidson`) | matrix-free only | block-sparse distributed (Tier A) or matrix-free (Tier B) |
| Parallelism | threads + BLAS | workers × threads | workers × threads |
| Communication | none | args out, partials back | ket-block fetches + k×k reductions |
| Peak memory bound | one node (vector + maybe dense H) | one node (vector), but no dense H and split scratch | **aggregate cluster memory** |
| Same physics kernels? | — | yes, byte-identical | yes, byte-identical |

The per-block kernels (`contract_matrix_element`, `form_sigma_block_expand`,
`compute_1rdm`, denominators, DIIS) are the same code in all three columns —
which is why every distributed routine is testable against its single-node
sibling to ~1e-12, and why multinode bugs live almost exclusively in the
*harness*: ownership, load paths, restart orthogonality, memory lifetime. Both
bugs found in this code's history were exactly that (the Davidson restart, and
the `Pkg.test` worker load path).

**When to use which:** single node whenever the vector (and, for
`tps_ci_direct`, dense H) fits and one node's cores suffice — it is simpler
and faster (zero serialization). Replicated multinode when dense H is the
blocker or you need more cores. Sharded multinode when the *vector* no longer
fits on a node. Below those thresholds, multinode is strictly slower.

### Known limitations (by design, current state)

- SPT results and RDMs are gathered on the master (they are small; the local
  Tucker CI solver is single-node).
- `tpsci_ci_multinode(use_sharded=true)` gathers between iterations (§8). A
  fully never-gather selected-CI loop is feasible with the existing sharded
  primitives (`compute_pt1_wavefunction_sharded`, `clip!`, `add!`,
  `orthonormalize!`, `tps_ci_davidson_sharded`) for the default path
  (`incremental=false`, `thresh_spin=nothing`); the open pieces are verifying
  the ownership-reconciling `add!` merge for newly appearing sectors and
  feature parity for the optional S²-extension/incremental paths.
- Diagnostic summaries (`sharded_state_summary`, `cluster_ops_summary`) and
  the debug gathers still loop over workers serially (rare calls; the hot
  reductions — `overlap`, `norm`, H0 expectations, PT1 denominators — fan out
  concurrently as of 2026-07-07).

---

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Package Pkg not found in current path` on a worker | Workers spawned under `Pkg.test` inherit a sandbox `JULIA_LOAD_PATH` without `@stdlib`. Fixed inside `ensure_tpsci_multinode_workers!` (restores the default `LOAD_PATH` first); also spawn test workers with `exeflags="--project=$(Base.active_project())"`. |
| `No sharded TPSCI state cached with id …` | Using a handle after `destroy!`, or on a worker list it was never distributed to. |
| `… requires identical worker lists` / `requires matching Fock ownership` | Mixing sharded states created with different `workers=` or strategies. Create related states from each other (`similar/copy/extract/restrict`) so ownership is inherited. |
| Davidson “did not converge” with energies diving far below the reference | You are on a pre-2026-07-07 build without the restart re-orthonormalization fix. |
| Worker memory grows across a script | Leaked sharded states — audit your `destroy!` calls (§9). |
| First distributed call is slow | One-time per-worker `Pkg.activate` + `using TPSChem` + JIT; subsequent calls reuse it. |

---

## 12. Minimal recipes

Local, end to end:

```julia
using Distributed
addprocs(2; exeflags="--project=$(Base.active_project())")
using TPSChem
# … build ints, clusters, cluster_bases, clustered_ham, d1 as usual …

# distributed cluster ops (sharded by cluster)
dops = TPSChem.compute_cluster_ops_distributed(cluster_bases, ints; workers=workers())
TPSChem.add_cmf_operators_distributed!(dops, cluster_bases, ints, d1.a, d1.b)

# properties on a solved reference psi (replicated backend)
γaa, γbb = TPSChem.compute_1rdm_distributed(psi, dops; workers=workers())

# across-node diagonalization (sharded backend)
dv = TPSChem.distribute_tpsci_state(guess; workers=workers(), strategy=:balanced)
e, dv_out = TPSChem.tps_ci_davidson_sharded(dv, dops, clustered_ham;
                                            nroots=3, h_storage=:auto)
TPSChem.destroy!(dv_out); TPSChem.destroy!(dv); TPSChem.destroy!(dops)
```

On a cluster: set `TPSCHEM_INPUT_JLD2` (+ machine file) and `sbatch` a script
from this directory; every driver funnels through `common.jl` →
`init_multinode_workers!` → the scatter/gather machinery above. See
`README.md` here for the per-driver environment variables.

---

## 13. One-paragraph mental model

Start one Julia worker per node. Shard the CI vector by Fock sector and the
cluster operators by cluster index — metadata on the master, payloads in
worker-global registries keyed by handle ids. Every operation scatters one
`remotecall_fetch` per worker over its owned slice, threads within the worker,
and gathers either a small summed partial (energies, RDMs, σ²), a set of owned
blocks/lengths (matvec outputs), or — in the sharded Davidson/CEPA solvers —
nothing but `k×k` overlap matrices, so no full vector ever exists on any
single machine. The physics kernels are byte-identical to the single-node
code; the multinode layer is ownership, message passing, and memory lifetime.
