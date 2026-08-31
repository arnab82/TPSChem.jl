# CEPA job submission: what each keyword does

Reference for the knobs that decide how a TPS-CEPA run behaves — how H is applied
on the Q space, which linear solver runs, whether the roots share a solve, and
where the memory goes. Applies to both the single-node entry point
(`do_fois_cepa`) and the across-nodes one (`do_tps_sharded_cepa`).

Environment variable names are the ones the launchers in `examples/multinode/`
read. Single-node runs are usually driven by your own script, so there the
keyword name in the second column is what matters — the `TPSCHEM_*` names below
exist only where a launcher in this repo defines them.

---

## Start here: the three choices that matter

Everything else is a threshold. These three decide the cost of the run.

### 1. How is H applied on the Q space?

The Q space has dimension `dim_q` (printed as `shared FOIS dim`). The whole
question is whether H_qq fits in memory.

| single node `build_hqq` | multinode `h_storage` | stores | when to use |
|---|---|---|---|
| `:sparse` | `:blocks` | the sparse matrix, `O(nnz)` | it fits — fastest applies by far |
| `:direct` / `:parallel` | — | dense, `dim_q² × 8 B` | small `dim_q` only |
| `:matvec` | — | nothing; entries recomputed per apply | matrix will not fit |
| `:fois` | `:matrixfree` | nothing; H applied through `open_matvec_thread` | last resort |
| `:auto` (default) | `:auto` (default) | — | picks for you (multinode checks the memory budget) |

**Check the fill before committing to a big run.** H_qq is *not* very sparse —
around 34–36% on the Cr2 systems measured. The solver prints

```
H_qq nnz = 17203358  (34.089% fill, 0.26 GiB CSC)
```

At 35% fill, storage is `dim_q² × 0.35 × 16 B`. That is 0.26 GiB at
`dim_q = 7104`, but **~180 GB at `dim_q = 181000`** — so past roughly
`dim_q ≈ 40000` on a 256 GB node, `:sparse`/`:blocks` stops being an option and
`:matvec` is the realistic choice. Multinode `:auto` runs this check itself and
announces a downgrade to `:matrixfree`; single node does not, so look at the
printed fill.

Relative cost of one apply at `dim_q = 7104` (4 threads, R = 4 roots):

| storage | one apply |
|---|---|
| `:sparse` | 0.011 s |
| `:direct` | 0.015 s |
| `:matvec` | 3.4 s |
| `:fois` | 41 s |

`:matvec` and `:fois` are three to four orders of magnitude slower per apply.
That is the cost of not storing the matrix, and it is what makes the other two
choices below matter.

### 2. Which linear solver?

| value | env | notes |
|---|---|---|
| `:pcg` | `TPSCHEM_SOLVER=pcg` | Jacobi-preconditioned CG. Fewest iterations when the shifted operator is positive definite. Falls back to MINRES automatically per solve when it is not. |
| `:minres` | `TPSCHEM_SOLVER=minres` | No definiteness assumption. The safe default for excited roots, where `H_qq - E0` is generally indefinite. |
| `:krylov` | `TPSCHEM_SOLVER=krylov` | Legacy single-node `KrylovKit.linsolve`. Single-RHS, so it cannot share a solve across roots. Prefer `:pcg` or `:minres`. |

`:pcg` needs a stored H on the multinode path (it takes the preconditioner from
`diag(H_qq)`); with `h_storage=:matrixfree` it announces a fall back to `:minres`.

An unrecognised solver name is now an error. It used to fall through silently to
the slowest path, which is how a run once spent 18 minutes per iteration.

### 3. Do the roots share the solve?

`block_roots` / `TPSCHEM_BLOCK_ROOTS`, default `true`.

The solver has always been multiroot — R roots in, R energies out. This decides
whether the roots are solved in **one pass** or one pass each. H_qq is the same
for every root and the shift is diagonal, so one apply can serve all R roots;
only the coupling vector `h[c]` and the shift `eshift[c]` differ, and each root
keeps its own Krylov scalars and convergence test either way.

Measured speedup of one apply for 4 roots against 4 sequential applies:

| storage | single node | multinode |
|---|---|---|
| `:sparse` / `:blocks` | 2.9× | 2.4× |
| `:direct` | 2.0× | — |
| `:matvec` | 3.2× | — |
| `:fois` / `:matrixfree` | 3.9× | 3.3× |

End-to-end that translates into a large win exactly where an apply is expensive,
and near nothing where it is cheap:

| run | one pass per root | all roots one pass |
|---|---|---|
| single node, `:matvec` | 494 s | 166 s |
| single node, `:fois` | 994 s | 258 s |
| multinode, `:matrixfree` | 640 s | 209 s |
| single node, `:sparse` | 12.9 s | 11.9 s |
| multinode, `:blocks` | 19.5 s | 18.9 s |

The stored tiers are a wash because at that size the solve is a second or two out
of a run dominated by the FOIS construction and the one-time H build — there is
simply not much solve to speed up. Leave `block_roots=true`; it never costs more
than the spread between the fastest and slowest root's iteration count.

Not available for `solver=:krylov`, which takes one right-hand side.

---

## Full keyword reference

### Both entry points

| env | keyword | default | what it does |
|---|---|---|---|
| `TPSCHEM_CEPA_SHIFT` | `cepa_shift` | `"cepa"` | `"cepa"` (CEPA-0, one macro-iteration), `"acpf"`, `"aqcc"`, `"cisd"`. The last three iterate the shift, so the solve is repeated while the FOIS and H build stay one-time. |
| `TPSCHEM_CEPA_MIT` | `cepa_mit` | 30 | Cap on shift macro-iterations. Ignored for `"cepa"`. The loop exits when **every** root's energy moves less than `tol`, so one non-converging root pins it at the cap. |
| `TPSCHEM_THRESH_FOI` | `thresh_foi` | 1e-6 | Screening threshold for the first-order interacting space. **The main cost dial**: it sets `dim_q`, which sets everything else. |
| `TPSCHEM_CEPA_TOL` | `tol` | 1e-8 | Convergence target for the amplitude solves and the shift loop. Measured against ‖h‖, so warm starts do not silently tighten it. |
| `TPSCHEM_MINRES_MAXITER` | `cg_maxiter` | 300 | Cap on Krylov iterations per solve. Worth lowering on `:matvec`/`:fois`, where each iteration is seconds. |
| `TPSCHEM_THRESH_SIGMA` | `thresh_sigma` | 1e-8 | Screening for the coupling vectors `h`. With `block_roots=true` the screen is max-over-roots, so the blocked `h` keeps the union of the roots' surviving terms — slightly more accurate, and not bitwise identical to the per-root result. Set `0.0` if you need them to match exactly. |
| `TPSCHEM_NBODY` | `nbody` | 4 | Highest cluster-term body order included. |
| `TPSCHEM_COMPRESS_Q` | `compress` | false | Clip the FOIS after building it. |
| `TPSCHEM_THRESH_CLIP` | `thresh_clip` | 1e-5 | Clipping threshold, only used when `compress` is on. |
| `TPSCHEM_VERBOSE` | `verbose` | 1 | `0` silent except decisions that change cost; `1` banner, dimensions, energies; `2` per-root detail; `3` every Krylov residual. |

### Multinode only

| env | keyword | default | what it does |
|---|---|---|---|
| `TPSCHEM_H_STORAGE` | `h_storage` | `:auto` | `:blocks`, `:matrixfree`, or `:auto` (memory-budget check). `:blocks` still runs the feasibility check and refuses up front rather than OOM-killing the job hours in. |
| `TPSCHEM_MAX_MEM_H` | `max_mem_H` | 50.0 | GB per worker that `:auto` may spend on the stored H. |
| `TPSCHEM_LINSOLVE_TOL` | `linsolve_tol` | `tol` | Inner Krylov tolerance. Worth loosening relative to `tol`: solving to 1e-8 while the ACPF shift is still moving is wasted work. |
| — | `warm_start` | true | Reuse the previous macro-iteration's amplitudes. Leave on — it is what makes ACPF/AQCC affordable (iteration counts fall roughly 36 → 28 → 19 → 14). |
| `TPSCHEM_WORKER_THREADS` | — | — | Julia threads per worker. One worker per node, threads = cores on that node. |
| `TPSCHEM_BLAS_THREADS` | `blas_threads` | 1 | Keep at 1. The solvers thread over Julia threads; BLAS threads on top oversubscribe. |
| `TPSCHEM_MACHINE_FILE` | — | — | Node list, one host per line. |
| `TPSCHEM_SHARD_STRATEGY` | — | `:balanced` | How Fock sectors are assigned to workers. |
| — | `reference_solver` | `:sharded_davidson` | `:direct`, `:distributed_davidson`, or `:sharded_davidson`. Skipped entirely if you pass `e0`. |
| `TPSCHEM_E0` | `e0` | — | Comma-separated reference energies. Passing them skips the reference solve — do this when you already have a converged TPSCI reference. |

### Threads

Pin these, and pin them the same way across runs you intend to compare:

```bash
export JULIA_NUM_THREADS=64          # single node: cores available
export TPSCHEM_WORKER_THREADS=64     # multinode: cores per node
export TPSCHEM_BLAS_THREADS=1
```

`H_qq * v` on a stored sparse matrix is threaded over Julia threads
(`ThreadedSymSpMV`) and is memory-bandwidth-bound — measured 2.2× at 4 threads and
2.6× at 8 on a 212 M-nonzero H_qq, and it should scale further on a many-socket
node. It falls back to `SparseArrays` when `JULIA_NUM_THREADS=1`, where the
single-threaded kernel is faster.

---

## Recipes

**Single node, Q space fits in memory** (`JULIA_NUM_THREADS=64`, BLAS at 1):

```julia
e_cepa, q = TPSChem.do_fois_cepa(ref, cluster_ops, clustered_ham;
                                 thresh_foi   = 8e-5,
                                 solver       = :pcg,
                                 build_hqq    = :sparse,
                                 block_roots  = true)
```

**Single node, Q space too large to store** — check the printed fill first:

```julia
e_cepa, q = TPSChem.do_fois_cepa(ref, cluster_ops, clustered_ham;
                                 thresh_foi   = 8e-5,
                                 solver       = :minres,   # no stored H, no PCG diagonal
                                 build_hqq    = :matvec,   # nothing stored, entries recomputed
                                 cg_maxiter   = 60,        # each iteration is seconds; cap it
                                 block_roots  = true)
```

Keep `block_roots=true` here above all: this is the regime where it is worth ~3×.

If your driver reads these from the environment, wire them yourself — e.g.
`solver = Symbol(get(ENV, "TPSCHEM_SOLVER", "pcg"))`,
`build_hqq = Symbol(get(ENV, "TPSCHEM_BUILD_HQQ", "auto"))`. The multinode
launchers in `examples/multinode/` already do this, including
`TPSCHEM_BLOCK_ROOTS`.

**Multinode, stored H:**

```bash
export TPSCHEM_MACHINE_FILE=nodes.txt
export TPSCHEM_WORKER_THREADS=64 TPSCHEM_BLAS_THREADS=1
export TPSCHEM_H_STORAGE=blocks
export TPSCHEM_MAX_MEM_H=200
export TPSCHEM_SOLVER=pcg
export TPSCHEM_LINSOLVE_TOL=1e-6
sbatch examples/multinode/run_cepa_sharded_minres_4nodes.slurm
```

**ACPF/AQCC:** set `TPSCHEM_CEPA_SHIFT=acpf` and a `TPSCHEM_CEPA_MIT`. Watch the
per-macro-iteration energies: if one root's shift oscillates instead of settling,
the loop will run to the cap and that root's energy is not converged, whatever the
others do.

---

## Reading the banner

Every solve prints the pathway it took. Check it before trusting a long run:

```
CEPA pathway: dim_q=181214  shift=cepa  solver=pcg  storage=sparse  roots=block(6)
H_qq nnz = 11500000000  (35.021% fill, 171.36 GiB CSC)
```

- `storage=fois` when you meant `sparse` means `build_hqq` was ignored — check the solver name.
- `roots=one-at-a-time(R)` with `R > 1` means blocking is off, either because you set it off or because `solver=:krylov`.
- A `nnz` line whose GiB does not fit the node is your cue to switch to `:matvec` before the build OOMs.
