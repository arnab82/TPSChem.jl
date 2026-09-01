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
| `:packed` | — | lower triangle, `dim_q(dim_q+1)/2 × 8 B` | **the default choice when a matrix fits** |
| `:sparse` | `:blocks` | CSC, `nnz × 16 B` | only below 25% fill |
| `:direct` / `:parallel` | — | dense, `dim_q² × 8 B` | superseded by `:packed`, which holds the same numbers in half the space |
| `:matvec` | — | nothing; entries recomputed per apply | no matrix fits |
| `:fois` | `:matrixfree` | nothing; H applied through `open_matvec_thread` | last resort |
| `:auto` (default) | `:auto` (default) | — | picks for you (multinode checks the memory budget) |

Measured, cr2_13 at `dim_q = 7104`, 4 threads, R = 4 roots:

| storage | build | stored | 1-root apply | block apply |
|---|---|---|---|---|
| `:sparse` (34% fill) | 4.21 s | 0.256 GiB | 0.0106 s | 0.0144 s |
| **`:packed`** | **3.57 s** | **0.188 GiB** | **0.0077 s** | **0.0105 s** |
| `:direct` | 3.73 s | 0.376 GiB | 0.0149 s | 0.0304 s |
| `:matvec` | 0.03 s | 0.001 GiB | 3.54 s | 4.41 s |
| `:fois` | 0.03 s | — | 40.7 s | 41.4 s |

`:packed` is the smallest *and* the fastest of the stored containers here, and the
quickest to build. Against `:direct` that is exactly the halving you would expect —
same numbers, half the bytes, and these applies are memory-bandwidth-bound, so
half the bytes is half the time. Against `:sparse` it wins because CSC costs
`16 × fill` bytes per entry-equivalent against packed's 4, so `:sparse` only
overtakes it below 25% fill.

**Size the storage to the node, then check the fill.** Once H_qq is stored the
solve costs almost nothing on top of it — a blocked MINRES holds seven `R × dim_q`
blocks, which is 61 MB at `dim_q = 181000` with 6 roots, against a matrix measured
in hundreds of GB. So the sane budget is most of the node for the matrix (say 80%)
and a rounding error for the solve. That is the whole argument for storing it: the
matrix-free path it replaces allocated 8.7 GiB *per matvec*.

Given a budget *B*, the question is which container fits — and **do not assume
`:sparse` is the sparse-friendly choice.** H_qq is often not very sparse: 34–36%
measured on the Cr2 systems here, and fill rises as the cluster count falls and
the local bases grow, because more configuration pairs stay Fock-connected. Fill
is system-specific and worth measuring rather than assuming.

That matters because **CSC only beats dense below 50% fill**. CSC stores 16 bytes
per entry (a `Float64` plus an `Int64` row index); dense stores 8 bytes per entry
and no indices. So:

| fill | `:sparse` CSC | `:direct` dense | which is smaller |
|---|---|---|---|
| 20% | `dim_q² × 3.2 B` | `dim_q² × 8 B` | sparse |
| 50% | `dim_q² × 8 B` | `dim_q² × 8 B` | break even |
| 75% | `dim_q² × 12 B` | `dim_q² × 8 B` | **dense** |

At `dim_q = 181000` that is 394 GB for `:sparse` against 263 GB for `:direct` —
`:sparse` is the worse of the two, and above 50% fill it is also slower to apply
(indirect indexing beats nothing about a contiguous column).

Worse, the `:sparse` **build** peaks well above its final size: it accumulates
per-thread `(I, J, V)` COO triplets at ~24 bytes per entry before `sparse()`
compresses them. At 75% fill and `dim_q = 181000` that is ~590 GB transient, on
top of the CSC. A `:sparse` job can therefore die in the builder even when the
final matrix would have fitted.

The solver prints what it found, and now says so when the choice was wrong:

```
 dense would be:              244.71 GiB  (CSC breaks even with dense at 50% fill)
 H_qq nnz = 24600000000  (75.000% fill, 367.28 GiB CSC)
 note: 75% fill — build_hqq=:direct would store this in 244.71 GiB
       instead of 367.28 GiB, and apply faster. :sparse only pays below 50%.
```

**Single-node storage rule**, given a measured fill *f* and a memory budget *B*
(80% of the node is a reasonable *B* — the solve needs well under a GB on top):

- *f* < 50% and `dim_q² × f × 16 B` < *B* → `:sparse`
- *f* ≥ 50% and `dim_q² × 8 B` < *B* → `:direct`
- otherwise → `:matvec`, and consider going multinode instead

Largest `dim_q` that fits a 176 GB budget (80% of a 220 GB node):

| container | bytes/entry | max dim_q |
|---|---|---|
| `:sparse` at 20% fill | 3.2 | 234000 |
| **`:packed`** | **4.0** | **209000** |
| `:sparse` at 35% fill | 5.6 | 177000 |
| `:sparse` at 50% fill | 8.0 | 148000 |
| `:direct` (full dense) | 8.0 | 148000 |

`:packed` is fill-independent, so it is the predictable choice when the fill is
unknown — and at `dim_q = 181214` it needs 131 GB where `:sparse` at 35% fill
needs 179 GB and `:direct` needs 263 GB. `:sparse` only beats it below 25% fill.

Multinode `:blocks` sidesteps the whole question: it stores **dense per-Fock-block**
(8 B per entry, no index overhead) and shards it across nodes, so its footprint is
`nnz × 8 B` divided by the node count. `:auto` runs the feasibility check itself
and announces a downgrade to `:matrixfree`; single node does not, so read the
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
