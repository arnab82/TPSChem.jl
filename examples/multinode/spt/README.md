# SPT sharding benchmarks

Two drivers for measuring the never-gather (sharded) SPT path on a cluster.
**Run the diagnostics first** — it is metadata-only and takes minutes, and it
tells you whether the 2D grid matvec can help on your system before you spend
node-hours on the scaling run.

## 1. Diagnostics (one node, no workers)

```bash
export TPSCHEM_INPUT_JLD2=/path/to/problem.jld2
export TPSCHEM_P_LIST=4,9,16,25,36        # worker counts to model
sbatch examples/multinode/spt/run_spt_diagnostics.slurm
```

Reports:

- **Fock-coupling density and bra in-degree.** The 2D grid only pays when a
  worker's slice of sectors does not already touch essentially every other
  sector. On h12 this is 6.8% dense with mean in-degree 142 of 2090 — dense
  enough that re-partitioning by *contribution* moves exactly the same bytes
  (it is the dual of the same cut), while the grid does help.
- **Predicted traffic per matvec** for destination / 2D grid / ring at each `P`,
  and which wins.
- **FOIS load spread** under the current assignment (count of contributing ket
  sectors) against a work-weighted one. On h12 these are within 1.03–1.12x, so
  re-weighting buys nothing; check whether that holds for you.

## 2. Scaling benchmark (multi-node)

```bash
export TPSCHEM_INPUT_JLD2=/path/to/problem.jld2
export TPSCHEM_NROOTS=4
export TPSCHEM_FOIS_THRESH=1e-4
export TPSCHEM_SKIP_SINGLE_NODE=0         # 1 if the state will not fit the master
sbatch examples/multinode/spt/run_spt_scaling.slurm
```

Times, and checks against the single-node result:

| what | partitioning | traffic |
| --- | --- | --- |
| `build_sigma_sharded` | destination Fock sector | `O(P |state|)` |
| `build_sigma_sharded_2d` | 2D worker grid | `O(sqrt(P) |state|)` |
| `build_compressed_1st_order_state_sharded` | destination | fetches the reference only |
| `compute_pt1_wavefunction_sharded` | `:dest` and `:grid` | see note below |

For a scaling curve submit with `--nodes=1,4,9,16`. **The grid needs a non-prime
worker count**; with a prime count it warns and falls back to the destination
partition, so 4/9/16/25 are the useful points.

## What to expect, and what not to

Measured on h12 (dim 56317, 2090 Fock sectors), processes on one machine:

| P | single-node | destination | 2D grid | 2D vs destination |
| --- | --- | --- | --- | --- |
| 4 | 14.0 s | 13.1 s | 9.4 s | 1.38x |
| 9 | 13.2 s | 6.7 s | 3.9 s | 1.71x |

- **The grid helps the CI matvec, where bra and ket are the same large state.**
  It trades ket traffic for bra traffic, so it only pays when both sides are
  large.
- **It does not help PT1** (`partition=:grid`), where the FOIS bra dwarfs the
  reference ket: 0.77 vs 0.77 s with a dim-121 reference, 2.70 vs 2.82 s with
  dim 2261. The option exists for the case where your reference approaches the
  FOIS in size. Default is `:dest`.
- **It does not apply to the FOIS builder at all** — that creates its output
  fresh and fetches only the (smaller) reference, so there is no bra traffic to
  trade.

## What these runs are actually asking

Every number below was taken with processes on one 11-core machine, so each is a
hypothesis, not a result. The point of running on the cluster is to confirm or
kill them. Each question says what to look at and what each outcome means.

**Q1. Does the 2D grid's advantage grow with P, as sqrt(P) predicts?**
Look at `2D vs destination` across `--nodes=4,9,16`. Loopback gave 1.38x at P=4
and 1.71x at P=9. If the margin keeps widening, the volume model holds. If it
flattens, the win was fetch balancing rather than reduced volume and the grid
stops paying beyond some P. *Real interconnect should make the grid look
better, not worse* — loopback has far more bandwidth and far less latency than
any network, so communication is under-weighted here.

**Q2. Does per-worker memory actually scale down?**
This is the claim the grid exists for. Look at `peak/worker` for the two
matvecs as P grows. Prediction: destination stays roughly flat (~O(|state|)
however many workers you add), grid falls as 1/sqrt(P). If destination's
peak/worker does *not* stay flat, the O(P|state|) diagnosis is wrong and the
whole motivation needs revisiting.

**Q3. Where does the missing ~50% go?**
Sharded matvec runs at ~49% parallel efficiency and I have not attributed the
loss — it could be serialization, Distributed's own overhead, load imbalance,
or genuine parallel loss. Profile on the real machine before optimizing; four
plausible fixes were already killed by measurement (contribution partitioning,
PT1 grid, FOIS grid, load re-weighting) and a fifth guess is not worth building.

**Q4. Is your coupling graph sparse enough for the grid at all?**
From the diagnostics run. h12 is 6.8% dense with mean bra in-degree 142 of 2090.
If yours is much denser, every worker touches everything under any partitioning
and the grid cannot help. If it is much sparser, the grid should beat the
loopback numbers.

**Q5. Does the `length(kets)` load proxy still hold?**
From the diagnostics run: `current` vs `work-weighted`. On h12 these are within
1.03-1.12x, so re-weighting buys nothing. A larger gap on your system makes a
work-weighted assignment worth implementing — the weighting machinery already
exists in `_spt_pt2_job_weight`.

**Q6. Does the single-node speedup carry over?**
The node-local work got ~2.4x faster this pass (R-fused contraction, the
transform_basis rewrite, dropping the global contribution buffer). Those are
node-local, so each worker should inherit them; `single-node build_sigma!` in
the scaling output is the check. If it does not, the worker processes are not
picking up the same code path.

**Q7. At what P does the destination partition actually fall over?**
Run it until it does. The prediction is that per-worker transient memory stays
~O(|state|), so it fails when |state| approaches one node's RAM regardless of
node count — that is the whole reason the grid was written. Knowing the real
crossover for your system sizes tells you which partitioning to use by default.

## Numbers to treat with suspicion

Everything above was measured with processes on a single 11-core machine, so
"P=9" was 9 processes on 11 cores. The `sqrt(P)` *scaling* should carry to real
hardware; the constants will not, and serialization and latency costs change
completely over a real interconnect. That is the reason these drivers exist:
re-measure on the machine you actually run on.

Known remaining gaps, in `../SHARDED_SPT_PLAN.md` §10:

- ~50% of the sharded time is still unattributed (serialization vs Distributed
  overhead vs genuine parallel loss) — needs a profile on real hardware.
- A ring/systolic rotation would give `O(|state|/P)` peak memory per worker
  rather than the grid's `O(|state|/sqrt(P))`; worth building only if memory,
  not bandwidth, is what stops you.
