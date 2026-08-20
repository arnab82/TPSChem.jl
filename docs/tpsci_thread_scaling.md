# TPSCI thread scaling is limited by allocator contention

Measured on an Apple M3 Pro (5 performance + 6 efficiency cores, Julia 1.11.7),
H12 / 6 clusters / R=4, `thresh_cipsi=1e-4`, FOIS = 661,467 configs.  All numbers
are `open_matvec_thread`, the FOIS build, which is the threaded hot spot.

## The observation

| threads | `open_matvec_thread` |
|--------:|---------------------:|
| 1       | 19.0 s |
| 4       | 13.3 s |
| 8       | 23.9 s |

Eight threads are slower than one.  Peak is around four.

## What it is not

**Not load imbalance.**  Instrumenting the loop per slot at 8 threads: every slot
finishes at the same wall time (23.860 s, identical to 3 decimals) having consumed
within 8% of the same estimated cost.  Balance is already near-perfect.

**Not heterogeneous cores.**  If three threads sat on efficiency cores, dynamic
job pulling would hand them proportionally less work; instead all eight consume
equal cost at equal speed.  A pure-compute microbenchmark scales 5.76x on 8
threads on this same machine.

**Not GC.**  Reported GC time is 1.3-1.6%.  `--gcthreads=8` (default is
nthreads/2) measured 24.2 s and `--gcthreads=8 --heap-size-hint=24G` 25.2 s,
against a 23.9 s baseline.  Neither helps.

**Not memory bandwidth.**  See the two-process result below: two processes share
bandwidth just as eight threads do, and do not slow down.

## What it is

Running the *same* eight hardware threads as two 4-thread processes instead of
one 8-thread process, on the identical workload:

| configuration | aggregate throughput |
|---|---:|
| 4 threads, one process  |   732k cost/s |
| 8 threads, one process  |   408k cost/s |
| 8 threads, two processes| 1,052k cost/s |

Two processes get 2.6x more out of the same cores than one process does.  The
serializing resource is therefore per-process, and it is not the GC.  A minimal
microbenchmark isolates it (no TPSCI code involved):

| threads | pure compute | allocation-heavy |
|--------:|-------------:|-----------------:|
| 1 | 1.00x | 1.00x |
| 2 | 1.96x | 1.68x |
| 4 | 3.79x | 1.31x |
| 8 | 5.76x | 1.39x |

Julia's allocator does not scale past ~2 threads for small short-lived objects.
`open_matvec_thread` allocates ~8 GB (249 M objects) per call, so it inherits
that ceiling.

## Consequences

**For cluster runs.**  One process with 96 threads is the wrong shape for this
workload.  Prefer several processes per node with fewer threads each -- the
existing multinode machinery already runs separate processes, so this costs
nothing to try.  The crossover was ~4 threads here and should be measured on the
target node rather than assumed; the two-process result is the reason to expect
a win at all.

**For optimization.**  Scheduling is not the lever -- balance is already perfect
and dynamic pulling measured at parity with cost-bucketed `:static` (23.9 s
both).  Cutting allocation is the lever.  Allocation profile of one call
(`Profile.Allocs`, sample_rate 5e-4):

| share | type |
|------:|------|
| 40.2% | `ReshapedArray{Float64,2,Vector{Float64}}` -- from `reshape2`, 46k sampled objects |
| 11.7% | `Pair{ClusterConfig{5}, MVector{4,Float64}}` |
| 10.0% | `ClusterConfig{5}` |
|  7.6% | `Adjoint{Float64, ReshapedArray{...}}` |
|  4.3% | `@NamedTuple{thresh::Float64, prescreen::Bool}` -- kwargs |

By first TPSChem frame: `_open_matvec_thread_job:297` (the
`for (config_ket, coeff_ket) in configs_ket` loop) 20.0%, then the `reshape2`
sites inside `contract_matvec_thread` (:517 18.2%, :622 9.3%, :599 6.9%,
:613 6.2%).

Two specific leads, neither attempted here:

- `ClusterConfig{5}` is `isbitstype`, so the 18k heap allocations of it are
  boxing -- there is type instability on that path worth tracking down.
- `MVector{4,Float64}` is mutable, so every config's coefficient vector is a
  separate heap object (661k of them for this FOIS).  An immutable `SVector`
  would make the dict values isbits and remove that whole class, but the code
  mutates coefficients in place (`.+=`), so it is not a local change.
