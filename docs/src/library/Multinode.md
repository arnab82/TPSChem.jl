```@meta
CurrentModule = TPSChem
```

# Multinode & Sharded

!!! note "multinode branch"
    This page documents functionality that lives on the `multinode` branch.
    It builds a distributed / sharded layer on top of TPSCI, TPS-CEPA, and SPT
    so that CI vectors larger than the RAM of any single node can be solved by
    spreading ("sharding") them across many nodes.

For a step-by-step walkthrough of *where* each stage runs (master vs. workers),
*what* crosses the network, and *how* the pieces recombine, see the
[Multinode / distributed](../generated/multinode_tutorial.md) tutorial.

## The model in one paragraph

One Julia worker process runs per node, with many threads inside each worker
(no MPI — this uses Julia's built-in `Distributed`). The CI vector is split by
Fock sector into **shards**; each shard lives on exactly one worker and *stays
there*. The master holds only small metadata (who owns what, sizes, energies)
and never materializes the full vector. Two distributed containers carry the
state: [`DistributedTPSCIstate`](@ref) (the sharded wavefunction) and
[`DistributedClusterOps`](@ref) (the per-cluster operator tables).

## Worked example — sharded CEPA

A complete driver lives at `examples/bimetallics/benchmark_pt2.jl`; the core of
it is:

```julia
using Distributed, TPSChem, JLD2

workers_ = init_multinode_workers!()          # 1 worker/node, N threads/worker

data     = JLD2.load(DATA_FILE)
ints, clusters, d1 = data["ints"], data["clusters"], data["d1"]

# local cluster basis + operators for the small reference solve
cluster_bases = TPSChem.compute_cluster_eigenbasis_spin(
    ints, clusters, d1, CLUSTER_SPIN_ROOTS, INIT_FSPACE; max_roots=M)
clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)

cluster_ops_ref = TPSChem.compute_cluster_ops(cluster_bases, ints)
TPSChem.add_cmf_operators!(cluster_ops_ref, cluster_bases, ints, d1.a, d1.b)

ci_vector = TPSChem.TPSCIstate(clusters, INIT_FSPACE, R=NROOTS)
ci_vector = TPSChem.add_spin_focksectors(ci_vector)
eci, ref_vec, _ = TPSChem.tps_ci_direct(ci_vector, cluster_ops_ref, clustered_ham)

# distributed operators for the sharded CEPA
cluster_ops = TPSChem.compute_cluster_ops_distributed(cluster_bases, ints; workers=workers_)
TPSChem.add_cmf_operators_distributed!(cluster_ops, cluster_bases, ints, d1.a, d1.b)

e_cepa, qspace = TPSChem.do_tps_sharded_cepa(
    ref_vec, cluster_ops, clustered_ham;
    e0=eci, thresh_foi=1e-5, solver=:minres, h_storage=:auto, workers=workers_)

TPSChem.destroy!(qspace)
TPSChem.destroy!(cluster_ops)
```

`init_multinode_workers!` is a small driver helper (see the example) that adds
one worker per host, activates the project on each, and calls
[`ensure_tpsci_multinode_workers!`](@ref).

## Index
```@index
Pages = ["Multinode.md"]
```

## Distributed TPSCI
```@autodocs
Modules = [TPSChem]
Pages   = ["tpsci_multinode.jl",
           "tpsci_property_multinode.jl"]
Order   = [:type, :function]
```

## Sharded solvers (never-gather)
```@autodocs
Modules = [TPSChem]
Pages   = ["tpsci_sharded_davidson.jl",
           "tps_cepa_sharded.jl",
           "spt_sharded.jl"]
Order   = [:type, :function]
```

## Distributed SPT
```@autodocs
Modules = [TPSChem]
Pages   = ["spt_multinode.jl",
           "spt_variance_multinode.jl"]
Order   = [:type, :function]
```
