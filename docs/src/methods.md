```@meta
CurrentModule = TPSChem
```

# Methods & Workflows

TPSChem implements a family of tensor-product-state methods that build on one
another. This page is a map: for each method it names the **entry-point
functions** and links to the detailed API. All correlated methods start from a
[CMF](#cmf) reference and its cluster basis / cluster operators.

The typical pipeline is:

```text
integrals ──► CMF ──► cluster basis + cluster operators ──►  TPSCI ─┐
                                                              SPT ───┼─► +PT2 / +CEPA ─► RDMs / properties
```

## The common preamble

Every correlated method needs three ingredients built once from a CMF solution:

```julia
# after a CMF solve gives `ints`, `clusters`, `init_fspace`, and 1-RDM `d1`
cluster_bases = compute_cluster_eigenbasis(ints, clusters; init_fspace=init_fspace,
                                           rdm1a=d1.a, rdm1b=d1.b, max_roots=M)
cluster_ops   = compute_cluster_ops(cluster_bases, ints)
add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b)
clustered_ham = extract_ClusteredTerms(ints, clusters)
```

`cluster_bases` are the local many-body states, `cluster_ops` their operator
matrix elements, and `clustered_ham` the Hamiltonian split into
[clustered terms](library/ClusteredTerms.md).

## CMF

**Cluster Mean Field** — jointly optimize orbitals and local cluster ground
states to get the variationally best single tensor product state. This is the
reference for everything below.

- [`cmf_ci`](@ref ClusterMeanField.cmf_ci) — self-consistent CI at fixed orbitals.
- [`cmf_oo`](@ref ClusterMeanField.cmf_oo) — CI + orbital optimization (gradient / CG).
- [`cmf_oo_newton`](@ref ClusterMeanField.cmf_oo_newton), [`cmf_oo_diis`](@ref ClusterMeanField.cmf_oo_diis) — Newton and DIIS optimizers.

See the [CMF example](cmf.md) and the [ClusterMeanField API](library/CMFs.md).

## CMF-PT2 / QDPT

Second-order perturbative correction on top of the CMF (or any small TPSCI)
reference, including the quasi-degenerate multi-state variant.

- [`compute_pt2_energy`](@ref) — Epstein–Nesbet PT2 on a reference `TPSCIstate`.
- [`compute_qdpt_energy`](@ref) — quasi-degenerate PT2 for several roots.

## TPSCI

**Tensor Product Selected CI** — iteratively grow a *sparse* set of important
tensor-product configurations toward FCI.

- [`tpsci_ci`](@ref) — the full selected-CI loop (diagonalize → PT1 → select → grow).
- [`tps_ci_direct`](@ref), [`tps_ci_davidson`](@ref) — variational diagonalization in a fixed model space.
- [`do_fois_ci`](@ref) — expand into and variationally treat the first-order interacting space.

See the [TPSCI API](library/TPSCI.md).

## TPSCI + PT2

- [`compute_pt1_wavefunction`](@ref) — first-order (PT1) wavefunction in the FOIS.
- [`compute_pt2_energy`](@ref) — PT2 energy correction to a TPSCI reference.

## SPT

**Subspace Product Tucker** — reintroduce discarded tensor products as
HOSVD-compressed blocks and solve variationally.

- [`subspace_product_tucker`](@ref) — the SPT driver.
- [`do_fois_ci`](@ref) — FOIS build/solve for `SPTstate`.

See the [SPT API](library/SPT.md).

## SPT + PT2

- [`do_fois_pt2`](@ref) — build the FOIS and take the PT2 correction.
- [`compute_pt2_energy`](@ref), [`compute_pt2_energy_blockwise`](@ref) — PT2 energy for an `SPTstate`.
- [`compute_pt1_wavefunction`](@ref) — the SPT PT1 wavefunction.
- [`compute_spt_sigma_norm_blockwise`](@ref) — blockwise σ-norm / variance estimate.

## TPS-CEPA

Coupled electron-pair-approximation-style corrections that resum higher-order
terms beyond PT2 on top of a TPSCI/SPT reference.

- [`do_fois_cepa`](@ref), [`tpsci_cepa_solve`](@ref) — CEPA over the TPSCI FOIS.
- [`tucker_cepa_solve`](@ref) — CEPA in the SPT (Tucker) representation.
- On the **multinode** version of these docs, the distributed/sharded solvers
  (e.g. `do_tps_sharded_cepa`) run CEPA over a sharded Q-space across nodes —
  see the *Multinode & Sharded* page in the API reference.

## RDMs

Reduced density matrices from a converged TPSCI wavefunction (see the
[RDMs & Properties API](library/Properties.md)).

- [`compute_1rdm`](@ref), [`compute_1rdm_sf`](@ref) — spin-resolved / spin-free 1-RDM.
- [`compute_2rdm`](@ref), [`compute_2rdm_blas`](@ref) — 2-RDM.
- `*_threaded` and `*_distributed` variants for large problems.

The building-block RDM types ([`RDM1`](@ref), [`RDM2`](@ref)) and orbital
gradient/Hessian machinery live in the [RDM submodule](library/RDM.md).

## 1-RDM properties

One-electron properties, transition properties, and spectra.

- [`contract_1rdm_property`](@ref), [`compute_1e_property_direct`](@ref) — expectation of a 1e operator.
- [`compute_transition_dipoles`](@ref), [`compute_oscillator_strengths`](@ref) — transition properties.
- [`absorption_spectrum`](@ref), [`print_stick_spectrum`](@ref) — spectra.
- [`correlation_functions`](@ref) — spin/charge correlation functions.
