```@meta
CurrentModule = TPSChem
```

# SPT
The Subspace Product Tucker (SPT) method (formerly called Block-Sparse Tucker, BST) approximates FCI as a linear combination of individually compressed (via HOSVD)
blocks of the Hilbert space.

## Background
Similar to TPSCI, the approach here starts with a CMF wavefunction, and systematically reintroduces the discarded tensor product
states to variationally approach FCI. Unlike with TPSCI, however, we don't assume that the final wavefunction is written as 
a purely sparse form (where only a few TPS's are needed), but rather we assume that collections of TPS's where certain numbers of clusters are "excited" can be efficiently compressed via HOSVD (although the basic idea would extend to other tensor decompositions, like CP or MPS). 

## Performance considerations 

## Index
```@index
Pages   = ["SPT.md"]
```

SPT is a **variational method**: [`subspace_product_tucker`](@ref) grows and
variationally solves over HOSVD-compressed blocks; [`do_fois_ci`](@ref) treats
the first-order interacting space, and [`tucker_cepa_solve`](@ref) applies a
CEPA correction in the Tucker representation.

## Variational SPT
```@autodocs
Modules = [TPSChem]
Pages   = ["spt.jl",
           "spt_helpers.jl",
           "tucker_inner.jl",
           "tucker_outer.jl",
           "tucker_build_dense_H_term.jl",
           "tucker_contract_dense_H_with_state.jl",
           "tucker_form_sigma_block_expand.jl"]
Order   = [:type, :function]
```

## Perturbation theory & variance

PT2 corrections ([`do_fois_pt2`](@ref), [`compute_pt2_energy`](@ref),
[`compute_pt2_energy_blockwise`](@ref)) and the blockwise σ-norm variance
estimate ([`compute_spt_sigma_norm_blockwise`](@ref)).

```@autodocs
Modules = [TPSChem]
Pages   = ["tucker_pt.jl",
           "spt_variance.jl"]
Order   = [:type, :function]
```

## HOSVD
```@autodocs
Modules = [TPSChem]
Pages   = ["hosvd.jl"]
Order   = [:type, :function]
```
