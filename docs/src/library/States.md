# States & Configurations

Each method in the package uses a dedicated data representation for the
wavefunction. These all rely on hash-table lookups keyed by compact,
stack-allocated *configuration* indices.

**Configuration (index) types**

- [`FockConfig`](@ref) — how the electrons are distributed over the clusters
  (per-cluster `(nα, nβ)`).
- [`ClusterConfig`](@ref) — which local many-body state each cluster is in.
- [`TuckerConfig`](@ref) — a range of local states per cluster (a block).
- `TransferConfig`, `OperatorConfig`, `SparseIndex` — bookkeeping for operator
  application between Fock sectors.

**Wavefunction types** (each maps configuration indices → coefficients)

- TPSCI: [`TPSCIstate`](@ref) maps `FockConfig` → `ClusterConfig` → a vector of
  coefficients (one per root).
- SPT: [`SPTstate`](@ref) maps `FockConfig` → `TuckerConfig` → a compressed
  `Tucker` factor for that block.
- `BSstate` — a block-sparse Tucker-form state.

The per-cluster basis and operator data live in [`ClusterBasis`](@ref),
`ClusterSubspace`, and `ClusterOps`.

## Index
```@index
Pages = ["States.md"]
```

## Types
```@autodocs
Modules = [TPSChem]
Pages   = ["type_AbstractState.jl",
           "type_BSstate.jl",
           "type_TPSCIstate.jl",
           "type_SPTstate.jl",
           "type_FockConfig.jl",
           "type_ClusterConfig.jl",
           "type_TuckerConfig.jl",
           "type_TransferConfig.jl",
           "type_OperatorConfig.jl",
           "type_SparseIndex.jl",
           "type_ClusterBasis.jl",
           "type_ClusterSubspace.jl",
           "type_ClusterOps.jl"]
Order   = [:type]
```

## Methods
```@autodocs
Modules = [TPSChem]
Pages   = ["type_AbstractState.jl",
           "type_BSstate.jl",
           "type_TPSCIstate.jl",
           "type_SPTstate.jl",
           "type_FockConfig.jl",
           "type_ClusterConfig.jl",
           "type_TuckerConfig.jl",
           "type_TransferConfig.jl",
           "type_OperatorConfig.jl",
           "type_SparseIndex.jl",
           "type_ClusterBasis.jl",
           "type_ClusterSubspace.jl",
           "type_ClusterOps.jl",
           "Indexing.jl",
           "build_local_quantities.jl"]
Order   = [:function]
```
