# Clustered Operators & Terms

The molecular Hamiltonian is decomposed into *clustered terms* — pieces labelled
by which clusters they act on and by the creation/annihilation operator strings
involved. `ClusteredOperator` collects these terms; each term carries the local
operator strings and integral tensors needed to build σ = H·v.

```@index
Pages = ["ClusteredTerms.md"]
```

## Documentation
```@autodocs
Modules = [TPSChem]
Pages   = ["type_ClusteredTerm.jl",
           "type_ClusteredOperator.jl"]
Order   = [:type, :function]
```
