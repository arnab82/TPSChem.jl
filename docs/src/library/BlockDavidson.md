```@meta
CurrentModule = TPSChem
```

# BlockDavidson

A lightweight block-Davidson eigensolver ([`Davidson`](@ref), [`eigs`](@ref))
and the matrix-free linear-operator wrapper [`LinOpMat`](@ref) it drives. This
is the workhorse used to diagonalize the (often matrix-free) Hamiltonians that
appear throughout the CI solvers.

```@index
Pages = ["BlockDavidson.md"]
```

## Documentation
```@autodocs
Modules = [TPSChem.BlockDavidson]
Order   = [:type, :function]
```
