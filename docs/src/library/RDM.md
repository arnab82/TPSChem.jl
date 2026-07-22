```@meta
CurrentModule = TPSChem
```

# RDM

The **RDM** submodule provides the reduced-density-matrix *types*
([`RDM1`](@ref), [`RDM2`](@ref) and their spin-summed variants `ssRDM1` /
`ssRDM2`) together with the orbital gradient, orbital Hessian, and generalized
Fock machinery that drive the [cluster mean-field](CMFs.md) orbital
optimization.

!!! note
    These are the *building-block* density matrices and the gradient/Hessian
    used by CMF orbital optimization. RDMs **measured from a converged TPSCI
    wavefunction** are computed separately — see
    [RDMs & Properties](Properties.md).

```@index
Pages = ["RDM.md"]
```

## Documentation
```@autodocs
Modules = [TPSChem.RDM]
Order   = [:type, :function]
```
