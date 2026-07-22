# API Overview

`TPSChem` consolidates the former FermiCG ecosystem into a single package. Each
former standalone package lives on as a submodule of the same name:

| Submodule | Purpose |
|-----------|---------|
| [`QCBase`](QCBase.md) | Atoms, molecules, MO clusters, generic interface |
| [`InCoreIntegrals`](InCoreIntegrals.md) | In-core 1e/2e integrals and transformations |
| [`RDM`](RDM.md) | Reduced density matrices, orbital gradients/Hessians |
| [`BlockDavidson`](BlockDavidson.md) | Matrix-free block-Davidson eigensolver |
| [`ActiveSpaceSolvers`](ActiveSpaceSolvers.md) | FCI / active-space solver interface |
| [`ClusterMeanField`](CMFs.md) | Cluster mean-field (CMF) orbital optimization |

The **core** of the package (the former FermiCG) implements the tensor-product
state methods on top of these submodules:
[TPSCI](TPSCI.md), [SPT](SPT.md), their [state representations](States.md), and
[clustered operators](ClusteredTerms.md).

The module-level docstring:

```@docs
TPSChem
```
