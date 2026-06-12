<div align="left">
  <img src="docs/src/logo1.png" height="60px"/>
</div>

# TPSChem.jl

[![Build Status](https://github.com/nmayhall/TPSChem.jl/workflows/CI/badge.svg)](https://github.com/nmayhall/TPSChem.jl/actions)
[![Coverage](https://codecov.io/gh/nmayhall/TPSChem.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/nmayhall/TPSChem.jl)

A Julia package for coarse-grained electronic structure calculations in a tensor product state (TPS) basis.

## Details

`TPSChem` computes high-accuracy electronic states for molecular systems in a tensor product state basis. Unlike in the traditional Slater determinant basis, a TPS basis can be chosen such that each basis vector has a considerable amount of electron correlation already included. As a result, the exact wavefunction in this basis can be considerably more compact. This increased compactness comes at the cost of a significant increase in complexity for determining matrix elements. Implemented methods include:

1. `CMF` - Cluster Mean Field, the reference product state the other methods build on.
1. `CMF-PT2` - Second order PT2 correction on top of `CMF` using a barycentric Moller-Plesset-type partitioning.
1. `CMF-CEPA` - A CEPA-type formalism on top of CMF. First published [here](https://arxiv.org/abs/2206.02333).
1. `TPSCI` - a generalization of the CIPSI method to a TPS basis. Essentially, one starts with a small number of TPS functions, solves the Schrodinger equation in this small subspace, then uses perturbation theory to determine which TPS's to add to improve the energy. This is done iteratively until the results stop changing. First published [here](https://pubs.acs.org/doi/10.1021/acs.jctc.0c00141).
1. `SPT` - Subspace Product Tucker compression of the TPS wavefunction.

## Package structure

`TPSChem` consolidates what was previously the multi-repo FermiCG ecosystem into a
single package. Each former package lives on as a submodule with its own README:

| Submodule | Provides | Formerly |
|---|---|---|
| [`TPSChem.QCBase`](src/QCBase/README.md) | `Molecule`, `Atom`, `MOCluster`, generic interfaces | QCBase.jl |
| [`TPSChem.InCoreIntegrals`](src/InCoreIntegrals/README.md) | `InCoreInts` 1-/2-electron integral container | InCoreIntegrals.jl |
| [`TPSChem.BlockDavidson`](src/BlockDavidson/README.md) | block-Davidson eigensolver (`Davidson`, `eigs`) | BlockDavidson.jl |
| [`TPSChem.RDM`](src/RDM/README.md) | `RDM1`/`RDM2` types, energies, orbital gradients | RDM.jl |
| [`TPSChem.ActiveSpaceSolvers`](src/ActiveSpaceSolvers/README.md) | FCI: `FCIAnsatz`, `solve`, `Solution`, RDMs, cluster operators | ActiveSpaceSolvers.jl |
| [`TPSChem.ClusterMeanField`](src/ClusterMeanField/README.md) | CMF reference states: `cmf_ci`, `cmf_oo` (+ PySCF helpers via extension) | ClusterMeanField.jl |

The TPSCI/SPT/CEPA methods themselves live in the top-level `TPSChem` module
([`src/core/`](src/core)), which re-exports the commonly used names from the
submodules — `using TPSChem` is all most scripts need.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/nmayhall/TPSChem.jl")
```

Or for development:

```
git clone https://github.com/nmayhall/TPSChem.jl
cd TPSChem.jl/
julia --project=./ -tauto
julia> using Pkg; Pkg.test()
```

## PySCF integration

Functions that call PySCF (`pyscf_do_scf`, `pyscf_build_ints`, `pyscf_fci`, ...)
live in a package extension that activates when [PyCall](https://github.com/JuliaPy/PyCall.jl)
is loaded, so the core package does not require Python:

```julia
using TPSChem
using PyCall   # activates the PySCF-backed functions (requires pyscf in your Python)
```
