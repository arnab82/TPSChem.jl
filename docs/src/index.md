```@meta
CurrentModule = TPSChem
```

# TPSChem.jl

*Coarse-grained electronic structure in a tensor product state (TPS) basis.*

**TPSChem** approximates full configuration interaction (FCI) on large active
spaces by partitioning the orbitals into disjoint **clusters**, diagonalizing
local (cluster) Hamiltonians, and building the wavefunction in the tensor
product space of the resulting many-body cluster states. Much of the electron
correlation is folded up into the basis vectors themselves, so a compact,
sparse (or block-compressed) representation of the global state suffices.

The package consolidates the former FermiCG ecosystem — `FermiCG`, `QCBase`,
`InCoreIntegrals`, `BlockDavidson`, `RDM`, `ActiveSpaceSolvers`, and
`ClusterMeanField` — into a single package. Each former standalone package lives
on as a submodule of the same name.

## Methods

- **[CMF](library/CMFs.md)** — Cluster Mean Field: jointly optimize the orbitals
  and the local cluster ground states to obtain the variationally best *single*
  tensor product state. This is the reference for the correlated methods.
- **[TPSCI](library/TPSCI.md)** — Tensor Product Selected CI: a selected-CI in
  the cluster-state basis that grows a **sparse** set of important tensor
  products toward FCI.
- **[SPT](library/SPT.md)** — Subspace Product Tucker: reintroduces the discarded
  tensor products as **HOSVD-compressed blocks** of the Hilbert space.
- **TPS-CEPA** — a CEPA-style correction on top of a TPSCI/CMF reference.

## Documentation versions

Docs are published per branch:

| Version | Branch | Contents |
|---------|--------|----------|
| `dev` | `main` | Core single-node CMF / TPSCI / SPT |
| `multinode` | `multinode` | Everything in `dev` **plus** the distributed / sharded solvers for CI vectors that exceed one node's RAM |

Use the version selector (bottom-left) to switch. The distributed workflow is
documented under **Multinode / distributed** and **API Reference → Multinode &
Sharded** on the `multinode` version.

## Getting started

See the [Installation](installation_instructions.md) page, then the worked
[CMF](cmf.md) and [FCI](fci.md) examples. The full API is under
[API Reference](library/TPSChem.md).

## Citing / background

- Cluster Mean Field: Jiménez-Hoyos and Scuseria, [arXiv:1505.05909](https://arxiv.org/abs/1505.05909).
