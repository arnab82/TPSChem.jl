# TPSChem.ClusterMeanField

*Formerly the standalone package [ClusterMeanField.jl](https://github.com/nmayhall-vt/ClusterMeanField.jl).*

Cluster Mean Field (CMF): variationally optimize a product of cluster states — the
reference wavefunction that TPSCI, SPT, and the CMF-PT2/CEPA methods all build on.
`cmf_ci` optimizes the cluster CI coefficients for fixed orbitals; `cmf_oo` also
optimizes the orbitals.

## Usage

```julia
using TPSChem
using TPSChem.ClusterMeanField

# ints::InCoreInts for the full active space, e.g. from PySCF (see below)
clusters    = [MOCluster(1, [1,2,3,4]), MOCluster(2, [5,6,7,8]), MOCluster(3, [9,10,11,12])]
init_fspace = [(2,2), (2,2), (2,2)]      # (nα, nβ) per cluster

d1 = RDM1(n_orb(ints))
e_cmf, U, d1 = cmf_oo(ints, clusters, init_fspace, d1,
                      max_iter_oo=60, verbose=0, gconv=1e-10, method="bfgs")

# rotate the integrals to the optimized CMF orbitals before building cluster bases
ints = orbital_rotation(ints, U)
```

Each cluster's CI problem is solved with
[`ActiveSpaceSolvers`](../ActiveSpaceSolvers/README.md) by default
(`use_pyscf=false`).

## PySCF integration (optional)

The `pyscf_*` helpers (`pyscf_do_scf`, `pyscf_build_ints`, `pyscf_fci`,
`pyscf_write_molden`, `localize`, ...) live in the `TPSChemPyCallExt` package
extension and only become available when PyCall is loaded — the core package needs no
Python:

```julia
using TPSChem
using PyCall          # activates the extension (requires pyscf in your Python)

mf   = ClusterMeanField.pyscf_do_scf(mol)
ints = ClusterMeanField.pyscf_build_ints(mol, mf.mo_coeff, zeros(nbas, nbas))
```

## Exported names

`cmf_ci`, `cmf_oo`, and (extension-provided)
`pyscf_do_scf`, `make_pyscf_mole`, `pyscf_build_1e`, `pyscf_build_eri`,
`pyscf_build_ints`, `pyscf_get_jk`, `pyscf_fci`, `pyscf_write_molden`,
`get_nuclear_rep`, `get_ovlp`, `localize`
