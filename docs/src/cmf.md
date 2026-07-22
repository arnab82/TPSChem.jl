```@meta
CurrentModule = TPSChem.ClusterMeanField
```

# Cluster Mean-Field (CMF)

This example clusters a chain of six hydrogen atoms into three H₂ pairs and
optimizes the orbitals and local cluster states self-consistently. It mirrors
the maintained test `test/upstream/ClusterMeanField/test_cmf.jl`.

!!! note "PySCF required"
    The `pyscf_*` and `localize` helpers come from the `TPSChemPyCallExt`
    extension, which activates when `PyCall` is loaded (see
    [Installation](installation_instructions.md)).

## Set up the workspace

```julia
using TPSChem.QCBase
using TPSChem.RDM
using TPSChem.InCoreIntegrals
using TPSChem.ActiveSpaceSolvers
using TPSChem.ClusterMeanField
using PyCall            # activates the PySCF-backed helpers
using LinearAlgebra
```

## Build the molecule and integrals

The basis is part of the [`Molecule`](@ref); `pyscf_do_scf` then takes the
molecule alone.

```julia
atoms = [Atom(i, "H", [Float64(i-1), 0.0, 0.0]) for i in 1:6]
basis = "sto-3g"
mol   = Molecule(0, 1, atoms, basis)

mf   = pyscf_do_scf(mol)
nbas = size(mf.mo_coeff, 1)
ints = pyscf_build_ints(mol, mf.mo_coeff, zeros(nbas, nbas))
```

Optionally get a reference FCI energy (3 α, 3 β electrons):

```julia
e_fci, d1a_fci, d1b_fci, d2_fci = pyscf_fci(ints, 3, 3)
```

## Localize and rotate the integrals

```julia
C  = mf.mo_coeff
Cl = localize(mf.mo_coeff, "lowdin", mf)
S  = get_ovlp(mf)
U  = C' * S * Cl
ints = orbital_rotation(ints, U)         # integrals in the localized basis
```

## Define the clustering

```julia
clusters    = [MOCluster(i, collect(r)) for (i, r) in enumerate([1:2, 3:4, 5:6])]
init_fspace = [(1, 1), (1, 1), (1, 1)]   # (nα, nβ) per cluster
```

## Run the CMF orbital optimization

Start from a mean-field 1-RDM guess and optimize:

```julia
rdm_mf   = C[:, 1:2] * C[:, 1:2]'
rdm1     = RDM1(rdm_mf, rdm_mf)

e_cmf, U = cmf_oo(ints, clusters, init_fspace, rdm1;
                  verbose=0, gconv=1e-6, method="cg")
```

For the H₆/STO-3G example above, `e_cmf ≈ -3.205983033016`. The rotation `U`
maps the localized orbitals to the CMF orbitals (`C_cmf = Cl * U`), which you can
write to a molden file with `pyscf_write_molden(mol, Cl*U, filename="cmf.molden")`.

See also [`cmf_ci`](@ref), [`cmf_oo_newton`](@ref), and [`cmf_oo_diis`](@ref) in
the [ClusterMeanField](library/CMFs.md) reference.
