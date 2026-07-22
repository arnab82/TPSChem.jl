```@meta
CurrentModule = TPSChem.ActiveSpaceSolvers
```

# Full CI (FCI)

The active-space FCI solver is exposed through the
[ActiveSpaceSolvers](library/ActiveSpaceSolvers.md) interface: build an
[`FCIAnsatz`](@ref), then call [`solve`](@ref). This example does FCI on the
same H₆/STO-3G integrals used in the [CMF example](cmf.md).

## Build integrals

```julia
using TPSChem.QCBase
using TPSChem.InCoreIntegrals
using TPSChem.ActiveSpaceSolvers
using PyCall            # activates the PySCF-backed helpers

atoms = [Atom(i, "H", [Float64(i-1), 0.0, 0.0]) for i in 1:6]
mol   = Molecule(0, 1, atoms, "sto-3g")

mf   = pyscf_do_scf(mol)
nbas = size(mf.mo_coeff, 1)
ints = pyscf_build_ints(mol, mf.mo_coeff, zeros(nbas, nbas))
```

## Solve FCI

`FCIAnsatz(norb, nα, nβ)` defines the determinant space; `SolverSettings`
controls the eigensolver.

```julia
ansatz = FCIAnsatz(6, 3, 3)          # 6 orbitals, 3 α + 3 β electrons
sol    = solve(ints, ansatz, SolverSettings())
display(sol)
```

`sol` is a [`Solution`](@ref) holding the energies and CI vectors; from it you
can compute 1- and 2-RDMs via [`compute_1rdm_2rdm`](@ref).

For a quick reference value you can compare against PySCF's FCI directly:

```julia
e_fci, d1a_fci, d1b_fci, d2_fci = pyscf_fci(ints, 3, 3)
```
