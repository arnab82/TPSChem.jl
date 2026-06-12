# TPSChem.BlockDavidson

*Formerly the standalone package [BlockDavidson.jl](https://github.com/nmayhall-vt/BlockDavidson.jl).*

Simple block-Davidson/Lanczos solver for the lowest eigenvalues and eigenvectors of a
matrix or matrix-free `LinearMap`. Used throughout TPSChem for the outer CI
diagonalizations (TPSCI, SPT, FCI).

## Usage

### Explicit matrix

```julia
using TPSChem.BlockDavidson

dav = Davidson(A)
e, v = eigs(dav)

# with diagonal preconditioning
e, v = eigs(dav, Adiag=diag(A))
```

### Matrix-free

Define the action of your operator on a block of vectors and wrap it in a `LinearMap`:

```julia
using LinearMaps
using TPSChem.BlockDavidson

lmap = LinearMap(matvec)   # matvec(v) returns A*v
dav = Davidson(lmap; max_iter=200, max_ss_vecs=8, tol=1e-6, nroots=6,
               v0=v_guess, lindep_thresh=1e-10)
e, v = eigs(dav)
```

## Exported names

`Davidson`, `LinOpMat`, `eigs`
