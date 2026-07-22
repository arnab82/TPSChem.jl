# Installation

TPSChem requires **Julia 1.10 or newer** (the LTS is 1.10; CI also tests the
current stable release).

## 1. Clone and instantiate

```bash
git clone https://github.com/nmayhall/TPSChem.jl
cd TPSChem.jl
julia --project=./ -e 'using Pkg; Pkg.instantiate()'
```

The `multinode` branch adds the distributed / sharded solvers:

```bash
git checkout multinode
julia --project=./ -e 'using Pkg; Pkg.instantiate()'
```

## 2. PySCF (optional, for building integrals)

The molecule/integral helpers (`pyscf_do_scf`, `pyscf_build_ints`,
`pyscf_fci`, `localize`, `pyscf_write_molden`, ...) are provided by the package
extension `TPSChemPyCallExt`, which activates automatically when
[`PyCall`](https://github.com/JuliaPy/PyCall.jl) is loaded. They require a
Python environment with [PySCF](https://pyscf.org/) installed.

```julia
using Pkg
Pkg.add("PyCall")

# point PyCall at a Python that has pyscf, e.g.:
ENV["PYTHON"] = "/path/to/python"   # a venv/conda python with `pip install pyscf`
Pkg.build("PyCall")
```

The tensor-product methods themselves (CMF/TPSCI/SPT solvers) work on any
`InCoreInts` object and do not require PySCF — it is only one convenient way to
produce integrals.

## 3. Run the tests

```bash
julia --project=./ -t auto
```

or from inside a Julia session started with `--project=./`:

```julia
using Pkg
Pkg.test()
```

## 4. Build the documentation locally (optional)

```bash
julia --project=docs docs/make.jl
```

The rendered site is written to `docs/build/`; open `docs/build/index.html`.
Building requires no GitHub credentials — the deploy step is a no-op outside of
GitHub Actions.
