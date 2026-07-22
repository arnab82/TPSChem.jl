# CMF 
## Background
The Cluster Mean Field (CMF) approach used in this work was developed by [Jimènez-Hoyos and Scuseria](https://arxiv.org/abs/1505.05909). 

PySCF-backed helpers (`pyscf_do_scf`, `pyscf_build_ints`, ...) are provided by the
package extension `TPSChemPyCallExt`, which activates when `PyCall` is loaded.

## Index
```@index
Pages = ["CMFs.md"]
```

## Documentation 
```@autodocs
Modules = [TPSChem.ClusterMeanField]
Order   = [:type, :function]
```
