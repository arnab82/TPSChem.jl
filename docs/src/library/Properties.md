# RDMs & Properties

Reduced density matrices and one-electron properties computed from a converged
TPSCI wavefunction, plus transition properties and absorption spectra. For the
underlying RDM *types* (`RDM1`, `RDM2`) and orbital gradient/Hessian machinery,
see the [RDM submodule](RDM.md).

```@index
Pages = ["Properties.md"]
```

## Reduced density matrices

Spin-resolved and spin-free 1- and 2-RDMs, with threaded and distributed
variants for large states.

```@autodocs
Modules = [TPSChem]
Pages   = ["tpsci_property.jl"]
Order   = [:function]
Filter  = t -> occursin("rdm", lowercase(string(nameof(t))))
```

## One-electron & transition properties, spectra

Expectation values of one-electron operators, transition dipoles / oscillator
strengths, and (stick / broadened) absorption spectra.

```@autodocs
Modules = [TPSChem]
Pages   = ["tpsci_property.jl"]
Order   = [:function]
Filter  = t -> !occursin("rdm", lowercase(string(nameof(t))))
```

```@autodocs
Modules = [TPSChem]
Pages   = ["absorption_spectrum.jl"]
Order   = [:type, :function]
```
