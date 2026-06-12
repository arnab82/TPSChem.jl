# One-Electron Properties in the Tensor Product State CI Framework

## Overview

This note derives the theory and working equations for computing one-electron
properties and absorption spectra from TPSCI (and BS/SPT) wavefunctions.
The key objects are the **transition one-particle reduced density matrix (1-RDM)**
and its contraction with one-electron property integrals.

---

## 1. The TPSCI Wavefunction

The system is partitioned into $N$ orbital clusters
$\{C_1, C_2, \ldots, C_N\}$.
Each cluster $C_I$ has a local Fock space labeled by
$(n_{I,\alpha}, n_{I,\beta})$ and a set of locally-diagonalized eigenstates
$\{|\mu_I\rangle\}$ obtained from a cluster mean-field or CASCI solve.

The TPSCI wavefunction for root $r$ is a sparse linear combination of
tensor-product cluster states:

$$
|\Psi_r\rangle
= \sum_{\mathbf{f}} \sum_{\boldsymbol{\mu}}
  C^{(r)}_{\mathbf{f},\boldsymbol{\mu}}\;
  |\mu_1^{(f_1)}\rangle \otimes |\mu_2^{(f_2)}\rangle
  \otimes \cdots \otimes |\mu_N^{(f_N)}\rangle
$$

where $\mathbf{f} = (f_1,\ldots,f_N)$ is a **Fock configuration**
(electron distribution across clusters) and
$\boldsymbol{\mu} = (\mu_1,\ldots,\mu_N)$ is a **cluster configuration**
(which eigenstate each cluster occupies).
The sparsity is controlled by the TPSCI threshold.

In the code: `psi.data[fock][config] = MVector{R,T}` stores the
$R$-root coefficient vector $C^{(1\ldots R)}_{\mathbf{f},\boldsymbol{\mu}}$.

---

## 2. One-Electron Operators and Properties

A general one-electron operator has the second-quantized form

$$
\hat{P} = \sum_{p,q,\sigma} P_{pq}\, \hat{p}^\dagger_{p\sigma} \hat{q}_{\sigma}
$$

where $P_{pq} = \langle p | \hat{P} | q \rangle$ are the **property integrals**
(a Hermitian $n_\text{orb} \times n_\text{orb}$ matrix in the MO basis),
$\sigma \in \{\alpha, \beta\}$ is spin, and the sum runs over all molecular
orbitals $p, q$.

Examples:
- **Electric dipole** (length gauge): $P_{pq} = \langle p | \mathbf{r} | q \rangle$
  (three components $\mu_x, \mu_y, \mu_z$)
- **Electric quadrupole**: $P_{pq} = \langle p | r_i r_j | q \rangle$
- **Momentum** (velocity gauge): $P_{pq} = \langle p | \nabla | q \rangle$
- **Spin-orbit coupling** (one-electron part): analogous form

The expectation value between two (possibly different) states $r_1, r_2$ is

$$
\langle \Psi_{r_1} | \hat{P} | \Psi_{r_2} \rangle
= \sum_{p,q} P_{pq}
  \underbrace{\sum_\sigma\langle \Psi_{r_1} |
    \hat{p}^\dagger_{p\sigma} \hat{q}_{\sigma}
  | \Psi_{r_2} \rangle}_{\gamma^{r_1 r_2}_{pq}}
= \mathrm{Tr}\!\left(P \cdot \gamma^{r_1 r_2}\right)
$$

---

## 3. The Transition One-Particle Reduced Density Matrix

### Definition

The **transition 1-RDM** between roots $r_1$ (bra) and $r_2$ (ket) is defined
spin-resolved as

$$
\gamma^{r_1 r_2}_{\alpha\alpha,pq}
= \langle \Psi_{r_1} | \hat{p}^\dagger_{p\alpha} \hat{q}_{\alpha} | \Psi_{r_2} \rangle,
\qquad
\gamma^{r_1 r_2}_{\beta\beta,pq}
= \langle \Psi_{r_1} | \hat{p}^\dagger_{p\beta} \hat{q}_{\beta} | \Psi_{r_2} \rangle
$$

For a spin-free property operator ($P_{\alpha\alpha} = P_{\beta\beta} = P$, no spin-flip):

$$
\langle \Psi_{r_1} | \hat{P} | \Psi_{r_2} \rangle
= \mathrm{Tr}\!\left(P \cdot \left(\gamma^{r_1 r_2}_{\alpha\alpha}
  + \gamma^{r_1 r_2}_{\beta\beta}\right)\right)
= \mathrm{Tr}(P \cdot \gamma^{r_1 r_2}_\text{total})
$$

When $r_1 = r_2 = r$, $\gamma^{rr}$ is the ordinary **1-RDM** of root $r$.
When $r_1 \neq r_2$, $\gamma^{r_1 r_2}$ is the **transition 1-RDM**.

### Storage

In the code: `compute_1rdm(bra, ket, cluster_ops)` returns tensors
`γ_aa[p, q, r1, r2]` and `γ_bb[p, q, r1, r2]` of shape
$(n_\text{orb}, n_\text{orb}, R_1, R_2)$.
Orbital indices $p, q$ run over the full MO space with clusters concatenated
in order: $C_1 \| C_2 \| \cdots \| C_N$.

---

## 4. Evaluation of the 1-RDM in the Clustered Basis

Substituting the TPSCI expansion, the transition 1-RDM splits into
**two topologically distinct contributions** depending on whether $p$ and $q$
reside on the same cluster or different clusters.

### 4.1 On-Cluster Contribution ($p, q \in C_I$)

When both orbitals belong to cluster $I$:

$$
\gamma^{r_1 r_2}_{\alpha\alpha,pq}\Big|_{p,q\in C_I}
= \sum_{\mathbf{f}}
  \underbrace{\left[\sum_{\substack{\boldsymbol{\mu}, \boldsymbol{\nu} \\
    \mu_K = \nu_K,\; K\neq I}}
    C^{(r_1)}_{\mathbf{f},\boldsymbol{\mu}}\,
    C^{(r_2)}_{\mathbf{f},\boldsymbol{\nu}}\right]}_{\rho^{r_1 r_2}_{I}[s, t;\,\mathbf{f}]}
  \langle s_I | \hat{p}^\dagger_{p\alpha} \hat{q}_\alpha | t_I \rangle
$$

The object $\rho^{r_1 r_2}_I[s,t;\mathbf{f}]$ is the **cluster reduced density
matrix** (cRDM) for cluster $I$ in Fock sector $\mathbf{f}$:

$$
\rho^{r_1 r_2}_I[s, t] = \sum_{\substack{\text{configs} \\ \text{all } K \neq I \text{ fixed}}}
C^{(r_1)}_{\ldots s \ldots}\, C^{(r_2)}_{\ldots t \ldots}
$$

The local transition density matrix $\langle s | \hat{p}^\dagger_\alpha \hat{q}_\alpha | t \rangle$
is the `"Aa"` operator stored in `cluster_ops[I]["Aa"][(f_I, f_I)]`,
a tensor of shape $(n_p \cdot n_q,\, n_s,\, n_t)$ (orbital indices flattened,
$n_{p/q}$ = number of orbitals in $C_I$).

In matrix notation (suppressing the Fock-sector sum for clarity):

$$
\gamma^{r_1 r_2}_{\alpha\alpha}\Big|_{C_I} =
\sum_{s,t} \rho_I[s,t,r_1,r_2]\; \widetilde{\Gamma}^{I,\alpha\alpha}[p,q,s,t]
$$

where $\widetilde{\Gamma}^{I,\alpha\alpha}[p,q,s,t]
= \langle s_I | \hat{p}^\dagger_\alpha \hat{q}_\alpha | t_I \rangle$
is reshaped from `"Aa"` after undoing the code's flattening:
`reshape(Aa_i, norb_I, norb_I, n_s, n_t)`.

**Implementation note.** The cRDM is built by grouping configs that agree at
all clusters $K \neq I$; matching groups are then outer-multiplied over roots.

### 4.2 Inter-Cluster Contribution ($p \in C_I$, $q \in C_J$, $I \neq J$)

When $p$ and $q$ reside on *different* clusters, one electron must be
transferred from cluster $J$ to cluster $I$ (for the alpha-alpha case).
The Fock configurations of bra and ket therefore differ:

$$
f_{I,\text{bra}} = f_{I,\text{ket}} + (1, 0),\qquad
f_{J,\text{bra}} = f_{J,\text{ket}} - (1, 0),\qquad
f_{K,\text{bra}} = f_{K,\text{ket}},\quad K \neq I, J
$$

The matrix element factorizes over clusters:

$$
\gamma^{r_1 r_2}_{\alpha\alpha,pq}\Big|_{p\in C_I,\, q\in C_J}
= \epsilon_{IJ}\sum_{\substack{\mathbf{f}_\text{ket} \\ \text{configs}}}
  C^{(r_1)}_{\mathbf{f}_\text{bra},\boldsymbol{s}}\,
  C^{(r_2)}_{\mathbf{f}_\text{ket},\boldsymbol{t}}\;
  \langle s_I | \hat{p}^\dagger_{p\alpha} | t_I \rangle\;
  \langle s_J | \hat{q}_\alpha | t_J \rangle
$$

where the condition $s_K = t_K$ for all $K \neq I, J$ must hold,
and $\epsilon_{IJ}$ is the **fermionic sign**.

The two local tensors are:

$$
\langle s_I | \hat{p}^\dagger_{p\alpha} | t_I \rangle
= A^I_\alpha[p, s_I, t_I]
\qquad \text{(creation; stored as \texttt{"A"})}
$$

$$
\langle s_J | \hat{q}_\alpha | t_J \rangle
= a^J_\alpha[q, s_J, t_J]
\qquad \text{(annihilation; stored as \texttt{"a"}, adjoint of \texttt{"A"})}
$$

These are the **one-open-orbital-index** gamma tensors: each carries one
uncontracted orbital label ($p$ or $q$) alongside two cluster state indices.

#### Fermionic Sign

Placing $\hat{p}^\dagger_{I}$ and $\hat{q}_{J}$ in their canonical (ascending
cluster index) positions requires commuting through the occupation-number
strings of intermediate clusters. Following the convention in the code:

$$
\epsilon_{IJ} = (-1)^{\,n^\text{ket}_{<I}} \cdot (-1)^{\,n^\text{ket}_{<J}}
$$

where $n^\text{ket}_{<K} = \sum_{L < K}(n^\text{ket}_{L,\alpha} + n^\text{ket}_{L,\beta})$
is the total electron count before cluster $K$ in the **ket** Fock configuration.
The formula is symmetric in $I, J$: swapping them changes both factors but
leaves the product unchanged (by inspection for the typical cases).

---

## 5. Summary: Full Working Equations

Collecting both contributions and both spins:

$$
\boxed{
\gamma^{r_1 r_2}_{\text{total},pq}
= \underbrace{\sum_I \sum_{s,t} \rho_I^{r_1 r_2}[s,t]
  \left(\widetilde{\Gamma}^{I,\alpha\alpha}_{pq,st}
       + \widetilde{\Gamma}^{I,\beta\beta}_{pq,st}\right)}_{\text{on-cluster}}
+ \underbrace{\sum_{I \neq J} \epsilon_{IJ}\sum_{\text{configs}}
  C^{(r_1)}_\text{bra} C^{(r_2)}_\text{ket}
  \left(A^I_\alpha[p,s_I,t_I]\,a^J_\alpha[q,s_J,t_J]
       + B^I_\beta[p,s_I,t_I]\,b^J_\beta[q,s_J,t_J]\right)}_{\text{inter-cluster}}
}
$$

---

## 6. Absorption Spectra from Dipole Integrals

### Electric Dipole Operator

In the length gauge, the electric-dipole operator is

$$
\hat{\mu}_\alpha = -e\sum_{p,q} \langle p | r_\alpha | q \rangle
                  \sum_\sigma \hat{p}^\dagger_\sigma \hat{q}_\sigma,
\qquad \alpha \in \{x, y, z\}
$$

The integrals $\langle p | r_\alpha | q \rangle$ are **one-electron integrals**
computable from any standard quantum chemistry package.

### Transition Dipole Moment

The transition dipole moment between the reference state $|0\rangle$ (root
$r_0$) and excited state $|n\rangle$ (root $n$) is

$$
\mu^{0n}_\alpha
= \langle \Psi_0 | \hat{\mu}_\alpha | \Psi_n \rangle
= \sum_{p,q} \langle p | r_\alpha | q \rangle \cdot \gamma^{0n}_{\text{total},pq}
= \mathrm{Tr}\!\left(\mu_\alpha \cdot \gamma^{0n}_\text{total}\right)
$$

This is computed by `compute_transition_dipoles`.

### Oscillator Strength

The **electric-dipole oscillator strength** in the length gauge is

$$
f_{0n} = \frac{2}{3}\,\Delta E_{0n}\,
         \left(|\mu^{0n}_x|^2 + |\mu^{0n}_y|^2 + |\mu^{0n}_z|^2\right)
$$

where $\Delta E_{0n} = E_n - E_0 > 0$.  The factor of $\frac{2}{3}$ arises from
averaging over the three Cartesian directions and from the dipole-length /
oscillator-strength relation in atomic units.

The oscillator strengths obey the **Thomas–Reiche–Kuhn sum rule**:

$$
\sum_n f_{0n} = N_\text{elec}
$$

where $N_\text{elec}$ is the total number of electrons (a useful check).

### Absorption Spectrum

The theoretical absorption cross-section is a sum of delta functions at
each transition energy. In practice, the delta functions are replaced by
a normalized line shape $g(\omega - \omega_{0n}; \sigma)$:

**Lorentzian** (natural/collisional broadening):
$$
g_L(\omega) = \frac{\sigma/\pi}{\omega^2 + \sigma^2}
$$

**Gaussian** (inhomogeneous broadening):
$$
g_G(\omega) = \frac{1}{\sigma\sqrt{2\pi}}\exp\!\left(-\frac{\omega^2}{2\sigma^2}\right)
$$

The stick spectrum broadened to a continuous absorption profile is:

$$
I(\omega) = \sum_{n \neq 0} f_{0n}\, g(\omega - \Delta E_{0n};\,\sigma)
$$

In the code $\sigma$ is in Hartree; a typical value of $\sigma = 0.005\,\text{Ha}
\approx 0.14\,\text{eV}$ gives a reasonable linewidth.

---

## 7. Code Workflow

```julia
# After running TPSCI to get R roots:
E, psi = tpsci_ci(ci_vector, cluster_ops, clustered_ham; nroots=R)

# Compute transition 1-RDM for all root pairs at once (shape: norb×norb×R×R)
γ_aa, γ_bb = compute_1rdm(psi, cluster_ops)

# Reuse γ for any one-electron property:
# e.g., expectation of an operator with integrals P_mat (norb×norb):
P = contract_1rdm_property(γ_aa, γ_bb, P_mat)    # R×R matrix

# For absorption spectra, supply dipole integrals (norb×norb each):
f = compute_oscillator_strengths(E, γ_aa, γ_bb, μ_x, μ_y, μ_z; ref_root=1)

print_stick_spectrum(E, f; units=:ev)

ω, I = absorption_spectrum(E, f; σ=0.005, lineshape=:lorentzian)
```

**Key design choice**: $\gamma$ is computed *once* with all orbital indices open
($p, q$ uncontracted) and stored as a full $(n_\text{orb} \times n_\text{orb}
\times R \times R)$ tensor. Any number of properties are then obtained by cheap
matrix traces $\mathrm{Tr}(P \cdot \gamma^{r_1 r_2})$, avoiding repeated passes
through the wavefunction.

---

## 8. Implementation Details

### Orbital Index Convention

Clusters are concatenated in order. For clusters with $n_1, n_2, \ldots, n_N$
orbitals respectively:

$$
p \in C_I \iff p \in \left[1 + \sum_{K<I} n_K,\; \sum_{K \leq I} n_K\right]
$$

The orbital offset of cluster $I$ is `orb_offsets[I]` $= \sum_{K<I} n_K$.

### Shape of Stored Operators

After the reshape step in `compute_cluster_ops`, operators in `cluster_ops`
are stored as:

| Operator | Key | Raw shape | Stored shape | Fock transition |
|---|---|---|---|---|
| $\langle s\|p'_\alpha q_\alpha\|t\rangle$ | `"Aa"` | $(n_p, n_q, n_s, n_t)$ | $(n_p n_q, n_s, n_t)$ | $(f, f)$ diagonal |
| $\langle s\|p'_\beta q_\beta\|t\rangle$ | `"Bb"` | $(n_p, n_q, n_s, n_t)$ | $(n_p n_q, n_s, n_t)$ | $(f, f)$ diagonal |
| $\langle s\|p'_\alpha\|t\rangle$ | `"A"` | $(n_p, n_s, n_t)$ | $(n_p, n_s, n_t)$ | $f_\alpha + 1$ |
| $\langle s\|p_\alpha\|t\rangle$ | `"a"` | $(n_p, n_s, n_t)$ | $(n_p, n_s, n_t)$ | $f_\alpha - 1$ |
| $\langle s\|p'_\beta\|t\rangle$ | `"B"` | $(n_p, n_s, n_t)$ | $(n_p, n_s, n_t)$ | $f_\beta + 1$ |
| $\langle s\|p_\beta\|t\rangle$ | `"b"` | $(n_p, n_s, n_t)$ | $(n_p, n_s, n_t)$ | $f_\beta - 1$ |

The `"Aa"` operator is reshaped back to $(n_p, n_q, n_s, n_t)$ during the
1-RDM computation using Julia's column-major convention:
$\text{pq\_flat} = p + (q-1) \cdot n_\text{orb}$.

### Computational Complexity

| Step | Cost |
|---|---|
| Cluster RDM $\rho_I$ | $O(M^2)$ per fock sector, where $M$ = configs per sector |
| On-cluster contraction | $O(n_I^2 \cdot n_s \cdot n_t)$ per cluster per fock |
| Inter-cluster | $O(N^2 \cdot M^2 \cdot n_I \cdot n_J)$ per fock-pair |
| Property contraction | $O(n_\text{orb}^2 \cdot R^2)$ (cheap; done after) |

The inter-cluster loop dominates for large $N$ but is the same order as a
single Hamiltonian matrix-vector product, since both require the same
config-pair enumeration.

---

## 9. Spin-Flip Transition 1-RDM and Spin-Orbit Coupling

### 9.1 Definition

For spin-orbit coupling (SOC) or any response involving $\Delta M_S = \pm 1$,
two additional spin-off-diagonal transition 1-RDMs are needed:

$$
\gamma^{r_1 r_2}_{\alpha\beta,pq}
= \langle \Psi_{r_1} | \hat{p}^\dagger_{p\alpha} \hat{q}_\beta | \Psi_{r_2} \rangle,
\qquad
\gamma^{r_1 r_2}_{\beta\alpha,pq}
= \langle \Psi_{r_1} | \hat{p}^\dagger_{p\beta} \hat{q}_\alpha | \Psi_{r_2} \rangle
$$

Both have shape $(n_\text{orb}, n_\text{orb}, R_1, R_2)$.
Note $\gamma_{\beta\alpha} = (\gamma_{\alpha\beta})^\dagger$ for a single state
($r_1 = r_2$), but they are independent for transition matrix elements.

### 9.2 Clustered Evaluation

**On-cluster** ($p, q \in C_I$): The bra and ket Fock sectors at cluster $I$
differ — unlike the spin-conserving case — because creating an $\alpha$ and
annihilating a $\beta$ changes the local spin configuration:

$$
f_{I,\text{bra}} = (n_{I,\alpha}+1,\, n_{I,\beta}-1),\qquad
f_{I,\text{bra}} = (n_{I,\alpha}-1,\, n_{I,\beta}+1)
$$

respectively. All other clusters share the same Fock sector.
The local operators used are the spin-flip TDMs stored in `cluster_ops`:

| Contribution | Operator key | Fock transition | Shape stored |
|---|---|---|---|
| $\gamma_{\alpha\beta}$ on-cluster | `"Ab"` | $((n_\alpha+1,n_\beta-1),(n_\alpha,n_\beta))$ | $(n_p^2,\,n_s,\,n_t)$ |
| $\gamma_{\beta\alpha}$ on-cluster | `"Ba"` | $((n_\alpha-1,n_\beta+1),(n_\alpha,n_\beta))$ | $(n_p^2,\,n_s,\,n_t)$ |

After retrieval both are reshaped to $(n_p, n_p, n_s, n_t)$ using Julia's
column-major convention before contraction.

**Inter-cluster** ($p \in C_I$, $q \in C_J$, $I \neq J$):

| Contribution | Operators | Fock at $I$ | Fock at $J$ | Sign |
|---|---|---|---|---|
| $\gamma_{\alpha\beta}$, $p\in I$, $q\in J$ | `"A"`$[I]$ $\times$ `"b"`$[J]$ | $+1\alpha$ | $-1\beta$ | $(-1)^\chi$ |
| $\gamma_{\beta\alpha}$, $p\in I$, $q\in J$ | `"B"`$[I]$ $\times$ `"a"`$[J]$ | $+1\beta$ | $-1\alpha$ | $(-1)^\chi$ |

where $\chi = \sum_{K<I} N_K + \sum_{K<J} N_K$ as in the spin-conserving case
(`_rdm_sign`).

### 9.3 Spin-Orbit Coupling Matrix Elements

The one-electron Breit-Pauli SOC Hamiltonian in the MO basis is

$$
\hat{H}_\text{SOC}
= \sum_{pq} h^z_{pq}\,\bigl(\hat{p}^\dagger_{p\alpha}\hat{q}_\alpha
                            - \hat{p}^\dagger_{p\beta}\hat{q}_\beta\bigr)
+ \sum_{pq} h^+_{pq}\,\hat{p}^\dagger_{p\alpha}\hat{q}_\beta
+ \sum_{pq} h^-_{pq}\,\hat{p}^\dagger_{p\beta}\hat{q}_\alpha
$$

where the integral matrices come from `int1e_spnucsp` (nuclear) plus
`int2e_p4` (two-electron mean-field screening); in practice the
**SOMF** (spin-orbit mean-field) approximation is used. The matrix element
between states $r_1$ and $r_2$ is

$$
\langle \Psi_{r_1} | \hat{H}_\text{SOC} | \Psi_{r_2} \rangle
= \mathrm{Tr}\!\bigl(h^z\cdot(\gamma^{r_1 r_2}_{\alpha\alpha}
                              -\gamma^{r_1 r_2}_{\beta\beta})\bigr)
+ \mathrm{Tr}\!\bigl(h^+\cdot\gamma^{r_1 r_2}_{\alpha\beta}\bigr)
+ \mathrm{Tr}\!\bigl(h^-\cdot\gamma^{r_1 r_2}_{\beta\alpha}\bigr)
$$

In code:

```julia
γ_aa, γ_bb = TPSChem.compute_1rdm(psi, cluster_ops)
γ_ab, γ_ba = TPSChem.compute_1rdm_sf(psi, cluster_ops)

# SOC matrix (R×R, complex in general; here real-valued approximation)
SOC = zeros(nroots, nroots)
for r2 in 1:nroots, r1 in 1:nroots
    SOC[r1,r2] = ( dot(vec(hz),  vec(γ_aa[:,:,r1,r2] - γ_bb[:,:,r1,r2]))
                 + dot(vec(hpos), vec(γ_ab[:,:,r1,r2]))
                 + dot(vec(hneg), vec(γ_ba[:,:,r1,r2])) )
end
```

---

## 10. Two-Particle Reduced Density Matrix

### 10.1 Definition

The **spin-free transition 2-RDM** is

$$
\Gamma^{r_1 r_2}_{pq,rs}
= \sum_{\sigma,\tau}\langle \Psi_{r_1} |
  \hat{p}^\dagger_\sigma \hat{q}^\dagger_\tau \hat{s}_\tau \hat{r}_\sigma
  | \Psi_{r_2} \rangle
$$

Shape: $(n_\text{orb}, n_\text{orb}, n_\text{orb}, n_\text{orb}, R_1, R_2)$.

For a two-electron operator $\hat{W} = \frac{1}{2}\sum_{pqrs} w_{pq,rs}
\hat{p}^\dagger\hat{q}^\dagger\hat{s}\hat{r}$:

$$
\langle \Psi_{r_1} | \hat{W} | \Psi_{r_2} \rangle
= \frac{1}{2}\sum_{pqrs} w_{pq,rs}\,\Gamma^{r_1 r_2}_{pq,rs}
$$

### 10.2 Topological Cases in the Clustered Basis

The 2-RDM receives contributions from three charge-conserving cluster
topologies. Charge conservation at each cluster requires that the number
of creation operators equals the number of annihilation operators per cluster.

#### Case (I,I,I,I) — On-cluster

All four orbital indices on the same cluster $I$. The 4-body matrix element
is expressed via the **composition identity**

$$
\langle u|\hat{p}^\dagger\hat{q}^\dagger\hat{s}\hat{r}|v\rangle
= \sum_w \langle u|\hat{p}^\dagger\hat{r}|w\rangle
         \langle w|\hat{q}^\dagger\hat{s}|v\rangle
- \delta_{qr}\,\langle u|\hat{p}^\dagger\hat{s}|v\rangle
$$

where $w$ runs over the full cluster state space (resolution of identity),
and $\delta_{qr}$ applies **only to same-spin pairs** (the cross-spin
operators commute). The spin-summed formula is

$$
\Gamma^{r_1 r_2}_{pq,rs}\Big|_{I,I,I,I}
= \sum_{u,v,r_1,r_2} \rho_I[u,v,r_1,r_2]
  \Bigl[\sum_w
    \bigl(\widetilde\Gamma^{I,\alpha\alpha}_{pr,uw}
          \widetilde\Gamma^{I,\alpha\alpha}_{qs,wv}
         +\widetilde\Gamma^{I,\alpha\alpha}_{pr,uw}
          \widetilde\Gamma^{I,\beta\beta}_{qs,wv}
         +\widetilde\Gamma^{I,\beta\beta}_{pr,uw}
          \widetilde\Gamma^{I,\alpha\alpha}_{qs,wv}
         +\widetilde\Gamma^{I,\beta\beta}_{pr,uw}
          \widetilde\Gamma^{I,\beta\beta}_{qs,wv}
    \Bigr)
  - \delta_{qr}\bigl(
      \widetilde\Gamma^{I,\alpha\alpha}_{ps,uv}
    + \widetilde\Gamma^{I,\beta\beta}_{ps,uv}
    \bigr)\Bigr]
$$

where $\widetilde\Gamma^{I,\sigma\sigma}_{pq,st}
= \langle s_I|\hat{p}^\dagger_\sigma\hat{q}_\sigma|t_I\rangle$
is the reshaped `"Aa"` (or `"Bb"`) tensor.

#### Case (I,J,I,J) — Inter-cluster Fock-diagonal

$p, r \in C_I$ and $q, s \in C_J$ ($I \neq J$). Each cluster has one
creation and one annihilation operator, so both remain **Fock-neutral**.
The inter-cluster Jordan–Wigner string is trivially $\pm 1$ for
number-preserving operators, so the **sign is always $+1$**.

$$
\Gamma^{r_1 r_2}_{pq,rs}\Big|_{I,J,I,J}
= \sum_{\substack{\text{configs}\\K\neq I,J}}
  C^{(r_1)}_\text{bra}\,C^{(r_2)}_\text{ket}
  \sum_{\sigma,\tau}
  \widetilde\Gamma^{I,\sigma\sigma}_{pr,s_I t_I}\,
  \widetilde\Gamma^{J,\tau\tau}_{qs,s_J t_J}
$$

Spin contributions and the cluster operators they use:

| $\sigma$ | $\tau$ | At $I$ | At $J$ |
|---|---|---|---|
| $\alpha$ | $\alpha$ | `"Aa"`$[I]$ | `"Aa"`$[J]$ |
| $\alpha$ | $\beta$ | `"Aa"`$[I]$ | `"Bb"`$[J]$ |
| $\beta$ | $\alpha$ | `"Bb"`$[I]$ | `"Aa"`$[J]$ |
| $\beta$ | $\beta$ | `"Bb"`$[I]$ | `"Bb"`$[J]$ |

This is the dominant term for **inter-cluster Coulomb and exchange** (the
Heisenberg $J$ coupling).

#### Case (I,I,J,J) — Charge-transfer (CT)

$p, q \in C_I$ (both creations) and $r, s \in C_J$ (both annihilations).
The bra and ket occupy different Fock sectors:
$f_{I,\text{bra}} = f_{I,\text{ket}} + \Delta$,
$f_{J,\text{bra}} = f_{J,\text{ket}} - \Delta$.

**Fermionic sign** (from the image formula, $\chi = \sum_{K=I}^{J-1} N_K$):

| Spin pair | $\Delta$ at $I$ / $J$ | Operators | Sign |
|---|---|---|---|
| $\alpha\alpha$ | $(+2,0)$ / $(-2,0)$ | `"AA"`$[I]$ × `"aa"`$[J]$ | $+1$ |
| $\beta\beta$ | $(0,+2)$ / $(0,-2)$ | `"BB"`$[I]$ × `"bb"`$[J]$ | $+1$ |
| $\alpha\beta$ | $(+1,+1)$ / $(-1,-1)$ | `"AB"`$[I]$ × `"ba"`$[J]$ | $(-1)^\chi$ |
| $\beta\alpha$ | $(+1,+1)$ / $(-1,-1)$ | `"BA"`$[I]$ × `"ab"`$[J]$ | $(-1)^\chi$ |

The $+1$ sign for same-spin pairs follows from the cancellation of two
Jordan–Wigner string factors: each pair of operators (both at $I$ or both at
$J$) traverses the same inter-cluster string and the two resulting $(-1)$
factors cancel.

### 10.3 Exchange Coupling from the 2-RDM

The Heisenberg exchange coupling between clusters $I$ and $J$ is

$$
J_{IJ}
= \frac{1}{2}\sum_{\substack{p,r \in C_I \\ q,s \in C_J}}
  (pr|qs)\;\Gamma_{pq,rs}
$$

where $(pr|qs) = \int\!\int \phi_p(1)\phi_r(1)\,r_{12}^{-1}\,\phi_q(2)\phi_s(2)\,d1\,d2$
are the two-electron Coulomb integrals. In code:

```julia
Γ = TPSChem.compute_2rdm(v0a, cluster_ops)

# Extract cluster orbital ranges
I_orbs = (orb_offsets[1]+1):(orb_offsets[1]+norb_I)  # cluster 1
J_orbs = (orb_offsets[2]+1):(orb_offsets[2]+norb_J)  # cluster 2

J12 = 0.0
for s in J_orbs, r in I_orbs, q in J_orbs, p in I_orbs
    J12 += eri[p,r,q,s] * Γ[p,q,r,s,1,1]
end
J12 /= 2
```

where `eri[p,r,q,s]` is the two-electron repulsion integral from `ints_h2`.

---

## 11. Natural Transition Orbitals (NTOs)

The singular value decomposition of the transition 1-RDM,

$$
\gamma^{0n}_\text{total} = U\, \Sigma\, V^\dagger
$$

gives **natural transition orbitals**: the columns of $U$ (hole NTOs) and $V$
(particle NTOs) diagonalize the transition density and visualize which
orbitals participate in the excitation.
