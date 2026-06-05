# Hellmann–Feynman Forces for ACSF Tight-Binding: Derivation, Implementation, and Testing

## 1. Setup and Notation

### Tight-binding Hamiltonian

The model uses a **half-list** of bonds: each directed pair $(i, j)$ with $j > i$ appears exactly once, stored in arrays `pair_i`, `pair_j`, `pair_v`. The bond displacement vector is:

$$\mathbf{r}_p = \mathbf{R}_{j_p} - \mathbf{R}_{i_p} = \texttt{pair\_v}[p]$$

The Bloch Hamiltonian is built with the convention:

$$H(\mathbf{k})[i, j] = t_p \, e^{i\mathbf{k}\cdot\mathbf{r}_p}, \qquad H(\mathbf{k})[j, i] = t_p \, e^{-i\mathbf{k}\cdot\mathbf{r}_p}$$

i.e., the phase $e^{+i\mathbf{k}\cdot\mathbf{r}_p}$ goes in the `[pair_i, pair_j]` slot (source row, target column), and the Hermitian conjugate is added.

### Band energy

At half-filling with a k-point mesh of $N_k$ points:

$$E_{\text{band}} = \frac{1}{N_k}\sum_{\mathbf{k}} \text{Re}\!\left[\text{Tr}\!\left(H(\mathbf{k})\,\rho(\mathbf{k})\right)\right]$$

Using the Hermitian structure of $H$ and $\rho$, this simplifies to:

$$\boxed{E_{\text{band}} = \frac{2}{N_k}\sum_{\mathbf{k}}\sum_p \text{Re}\!\left[t_p \, e^{i\mathbf{k}\cdot\mathbf{r}_p}\,\rho_\mathbf{k}[j_p, i_p]\right]}$$

where $\rho_\mathbf{k}[j, i]$ denotes the $(j, i)$ element of the density matrix at k-point $\mathbf{k}$ — **note the DM index order**: `DM_all[:, pair_j, pair_i]`.

### Density matrix

At half-filling (spin-summed):

$$\rho(\mathbf{k}) = 2 \, V_{\text{occ}}(\mathbf{k}) \, V_{\text{occ}}(\mathbf{k})^\dagger$$

where $V_{\text{occ}}$ contains the $N/2$ lowest eigenvectors of $H(\mathbf{k})$.

### ACSF hopping amplitudes

Each hopping is a linear model over ACSF descriptors:

$$t_p = D_p \cdot \boldsymbol{\theta} = \sum_m \theta^{2b}_m \, T_m(x_{ij}) f_c(r_{ij}) + \sum_m \sum_w \theta^{3b}_{m,w} \, T_m(x_{ij}) \sum_{k \neq j} T_m(x_{ik})\cos^{w+1}(\theta_{ijk})$$

where:
- $T_m(x)$ is the $m$-th Chebyshev polynomial
- $x_{ij} = \dfrac{2r_{ij} - (r_{\text{in}} + r_{\text{cut}})}{r_{\text{cut}} - r_{\text{in}}}$ maps bond lengths to $[-1, 1]$
- $f_c(r) = \dfrac{1}{2}\!\left(\cos\dfrac{\pi r}{r_{\text{cut}}} + 1\right)$ is the cosine cutoff
- $\theta_{ijk}$ is the angle at atom $i$ between bonds $(i,j)$ and $(i,k)$

The three-body sum runs over all **k-leg bonds** $q = (i, k)$ with $k \neq j$ and $r_{ik} < r_{\text{cut}}$.

---

## 2. Hellmann–Feynman Force Derivation

The Hellmann–Feynman theorem holds exactly for the orthogonal tight-binding model (no Pulay forces) because the basis functions are site-local and geometry-independent:

$$\mathbf{F}_I = -\frac{\partial E_{\text{band}}}{\partial \mathbf{R}_I} = -\frac{2}{N_k}\sum_\mathbf{k}\sum_p \text{Re}\!\left[\frac{\partial t_p}{\partial \mathbf{R}_I}\,e^{i\mathbf{k}\cdot\mathbf{r}_p}\,\rho_\mathbf{k}[j_p,i_p] + t_p\,\frac{\partial e^{i\mathbf{k}\cdot\mathbf{r}_p}}{\partial \mathbf{R}_I}\,\rho_\mathbf{k}[j_p,i_p]\right]$$

### Which atoms does bond $p = (i_p, j_p)$ affect?

Each bond $p$ has three classes of atom whose displacement changes the energy:

1. **Atom $j_p$** (the bond target): $\partial \mathbf{r}_p/\partial \mathbf{R}_{j_p} = +1$. Both the hopping and the Bloch phase change.
2. **Atom $i_p$** (the bond centre): $\partial \mathbf{r}_p/\partial \mathbf{R}_{i_p} = -1$, and additionally all k-leg bond vectors $\mathbf{r}_q = \mathbf{R}_k - \mathbf{R}_{i_p}$ change.
3. **Atom $k = j_q$** (a k-leg neighbour of $i_p$): $\partial \mathbf{r}_q/\partial \mathbf{R}_k = +1$, changing $t_p$ via the 3-body angular sum.

### 2.1 J-leg contribution

Define the **bond force** for bond $p$ in direction $\alpha$:

$$\text{bond\_force}[p,\alpha] = \frac{2}{N_k}\sum_\mathbf{k} \text{Re}\!\left[\left(\frac{\partial t_p}{\partial r_{p,\alpha}} + ik_\alpha\,t_p\right) e^{i\mathbf{k}\cdot\mathbf{r}_p}\,\rho_\mathbf{k}[j_p, i_p]\right]$$

Then:

$$\boxed{F_{i_p,\alpha} \mathrel{+}= +\text{bond\_force}[p,\alpha], \qquad F_{j_p,\alpha} \mathrel{+}= -\text{bond\_force}[p,\alpha]}$$

**Sign derivation:**
- Force on $j_p$: $\partial\mathbf{r}_p/\partial\mathbf{R}_{j_p} = +1$, so $-\partial E_p/\partial R_{j_p,\alpha} = -\text{bond\_force}$ ✓
- Force on $i_p$: $\partial\mathbf{r}_p/\partial\mathbf{R}_{i_p} = -1$ flips both the hopping derivative and the phase derivative, giving $+\text{bond\_force}$ ✓

Newton's third law: $F_{i_p} + F_{j_p} = 0$ per bond. ✓

### 2.2 K-leg contribution

For every ordered triplet $(p, q)$ where $p = (i, j)$ is the **j-leg bond** and $q = (i, k)$ is the **k-leg bond** (same centre $i$, different target), the 3-body part of $t_p$ depends on $\mathbf{r}_q$.

Define the **hopping-only kernel** for bond $p$:

$$K_p = \frac{2}{N_k}\sum_\mathbf{k} \text{Re}\!\left[e^{i\mathbf{k}\cdot\mathbf{r}_p}\,\rho_\mathbf{k}[j_p, i_p]\right]$$

**Critical point:** The Bloch phase $e^{i\mathbf{k}\cdot\mathbf{r}_p}$ depends on $\mathbf{r}_p$ only — not on $\mathbf{r}_q$ — so $\partial e^{i\mathbf{k}\cdot\mathbf{r}_p}/\partial\mathbf{R}_k = 0$. There is **no** $ik_\alpha t_p$ term in $K_p$.

The k-leg force contributions are:

$$\boxed{F_{k,\alpha} \mathrel{+}= -K_p \cdot \frac{\partial t_p}{\partial r_{q,\alpha}}, \qquad F_{i,\alpha} \mathrel{+}= +K_p \cdot \frac{\partial t_p}{\partial r_{q,\alpha}}}$$

Newton's third law: $F_k + F_i = 0$ per triplet. ✓

**Why the triplet list must be ordered:** `_build_triplet_indices` returns all ordered pairs $(p, q)$ **and** $(q, p)$ for every unordered pair of bonds from the same centre. This means:
- Triplet $(p, q)$: atom $k = j_q$ gets a k-leg force; atom $i$ gets the reaction.
- Triplet $(q, p)$: atom $j = j_p$ gets a k-leg force (from the triplet where $p$ is the k-leg of $q$); atom $i$ gets the reaction.

This distributes forces correctly to all three atoms $(i, j, k)$ in every angular triplet.

---

## 3. Hopping Gradient Derivation

### 3.1 Descriptor basis

For bond $p = (i,j)$, define the scalar Chebyshev-cutoff product and its radial derivative:

$$\phi_m(r) = T_m(x) \, f_c(r), \qquad \frac{d\phi_m}{dr} = \frac{dT_m}{dx}\frac{dx}{dr}f_c(r) + T_m(x)\frac{df_c}{dr}$$

with $\dfrac{dx}{dr} = \dfrac{2}{r_{\text{cut}} - r_{\text{in}}}$ and $\dfrac{df_c}{dr} = -\dfrac{\pi}{2r_{\text{cut}}}\sin\!\dfrac{\pi r}{r_{\text{cut}}}$.

### 3.2 J-leg gradient $\partial t_p / \partial r_{p,\alpha}$

Differentiating $t_p$ w.r.t. the j-leg bond vector $\mathbf{r}_p = \mathbf{R}_j - \mathbf{R}_i$:

**Two-body part** (depends only on $r_p = |\mathbf{r}_p|$):

$$\frac{\partial t_p^{2b}}{\partial r_{p,\alpha}} = \hat{r}_{p,\alpha} \sum_m \theta^{2b}_m \frac{d\phi_m}{dr_p}$$

**Three-body part — Part 1** (radial change of the j-leg Chebyshev $T_m(x_{ij})$):

$$\left.\frac{\partial t_p^{3b}}{\partial r_{p,\alpha}}\right|_{\text{Part 1}} = \hat{r}_{p,\alpha} \sum_m \frac{d\phi_m(r_p)}{dr_p} \sum_w \theta^{3b}_{m,w}\,\phi_m(r_q)\cos^{w+1}(\theta)$$

**Three-body part — Part 2** (angular change of $\cos\theta$ through $\mathbf{r}_p$):

$$\frac{\partial \cos\theta_{ijk}}{\partial r_{p,\alpha}} = \frac{\hat{r}_{q,\alpha} - \cos\theta\,\hat{r}_{p,\alpha}}{r_p}$$

$$\left.\frac{\partial t_p^{3b}}{\partial r_{p,\alpha}}\right|_{\text{Part 2}} = \frac{\partial \cos\theta}{\partial r_{p,\alpha}} \cdot \underbrace{\sum_m \phi_m(r_p)\,\phi_m(r_q) \sum_w \theta^{3b}_{m,w}(w+1)\cos^w(\theta)}_{\text{shared angular-derivative scalar } S}$$

### 3.3 K-leg gradient $\partial t_p / \partial r_{q,\alpha}$

Differentiating the 3-body part of $t_p$ w.r.t. the k-leg bond vector $\mathbf{r}_q = \mathbf{R}_k - \mathbf{R}_i$:

**The angular-derivative scalar $S$ is identical** to Part 2 of the j-leg (same expression, summed over $m$ and $w$).

**The direction vector is swapped** relative to the j-leg:

$$\frac{\partial \cos\theta_{ijk}}{\partial r_{q,\alpha}} = \frac{\hat{r}_{p,\alpha} - \cos\theta\,\hat{r}_{q,\alpha}}{r_q}$$

Note the swap: $r_p \leftrightarrow r_q$ compared to the j-leg formula. The full k-leg gradient is:

$$\frac{\partial t_p}{\partial r_{q,\alpha}} = \underbrace{\hat{r}_{q,\alpha} \sum_m \frac{d\phi_m(r_q)}{dr_q} \sum_w \theta^{3b}_{m,w}\,\phi_m(r_p)\cos^{w+1}(\theta)}_{\text{Part 1: radial deriv. of }T_m(x_{ik})} + \underbrace{\frac{\partial\cos\theta}{\partial r_{q,\alpha}} \cdot S}_{\text{Part 2: angle via }r_q}$$

**Summary of asymmetry between j-leg and k-leg:**

| Component | J-leg | K-leg |
|-----------|-------|-------|
| Part 1 radial derivative | $d\phi_m(r_p)/dr_p$ | $d\phi_m(r_q)/dr_q$ |
| Part 1 angular factor | $\phi_m(r_q) \cdot \cos^{w+1}(\theta)$ | $\phi_m(r_p) \cdot \cos^{w+1}(\theta)$ |
| Part 1 direction | $\hat{r}_{p,\alpha}$ | $\hat{r}_{q,\alpha}$ |
| Part 2 scalar $S$ | same | same |
| Part 2 direction | $(\hat{r}_q - \cos\theta\,\hat{r}_p)/r_p$ | $(\hat{r}_p - \cos\theta\,\hat{r}_q)/r_q$ |

---

## 4. Implementation

### `_acsf_hopping_gradient_from_pairs` — returns 4 values

```python
grad_t, kleg_t_p, kleg_t_q, kleg_grad = _acsf_hopping_gradient_from_pairs(
    M, W, r_cut, tb_params, pair_i, pair_j, pair_v, N
)
```

| Output | Shape | Meaning |
|--------|-------|---------|
| `grad_t` | `(n_pairs, 3)` | $\partial t_p / \partial r_{p,\alpha}$ — j-leg gradient |
| `kleg_t_p` | `(n_triplets,)` | Bond index of the j-leg bond in each triplet |
| `kleg_t_q` | `(n_triplets,)` | Bond index of the k-leg bond in each triplet |
| `kleg_grad` | `(n_triplets, 3)` | $\partial t_{p} / \partial r_{q,\alpha}$ — k-leg gradient |

### `_compute_band_forces` — j-leg + k-leg

```python
F_band = _compute_band_forces(
    pair_i, pair_j, pair_v, t_ij, grad_t, DM_all, kpoints, N,
    kleg_t_p=kleg_t_p, kleg_t_q=kleg_t_q, grad_kleg=kleg_grad,
)
```

#### J-leg loop (vectorised over k-points):

```python
phases = exp(1j * kpoints @ pair_v.T)           # (n_kp, n_pairs)
DM_nm  = DM_all[:, pair_j, pair_i]              # (n_kp, n_pairs)  ← j before i

kernel     = (grad_t[alpha] + 1j * k_alpha * t_ij) * phases * DM_nm
bond_force = 2 * Re(sum(kernel, axis=0)) / n_kp

F[pair_i, alpha] += +bond_force
F[pair_j, alpha] += -bond_force
```

**DM index order:** `DM_all[:, pair_j, pair_i]` uses the $(j, i)$ entry because the Hamiltonian was built as `H[pair_i, pair_j] = t * phase`, and $\text{Tr}(H\rho) = \sum_{m,n}H[m,n]\rho[n,m]$ requires `DM[pair_j, pair_i]`. Using the reversed order `DM_all[:, pair_i, pair_j]` gives the complex conjugate and produces direction-dependent force errors.

#### K-leg block:

```python
phases_p  = exp(1j * kpoints @ pair_v[kleg_t_p].T)        # uses r_p (j-leg bond)
DM_nm_p   = DM_all[:, pair_j[kleg_t_p], pair_i[kleg_t_p]] # DM[j_p, i_p]
kleg_kern = 2 * Re(sum(phases_p * DM_nm_p, axis=0)) / n_kp # no ik*t term!

F[pair_j[kleg_t_q], alpha] += -kleg_kern * kleg_grad[:, alpha]  # k-atom
F[pair_i[kleg_t_p], alpha] += +kleg_kern * kleg_grad[:, alpha]  # centre i
```

---

## 5. Testing Strategy

### Test 1: Band energy consistency (`TestTBEnergy`)

Verify the trace formula equals the eigenvalue sum at half-filling:

$$\frac{1}{N_k}\sum_\mathbf{k} \text{Re}\!\left[\text{Tr}\!\left(H_\mathbf{k}\,\rho_\mathbf{k}\right)\right] = \frac{2}{N_k}\sum_\mathbf{k}\sum_{n=1}^{N/2} \varepsilon_{n,\mathbf{k}}$$

Both expressions should agree to machine precision. Tested at the $\Gamma$-point and on a $3\times3$ Monkhorst-Pack mesh.

### Test 2: Hellmann–Feynman forces vs finite differences (`TestHellmannFeynmanForces`)

For each atom $I$ and direction $\alpha \in \{x, y\}$, compare:

$$F_{I,\alpha}^{\text{analytic}} \approx -\frac{E(\mathbf{R}_I + \delta\hat{e}_\alpha) - E(\mathbf{R}_I - \delta\hat{e}_\alpha)}{2\delta}$$

using $\delta = 10^{-4}$ Å. Tolerance: $\max_I |F_{I,\alpha}^{\text{analytic}} - F_{I,\alpha}^{\text{FD}}| < 5\times10^{-4}$ eV/Å.

**Force conservation:** also verify $\sum_I \mathbf{F}_I^{\text{analytic}} \approx \mathbf{0}$ (tolerance $10^{-10}$ eV/Å).

### Test 3: Hopping gradients vs finite differences (`TestHoppingGradient`)

**J-leg gradient** — for each bond $p = (i, j)$, perturb atom $j$ (the bond target):

$$\frac{\partial t_p}{\partial r_{p,\alpha}} \approx \frac{t_p(\mathbf{R}_{j} + \delta\hat{e}_\alpha) - t_p(\mathbf{R}_{j} - \delta\hat{e}_\alpha)}{2\delta}$$

Moving atom $j$ changes only $\mathbf{r}_p$, not any k-leg bond vector $\mathbf{r}_q$. Tolerance: $< 10^{-6}$.

**K-leg gradient** — for each triplet $(p, q)$, perturb atom $k = j_q$ (the k-leg target):

$$\frac{\partial t_p}{\partial r_{q,\alpha}} \approx \frac{t_p(\mathbf{R}_k + \delta\hat{e}_\alpha) - t_p(\mathbf{R}_k - \delta\hat{e}_\alpha)}{2\delta}$$

Moving atom $k$ changes only $\mathbf{r}_q = \mathbf{R}_k - \mathbf{R}_i$. The FD derivative of $t_p$ (not $t_q$) isolates the k-leg gradient exactly. Tolerance: $< 10^{-6}$.

---

## 6. Critical Implementation Pitfalls

### 6.1 DM index order

The Hamiltonian is built as `H[pair_i, pair_j] += t * phase`. Therefore:

$$\text{Tr}(H\,\rho) = \sum_{m,n} H[m,n]\,\rho[n,m]$$

Bond $p$ contributes $H[i_p, j_p]\,\rho[j_p, i_p]$, so the correct index is `DM_all[:, pair_j, pair_i]` (j before i). The wrong order gives the complex conjugate $\rho[i_p, j_p] = \rho[j_p, i_p]^*$, which has a flipped imaginary part and produces direction-dependent force errors (e.g., the $x$-force is wrong while the $y$-force may pass by coincidence).

### 6.2 No $ik_\alpha t_p$ term in the k-leg kernel

The Bloch phase in the energy is $e^{i\mathbf{k}\cdot\mathbf{r}_p}$, which depends on $\mathbf{r}_p$ only. When differentiating w.r.t. $\mathbf{R}_k$ (the k-leg atom), this phase is constant:

$$\frac{\partial}{\partial R_{k,\alpha}} e^{i\mathbf{k}\cdot\mathbf{r}_p} = 0 \quad \text{(since } \mathbf{r}_p = \mathbf{R}_j - \mathbf{R}_i \text{ does not involve } \mathbf{R}_k\text{)}$$

The k-leg kernel $K_p$ therefore contains only $e^{i\mathbf{k}\cdot\mathbf{r}_p}$ — no $ik_\alpha t_p$ factor.

### 6.3 J-leg force signs

$$F_{i_p,\alpha} = +\text{bond\_force}, \qquad F_{j_p,\alpha} = -\text{bond\_force}$$

Derivation:
- $j_p$: $\partial\mathbf{r}_p/\partial\mathbf{R}_{j_p} = +1$, so $\mathbf{F}_{j_p} = -\partial E/\partial \mathbf{R}_{j_p} = -\text{bond\_force}$
- $i_p$: $\partial\mathbf{r}_p/\partial\mathbf{R}_{i_p} = -1$ flips both the $\partial t_p/\partial r_p$ and the $ik_\alpha t_p$ contributions, so $\mathbf{F}_{i_p} = +\text{bond\_force}$

### 6.4 K-leg angle derivative: direction vectors are swapped

$$\frac{\partial\cos\theta}{\partial \mathbf{r}_p} = \frac{\hat{\mathbf{r}}_q - \cos\theta\,\hat{\mathbf{r}}_p}{r_p}, \qquad \frac{\partial\cos\theta}{\partial \mathbf{r}_q} = \frac{\hat{\mathbf{r}}_p - \cos\theta\,\hat{\mathbf{r}}_q}{r_q}$$

The unit vectors are exchanged and the denominator uses the respective bond length. The angular-derivative **scalar** $S$ is identical for both legs.

### 6.5 Periodic boundary pitfall in finite-difference tests

Placing an atom at $x \approx 0$ (the cell boundary) means a $-\delta$ perturbation wraps it to $x \approx L$, changing the neighbour list discontinuously. This makes $E(\mathbf{R}_I - \delta\hat{e}_x)$ represent a fundamentally different structure and produces FD force errors of order $10^2$–$10^3$ eV/Å.

**Fix:** use $(i + 0.5)\,a$ as the base coordinate so atoms are at least $a/2$ from any boundary:

```python
x = (i + 0.5) * a + rng.normal(0.0, perturb)
y = (j + 0.5) * a + rng.normal(0.0, perturb)
```

The failure mode is insidious: all other atoms may pass the test perfectly while only the boundary atom fails with a huge error, because the topology change is local.

### 6.6 Newton's third law as a sanity check

Force conservation is guaranteed analytically:

- **J-leg:** bond force appears with $+$ on $i_p$ and $-$ on $j_p$ → pairwise cancellation.
- **K-leg:** k-leg force appears with $-$ on $k$ and $+$ on $i$ → pairwise cancellation.

Therefore `np.sum(F_band, axis=0)` should vanish to floating-point precision ($\sim 10^{-15}$ eV/Å). This is a cheap, fast sanity check that catches sign errors and missing Newton's-third-law partners.

---

## 7. Summary of Force Contributions

| Source | Atoms affected | Formula |
|--------|---------------|---------|
| J-leg hopping | $i_p$, $j_p$ | $\displaystyle\pm\frac{2}{N_k}\sum_\mathbf{k}\text{Re}\!\left[\frac{\partial t_p}{\partial r_{p,\alpha}}\,e^{i\mathbf{k}\cdot\mathbf{r}_p}\,\rho_\mathbf{k}[j_p,i_p]\right]$ |
| Bloch phase | $i_p$, $j_p$ | $\displaystyle\pm\frac{2}{N_k}\sum_\mathbf{k}\text{Re}\!\left[ik_\alpha t_p\,e^{i\mathbf{k}\cdot\mathbf{r}_p}\,\rho_\mathbf{k}[j_p,i_p]\right]$ |
| K-leg 3-body | centre $i$, k-atom $k$ | $\displaystyle\pm K_p \cdot \frac{\partial t_p}{\partial r_{q,\alpha}}$ |

The j-leg and Bloch-phase contributions are combined into a single `bond_force` per bond. The k-leg contribution uses the separate hopping-only kernel $K_p$ with no phase-derivative term.