# Gaussian fPEPS with larger unit cells: construction, unified loss function and block-diagonalized Gaussian map

*July 2026*

## Abstract

These notes document the extension of the GfPEPS translation scheme from trivial $(1\times 1)$ unit cells to arbitrary rectangular $(L_x \times L_y)$ unit cells, as implemented in `GfPEPS.jl`. We review the construction of the unit-cell covariance matrix $\Gamma_{\mathrm{uc}}$ following Hackenbroich *et al.* [Phys. Rev. B **101**, 115134 (2020)], define the unified energy loss function used for both BCS and Kitaev-type Hamiltonians, and derive the *block-diagonalization* of the Gaussian map obtained from a smart ordering of the virtual modes: all momentum dependence is confined to the virtual modes on bonds that wrap around the unit-cell boundary, so the intra-cell modes can be eliminated once per loss evaluation by a $k$-independent Schur complement. This reduces the per-momentum inversion cost from $\mathcal{O}\big((8\Lambda L_x L_y)^3\big)$ to $\mathcal{O}\big((4\Lambda (L_x{+}L_y))^3\big)$.

## 1. Setup and conventions

Each lattice site $i$ carries $N_f$ physical fermions and $4\Lambda$ virtual fermions ($\Lambda$ flavors on each of the four bonds $l,r,u,d$), described by the local fiducial state $|Q_i\rangle$ with covariance matrix (CM)

$$
(\Gamma_Q)_{\mu\nu} = \tfrac{i}{2}\,
\langle Q | [c_\mu, c_\nu] | Q \rangle ,
\qquad
\Gamma_Q =
\begin{pmatrix} A & B \\ -B^{T} & D \end{pmatrix},
\tag{1}
$$

where $A \in \mathbb{R}^{2N_f \times 2N_f}$ acts on the physical Majorana modes, $D \in \mathbb{R}^{8\Lambda \times 8\Lambda}$ on the virtual ones, and $B$ couples the two sectors. The virtual Majorana modes of each site are ordered as

$$
\big(c^{(1)}_{l_1}, c^{(2)}_{l_1}, c^{(1)}_{r_1}, c^{(2)}_{r_1},
\dots,
c^{(1)}_{u_1}, c^{(2)}_{u_1}, c^{(1)}_{d_1}, c^{(2)}_{d_1}, \dots\big),
\tag{2}
$$

i.e. first the $2\Lambda$ pairs of left/right modes, then the $2\Lambda$ pairs of up/down modes. The physical state $|\Psi\rangle$ is obtained by projecting all virtual bonds onto maximally entangled states $|\omega_{ij}\rangle$; on the level of CMs this is the Gaussian map

$$
G^{(\Psi)}_{\bm k}
= B \big( D + G^{(\omega)}_{\bm k} \big)^{-1} B^{T} + A ,
\tag{3}
$$

with $G^{(\omega)}_{\bm k}$ the Fourier-transformed CM of the maximally entangled virtual bonds.

## 2. Construction for an $L_x \times L_y$ unit cell

For a translation-invariant state with an $L_x \times L_y$ unit cell we assign one local fiducial state $|Q_s\rangle$ to every site $s = i_x + (i_y - 1) L_x$ of the unit cell (sites related by the unit-cell layout may share the same variational tensor). The fiducial state of the whole unit cell is the product state of the per-site fiducial states, so its CM is block diagonal,

$$
A_{\mathrm{uc}} = \bigoplus_{s=1}^{L_x L_y} A_s , \qquad
B_{\mathrm{uc}} = \bigoplus_{s=1}^{L_x L_y} B_s , \qquad
D_{\mathrm{uc}} = \bigoplus_{s=1}^{L_x L_y} D_s ,
\tag{4}
$$

with the physical modes of all sites listed first (site-major ordering) and the virtual modes of all sites following in the same site-major order with the per-site ordering (2). Each per-site block is parametrized by an orthogonal matrix $X_s$ through the canonical form $\Gamma_s = X_s^{T} \big(\bigoplus_{j} i\sigma_y\big) X_s$, exactly as in the $1\times 1$ case, and the $X_s$ of all *distinct* sites of the unit cell are the variational parameters.

The virtual-bond CM factorizes into one term per nearest-neighbor bond. A horizontal bond couples the $r$ modes of site $(i_x, i_y)$ to the $l$ modes of site $(i_x{+}1, i_y)$; for $i_x < L_x$ both sites lie in the same unit cell and the bond contributes the *momentum-independent* block

$$
G^{(\omega)}\big[l_{(i_x+1,i_y)}, r_{(i_x,i_y)}\big] = -\sigma_x ,
\qquad
G^{(\omega)}\big[r_{(i_x,i_y)}, l_{(i_x+1,i_y)}\big] = +\sigma_x ,
$$

per flavor, whereas the wrapping bond $i_x = L_x$ connects to the neighboring unit cell and acquires the phase of one unit-cell translation,

$$
G^{(\omega)}_{\bm k}\big[l_{(1,i_y)}, r_{(L_x,i_y)}\big]
= -e^{i k_x L_x}\,\sigma_x ,
\qquad
G^{(\omega)}_{\bm k}\big[r_{(L_x,i_y)}, l_{(1,i_y)}\big]
= +e^{-i k_x L_x}\,\sigma_x ,
\tag{5}
$$

and analogously for vertical bonds with $e^{\pm i k_y L_y}$. For $L_x = L_y = 1$ this reduces to Eq. (C7) of *Hackenbroich et al.* The allowed momenta live in the folded Brillouin zone $k_\alpha \in [-\pi/L_\alpha, \pi/L_\alpha)$.

With (4) and (5), the physical CM at momentum $\bm k$ is again given by the Gaussian map (3), now of dimension $2 N_f L_x L_y$, obtained by inverting the $8\Lambda L_x L_y$-dimensional matrix $D_{\mathrm{uc}} + G^{(\omega)}_{\bm k}$ for every $\bm k$.

## 3. Unified loss function

The energy of any quadratic (BdG) Hamiltonian — including the Kitaev honeycomb model in the vortex-free sector, which maps onto a $N_f = 1$ BCS-type Hamiltonian on the square lattice — can be evaluated directly from $G^{(\Psi)}_{\bm k}$. Writing the unit-cell Bloch Hamiltonian in the Majorana basis as $h_{\bm k} = \tfrac{i}{2}\, \Omega H^{\mathrm{BdG}}_{\bm k} \Omega^\dagger$, the energy per unit cell is

$$
E = \frac{1}{N_k} \sum_{\bm k} \Big[
    \tfrac12 \operatorname{tr} \xi_{\bm k}
    - \tfrac14 \operatorname{tr}\!\big( h_{\bm k}\, G^{(\Psi)\,T}_{\bm k} \big)
    + E_{\mathrm{shift}}(\bm k) \Big],
\tag{6}
$$

which is a single trace formula independent of the specific pairing or hopping structure: BCS ($N_f = 2$) and Kitaev ($N_f = 1$, with $E_{\mathrm{shift}} = -J_z L_x L_y$ accounting for the $\mathbb{Z}_2$ background gauge field). Hamiltonians only differ in the matrices $\xi_{\bm k}$, $\Delta_{\bm k}$ entering $H^{\mathrm{BdG}}_{\bm k}$. This realizes the unified loss function: the same code path `energy_loss_X` optimizes any `MomentumSpaceBdGHamiltonian` on any rectangular unit cell.

## 4. Block-diagonalization via smart mode ordering

The naive evaluation of (3) requires factorizing the $d \times d$ matrix $D_{\mathrm{uc}} + G^{(\omega)}_{\bm k}$ with $d = 8\Lambda L_x L_y$ *for every momentum* $\bm k$, which dominated the cost of previous attempts at larger unit cells. The key observation is that the momentum dependence of $G^{(\omega)}_{\bm k}$ in (5) is carried *only* by the wrapping bonds. We therefore reorder the virtual Majorana modes into two groups:

- $\partial$ (**boundary**): $l$-modes of column $1$, $r$-modes of column $L_x$, $u$-modes of row $1$, $d$-modes of row $L_y$
- $\mathrm{in}$ (**inner**): all remaining virtual modes

with $d_\partial = 4\Lambda(L_x + L_y)$ and $d_{\mathrm{in}} = d - d_\partial$. In this ordering,

$$
D_{\mathrm{uc}} + G^{(\omega)}_{\bm k}
=
\begin{pmatrix}
  M_{\mathrm{in}\,\mathrm{in}} & D_{\mathrm{in}\,\partial} \\
  -D_{\mathrm{in}\,\partial}^{T} & D_{\partial\partial} + G^{\mathrm{wrap}}_{\bm k}
\end{pmatrix},
\qquad
M_{\mathrm{in}\,\mathrm{in}} = D_{\mathrm{in}\,\mathrm{in}} + G^{\mathrm{intra}}_{\mathrm{in}\,\mathrm{in}},
\tag{7}
$$

where $G^{\mathrm{intra}}$ collects the ($k$-independent) intra-cell bonds and $G^{\mathrm{wrap}}_{\bm k}$ the wrapping bonds (5). Crucially, $M_{\mathrm{in}\,\mathrm{in}}$, $D_{\mathrm{in}\,\partial}$ and $D_{\partial\partial}$ are all independent of $\bm k$: intra-cell bonds never couple to boundary modes, and $D_{\mathrm{uc}}$ couples modes within one site only.

Eliminating the inner modes by the Schur complement of (7) yields *effective* unit-cell blocks

$$
\begin{aligned}
  A_{\mathrm{eff}} &= A_{\mathrm{uc}} + B_{\mathrm{in}}\, M_{\mathrm{in}\,\mathrm{in}}^{-1} B_{\mathrm{in}}^{T} ,\\
  B_{\mathrm{eff}} &= B_\partial - B_{\mathrm{in}}\, M_{\mathrm{in}\,\mathrm{in}}^{-1} D_{\mathrm{in}\,\partial} ,\\
  D_{\mathrm{eff}} &= D_{\partial\partial} + D_{\mathrm{in}\,\partial}^{T}\, M_{\mathrm{in}\,\mathrm{in}}^{-1} D_{\mathrm{in}\,\partial} ,
\end{aligned}
\tag{8}
$$

such that for every momentum

$$
G^{(\Psi)}_{\bm k}
= B_{\mathrm{eff}} \big( D_{\mathrm{eff}} + G^{\mathrm{wrap}}_{\bm k} \big)^{-1}
  B_{\mathrm{eff}}^{T} + A_{\mathrm{eff}} .
\tag{9}
$$

Both $A_{\mathrm{eff}}$ and $D_{\mathrm{eff}}$ are real antisymmetric (using $M_{\mathrm{in}\,\mathrm{in}}^{-T} = -M_{\mathrm{in}\,\mathrm{in}}^{-1}$), so (9) has exactly the form (3): the contraction of the intra-cell bonds is itself a Gaussian map, and $(A_{\mathrm{eff}}, B_{\mathrm{eff}}, D_{\mathrm{eff}})$ are precisely the blocks of the CM $\Gamma_{\mathrm{uc}}$ of the *unit-cell fiducial state* obtained by contracting the virtual bonds inside the unit cell. Physically, (8) constructs $\Gamma_{\mathrm{uc}}$ from the per-site $\Gamma_s$, (9) then contracts only the bonds crossing the unit-cell boundary.

**Complexity.** Per loss evaluation, the elimination (8) costs one $\mathcal{O}(d_{\mathrm{in}}^3)$ factorization *independent of the number of momenta* $N_k$, after which each momentum requires only an $\mathcal{O}(d_\partial^3)$ solve. The per-momentum cost improves by a factor $\big(d / d_\partial\big)^3 = \big(2 L_x L_y / (L_x + L_y)\big)^3$: for a $2\times 2$ cell a factor $8$, for a $4 \times 4$ cell a factor $64$. The memory footprint of the precomputed bond CMs shrinks by $\big(d/d_\partial\big)^2$ as only the wrap-bond blocks are stored per momentum. For $L_x = L_y = 1$ every virtual mode is a boundary mode, the inner block is empty, and (9) reduces exactly to the previous implementation, so the trivial unit cell suffers no overhead.

**Gradients.** The elimination (8) consists of dense solves and products and is differentiated by automatic differentiation (Zygote) without custom rules; it is executed once per loss evaluation. The momentum loop (9) reuses the existing hand-written reverse rule of the Gaussian map, which stores the LU factorizations of $D_{\mathrm{eff}} + G^{\mathrm{wrap}}_{\bm k}$ from the forward pass and is threaded over momenta.
