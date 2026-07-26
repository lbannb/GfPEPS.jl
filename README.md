# GfPEPS.jl
A Julia package for constructing **Gaussian fermionic Projected Entangled Pair States (GfPEPS)**, built on top of the [TensorKit.jl](https://github.com/QuantumKitHub/TensorKit.jl) framework.

In the parton construction framework, local microscopic degrees of freedom (e.g., spins) are represented by bosonic or fermionic partons within an enlarged Hilbert space. 
Physical states are recovered by enforcing a local constraint, typically via Gutzwiller projection.

This approach enables the formulation of mean-field ansätze that are highly non-trivial in terms of the microscopic spins and allow for the description of exotic phenomena such as fractionalized excitations.
The ground- and thermal states of such mean-field Hamiltonians $H_\text{MF}$ (fermionic quadratic Hamiltonians) are known as fermionic Gaussian states.
These states can be completely described by their covariance matrix allowing for a very efficient implementation.

## Getting started

```julia
using Pkg; Pkg.add("GfPEPS")
```

## When to use GfPEPS.jl

This package enables the construction of these states in 2D systems, which can be used in the following ways:

<p align="center">
<img width="1580" height="795" alt="general_workflow_GfPEPS" src="https://github.com/user-attachments/assets/6379ba1d-23fa-44fd-8bc0-a7ba7093fe6a" />
</p>

* Accelerating ground-state search algorithms by using the ground states of mean-field Hamiltonians $\ket{\Psi_\text{trial}}$ as initial seeds for PEPS optimization
* Comparing mean-field ansätze to study the underlying physics of strongly correlated quantum systems
* Simulating models of free fermions in 2D systems

## Literature
The implementation of this package is based on the construction schemes in the following papers:

* Hackenbroich, A., Bernevig, B. A., Schuch, N. & Regnault, N. (2020) [Phys. Rev. B 101, 115134](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.101.115134)
* Mortier, Q., Schuch, N., Verstraete, F. & Haegeman, J. (2022) [Phys. Rev. Lett. 129, 206401](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.206401)
* Yang, Q., Zhang, X.-Y., Liao, H.-J., Tu, H.-H. & Wang, L. (2023) [Phys. Rev. B 107, 125128](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.125128)

## Quick start

The single entry point is the `Gaussian_fPEPS` constructor: it optimizes the covariance
matrix of the fiducial state for a given quadratic Hamiltonian and translates it into an
iPEPS (PEPSKit.jl format).

```julia
using GfPEPS

# infinite square lattice with a trivial (1x1) unit cell and a 24x24 momentum grid
lattice = InfiniteRectLattice(1, 1; N_kx=24, N_ky=24, bc=(:PBC, :APBC))

# d-wave BCS Hamiltonian (Nf = 2 spinful fermions per site)
t, Δ, μ = 1.0, 1.0, 1.0
hopping = Dict(1 => Dict((1,0) => t, (-1,0) => t, (0,1) => t,  (0,-1) => t))
pairing = Dict(1 => Dict((1,0) => Δ, (-1,0) => Δ, (0,1) => -Δ, (0,-1) => -Δ))
H_BdG = default_BCS_hamiltonian(hopping, pairing, μ, lattice)

# Λ = 2 virtual fermion flavors per bond (bond dimension D = 2^Λ = 4)
Ψ = Gaussian_fPEPS(2, 2, lattice, H_BdG)

Ψ.optim_energy  # variational energy from the covariance matrix
Ψ.exact_energy  # exact BCS ground-state energy
Ψ.peps          # InfinitePEPS, e.g. for CTMRG with PEPSKit.jl
```

Larger unit cells work the same way — pass the unit-cell size and layout, and give each
distinct site its own couplings (the Kitaev honeycomb model is available via
`kitaev_BCS_hamiltonian`, which brings it into BCS form):

```julia
# 2x2 unit cell with two sublattices in a checkerboard pattern
lattice = InfiniteRectLattice(2, 2; N_kx=12, N_ky=12, bc=(:PBC, :APBC), uc_layout=[1 2; 2 1])
hopping = Dict(s => Dict((1,0) => t, (-1,0) => t, (0,1) => t, (0,-1) => t) for s in 1:2)
pairing = Dict(s => Dict((1,0) => Δ, (-1,0) => Δ, (0,1) => -Δ, (0,-1) => -Δ) for s in 1:2)
H_BdG = default_BCS_hamiltonian(hopping, pairing, μ, lattice)

Ψ = Gaussian_fPEPS(2, 2, lattice, H_BdG)  # Ψ.X_opt: one orthogonal matrix per distinct site
```

To enforce a target hole density, pass `DopingSettings`:

```julia
Ψ = Gaussian_fPEPS(2, 2, lattice, H_BdG; doping_kwargs=DopingSettings(; δ=0.16, enforce_density=true))
```

## Example - Kitaev Honeycomb model

The Kitaev honeycomb model is a prominent example of an exactly solvable spin model in two dimensions.
It describes spin-1/2 degrees of freedom located on the sites of a honeycomb lattice and hosts a QSL as its ground state, featuring topological order and fractionalized excitations. The fact that this model is exactly solvable makes it a perfect candidate to benchmark our `GfPEPS.jl` implementation.

The groundstate admits an exact representation via Gutzwiller-projected parton wave functions and the best GfPEPS with fixed finite bond dimension is found within a few steps, much faster and accurate than via random initializations:

<img width="800" height="600" alt="compareStartStates_Kitaev" src="https://github.com/user-attachments/assets/56f0165e-ca4e-44b6-b153-5c01dd06b4d7" />

_Note: This plot was generated using a 1 x 1 unit cell, using Λ=2, χ_E=48 and N_k=96 k-points in both directions._

The phase diagram from the optimized CM of the local fiducial states can be determined using the quasiparticle excitation energies $E_k$ from
```math
    \langle H \rangle = \frac{1}{2} \sum_\mathbf{k} E_\mathbf{k} - J_z N.
```

Overall, the characteristic features of the phase diagram are successfully reproduced.
The gapless phase, which emerges when the triangle inequalities 
```math
    |J_x| \leq |J_y| + |J_z|, \quad |J_y| \leq |J_x| + |J_z|, \quad |J_z| \leq |J_x| + |J_y|
```
are satisfied, can clearly be identified by the dark red regions where $E_{\mathbf{k}} = 0$ for some $\mathbf{k}$.
The effect of an insufficiently large Brillouin zone becomes apparent, as it does not contain the momentum values for which $E_{\mathbf{k}} = 0$. 
This is reflected in the symmetric pattern of the brighter regions within the gapless phase. 
Increasing the Brillouin zone, reduces this finite-size artifact. 
The remaining outliers can likely be attributed either to insufficient convergence of the covariance matrix or to an insufficient satisfaction of $\det(D - G_{\mathbf{k}}^{(\omega)}) = 0$ for certain momentum values, coming from the random initialization of the covariance matrix.


## Example - $t-J$ model

The construction can also be employed to the $t-J$ model at finite doping, which is of interest for explaining superconducting phenomena.
Using a suitable mean-field ansatz as an initial state for the variational iPEPS significantly reduces the number of optimization steps required to match, and actually beat the energy of random initializations.
The latter, indeed, easily gets stuck in local minima, while seeding the simulation with a parton-GfPEPS seems to allow the system to escape such traps and to converge to a lower-energy ground state. Moreover, the comparison between different seeds immediately hints at the most probable underlying symmetry of the state, without the need for expensive analysis.

<img width="800" height="600" alt="compareStartStates_tJ" src="https://github.com/user-attachments/assets/3c57332c-da77-4020-9486-531fe9af526f" />

_Note: This plot was generated using a 5 x 2 unit cell at a doping level per unit cell of δ=0.20, using Λ=2, χ_E=40 and N_k=96 k-points in both directions. The d+s-wave ansatz were initialized with different doping layouts but same mean doping level per unit cell._
