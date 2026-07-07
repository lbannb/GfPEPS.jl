<img width="2000" height="400" alt="GfPEPS_with_Gutzwiller_projector" src="https://github.com/user-attachments/assets/a2d804b4-15df-434e-a74d-bc40ea559f52" />

# GfPEPS.jl
A julia package for creating **Gaussian fermionic Projected Entangled Pair States (GfPEPS)**, built on top of the [TensorKit.jl](https://github.com/QuantumKitHub/TensorKit.jl) framework.

## 🎯 When to use GfPEPS.jl

GfPEPS are approximations to the ground states and thermal states of fermionic quadratic Hamiltonians.
Using the parton construction 


This package enables the construction of these states which can be used in the following ways:

<p align="center">
<img width="400" height="400" alt="general_workflow_GfPEPS" src="https://github.com/user-attachments/assets/232391b4-d986-40e9-9f4d-a616af69a796" /> 
</p>

* Computional speedup for ground state search algorithms by using these states as initial states
* Comparison of mean-field ansätze resulting in an understanding of the underlying physics of the model
* Simulating models of free fermions

## 📖 Literature
The implementation of this package is based on the construction schemes in the following papers:

* Hackenbroich, A., Bernevig, B. A., Schuch, N. & Regnault, N. (2020) [Phys. Rev. B 101, 115134](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.101.115134)
* Mortier, Q., Schuch, N., Verstraete, F. & Haegeman, J. (2022) [Phys. Rev. Lett. 129, 206401](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.206401)
* Yang, Q., Zhang, X.-Y., Liao, H.-J., Tu, H.-H. & Wang, L. (2023) [Phys. Rev. B 107, 125128](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.125128)

## 🚀 Quick start

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

See `docs/larger_unit_cells.tex` for the construction scheme behind larger unit cells and
the block-diagonalized Gaussian map used to make them fast.
