# GfPEPS.jl
A Julia package for constructing **Gaussian fermionic Projected Entangled Pair States (GfPEPS)**, built on top of the [TensorKit.jl](https://github.com/QuantumKitHub/TensorKit.jl) framework.

In the parton construction framework, local microscopic degrees of freedom (e.g., spins) are represented by bosonic or fermionic partons within an enlarged Hilbert space. 
Physical states are recovered by enforcing a local constraint, typically via Gutzwiller projection.

This approach enables the formulation of mean-field ansätze that are highly non-trivial in terms of the microscopic spins and allow for the description of exotic phenomena such as fractionalized excitations.
The ground- and thermal states of such mean-field Hamiltonians $H_\text{MF}$ (fermionic quadratic Hamiltonians) are known as fermionic Gaussian states.
These states can be completely described by their covariance matrix allowing for a very efficient implementation.

## When to use GfPEPS.jl

This package enables the construction of these states in 2D systems, which can be used in the following ways:

<p align="center">
<img width="1580" height="795" alt="general_workflow_GfPEPS" src="https://github.com/user-attachments/assets/6379ba1d-23fa-44fd-8bc0-a7ba7093fe6a" />
</p>

* Accelerating ground-state search algorithms by using the ground states of mean-field Hamiltonians as trial states $\ket{\Psi_\text{trial}}$
* Comparing mean-field ansätze to study the underlying physics of strongly correlated quantum systems
* Simulating models of free fermions in 2D systems

## Literature
The implementation of this package is based on the construction schemes in the following papers:

* Hackenbroich, A., Bernevig, B. A., Schuch, N. & Regnault, N. (2020) [Phys. Rev. B 101, 115134](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.101.115134)
* Mortier, Q., Schuch, N., Verstraete, F. & Haegeman, J. (2022) [Phys. Rev. Lett. 129, 206401](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.206401)
* Yang, Q., Zhang, X.-Y., Liao, H.-J., Tu, H.-H. & Wang, L. (2023) [Phys. Rev. B 107, 125128](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.125128)

## Quick start
Coming soon...
