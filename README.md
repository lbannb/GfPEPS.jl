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
Coming soon...
