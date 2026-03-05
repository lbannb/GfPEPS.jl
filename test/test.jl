using Revise
using Profile
using Test
using GfPEPS
using JSON: parsefile
using Random

Random.seed!(1234) # for reproducibility

#= Test settings =#
Nf = 2
Λ = 2
lattice = InfiniteRectLattice(1,1;N_kx=6, N_ky=6, bc=(:PBC, :APBC))
doping_kwargs = DopingSettings(; δ=0.16, enforce_density=false)

t1 = 1.0
hopping = get_isotropic_coupling_dict(lattice, [t1]; interaction_type=["NN"])
μ = 1.0

# dwave pairing: Δ_y = -Δ_x 
Δ1_x = 1.0
Δ1_y = -Δ1_x
pairing = get_anisotropic_coupling_dict(lattice, [[Δ1_x,Δ1_x,Δ1_y,Δ1_y]]; interaction_type=["NN"])
H_BdG = default_BCS_hamiltonian(hopping, pairing, μ, lattice; interaction_type=["NN"])

GfPEPS.H_BdG_majorana_k(Nf, lattice.kvals[:, 1], H_BdG, lattice)

Ψ_trial = Gaussian_fPEPS(Nf, Λ, lattice, H_BdG, doping_kwargs=doping_kwargs);

# Profile.clear()
# @profile Gaussian_fPEPS(Nf, Λ, lattice, H_BdG, doping_kwargs=doping_kwargs)

# Profile.print(format=:flat, sortedby=:count, maxdepth=20)
# Profile.print(format=:tree, maxdepth=12)

# # test energy from CM
# @test Ψ_trial.exact_energy ≈ Ψ_trial.optim_energy atol=1e-5
;