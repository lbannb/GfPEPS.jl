using Revise
using Test
using Statistics
using GfPEPS
using TensorKit
using PEPSKit
using Random

Random.seed!(1234) # for reproducibility

#= Test settings =#
Nf = 2
Λ = 2
lattice = InfiniteRectLattice(2,2;N_kx=6, N_ky=6, bc=(:PBC, :APBC), uc_layout=[1 2; 2 1])

doping_layout = [0.11 0.21; 0.21 0.11]
doping_kwargs = DopingSettings(; δ=mean(doping_layout), enforce_density=true, doping_layout=doping_layout);

# BCS parameters
t = 1.0
# dwave pairing: Δ_y = -Δ_x 
Δ1_x = 1.0
Δ1_y = -Δ1_x
μ = 2.0

@testset "Average doping in unit cell" begin
     # different couplings on the two sublattices
    hopping = Dict( 
        1 => Dict((2, 0) => t, (0, 2) => t, (-2, 0) => t, (0, -2) => t),
        2 => Dict((2, 0) => 2t, (0, 2) => 2t, (-2, 0) => 2t, (0, -2) => 2t)
    )
    pairing = Dict(
        1 => Dict((2, 0) => Δ1_x, (-2, 0) => Δ1_x, (0, 2) => Δ1_y, (0, -2) => Δ1_y),
        2 => Dict((2, 0) => 2Δ1_x, (-2, 0) => 2Δ1_x, (0, 2) => 2Δ1_y, (0, -2) => 2Δ1_y)
    )
    H_BdG = default_BCS_hamiltonian(hopping, pairing, μ, lattice)
    Ψ_trial = Gaussian_fPEPS(Nf, Λ, lattice, H_BdG; doping_kwargs=doping_kwargs);
end;
