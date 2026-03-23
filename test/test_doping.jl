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
doping_kwargs = DopingSettings(; enforce_density=false, doping_layout=doping_layout);

# BCS parameters
t1 = 1.0
t2 = 1.0
# dwave pairing: Δ_y = -Δ_x 
Δ1_x = 1.0
Δ1_y = -Δ1_x
Δ2_x = 1.0
Δ2_y = -Δ2_x
μ = [1.0, 2.0, 2.0, 1.0]

# @testset "Average doping in unit cell" begin
     # different couplings on the two sublattices
    hopping = Dict( 
        1 => Dict((1, 0) => t1, (0, 1) => t1, (-1, 0) => t1, (0, -1) => t1),
        2 => Dict((1, 0) => t2, (0, 1) => t2, (-1, 0) => t2, (0, -1) => t2)
    )
    pairing = Dict(
        1 => Dict((1, 0) => Δ1_x, (-1, 0) => Δ1_x, (0, 1) => Δ1_y, (0, -1) => Δ1_y),
        2 => Dict((1, 0) => Δ2_x, (-1, 0) => Δ2_x, (0, 1) => Δ2_y, (0, -1) => Δ2_y)
    )
    H_BdG = default_BCS_hamiltonian(hopping, pairing, μ, lattice)
    _ = GfPEPS.solve_for_mu(lattice, doping_layout, H_BdG)

    Ψ_trial = Gaussian_fPEPS(Nf, Λ, lattice, H_BdG; doping_kwargs=doping_kwargs);

    doping_layout_opt = [GfPEPS.doping_CM(Ψ_trial.X_opt, Nf, Λ, lattice)[lattice.uc_layout[r, c]] for c in 1:lattice.Lx, r in 1:lattice.Ly]
    @test doping_kwargs.doping_layout ≈ doping_layout_opt atol=doping_kwargs.density_tol

    # test average doping from CM   
    @test mean(doping_layout_opt) ≈ GfPEPS.avg_doping_CM(Ψ_trial.X_opt, Nf, Λ, lattice) atol=doping_kwargs.density_tol

# end;
;