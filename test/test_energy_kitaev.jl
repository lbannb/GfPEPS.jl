using Revise
using Test
using GfPEPS
using Random
# using PEPSKit

Random.seed!(1234) # for reproducibility

# fluxfree sector: E_exact = -1.5746 per unit cell
# @testset "Kitaev HC model" begin
    # @testset "Single unit cell (square lattice)" begin
        # Nf = 1
        # Λ = 2
        # lattice = InfiniteRectLattice(1,1;N_kx=24, N_ky=24, bc=(:PBC, :APBC))

        # Jx = 1.0
        # Jy = 1.0
        # Jz = 1.0
        # H_BdG = kitaev_BCS_hamiltonian(Jx, Jy, Jz, lattice; interaction_type=["NN"])

        # E = GfPEPS.exact_energy(lattice.kvals, H_BdG, Nf)

    #     Ψ_trial = Gaussian_fPEPS(Nf, Λ, lattice, H_BdG)

    #     # test energy from CM
    #     @test Ψ_trial.exact_energy ≈ Ψ_trial.optim_energy atol=1e-5
    # end
    # @testset "Trivial Brickwall unit cell" begin
        Nf = 1
        Λ = 2
        lattice = InfiniteBrickWallLattice(1,1;N_kx=24, N_ky=24, bc=(:PBC, :APBC))

        Jx = 1.0
        Jy = 1.0
        Jz = 1.0
        H_BdG = kitaev_BCS_hamiltonian(Jx, Jy, Jz, lattice; interaction_type=["NN"])

        E = GfPEPS.exact_energy(lattice.kvals, H_BdG, Nf)

        # Ψ_trial = Gaussian_fPEPS(Nf, Λ, lattice, H_BdG)

        # # test energy from CM
        # @test Ψ_trial.exact_energy ≈ Ψ_trial.optim_energy atol=1e-5
    # end
# end;
