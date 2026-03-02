using Revise
using Test
using GfPEPS
using Random
# using PEPSKit

Random.seed!(1234) # for reproducibility

@testset "Larger unit cell" begin
    @testset "π-flux AFM S=1/2 Heisenberg model" begin
        # Exact energy = -0.172 J
        # see https://www.mit.edu/~8.334/grades/projects/projects24/LukeKim.pdf for exact energy

        Nf = 2
        Λ = 2
        lattice = InfiniteRectLattice(2,2;N_kx=6, N_ky=6, bc=(:PBC, :APBC))

        Jx = 1.0
        Jy = 1.0
        Jz = 1.0
        H_BdG = kitaev_BCS_hamiltonian(Jx, Jy, Jz, lattice; interaction_type=["NN"])

        E = GfPEPS.exact_energy(lattice.kvals, H_BdG, Nf)

        Ψ_trial = Gaussian_fPEPS(Nf, Λ, lattice, H_BdG)

        # test energy from CM
        @test Ψ_trial.exact_energy ≈ Ψ_trial.optim_energy atol=1e-5
    end
end;
