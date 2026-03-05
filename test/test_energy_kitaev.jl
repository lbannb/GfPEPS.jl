using Revise
using Test
using GfPEPS
using Random
using PEPSKit

Random.seed!(1234) # for reproducibility

χ_0 = 4
χenv_max = 8
boundary_alg = (; tol = 1e-8, maxiter=1000, alg = :simultaneous)

# fluxfree sector: E_exact = -1.5746 per unit cell
@testset "Kitaev HC model" begin
    @testset "Single unit cell (square lattice)" begin
        Nf = 1
        Λ = 2
        lattice = InfiniteRectLattice(1,1;N_kx=24, N_ky=24, bc=(:PBC, :APBC))

        Jx = 1.0
        Jy = 1.0
        Jz = 1.0
        H_BdG = kitaev_BCS_hamiltonian(Jx, Jy, Jz, lattice; interaction_type=["NN"])

        Ψ_trial = Gaussian_fPEPS(Nf, Λ, lattice, H_BdG)

        # test energy from CM
        @test Ψ_trial.exact_energy ≈ Ψ_trial.optim_energy atol=1e-4

        # test energy from iPEPS
        env = GfPEPS.init_ctmrg_env(Ψ_trial.peps);
        env, _ = GfPEPS.grow_env(Ψ_trial.peps, env, χ_0, χenv_max; boundary_alg...);
        ham = GfPEPS.Kitaev_Hamiltonian(ComplexF64, InfiniteSquare(lattice.Lx, lattice.Ly); Jx=Jx, Jy=Jy, Jz=Jz)
        E_PEPS = real(expectation_value(Ψ_trial.peps, ham, env))
        
        @show Ψ_trial.exact_energy
        @show Ψ_trial.optim_energy
        @show E_PEPS

        @test E_PEPS ≈ Ψ_trial.optim_energy atol=2e-2 # depends on χenv_max
    end
end;