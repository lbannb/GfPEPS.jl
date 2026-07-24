using Revise
using Test
using GfPEPS
using Random
using PEPSKit

Random.seed!(1234) # for reproducibility

χenv_max = 12
boundary_alg = (; tol = 1e-8, maxiter=200, alg = :SimultaneousCTMRG, trunc = truncrank(χenv_max))

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
        env0 = initialize_ctmrg_environment(Ψ_trial.peps, IdentityInitialization())
        env, _ = leading_boundary(env0, Ψ_trial.peps; boundary_alg...)
        ham = GfPEPS.Kitaev_Hamiltonian(ComplexF64, InfiniteSquare(lattice.Lx, lattice.Ly); Jx=Jx, Jy=Jy, Jz=Jz)
        E_PEPS = real(expectation_value(Ψ_trial.peps, ham, env))
        
        @show Ψ_trial.exact_energy
        @show Ψ_trial.optim_energy
        @show E_PEPS

        @test E_PEPS ≈ Ψ_trial.optim_energy atol=1e-2 # depends on χenv_max
    end
end;