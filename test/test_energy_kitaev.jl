using Revise
using Test
using GfPEPS
using Random
using TensorKit
using PEPSKit

Random.seed!(1234) # for reproducibility

χenv_max = 8
boundary_alg = (; tol = 1e-8, maxiter=100, alg = :SimultaneousCTMRG, trunc = truncrank(χenv_max))

# fluxfree sector: E_exact = -1.5746 per unit cell
@testset "Kitaev HC model" begin
    Jx = Jy = Jz = 1.0
    E_exact = -1.5746 # fluxfree sector reference value per (honeycomb) unit cell
    Nf, Λ = 1, 2

    @testset "Exact energy folding (1x2 vs 1x1)" begin
        lattice_1x1 = InfiniteRectLattice(1, 1; N_kx=100, N_ky=100, bc=(:PBC, :APBC))
        lattice_1x2 = InfiniteRectLattice(2, 1; N_kx=50, N_ky=100, bc=(:PBC, :APBC), uc_layout=[1 2])

        H_1x1 = kitaev_BCS_hamiltonian(Jx, Jy, Jz, lattice_1x1)
        H_1x2 = kitaev_BCS_hamiltonian(Jx, Jy, Jz, lattice_1x2)

        E_1x1 = GfPEPS.exact_energy(lattice_1x1, H_1x1)
        E_1x2 = GfPEPS.exact_energy(lattice_1x2, H_1x2)

        @test E_1x1 ≈ E_exact atol=1e-4 # fluxfree sector reference value per (honeycomb) unit cell
        @test E_1x2 / GfPEPS.get_number_of_sites(lattice_1x2) ≈ E_1x1 atol=1e-10
    end;

    #=
        Kitaev honeycomb model (vortex-free sector) on larger unit cells via the unified
        BCS-form loss. The 1x2 unit cell must reproduce the 1x1 result
        per site: with (N_kx, N_ky/2) k-points its folded Brillouin zone contains exactly
        the same momenta as the 1x1 cell with (N_kx, N_ky) points.
    =#
    @testset "CM optimization (1x2 vs 1x1)" begin
        lattice_1x1 = InfiniteRectLattice(1, 1; N_kx=24, N_ky=24, bc=(:PBC, :APBC))
        lattice_1x2 = InfiniteRectLattice(2, 1; N_kx=12, N_ky=24, bc=(:PBC, :APBC), uc_layout=[1 2])

        H_1x1 = kitaev_BCS_hamiltonian(Jx, Jy, Jz, lattice_1x1)
        H_1x2 = kitaev_BCS_hamiltonian(Jx, Jy, Jz, lattice_1x2)

        Ψ_1x1 = Gaussian_fPEPS(Nf, Λ, lattice_1x1, H_1x1)
        Ψ_1x2 = Gaussian_fPEPS(Nf, Λ, lattice_1x2, H_1x2)

        # the folded exact energies agree per construction; both optimizations must reach them
        @test Ψ_1x1.optim_energy ≈ Ψ_1x1.exact_energy atol=1e-3
        @test Ψ_1x2.optim_energy ≈ Ψ_1x2.exact_energy atol=1e-3
        @test Ψ_1x2.optim_energy / GfPEPS.get_number_of_sites(lattice_1x2) ≈ Ψ_1x1.optim_energy atol=1e-3
    end;
end;