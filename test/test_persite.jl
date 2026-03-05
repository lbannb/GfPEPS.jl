using Revise
using Test
using GfPEPS
using Random
using LinearAlgebra

Random.seed!(42)

@testset "Per-site ansatz" begin

    @testset "G_in_persite matches G_in for 1×1 unit cell" begin
        Nf = 1
        Λ  = 2
        lattice = InfiniteRectLattice(1, 1; N_kx=6, N_ky=6, bc=(:PBC, :APBC))

        G_in_std = GfPEPS.G_in_Fourier(Λ, lattice)
        G_in_ps  = GfPEPS.G_in_Fourier_persite(Λ, lattice)

        @test size(G_in_std) == size(G_in_ps)
        @test G_in_std ≈ G_in_ps atol=1e-14
    end

    @testset "Γ_fiducial_blocks matches single-X for 1×1" begin
        Nf = 1
        Λ  = 2
        X = GfPEPS.rand_CM(Nf, Λ; parity=1)[2]

        # Standard: single covariance matrix
        Γ = GfPEPS.Γ_fiducial(X, Nf, Λ)
        A_std, B_std, D_std = GfPEPS.get_Γ_blocks(Γ, Nf)

        # Per-site: array with one element
        A_ps, B_ps, D_ps = GfPEPS.Γ_fiducial_blocks([X], Nf, Λ)

        @test A_std ≈ A_ps atol=1e-14
        @test B_std ≈ B_ps atol=1e-14
        @test D_std ≈ D_ps atol=1e-14
    end

    @testset "Per-site energy matches standard for 1×1 (Kitaev)" begin
        Nf = 1
        Λ  = 2
        lattice = InfiniteRectLattice(1, 1; N_kx=24, N_ky=24, bc=(:PBC, :APBC))

        Jx, Jy, Jz = 1.0, 1.0, 1.0
        H_BdG = kitaev_BCS_hamiltonian(Jx, Jy, Jz, lattice; interaction_type=["NN"])

        # Standard loss (single X)
        loss_std = GfPEPS.energy_loss_X(lattice, Nf, Λ, H_BdG)

        # Per-site loss (vector of per-site X matrices)
        loss_ps  = GfPEPS.energy_loss_X_persite(lattice, Nf, Λ, H_BdG)

        X = GfPEPS.rand_CM(Nf, Λ; parity=1)[2]

        E_std = loss_std(X)
        E_ps  = loss_ps([X])

        @show E_std, E_ps
        @test E_std ≈ E_ps atol=1e-12
    end

    @testset "Per-site G_in for 1×2 unit cell (structure check)" begin
        Nf = 2
        Λ  = 1
        lattice = InfiniteRectLattice(1, 2; N_kx=6, N_ky=6, bc=(:APBC, :PBC))

        G_in = GfPEPS.G_in_Fourier_persite(Λ, lattice)

        # For 1×2, Λ=1: 2 sites, 8Λ=8 virtual Majorana modes per site → 16×16 per k
        @test size(G_in, 2) == 16
        @test size(G_in, 3) == 16

        # Check anti-Hermiticity G(k) = -G(k)†  (virtual bond CM in Majorana basis)
        for ki in 1:size(G_in, 1)
            G_k = G_in[ki, :, :]
            @test G_k ≈ -adjoint(G_k) atol=1e-14
        end
    end

    @testset "Per-site loss evaluable for 1×2 unit cell" begin
        Nf = 2
        Λ  = 1
        lattice = InfiniteRectLattice(1, 2; N_kx=6, N_ky=6, bc=(:APBC, :PBC))

        # Build a d-wave BCS Hamiltonian with proper arguments
        hopping = GfPEPS.get_isotropic_coupling_dict(lattice, [1.0]; interaction_type=["NN"])
        pairing = GfPEPS.get_anisotropic_coupling_dict(lattice, [[1.0, 1.0, -1.0, -1.0]]; interaction_type=["NN"])
        H_BdG = GfPEPS.default_BCS_hamiltonian(hopping, pairing, 0.0, lattice; interaction_type=["NN"])

        loss_ps = GfPEPS.energy_loss_X_persite(lattice, Nf, Λ, H_BdG)

        # Two per-site X matrices (each of size 2(Nf+4Λ) = 12 → 12×12)
        X1 = GfPEPS.rand_CM(Nf, Λ; parity=1)[2]
        X2 = GfPEPS.rand_CM(Nf, Λ; parity=1)[2]

        E = loss_ps([X1, X2])
        @show E
        @test isfinite(E)
        @test E isa Real
    end
end;
