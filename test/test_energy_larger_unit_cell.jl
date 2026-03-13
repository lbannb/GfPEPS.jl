using Revise
using Test
using GfPEPS

@testset "Larger unit cell exact energy" begin
    # fluxfree sector: E_exact = -1.5746 per unit cell (sanity check)
    @testset "Kitaev HC model (1x1 unit cell)" begin
        lattice = InfiniteRectLattice(1,1;N_kx=100, N_ky=100, bc=(:PBC, :APBC))

        Jx = 1.0
        Jy = 1.0
        Jz = 1.0
        H_BdG = kitaev_BCS_hamiltonian(Jx, Jy, Jz, lattice; interaction_type=["NN"])
        E = GfPEPS.exact_energy(lattice, H_BdG)

        @test E ≈ -1.5746 atol=1e-4
    end;

    @testset "BCS uniform coupling (2x2 unit cell)" begin
        lattice_1x1 = InfiniteRectLattice(1,1;N_kx=100, N_ky=100, bc=(:PBC, :APBC))

        # lattice with 4 different sites in unit cell but all with the same coupling
        lattice_2x2 = InfiniteRectLattice(2,2;N_kx=100, N_ky=100, bc=(:PBC, :APBC), uc_layout=[1 2; 3 4])

        t = 1.0
        Δ = 2.0
        μ = 3.0

        hopping_1x1 = Dict(1 => Dict((1, 0) => t, (0, 1) => t, (-1, 0) => t, (0, -1) => t))
        pairing_1x1 = Dict(1 => Dict((1, 0) => Δ, (-1, 0) => Δ, (0, 1) => Δ, (0, -1) => Δ))

        hopping_2x2 = Dict(
            1 => Dict((1, 0) => t, (-1, 0) => t, (0, 1) => t, (0, -1) => t),
            2 => Dict((1, 0) => t, (-1, 0) => t, (0, 1) => t, (0, -1) => t),
            3 => Dict((1, 0) => t, (-1, 0) => t, (0, 1) => t, (0, -1) => t),
            4 => Dict((1, 0) => t, (-1, 0) => t, (0, 1) => t, (0, -1) => t)
        )
        pairing_2x2 = Dict(
            1 => Dict((1, 0) => Δ, (-1, 0) => Δ, (0, 1) => Δ, (0, -1) => Δ),
            2 => Dict((1, 0) => Δ, (-1, 0) => Δ, (0, 1) => Δ, (0, -1) => Δ),
            3 => Dict((1, 0) => Δ, (-1, 0) => Δ, (0, 1) => Δ, (0, -1) => Δ),
            4 => Dict((1, 0) => Δ, (-1, 0) => Δ, (0, 1) => Δ, (0, -1) => Δ)
        )

        H_BdG_1x1 = default_BCS_hamiltonian(hopping_1x1, pairing_1x1, μ, lattice_1x1)
        H_BdG_2x2 = default_BCS_hamiltonian(hopping_2x2, pairing_2x2, μ, lattice_2x2)

        E_1x1 = GfPEPS.exact_energy(lattice_1x1, H_BdG_1x1)
        E_2x2 = GfPEPS.exact_energy(lattice_2x2, H_BdG_2x2)

        @test E_1x1 ≈ E_2x2 atol=1e-10
    end;

    @testset "BCS different sublattice coupling (2x2 unit cell)" begin
        lattice_1x1 = InfiniteRectLattice(1,1;N_kx=100, N_ky=100, bc=(:PBC, :APBC))

        # 2 sublattices with different couplings 
        lattice_2x2 = InfiniteRectLattice(2,2;N_kx=100, N_ky=100, bc=(:PBC, :APBC), uc_layout=[1 2; 2 1])
        
        t_1 = 1.0
        t_2 = -1.0
        Δ_1 = 2.0
        Δ_2 = -2.0
        μ = 3.0

        hopping_1x1 = Dict(1 => Dict((1, 0) => t_1, (0, 1) => t_1, (-1, 0) => t_1, (0, -1) => t_1))
        pairing_1x1 = Dict(1 => Dict((1, 0) => Δ_1, (-1, 0) => Δ_1, (0, 1) => Δ_1, (0, -1) => Δ_1))

        # negative 
        hopping_2x2 = Dict(
            1 => Dict((2, 0) => t_1, (-2, 0) => t_1, (0, 2) => t_1, (0, -2) => t_1),
            2 => Dict((2, 0) => t_2, (-2, 0) => t_2, (0, 2) => -t_2, (0, -2) => t_2)
        )
        pairing_2x2 = Dict(
            1 => Dict((2, 0) => Δ_1, (-2, 0) => Δ_1, (0, 2) => Δ_1, (0, -2) => Δ_1),
            2 => Dict((2, 0) => Δ_2, (-2, 0) => Δ_2, (0, 2) => Δ_2, (0, -2) => Δ_2)
        )

        H_BdG_1x1 = default_BCS_hamiltonian(hopping_1x1, pairing_1x1, μ, lattice_1x1)
        H_BdG_2x2 = default_BCS_hamiltonian(hopping_2x2, pairing_2x2, μ, lattice_2x2)

        E_1x1 = GfPEPS.exact_energy(lattice_1x1, H_BdG_1x1)
        E_2x2 = GfPEPS.exact_energy(lattice_2x2, H_BdG_2x2)

        @test E_1x1 != E_2x2
    end;
end;