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
        lattice_2x2 = InfiniteRectLattice(2,2;N_kx=100, N_ky=100, bc=(:PBC, :APBC))

        interaction_type = ["NN"]
        t = 1.0
        Δ = 2.0
        μ = 3.0
        hopping = get_isotropic_coupling_dict(lattice_1x1, [t]; interaction_type=interaction_type)
        pairing = get_isotropic_coupling_dict(lattice_1x1, [Δ]; interaction_type=interaction_type)
        H_BdG_1x1 = default_BCS_hamiltonian(hopping, pairing, μ, lattice_1x1; interaction_type=interaction_type)
        H_BdG_2x2 = default_BCS_hamiltonian(hopping, pairing, μ, lattice_2x2; interaction_type=interaction_type)

        E_1x1 = GfPEPS.exact_energy(lattice_1x1, H_BdG_1x1)
        E_2x2 = GfPEPS.exact_energy(lattice_2x2, H_BdG_2x2)

        @test E_1x1 ≈ E_2x2 atol=1e-10
    end;
end;