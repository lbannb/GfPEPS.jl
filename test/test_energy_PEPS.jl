using Revise
using Test
using GfPEPS
using TensorKit
using PEPSKit
using Random 

# CTMRG settings
# χenv_max = 64
χenv_max = 32
boundary_alg = (; tol = 1e-8, maxiter=500, alg = :simultaneous)

# BCS parameters
t = 1.0
# dwave pairing: Δ_y = -Δ_x 
Δ1_x = 1.0
Δ1_y = -Δ1_x
μ = 2.0

# @testset "Energy after translation to fPEPS" begin
    @testset "Trivial unit cell (1x1)" begin
        χenv_max = 32
        Random.seed!(12345) # for reproducibility
        Nf = 2
        Λ = 2
        lattice = InfiniteRectLattice(1,1;N_kx=256, N_ky=256, bc=(:APBC, :PBC))
        N = GfPEPS.get_number_of_modes(Nf, Λ, lattice)

        X_vec = [GfPEPS.rand_CM(Nf, Λ)[1] for i in 1:GfPEPS.get_number_of_distinct_sites_in_uc(lattice)]
        peps = GfPEPS.translate(X_vec, Nf, Λ, lattice);

        env = GfPEPS.init_ctmrg_env(peps);
        env, _ = GfPEPS.grow_env(peps, env, 6, χenv_max; boundary_alg...)

        hopping = Dict(1 => Dict((1, 0) => t, (0, 1) => t, (-1, 0) => t, (0, -1) => t))
        pairing = Dict(1 => Dict((1, 0) => Δ1_x, (-1, 0) => Δ1_x, (0, 1) => Δ1_y, (0, -1) => Δ1_y))

        H_BdG = default_BCS_hamiltonian(hopping, pairing, μ, lattice)

        ham = GfPEPS.BCS_spin_hamiltonian(ComplexF64, InfiniteSquare(lattice.Lx, lattice.Ly), H_BdG, lattice.uc_layout)
        energy1 = real(expectation_value(peps, ham, env))

        energy2 = GfPEPS.energy_CM(X_vec, Nf, Λ, H_BdG, lattice)

        @show energy1
        @show energy2

        @test energy1 ≈ energy2 atol=1e-3 # depends on χenv_max
    end
    @testset "1x2 unit cell" begin
        Random.seed!(123) # for reproducibility
        Nf = 2
        Λ = 2
        lattice = InfiniteRectLattice(2,1;N_kx=256, N_ky=256, bc=(:APBC, :PBC), uc_layout=[1 2])
        N = GfPEPS.get_number_of_modes(Nf, Λ, lattice)

        X_vec = [GfPEPS.rand_CM(Nf, Λ)[1] for i in 1:GfPEPS.get_number_of_distinct_sites_in_uc(lattice)]
        peps = GfPEPS.translate(X_vec, Nf, Λ, lattice);

        env = GfPEPS.init_ctmrg_env(peps);
        env, _ = GfPEPS.grow_env(peps, env, 6, χenv_max; boundary_alg...)

        # different couplings on the two sublattices
        hopping = Dict(
            1 => Dict((1, 0) => t, (0, 1) => t, (-1, 0) => t, (0, -1) => t),
            2 => Dict((1, 0) => t, (0, 1) => t, (-1, 0) => t, (0, -1) => t)
        )
        pairing = Dict(
            1 => Dict((1, 0) => Δ1_x, (-1, 0) => Δ1_x, (0, 1) => Δ1_y, (0, -1) => Δ1_y),
            2 => Dict((1, 0) => Δ1_x, (-1, 0) => Δ1_x, (0, 1) => Δ1_y, (0, -1) => Δ1_y)
        )
        # hopping = Dict( 
        #     1 => Dict((2, 0) => t, (0, 2) => t, (-2, 0) => t, (0, -2) => t),
        #     2 => Dict((2, 0) => 2t, (0, 2) => 2t, (-2, 0) => 2t, (0, -2) => 2t)
        # )
        # pairing = Dict(
        #     1 => Dict((2, 0) => Δ1_x, (-2, 0) => Δ1_x, (0, 2) => Δ1_y, (0, -2) => Δ1_y),
        #     2 => Dict((2, 0) => 2Δ1_x, (-2, 0) => 2Δ1_x, (0, 2) => 2Δ1_y, (0, -2) => 2Δ1_y)
        # )

        H_BdG = default_BCS_hamiltonian(hopping, pairing, μ, lattice)

        ham = GfPEPS.BCS_spin_hamiltonian(ComplexF64, InfiniteSquare(lattice.Ly, lattice.Lx), H_BdG, lattice.uc_layout)
        energy1 = real(expectation_value(peps, ham, env))

        energy2 = GfPEPS.energy_CM(X_vec, Nf, Λ, H_BdG, lattice)

        @show energy1
        @show energy2

        @test energy1 ≈ energy2 atol=1e-3; # depends on χenv_max
    end
# end;
