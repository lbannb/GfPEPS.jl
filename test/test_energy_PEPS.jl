using Revise
using Test
using GfPEPS
using TensorKit
using PEPSKit
using Random 

@testset "Energy after translation to fPEPS" begin
    Random.seed!(1234)

    Nf = 2
    Λ = 2
    lattice = InfiniteRectLattice(1,1;N_kx=128, N_ky=128, bc=(:APBC, :PBC))
    N = GfPEPS.get_number_of_modes(Nf, Λ, lattice)

    Γ_fiducial, X = GfPEPS.rand_CM(Nf, Λ, lattice)

    # peps = GfPEPS.translate_naive(X, Nf, Nv)
    peps = GfPEPS.translate(X, Nf, Λ, lattice);

    χenv_max = 8
    boundary_alg = (; tol = 1e-8, maxiter=100, alg = :simultaneous)
    
    env = GfPEPS.init_ctmrg_env(peps);
    env, _ = GfPEPS.grow_env(peps, env, 4, χenv_max; boundary_alg...)

    t = 1.0
    hopping = get_isotropic_coupling_dict(lattice, [t]; interaction_type=["NN"])
    # dwave pairing: Δ_y = -Δ_x 
    Δ1_x = 1.0
    Δ1_y = -Δ1_x
    μ = 1.0
    pairing = get_anisotropic_coupling_dict(lattice, [[Δ1_x,Δ1_x,Δ1_y,Δ1_y]]; interaction_type=["NN"])
    H_BdG = default_BCS_hamiltonian(hopping, pairing, μ, lattice; interaction_type=["NN"])

    ham = GfPEPS.BCS_spin_hamiltonian(ComplexF64, InfiniteSquare(1, 1); t=t, Δ_0 = Δ1_x, μ = μ)
    energy1 = real(expectation_value(peps, ham, env))

    energy2 = GfPEPS.energy_CM(Γ_fiducial, Nf, H_BdG, lattice)

    @show energy1
    @show energy2

    @test energy1 ≈ energy2 atol=1e-2 # depends on Λ
end