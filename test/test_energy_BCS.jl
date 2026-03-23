using Revise
using Test
using GfPEPS
using Random

Random.seed!(1234) # for reproducibility

#= Test settings =#
Nf = 2
Λ = 4
lattice = InfiniteRectLattice(1,1;N_kx=6, N_ky=6, bc=(:PBC, :APBC))
doping_kwargs = DopingSettings(; doping_layout=[0.16;;], enforce_density=true)

@testset "BCS (trivial unit cell)" begin
    t1 = 1.0
    hopping = Dict(1 => Dict((1, 0) => t1, (0, 1) => t1, (-1, 0) => t1, (0, -1) => t1))
    μ = [1.0]

    @testset "d-wave pairing" begin
        # dwave pairing: Δ_y = -Δ_x 
        Δ1_x = 1.0
        Δ1_y = -Δ1_x
        pairing = Dict(1 => Dict((1, 0) => Δ1_x, (-1, 0) => Δ1_x, (0, 1) => Δ1_y, (0, -1) => Δ1_y))
        H_BdG = default_BCS_hamiltonian(hopping, pairing, μ, lattice)

        # @testset "Energy" begin
        #     Ψ_trial = Gaussian_fPEPS(Nf, Λ, lattice, H_BdG)
            
        #     # test energy from CM
        #     @test Ψ_trial.exact_energy ≈ Ψ_trial.optim_energy atol=1e-5
        # end;

        @testset "Doping" begin
            Ψ_trial = Gaussian_fPEPS(Nf, Λ, lattice, H_BdG; doping_kwargs=doping_kwargs);
            δ_opt = GfPEPS.doping_CM(Ψ_trial.X_opt, Nf, Λ, lattice)

            # test doping from CM
            @test δ_opt ≈ doping_kwargs.doping_layout atol=doping_kwargs.density_tol

            # test energy from CM
            @test Ψ_trial.exact_energy ≈ Ψ_trial.optim_energy atol=1e-5
        end;
    end;

    # @testset "s-wave pairing" begin
    #     # swave pairing: Δ_y = Δ_x 
    #     Δ = 1.0
    #     pairing = Dict(1 => Dict((1, 0) => Δ, (-1, 0) => Δ, (0, 1) => Δ, (0, -1) => Δ))
    #     H_BdG = default_BCS_hamiltonian(hopping, pairing, μ, lattice)

    #     @testset "Energy" begin
    #         Ψ_trial = Gaussian_fPEPS(Nf, Λ, lattice, H_BdG)
            
    #         # test energy from CM
    #         @test Ψ_trial.exact_energy ≈ Ψ_trial.optim_energy atol=1e-5
    #     end;

    #     @testset "Doping" begin
    #         Ψ_trial = Gaussian_fPEPS(Nf, Λ, lattice, H_BdG; doping_kwargs=doping_kwargs);
    #         δ_opt = GfPEPS.doping_CM(Ψ_trial.X_opt, Nf, Λ, lattice)

    #         # test doping from CM
    #         @test δ_opt ≈ doping_kwargs.δ atol=doping_kwargs.density_tol

    #         # test energy from CM
    #         @test Ψ_trial.exact_energy ≈ Ψ_trial.optim_energy atol=1e-5
    #     end;
    # end;

    # @testset "px+ipy-wave pairing" begin
    #     # px+ipy pairing: 
    #     Δ = 1.0
    #     pairing = Dict(1 => Dict((1, 0) => Δ, (-1, 0) => 1im*Δ, (0, 1) => -Δ, (0, -1) => -1im*Δ))
    #     H_BdG = default_BCS_hamiltonian(hopping, pairing, μ, lattice)

    #     @testset "Energy" begin
    #         Ψ_trial = Gaussian_fPEPS(Nf, Λ, lattice, H_BdG)
            
    #         # test energy from CM
    #         @test Ψ_trial.exact_energy ≈ Ψ_trial.optim_energy atol=1e-5
    #     end;

    #     @testset "Doping" begin
    #         Ψ_trial = Gaussian_fPEPS(Nf, Λ, lattice, H_BdG; doping_kwargs=doping_kwargs);
    #         δ_opt = GfPEPS.doping_CM(Ψ_trial.X_opt, Nf, Λ, lattice)

    #         # test doping from CM
    #         @test δ_opt ≈ doping_kwargs.δ atol=doping_kwargs.density_tol

    #         # test energy from CM
    #         @test Ψ_trial.exact_energy ≈ Ψ_trial.optim_energy atol=1e-5
    #     end;
    # end;
end;