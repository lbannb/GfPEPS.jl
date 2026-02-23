using Revise
using Test
using GfPEPS
using JSON: parsefile
using Random

Random.seed!(1234) # for reproducibility

#= Test settings =#
Nf = 2
Λ = 4
lattice = InfiniteRectLattice(1,1;N_kx=6, N_ky=6, bc=(:PBC, :APBC))
doping_kwargs = DopingSettings(; δ=0.16, enforce_density=true)

@testset "BCS (trivial unit cell)" begin
    t1 = 1.0
    hopping = get_isotropic_coupling_dict(lattice, [t1]; interaction_type=["NN"])
    μ = 1.0

    @testset "d-wave pairing" begin
        # dwave pairing: Δ_y = -Δ_x 
        Δ1_x = 1.0
        Δ1_y = -Δ1_x
        pairing = get_anisotropic_coupling_dict(lattice, [[Δ1_x,Δ1_x,Δ1_y,Δ1_y]]; interaction_type=["NN"])
        H_BdG = default_BCS_hamiltonian(hopping, pairing, μ, lattice; interaction_type=["NN"])

        @testset "Energy" begin
            Ψ_trial = Gaussian_fPEPS(Nf, Λ, lattice, H_BdG)
            
            # test energy from CM
            @test Ψ_trial.exact_energy ≈ Ψ_trial.optim_energy atol=1e-5
        end;

        @testset "Doping" begin
            Ψ_trial = Gaussian_fPEPS(Nf, Λ, lattice, H_BdG; doping_kwargs=doping_kwargs);
            δ_opt = GfPEPS.doping_CM(Ψ_trial.Γ_fiducial, Nf, lattice)

            # test doping from CM
            @test δ_opt ≈ doping_kwargs.δ atol=doping_kwargs.density_tol

            # test energy from CM
            @test Ψ_trial.exact_energy ≈ Ψ_trial.optim_energy atol=1e-2
        end;
    end;

    @testset "s-wave pairing" begin
        # swave pairing: Δ_y = Δ_x 
        Δ = 1.0
        pairing = get_isotropic_coupling_dict(lattice, [Δ]; interaction_type=["NN"])
        H_BdG = default_BCS_hamiltonian(hopping, pairing, μ, lattice; interaction_type=["NN"])

        @testset "Energy" begin
            Ψ_trial = Gaussian_fPEPS(Nf, Λ, lattice, H_BdG)
            
            # test energy from CM
            @test Ψ_trial.exact_energy ≈ Ψ_trial.optim_energy atol=1e-5
        end;

        @testset "Doping" begin
            Ψ_trial = Gaussian_fPEPS(Nf, Λ, lattice, H_BdG; doping_kwargs=doping_kwargs);
            δ_opt = GfPEPS.doping_CM(Ψ_trial.Γ_fiducial, Nf, lattice)

            # test doping from CM
            @test δ_opt ≈ doping_kwargs.δ atol=doping_kwargs.density_tol

            # test energy from CM
            @test Ψ_trial.exact_energy ≈ Ψ_trial.optim_energy atol=1e-2
        end;
    end;

    @testset "px+ipy-wave pairing" begin
        # px+ipy pairing: 
        Δ = 1.0
        pairing = get_anisotropic_coupling_dict(lattice, [[Δ,1im*Δ,-Δ,-1im*Δ]]; interaction_type=["NN"])
        H_BdG = default_BCS_hamiltonian(hopping, pairing, μ, lattice; interaction_type=["NN"])

        @testset "Energy" begin
            Ψ_trial = Gaussian_fPEPS(Nf, Λ, lattice, H_BdG)
            
            # test energy from CM
            @test Ψ_trial.exact_energy ≈ Ψ_trial.optim_energy atol=1e-5
        end;

        @testset "Doping" begin
            Ψ_trial = Gaussian_fPEPS(Nf, Λ, lattice, H_BdG; doping_kwargs=doping_kwargs);
            δ_opt = GfPEPS.doping_CM(Ψ_trial.Γ_fiducial, Nf, lattice)

            # test doping from CM
            @test δ_opt ≈ doping_kwargs.δ atol=doping_kwargs.density_tol

            # test energy from CM
            @test Ψ_trial.exact_energy ≈ Ψ_trial.optim_energy atol=1e-2
        end;
    end;
end;