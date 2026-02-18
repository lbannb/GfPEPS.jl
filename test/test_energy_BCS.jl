using Revise
using Test
using GfPEPS
using JSON: parsefile
using Random

Random.seed!(1234) # for reproducibility

#= Test settings =#
Nf = 2
Λ = 2
lattice = InfiniteRectLattice(1,1;N_kx=12, N_ky=12, bc=(:PBC, :APBC))
doping_kwargs = DopingSettings(; δ=0.16, enforce_density=true)

# BCS parameters
t = 1.0
μ = 1.0
Δ_0 = 1.0

@testset "BCS (trivial unit cell)" begin
    @testset "d-wave pairing" begin
        BCS_params = BCS(t, μ, "d_wave", Δ_0)

        # @testset "Energy" begin
        #     Ψ_trial = Gaussian_fPEPS(Nf, Λ, lattice, BCS_params)
            
        #     # test energy from CM
        #     @test Ψ_trial.exact_energy ≈ Ψ_trial.optim_energy atol=1e-5
        # end;

        @testset "Doping" begin
            Ψ_trial = Gaussian_fPEPS(Nf, Λ, lattice, BCS_params; doping_kwargs=doping_kwargs);
            δ_opt = GfPEPS.doping_CM(Ψ_trial.Γ_fiducial, Nf, lattice)
            # doping_fct =  X_mat -> GfPEPS.doping_CM(X_mat, Nf, Λ, lattice)
            # δ_opt = doping_fct(Ψ_trial.X_opt)

            # test doping from CM
            @test δ_opt ≈ doping_kwargs.δ atol=doping_kwargs.density_tol

            # test energy from CM
            @test Ψ_trial.exact_energy ≈ Ψ_trial.optim_energy atol=1e-2
        end;
    end;




    # @testset "p-wave pairing" begin
    #     config = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_p_wave.json"))
    #     X_opt, optim_energy, E_exact, _ = GfPEPS.get_X_opt(;conf=config)
    #     # test energy from CM
    #     @test optim_energy ≈ E_exact
    # end;

    # @testset "s-wave pairing" begin
    #     config = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_s_wave.json"))
    #     X_opt, optim_energy, E_exact, _ = GfPEPS.get_X_opt(;conf=config)
    #     # test energy from CM
    #     @test optim_energy ≈ E_exact
    # end;
end;