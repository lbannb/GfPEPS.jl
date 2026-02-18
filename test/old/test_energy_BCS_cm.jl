using Revise
using Test
using GfPEPS
using JSON: parsefile

@testset "Energy BCS" begin
    @testset "d_wave" begin
        config = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_d_wave.json"))
        X_opt, optim_energy, E_exact, _ = GfPEPS.get_X_opt(;conf=config)
        # test energy from CM
        @test optim_energy ≈ E_exact
    end;

    @testset "p_wave" begin
        config = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_p_wave.json"))
        X_opt, optim_energy, E_exact, _ = GfPEPS.get_X_opt(;conf=config)
        # test energy from CM
        @test optim_energy ≈ E_exact
    end;

    @testset "s_wave" begin
        config = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_s_wave.json"))
        X_opt, optim_energy, E_exact, _ = GfPEPS.get_X_opt(;conf=config)
        # test energy from CM
        @test optim_energy ≈ E_exact
    end;
end;