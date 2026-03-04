using GfPEPS
using Test

@testset "GfPEPS tests" begin
    tests = [
        "test_Bogoliubov.jl",
        "test_energy_BCS.jl",
        "test_energy_kitaev.jl",
        # "test_energy_PEPS.jl",
        # "test_env_init.jl",
        # "test_hole_density_BCS.jl",
        # "test_hole_density_BCS_GW.jl",
        # "test_kitaev_cm_energy.jl",
        # "test_optimal_gap_param.jl",
    ]
    @time for t in tests
        println("\nRunning test file: $(t)")
        @time include(t)
    end
end;
