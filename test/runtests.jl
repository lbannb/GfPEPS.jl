using Test

@testset "GfPEPS tests" begin
    tests = [
        "test_Bogoliubov.jl",
        "test_blocked_GaussianMap.jl",
        "test_doping.jl",
        "test_energy_BCS.jl",
        "test_energy_kitaev.jl",
        "test_energy_PEPS.jl",
        "test_energy_larger_unit_cell.jl",
        "test_energy_kitaev_larger_uc.jl",
        "test_hole_density_BCS_GW.jl",
    ]
    for t in tests
        println("\nRunning test file: $(t)")
        @time include(t)
    end
end;
