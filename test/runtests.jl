using Test

@testset "GfPEPS tests" begin
    tests = [
        "test_Bogoliubov.jl",
        "test_blocked_GaussianMap.jl",
        "test_doping.jl",
        "test_energy_BCS.jl",
        "test_energy_kitaev.jl",
        "test_energy_PEPS.jl",
    ]
    for t in tests
        println("\nRunning test file: $(t)")
        @time include(t)
    end
end;
