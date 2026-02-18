using Revise
using Test
using GfPEPS
using PEPSKit
using JSON: parsefile

@testset "Energy BCS" begin
    @testset "s_wave" begin
        config = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_s_wave.json"))
        config["params"]["N_virtual_fermions_on_bond"] = 2
        config["system_params"]["Lx"] = 24
        config["system_params"]["Ly"] = 24
        X_opt, optim_energy, E_exact, _ = GfPEPS.get_X_opt(;conf=config)

        peps = GfPEPS.translate(X_opt, config["params"]["N_physical_fermions_on_site"], config["params"]["N_virtual_fermions_on_bond"])

        bz = BrillouinZone2D(24, 24, (Symbol(config["system_params"]["x_bc"]), Symbol(config["system_params"]["y_bc"])))
        μ_from_δ = GfPEPS.solve_for_mu(bz, config["hamiltonian"]["hole_density"], config["hamiltonian"]["t"], config["hamiltonian"]["pairing_type"], config["hamiltonian"]["Δ_0"])

        χenv_max = 12
        boundary_alg = (; tol = 1e-8, maxiter=100, alg = :simultaneous)
        env = init_ctmrg_env(peps)
        env, _ = grow_env(peps, env, 4, χenv_max; boundary_alg...)

        H = GfPEPS.BCS_spin_hamiltonian(ComplexF64, InfiniteSquare(1, 1); pairing_type = config["hamiltonian"]["pairing_type"], t=config["hamiltonian"]["t"], Δ_0 = config["hamiltonian"]["Δ_0"], μ = μ_from_δ)
        energy_peps = real(expectation_value(peps, H, env))
        @test energy_peps ≈ optim_energy atol=1e-2
    end;
end;