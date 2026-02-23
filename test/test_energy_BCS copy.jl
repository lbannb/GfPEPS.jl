using Revise
using Test
using GfPEPS
using JSON: parsefile
using Random

Random.seed!(1234) # for reproducibility

#= Test settings =#
Nf = 2
Λ = 2
lattice = InfiniteRectLattice(1,1;N_kx=6, N_ky=6, bc=(:PBC, :APBC))
doping_kwargs = DopingSettings(; δ=0.16, enforce_density=true)

# BCS parameters
t1 = 1.0
hopping = get_isotropic_coupling_dict(lattice, [t1]; interaction_type=["NN"])
# dwave pairing: Δ_y = -Δ_x 
Δ1_x = 1.0
Δ1_y = -Δ1_x
pairing = get_anisotropic_coupling_dict(lattice, [[Δ1_x,Δ1_x,Δ1_y,Δ1_y]]; interaction_type=["NN"])
μ = 1.0

H_BdG = default_BCS_hamiltonian(hopping, pairing, μ, lattice; interaction_type=["NN"])

Ψ_trial = Gaussian_fPEPS(Nf, Λ, lattice, H_BdG)

# test energy from CM
@test Ψ_trial.exact_energy ≈ Ψ_trial.optim_energy atol=1e-5

# Ψ_trial = Gaussian_fPEPS(Nf, Λ, lattice, BCS_params; doping_kwargs=doping_kwargs);
# δ_opt = GfPEPS.doping_CM(Ψ_trial.Γ_fiducial, Nf, lattice)
# # doping_fct =  X_mat -> GfPEPS.doping_CM(X_mat, Nf, Λ, lattice)
# # δ_opt = doping_fct(Ψ_trial.X_opt)

# # test doping from CM
# @test δ_opt ≈ doping_kwargs.δ atol=doping_kwargs.density_tol

# # test energy from CM
# @test Ψ_trial.exact_energy ≈ Ψ_trial.optim_energy atol=1e-2




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