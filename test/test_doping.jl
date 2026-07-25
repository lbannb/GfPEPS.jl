using Revise
using Test
using Statistics
using GfPEPS
using TensorKit
using PEPSKit
using Random

Random.seed!(1234) # for reproducibility

#= Test settings =#
Nf = 2
Λ = 2
lattice = InfiniteRectLattice(2,2;N_kx=6, N_ky=6, bc=(:PBC, :APBC), uc_layout=[1 2; 2 1]) # unit cell layout with two sublattices A, B

δ_target = 0.10
doping_kwargs = DopingSettings(; enforce_density=true, δ=δ_target);

# BCS parameters
t1 = 1.0
t2 = -2.0
# dwave pairing: Δ_y = -Δ_x 
Δ1_x = 1.0
Δ1_y = -Δ1_x
Δ2_x = -2.0
Δ2_y = -Δ2_x
μ = 1.0

# CTMRG settings
χ_env_max = 32
boundary_alg = (; tol = 1e-8, alg=:SimultaneousCTMRG, maxiter=100, trunc = trunctol(; rtol=1e-4) & truncrank(χ_env_max))

@testset "Optimization with doping test (2x2 unit cell)" begin
    # different couplings on the two sublattices to have a non uniform doping layout
    hopping = Dict( 
        1 => Dict((2, 0) => t1, (0, 2) => t1, (-2, 0) => t1, (0, -2) => t1),
        2 => Dict((2, 0) => t2, (0, 2) => t2, (-2, 0) => t2, (0, -2) => t2)
    )
    pairing = Dict(
        1 => Dict((2, 0) => Δ1_x, (-2, 0) => Δ1_x, (0, 2) => Δ1_y, (0, -2) => Δ1_y),
        2 => Dict((2, 0) => Δ2_x, (-2, 0) => Δ2_x, (0, 2) => Δ2_y, (0, -2) => Δ2_y)
    )
    H_BdG = default_BCS_hamiltonian(hopping, pairing, μ, lattice)
    _ = GfPEPS.solve_for_mu(lattice, δ_target, H_BdG)
    Ψ_trial = Gaussian_fPEPS(Nf, Λ, lattice, H_BdG; doping_kwargs=doping_kwargs);

    @testset "Average doping in unit cell" begin
        # test if mean doping in unit cell is correct
        @test doping_kwargs.δ ≈ δ_target atol=doping_kwargs.density_tol
        # test if we can extract the correct doping values from the unit cell and that the mean matches δ_target
        doping_layout = GfPEPS.get_doping_layout(Nf, Λ, lattice, Ψ_trial.X_opt)
        @show doping_layout
        @show mean(doping_layout)

        @test mean(doping_layout) ≈ δ_target atol=doping_kwargs.density_tol
    end;

    @testset "Doping after Gutzwiller projection" begin

        env0 = initialize_ctmrg_environment(Ψ_trial.peps, IdentityInitialization())
        env, info = leading_boundary(env0, Ψ_trial.peps; boundary_alg...)
        @test info.convergence_error ≤ 1e-8
        δ_PEPS, doping_layout = GfPEPS.doping_peps(Ψ_trial.peps,env)
        @test δ_PEPS ≈ δ_target atol=1e-3 # depends on the CTMRG convergence and χ_env_max

        # Find a fugacity *layout* z such that the doping after projection matches the
        # unprojected doping site by site. 
        z, env_projected = GfPEPS.solve_for_fugacity(
            Ψ_trial.peps, doping_layout; χ_env_max=12, atol=doping_kwargs.density_tol
        )
        peps_projected = GfPEPS.gutzwiller_project(z, Ψ_trial.peps)

        δ_PEPS_projected, doping_layout_projected = GfPEPS.doping_pepsGW(peps_projected,env_projected)
        # mean doping is preserved by the projection ...
        @test δ_PEPS_projected ≈ δ_PEPS atol=1e-3
        # and so is the site-resolved layout, to the tolerance the fugacity solve was run at
        @test doping_layout_projected ≈ doping_layout atol=1e-3
    end
end;