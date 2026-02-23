const DEFAULT_PENALTY_FALLBACK = 1.0 # fallback value for penalty parameter in the augmented Lagrangian method

""" 
    DopingSettings

Struct to hold settings for doping optimization in the augmented Lagrangian method.
# Fields
- `δ::Float64=0.0`: Target hole density to enforce.
- `density_tol::Float64=1e-6`: Tolerance for how close the doping must be to the target δ to consider the constraint satisfied.
- `penalty_growth::Float64=1e2`: Factor by which to increase the penalty parameter λ if the constraint is not satisfied.
- `enforce_density::Bool=false`: Whether to enforce the density constraint or not.
- `density_opt_iters::Int=8`: Maximum number of iterations for the density optimization loop.
- `λ::Float64=1e2`: Initial penalty parameter for the augmented Lagrangian method. If not positive and finite, a default fallback value will be used.
"""
struct DopingSettings
    δ::Float64
    density_tol::Float64
    penalty_growth::Float64
    enforce_density::Bool
    density_opt_iters::Int
    λ::Float64
    solve_μ_from_δ::Bool

    function DopingSettings(; 
        δ=0.0, 
        density_tol=1e-6, 
        penalty_growth=1e1, 
        enforce_density=false, 
        density_opt_iters=10,
        λ=1e2,
        solve_μ_from_δ=true)

        new(δ, density_tol, penalty_growth, enforce_density, density_opt_iters, λ, solve_μ_from_δ)
    end
end

"""
    get_X_opt(lattice::AbstractInfiniteLattice, Nf::Int, Λ::Int, BCS_params::BCS; 
        X_init::Union{AbstractMatrix, Nothing}=nothing,
        doping_kwargs::DopingSettings=DopingSettings(),
        optim_LBFGS::Optim.LBFGS = Optim.LBFGS(; m=20, manifold = Optim.Stiefel()),
        optim_options::Optim.Options = Optim.Options(; iterations=1000, g_tol=1e-8, f_reltol=1e-10, successive_f_tol = 10, show_trace=false, extended_trace=false))

Get the optimal orthogonal X matrix for the GfPEPS approximation of the ground state of a BCS Hamiltonian, such that the Covariance matrix `Γ=Γ_fiducial(X,Λ,Nf)` minimizes the energy between the BCS Hamiltonian and this GfPEPS approximation.

# Keyword Arguments
- `lattice::AbstractInfiniteLattice`: The lattice for which to optimize X.
- `Nf::Int`: Number of physical fermions.
- `Λ::Int`: The bond dimension parameter for the PEPS ansatz.
- `H_bdg::AbstractBdGHamiltonian`: The BdG Hamiltonian object.

# Optional Keyword arguments
- `X_init::Union{AbstractMatrix, Nothing}=nothing`: Optional initial guess for the X matrix. If not provided, a random X matrix will be generated. If provided, we start the optimization for the full system size directly with this initial X for the case we want to continue optimization from a previous run.
- `doping_kwargs::DopingSettings=DopingSettings()`: Settings for doping optimization in the augmented Lagrangian method.
- `optim_LBFGS::Optim.LBFGS = Optim.LBFGS(; m=20, manifold = Optim.Stiefel())`: The LBFGS optimizer to use for optimization on the Stiefel manifold.
- `optim_options::Optim.Options = Optim.Options(; iterations=1000, g_tol=1e-8, f_reltol=1e-10, successive_f_tol = 10, show_trace=false, extended_trace=false)`: Options for the Optim optimizer.

# Returns
- `X_opt::AbstractMatrix`: The optimal orthogonal X matrix found by the optimization.
- `optim_energy::Float64`: The energy corresponding to the optimal orthogonal X matrix.
- `E_exact::Float64`: The exact energy for the given parameters of the quadratic Hamiltonian from analytic formula.
- `info_obj::NamedTuple`: A named tuple containing additional information about the optimization where the returned values are from Optim.

"""
function get_X_opt(
    lattice::AbstractInfiniteLattice, 
    Nf::Int, 
    Λ::Int,
    H_bdg::AbstractBdGHamiltonian;
    X_init::Union{AbstractMatrix, Nothing}=nothing,
    doping_kwargs::DopingSettings=DopingSettings(),
    optim_alg_options::Union{Optim.LBFGS, Optim.BFGS} = Optim.LBFGS(; m=20, manifold = Optim.Stiefel()),
    optim_options::Optim.Options = Optim.Options(; iterations=1000, g_tol=1e-6, f_reltol=1e-8, successive_f_tol = 10, show_trace=false, extended_trace=false, store_trace=true))

    # TODO: implement / test odd parity optimization
    # initial ortogonal matrix X to construct Γ_Q with correct parity sector (even = 1, odd = -1)
    X_opt = begin
        if isnothing(X_init)
            rand_CM(Nf, Λ, lattice; parity=1)[2]
        else
            @info "Using initial X matrix for optimization."
            X_init
        end
    end

    # warn if dirac points are present -> then optimization of Γ can be harder, so user can adjust different kvals set
    has_dirac_points(lattice.kvals, H_bdg) 

    if doping_kwargs.enforce_density && doping_kwargs.solve_μ_from_δ
        H_bdg.μ = solve_for_mu(lattice.kvals, doping_kwargs.δ, H_bdg)
    end

    # smaller set of momentum pairs for initial optimization for faster convergence
    N_kx_inits, N_ky_inits = isnothing(X_init) ? get_kgrids(lattice) : ([lattice.N_kx], [lattice.N_ky])
    
    if doping_kwargs.enforce_density
        @info "Target hole density δ = $(doping_kwargs.δ) will be enforced with tolerance $(doping_kwargs.density_tol)."
    end

    if !isempty(N_kx_inits) && !isempty(N_ky_inits)
       @info "Finding better initial guess for X by solving smaller system sizes..."
    end

    size_stages = collect(zip(N_kx_inits, N_ky_inits))
    stage_res = nothing
    stage_doping = nothing
    for (stage_idx, (N_kx_init, N_ky_init)) in enumerate(size_stages)
        stage_label = !(N_kx_init==lattice.N_kx) ? "Warmup stage $(stage_idx) (N_kx=$(N_kx_init), N_ky=$(N_ky_init))" : "Final optimization stage (N_kx=$(N_kx_init), N_ky=$(N_ky_init))"
        @info "Optimize X for: N_kx = $(N_kx_init), N_ky = $(N_ky_init)"

        training_lattice = InfiniteRectLattice(lattice.Lx, lattice.Ly; N_kx=N_kx_init, N_ky=N_ky_init, bc=lattice.bc, shift_x=lattice.shift_x, shift_y=lattice.shift_y)
        has_dirac_points(training_lattice.kvals, H_bdg)

        loss_fct = energy_loss_X(training_lattice, Nf, Λ, H_bdg)
        doping_fct = doping_loss_X(training_lattice, Nf, Λ)

        # optimize X for current stage and get energy and doping results
        X_opt, stage_res, stage_doping = optimize_X(X_opt, loss_fct, doping_fct; doping_kwargs=doping_kwargs, optim_alg_options=optim_alg_options, optim_options=optim_options)

        if Optim.converged(stage_res)
            if doping_kwargs.enforce_density
                @info "$(stage_label) converged after $(stage_res.iterations) iterations." energy=Optim.minimum(stage_res) doping=stage_doping
            else
                @info "$(stage_label) converged after $(stage_res.iterations) iterations." energy=Optim.minimum(stage_res)
            end
        else
            if doping_kwargs.enforce_density
                @warn "$(stage_label) did not converge after $(stage_res.iterations) iterations." gradient_norm=stage_res.g_residual energy=Optim.minimum(stage_res) doping=stage_doping
            else
                @warn "$(stage_label) did not converge after $(stage_res.iterations) iterations." gradient_norm=stage_res.g_residual energy=Optim.minimum(stage_res)
            end
        end
    end

    # final results summary and check if density constraint is satisfied
    constraint_final = doping_kwargs.enforce_density ? stage_doping - doping_kwargs.δ : nothing
    if doping_kwargs.enforce_density
        @info "Final doping summary" target=doping_kwargs.δ achieved=stage_doping deviation=constraint_final
    end
    if doping_kwargs.enforce_density && abs(constraint_final) > doping_kwargs.density_tol
        @warn "Final doping deviates from target by $(constraint_final). Consider increasing density_opt_iters or penalty_growth."
    end

    # compute final energy and compare to exact energy
    E_exact = exact_energy(lattice.kvals, H_bdg, Nf)
    optim_energy = Optim.minimum(stage_res)
    deviation = abs(optim_energy - E_exact)
    
    @info "Final energy summary" target=E_exact achieved=optim_energy deviation=deviation
    println()

    # return final results and optimization info
    info_obj = (
        converged = Optim.converged(stage_res),
        trace = stage_res.trace
    )

    return X_opt, optim_energy, E_exact, info_obj
end


"""
    get_kgrids(lattice::AbstractInfiniteLattice)

Returns an array which also includes smaller k arrays for initial optimization stages, starting from a small size and growing until just below the target lattice size. 
This helps find a better initial guess for X before optimizing on the full lattice.
"""
function get_kgrids(lattice::AbstractInfiniteLattice)
    function grow_sizes(N_final, N_start)
        res = Int[]
        N_final > N_start && push!(res, N_start)
        while !isempty(res) && 2 * res[end] < N_final
            push!(res, isodd(res[end]) ? res[end] * 2 - 1 : res[end] * 2)
        end
        return res
    end

    N_kx_inits = grow_sizes(lattice.N_kx, isodd(lattice.N_kx) ? 5 : 6)
    N_ky_inits = grow_sizes(lattice.N_ky, isodd(lattice.N_ky) ? 5 : 6)
    
    # add final sizes
    push!(N_kx_inits, lattice.N_kx)
    push!(N_ky_inits, lattice.N_ky)

    return N_kx_inits, N_ky_inits
end

"""
    optimize_X(X_init::AbstractMatrix, loss_fct::Function, doping_fct::Function; 
        doping_kwargs::DopingSettings=DopingSettings(),
        optim_LBFGS::Optim.LBFGS=Optim.LBFGS(; m=20, manifold = Optim.Stiefel()),
        optim_options::Optim.Options=Optim.Options(; iterations=1000, g_tol=1e-8, f_reltol=1e-10, successive_f_tol = 10, show_trace=false, extended_trace=false))

Optimize the orthogonal X matrix to minimize the given loss function, optionally enforcing a density constraint using an augmented Lagrangian method.

# Arguments
- `X_init::AbstractMatrix`: Initial guess for the orthogonal X matrix.
- `loss_fct::Function`: A function that takes an orthogonal X matrix and returns the energy loss to minimize.
- `doping_fct::Function`: A function that takes an orthogonal X matrix and returns the doping level, used for enforcing the density constraint. If `doping_kwargs.enforce_density` is false, this can be set to `nothing`.

# Optional Keyword Arguments
- `doping_kwargs::DopingSettings=DopingSettings()`: Settings for doping optimization in the augmented Lagrangian method.
- `optim_LBFGS::Optim.LBFGS`: The LBFGS optimizer to use for optimization on the Stiefel manifold.
- `optim_options::Optim.Options`: Options for the Optim optimizer.

# Returns
- `X_opt::AbstractMatrix`: The optimal orthogonal X matrix found by the optimization.
- `res::Optim.MultivariateOptimizationResults`: The result object returned by the Optim optimization, containing information about convergence and the optimization trace.
- `final_doping::Union{Float64, Nothing}`: The doping level corresponding to the optimal orthogonal X matrix if density constraint is enforced, otherwise `nothing`.

"""
function optimize_X(X::AbstractMatrix, loss_fct::Function, doping_fct::Function;
    doping_kwargs::DopingSettings=DopingSettings(),
    optim_alg_options::Union{Optim.LBFGS, Optim.BFGS},
    optim_options::Optim.Options)

     # No density constraint: minimize the pure energy objective on the Stiefel manifold.
    if !doping_kwargs.enforce_density
        grad_energy(x) = first(Zygote.gradient(loss_fct, x))
        grad_energy!(G, x) = copyto!(G, grad_energy(x))

        res = Optim.optimize(loss_fct, grad_energy!, X, optim_alg_options, optim_options)
        X_opt = Optim.minimizer(res)
        return X_opt, res, nothing
    end

    # Density constraint: use a penalty term to enforce the density constraint while minimizing the energy.

    # Set up augmented-Lagrangian variables.
    η = 0.0
    λ = doping_kwargs.λ

    # run density optimization loop, where in each iteration we minimize the augmented Lagrangian with the current penalty and multiplier, then update those based on the constraint violation.
    X_current = X
    last_res = nothing
    last_doping = doping_fct(X_current)
    total_iters = 0
    total_trace = []
    for _ in 1:max(doping_kwargs.density_opt_iters, 1) # usually only a few iterations are needed
        η_local = η
        λ_local = λ

        # new loss funciton with penalty term for density constraint
        loss_augmented(x) = begin
            dens = doping_fct(x)
            constraint = dens - doping_kwargs.δ
            return loss_fct(x) + η_local * constraint + 0.5 * λ_local * constraint^2
        end
        grad_aug(x) = first(Zygote.gradient(loss_augmented, x))
        grad_aug!(G, x) = copyto!(G, grad_aug(x))

        res = Optim.optimize(loss_augmented, grad_aug!, X_current, optim_alg_options, optim_options)
        total_iters += res.iterations
        total_trace = vcat(total_trace, res.trace)

        # update augmented Lagrangian parameters based on constraint violation
        last_res = res
        X_current = Optim.minimizer(res)
        last_doping = doping_fct(X_current)
        constraint = last_doping - doping_kwargs.δ

        if abs(constraint) <= doping_kwargs.density_tol
            λ = λ_local
            break
        end

        η = η_local + λ_local * constraint
        λ = max(λ_local * doping_kwargs.penalty_growth, DEFAULT_PENALTY_FALLBACK)
    end
    last_res === nothing && error("Augmented Lagrangian did not run for stage $(stage_label).")

    # update iterations
    last_res.iterations = total_iters
    last_res.trace = total_trace

    return X_current, last_res, last_doping
end