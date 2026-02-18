const DEFAULT_PENALTY_FALLBACK = 1.0

"""
        optimize_stage_with_density(loss_energy, doping_fn, X_init; kwargs...)

Run one optimization stage on the Stiefel manifold and minimize the energy (given by `loss_energy`) and enforce a
target hole density. When `enforce_density` is false the routine solves the
unconstrained problem to obtain the best energy-only optimum.

Arguments
---------
- `loss_energy`: Scalar-valued objective returning the mean energy for a given `X`.
- `doping_fn`: function to compute the doping from `X`.
- `X_init`: Starting point.
- `δ`: Desired hole density.
- `λ`: Initial penalty parameter for the augmented loss function.
- `density_tol`: Absolute tolerance on the density constraint.
- `penalty_growth`: Factor controlling how aggressively the penalty grows when
    the constraint is violated.
- `outer_iters`: Number of outer augmented-loss function updates.
- `enforce_density`: If set to false, no density constraint is enforced.

Returns the optimized `X`, the `Optim` result struct, and the final hole density.
"""
function optimize_stage_with_density(loss_energy::Function, doping_fn::Union{Function, Nothing}, X_init::AbstractMatrix;
    δ::Float64,
    λ::Real,
    grad_tol::Float64,
    f_reltol::Float64,
    maxiter::Int,
    show_trace::Bool,
    stage_label::AbstractString,
    density_tol::Float64,
    penalty_growth::Float64,
    outer_iters::Int,
    enforce_density::Bool)

    if !enforce_density
        # No constraint: minimize the pure energy objective on the Stiefel manifold.
        grad_energy(x) = first(Zygote.gradient(loss_energy, x))
        grad_energy!(G, x) = copyto!(G, grad_energy(x))
        res = Optim.optimize(loss_energy, grad_energy!, X_init, Optim.BFGS(; manifold=Optim.Stiefel()), Optim.Options(
            iterations = maxiter,
            g_tol = grad_tol,
            show_trace = show_trace,
            successive_f_tol = 10,
            f_reltol = f_reltol,
            store_trace = true,
            extended_trace = false
        ))
        X_opt = Optim.minimizer(res)
        return X_opt, res, nothing
    end

    # Set up augmented-Lagrangian variables.
    η = 0.0
    ρ = (λ isa Real && λ > 0) ? float(λ) : DEFAULT_PENALTY_FALLBACK
    if !isfinite(ρ) || ρ <= 0
        ρ = DEFAULT_PENALTY_FALLBACK
    end

    X_current = X_init
    last_res = nothing
    last_doping = doping_fn(X_current)

    for outer_iter in 1:max(outer_iters, 1)
        η_local = η
        ρ_local = ρ

        loss_augmented(x) = begin
            dens = doping_fn(x)
            constraint = dens - δ
            return loss_energy(x) + η_local * constraint + 0.5 * ρ_local * constraint^2
        end
        grad_aug(x) = first(Zygote.gradient(loss_augmented, x))
        grad_aug!(G, x) = copyto!(G, grad_aug(x))

        res = Optim.optimize(loss_augmented, grad_aug!, X_current, Optim.BFGS(; manifold=Optim.Stiefel()), Optim.Options(
            iterations = maxiter,
            g_tol = grad_tol,
            show_trace = show_trace,
            successive_f_tol = 10,
            f_reltol = f_reltol,
            store_trace = true,
            extended_trace = false
        ))

        last_res = res
        X_current = Optim.minimizer(res)
        last_doping = doping_fn(X_current)
        constraint = last_doping - δ

        if abs(constraint) <= density_tol
            ρ = ρ_local
            break
        end

        η = η_local + ρ_local * constraint
        ρ = max(ρ_local * penalty_growth, DEFAULT_PENALTY_FALLBACK)
    end

    last_res === nothing && error("Augmented Lagrangian did not run for stage $(stage_label).")

    return X_current, last_res, last_doping
end

"""
    get_X_opt(Nf, Nv, t, μ, pairing_type, Δ_0; kwargs...)

Optimize the fiducial correlation matrix parameterized by an orthogonal matrix `X`
so that the resulting Gaussian state minimizes the mean energy density for the
specified BCS Hamiltonian. When a non-zero target hole density `δ` is provided,
the routine enforces it via an augmented Lagrangian across a hierarchy of system
sizes before tackling the full lattice.

Keyword arguments include lattice dimensions, optimization tolerances, and
augmented-Lagrangian controls (`density_tol`, `density_outer_iters`, `penalty_growth`).
The function returns the optimized matrix `X`, the resulting energy, and the
exact BCS energy for comparison.
"""
function get_X_opt(Nf::Int, Nv::Int, params::Union{BCS,Kitaev};
    δ::Float64 = 0.0,
    solve_μ_from_δ::Bool = false,
    enforce_density::Bool = false,
    λ::Float64 = 1e2, # initial penalty parameter for hole density constraint
    Lx::Int = 6, 
    Ly::Int = 6,
    bc::Tuple{Symbol, Symbol}=(:APBC, :PBC),
    parity::Int = 1, # 1 for even, -1 for odd
    maxiter::Int=500,
    show_trace::Bool=false,
    grad_tol::Float64=1.0e-8,
    f_reltol::Float64=1.0e-10,
    seed::Int=1234,
    density_tol::Float64=1.0e-6,
    density_outer_iters::Int=8,
    penalty_growth::Float64=10.0,
    X_init::Union{AbstractMatrix, Nothing}=nothing
    )

    MKL.set_num_threads(Sys.CPU_THREADS) 

    Random.seed!(seed)

    # initial ortogonal matrix X to construct Γ_out with correct parity sector (even)
    X = begin
        if isnothing(X_init)
            rand_CM(Nf,Nv; parity=parity)[2]
        else
            @info "Using initial X matrix for optimization."
            X_init
        end
    end

    # construct Brillouin zone
    bz = BrillouinZone2D(Lx,Ly,bc)
    has_dirac_points(bz,params) # warn if dirac points are present

    if solve_μ_from_δ && params isa BCS
        μ = solve_for_mu(bz, δ, params.t, params.pairing_type, params.Δ_0)
        params = BCS(params.t, μ, params.pairing_type, params.Δ_0)
    end

    # smaller system size for initial optimization to find better starting point
    L_init_even = 6
    L_init_odd = 5
    Lx_inits = begin
        if Lx > L_init_even && isnothing(X_init)
            [isodd(Lx) ? L_init_odd : L_init_even]
        else
            []
        end
    end
    # Lx_inits = [(isodd(Lx) && Lx > L_init_even) ? L_init_odd : L_init_even]
    Ly_inits = begin
        if Ly > L_init_even && isnothing(X_init)
            [isodd(Ly) ? L_init_odd : L_init_even]
        else
            []
        end
    end

    try
        while 2*Lx_inits[end] < Lx 
            if isodd(Lx_inits[end])
                push!(Lx_inits, Lx_inits[end]*2 - 1)
            else
                push!(Lx_inits, Lx_inits[end] * 2)
            end
        end
        while 2*Ly_inits[end] < Ly
            if isodd(Ly_inits[end])
                push!(Ly_inits, Ly_inits[end]*2 - 1)
            else
                push!(Ly_inits, Ly_inits[end] * 2)
            end
        end
    catch
        
    end

    if enforce_density
        @info "Target hole density δ = $(δ) will be enforced with tolerance $(density_tol)."
    end

    if !isempty(Lx_inits)
       @info "Finding better initial guess for X by solving smaller system sizes..."
    end

    warmup_sizes = collect(zip(Lx_inits, Ly_inits))
    warmup_iterations = 1000

    for (stage_idx, (Lx_init, Ly_init)) in enumerate(warmup_sizes)

        stage_label = "Warmup stage $(stage_idx) (Lx=$(Lx_init), Ly=$(Ly_init))"
        @info "Optimize X for: Lx = $(Lx_init), Ly = $(Ly_init)"

        bz_init = BrillouinZone2D(Lx_init, Ly_init, bc)
        has_dirac_points(bz_init, params)

        loss_init_no_dens = optimize_loss(bz_init, Nf, Nv, params)
        doping_fn_init = enforce_density ?  X_mat -> doping_bcs(X_mat, bz_init, Nf, Nv) : nothing

        # Use the stage optimizer to refine the initial guess before scaling up.
        X, res_stage, stage_doping = optimize_stage_with_density(loss_init_no_dens, doping_fn_init, X;
            δ = δ,
            λ = λ,
            grad_tol = grad_tol,
            f_reltol = f_reltol,
            show_trace = show_trace,
            maxiter = warmup_iterations,
            stage_label = stage_label,
            density_tol = density_tol,
            penalty_growth = penalty_growth,
            outer_iters = density_outer_iters,
            enforce_density = enforce_density
        )

        if Optim.converged(res_stage)
            if enforce_density
                @info "$(stage_label) converged after $(res_stage.iterations) iterations." energy=Optim.minimum(res_stage) doping=stage_doping
            else
                @info "$(stage_label) converged after $(res_stage.iterations) iterations." energy=Optim.minimum(res_stage)
            end
        else
            if enforce_density
                @warn "$(stage_label) did not converge." gradient_norm=res_stage.g_residual energy=Optim.minimum(res_stage) doping=stage_doping
            else
                @warn "$(stage_label) did not converge." gradient_norm=res_stage.g_residual energy=Optim.minimum(res_stage)
            end
        end
    end

    loss_no_dens = optimize_loss(bz, Nf, Nv, params)
    doping_fn = enforce_density ? X_val -> doping_bcs(X_val, bz, Nf, Nv) : nothing

    @info "Finding optimal X for full system size..."
    stage_label = "Final optimization (Lx=$(Lx), Ly=$(Ly))"
    @info "Optimize X for: Lx = $(Lx), Ly = $(Ly)"

    # Final pass on the target lattice size.
    X_opt, res_final, final_doping = optimize_stage_with_density(loss_no_dens, doping_fn, X;
        δ = δ,
        λ = λ,
        grad_tol = grad_tol,
        f_reltol = f_reltol,
        show_trace = show_trace,
        maxiter = maxiter,
        stage_label = stage_label,
        density_tol = density_tol,
        penalty_growth = penalty_growth,
        outer_iters = density_outer_iters,
        enforce_density = enforce_density
    )

    if Optim.converged(res_final)
        @info "$(stage_label) converged after $(res_final.iterations) iterations."
    else
        @warn "$(stage_label) did not converge." gradient_norm=res_final.g_residual
    end

    constraint_final = enforce_density ? final_doping - δ : nothing
    if enforce_density
        @info "Final doping summary" target=δ achieved=final_doping deviation=constraint_final
    end
    if enforce_density && abs(constraint_final) > density_tol
        @warn "Final doping deviates from target by $(constraint_final). Consider increasing density_outer_iters or penalty_growth."
    end

    E_exact = exact_energy(params, bz)
    optim_energy = Optim.minimum(res_final)
    deviation = abs(optim_energy - E_exact)
    
    @info "Final energy summary" target=E_exact achieved=optim_energy deviation=deviation
    println()

    info_obj = (
        converged = Optim.converged(res_final),
        trace = res_final.trace
    )

    return X_opt, optim_energy, E_exact, info_obj
end

function get_X_opt(;conf=parsefile(joinpath(GfPEPS.config_path, "conf_BCS_d_wave.json")), X_init::Union{AbstractMatrix, Nothing}=nothing) 
    params = begin
        if conf["hamiltonian"]["type"] == "BCS"
            BCS(
                conf["hamiltonian"]["t"],
                conf["hamiltonian"]["μ"],
                conf["hamiltonian"]["pairing_type"],
                conf["hamiltonian"]["Δ_0"],
                get(get(conf, "params", Dict()), "Δ_02", 0.0)
            )
        elseif conf["hamiltonian"]["type"] == "Kitaev"
            Kitaev(
                conf["hamiltonian"]["Jx"],
                conf["hamiltonian"]["Jy"],
                conf["hamiltonian"]["Jz"]
            )
        else
            error("Unknown Hamiltonian type: $(conf["hamiltonian"]["type"])")
        end
    end
    
    return get_X_opt(
        conf["params"]["N_physical_fermions_on_site"],
        conf["params"]["N_virtual_fermions_on_bond"],
        params;
        δ = get(get(conf, "hamiltonian", Dict()), "hole_density", 0.0),
        solve_μ_from_δ = get(get(conf, "hamiltonian", Dict()), "μ_from_hole_density", false),
        enforce_density = get(get(conf, "hamiltonian", Dict()), "enforce_density", false),
        λ = get(get(conf, "hamiltonian", Dict()), "lagrange_multiplier_density", 1e2),
        Lx = conf["system_params"]["Lx"],
        Ly = conf["system_params"]["Ly"],
        bc = (Symbol(conf["system_params"]["x_bc"]), Symbol(conf["system_params"]["y_bc"])),
        parity = get(get(conf, "system_params", Dict()), "parity", 1),
        maxiter = get(get(conf, "params", Dict()), "maxiter", 1000),
        show_trace = get(get(conf, "params", Dict()), "show_trace", false),
        grad_tol = get(get(conf, "params", Dict()), "grad_tol", 1e-8),
        f_reltol = get(get(conf, "params", Dict()), "f_reltol", 1e-10),
        seed = get(get(conf, "params", Dict()), "seed", 1234),
        density_tol =  get(get(conf, "hamiltonian", Dict()), "density_tol", 1e-6),
        density_outer_iters = get(get(conf, "hamiltonian", Dict()), "density_outer_iters", 10),
        penalty_growth = get(get(conf, "hamiltonian", Dict()), "penalty_growth", 1e1),
        X_init = X_init
    )
end