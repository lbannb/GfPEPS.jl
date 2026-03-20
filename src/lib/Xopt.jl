const DEFAULT_PENALTY_FALLBACK = 1.0 # fallback value for penalty parameter in the augmented Lagrangian method
""" 
    DopingSettings

Struct to hold settings for doping optimization in the augmented Lagrangian method.
# Fields
- `δ::Float64=0.0`: Target hole density to enforce.
- `doping_layout::Matrix{Float64}`: Optional matrix specifying the target doping for each site in the unit cell. If not provided, a uniform doping of δ will be assumed for all sites.
- `density_tol::Float64=1e-6`: Tolerance for how close the doping must be to the target δ to consider the constraint satisfied.
- `penalty_growth::Float64=1e2`: Factor by which to increase the penalty parameter λ if the constraint is not satisfied.
- `enforce_density::Bool=false`: Whether to enforce the density constraint or not.
- `density_opt_iters::Int=8`: Maximum number of iterations for the density optimization loop.
- `λ::Float64=1e2`: Initial penalty parameter for the augmented Lagrangian method. If not positive and finite, a default fallback value will be used.
- `solve_μ_from_δ::Bool=true`: Whether to solve for the chemical potential μ from the target doping δ before optimization. If false, the provided μ in the Hamiltonian will be used and the doping constraint will be enforced around that.
"""
mutable struct DopingSettings
    δ::Float64
    doping_layout::Union{Nothing, Matrix{Float64}}
    density_tol::Float64
    penalty_growth::Float64
    enforce_density::Bool
    density_opt_iters::Int
    λ::Float64
    solve_μ_from_δ::Bool

    function DopingSettings(; 
        δ=0.0, 
        doping_layout=nothing,
        density_tol=1e-6, 
        penalty_growth=5, 
        enforce_density=false, 
        density_opt_iters=10,
        λ=1e2,
        solve_μ_from_δ=true)

        if !isnothing(doping_layout)
            @assert δ ≈ mean(doping_layout) "The target doping δ should match the average of the provided doping_layout."
        end

        new(δ, doping_layout, density_tol, penalty_growth, enforce_density, density_opt_iters, λ, solve_μ_from_δ)
    end
end

"""
    get_X_opt(lattice::AbstractInfiniteLattice, Nf::Int, Λ::Int, BCS_params::BCS; 
        X_init::Union{AbstractMatrix, Nothing}=nothing,
        doping_kwargs::DopingSettings=DopingSettings(),
        optim_alg_options::Union{Optim.LBFGS, Optim.BFGS} = Optim.LBFGS(; m=20, manifold = Optim.Stiefel(:CholQR)),
        optim_options::Optim.Options = Optim.Options(; iterations=1000, g_tol=1e-6, f_reltol=1e-8, successive_f_tol = 10, show_trace=false, extended_trace=false, store_trace=true))

Get the optimal orthogonal X matrix for the GfPEPS approximation of the ground state of a BCS Hamiltonian, such that the Covariance matrix `Γ=Γ_fiducial(X,Λ,Nf)` minimizes the energy between the BCS Hamiltonian and this GfPEPS approximation.

# Keyword Arguments
- `lattice::AbstractInfiniteLattice`: The lattice for which to optimize X.
- `Nf::Int`: Number of physical fermions.
- `Λ::Int`: The bond dimension parameter for the PEPS ansatz.
- `H_bdg::AbstractBdGHamiltonian`: The BdG Hamiltonian object.

# Optional Keyword arguments
- `X_init::Union{AbstractMatrix, Nothing}=nothing`: Optional initial guess for the X matrix. If not provided, a random X matrix will be generated. If provided, we start the optimization for the full system size directly with this initial X for the case we want to continue optimization from a previous run.
- `doping_kwargs::DopingSettings=DopingSettings()`: Settings for doping optimization in the augmented Lagrangian method.
- `optim_alg_options::Union{Optim.LBFGS, Optim.BFGS} = Optim.LBFGS(; m=15, manifold = Optim.Stiefel(:CholQR))`: LBFGS optimization options, with a custom manifold that includes gauge projection. By default, uses L-BFGS with memory 15 and CholQR retraction.
- `optim_options::Optim.Options = Optim.Options(; iterations=1000, g_tol=1e-6, f_reltol=1e-8, successive_f_tol = 10, show_trace=false, extended_trace=false, store_trace=true)`: Options for the Optim optimizer.

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
    X_init::Union{AbstractVector{<:AbstractMatrix}, Nothing}=nothing,
    doping_kwargs::DopingSettings=DopingSettings(),
    optim_alg_options::Union{Optim.LBFGS, Optim.BFGS} = Optim.LBFGS(; m=15, manifold = Optim.Stiefel(:CholQR)),
    optim_options::Optim.Options = Optim.Options(; iterations=1000, g_tol=1e-6, f_reltol=1e-8, successive_f_tol = 10, show_trace=false, extended_trace=false, store_trace=true))

    #= 
    build gauge-projected manifold for better L-BFGS convergence.
    For per-site ansatz the optimisation variable is a block-diagonal
    matrix X_total = diag(X_1, …, X_N) living on the product Stiefel manifold.
    The block-diagonal J_total preserves block-diagonality. 
    =#
    J_single = build_J(Λ, Nf) # 2(Nf+4Λ) × 2(Nf+4Λ)
    gauge_manifold = GaugeFixedStiefel(J_single)
    optim_alg_options = with_manifold(optim_alg_options, gauge_manifold)

    # each distinct site in the unit cell has its own X matrix
    X_opt = isnothing(X_init) ? [rand_CM(Nf, Λ; parity=1)[2] for i in 1:get_number_of_distinct_sites_in_uc(lattice)] : X_init

    # warn if dirac points are present -> then optimization of Γ can be harder, so user can adjust different kvals set
    has_dirac_points(lattice.kvals, H_bdg) 

    # set doping layout to uniform if not provided
    doping_kwargs.doping_layout = isnothing(doping_kwargs.doping_layout) ? fill(doping_kwargs.δ, size(lattice.uc_layout)) : doping_kwargs.doping_layout

    if doping_kwargs.enforce_density && doping_kwargs.solve_μ_from_δ
        H_bdg.μ = solve_for_mu(lattice, doping_kwargs.δ, H_bdg)
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

        training_lattice = InfiniteRectLattice(lattice.Lx, lattice.Ly;
            uc_layout=lattice.uc_layout,
            N_kx=N_kx_init,
            N_ky=N_ky_init,
            bc=lattice.bc,
            shift_x=lattice.shift_x,
            shift_y=lattice.shift_y)
        has_dirac_points(training_lattice.kvals, H_bdg)

        loss_fct = energy_loss_X(training_lattice, Nf, Λ, H_bdg)
        doping_fct = doping_loss_X(training_lattice, Nf, Λ)

        # optimize the packed (block-diagonal) X on the product Stiefel manifold
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
    E_exact = exact_energy(lattice, H_bdg)
    optim_energy = Optim.minimum(stage_res)
    deviation = abs(optim_energy - E_exact)
    
    @info "Final energy summary" target=E_exact achieved=optim_energy deviation=deviation
    println()

    # unpack the optimised block-diagonal matrix back into per-site X matrices
    X_mat = X_matrix_form(X_opt, lattice)

    # return final results and optimization info
    info_obj = (
        converged = Optim.converged(stage_res),
        trace = stage_res.trace
    )

    return X_mat, optim_energy, E_exact, info_obj
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
    optimize_X(Xs, loss_fct, doping_fct; kwargs...)

Optimize a vector of per-site orthogonal matrices `Xs` on the product Stiefel
manifold to minimize `loss_fct`, optionally enforcing a density constraint via
an augmented Lagrangian.

Internally the `Vector{Matrix}` is packed into a 3D `Array{Float64,3}` of shape
`(n, n, Nsites)` so that `eltype` is `Float64`, which is required by Optim.jl.
Loss closures that accept `AbstractVector{<:AbstractMatrix}` are wrapped
automatically to slice the 3D array before evaluation.

# Arguments
- `Xs::AbstractVector{<:AbstractMatrix}`: Initial guess — one orthogonal matrix per distinct site in the unit cell.
- `loss_fct::Function`: `Xs -> energy` loss (accepts a vector of per-site X matrices).
- `doping_fct::Function`: `Xs -> doping` (can be `nothing` when density is not enforced).

# Keyword Arguments
- `doping_kwargs::DopingSettings`: Augmented Lagrangian settings.
- `optim_alg_options::Union{Optim.LBFGS, Optim.BFGS}`: Optimizer with manifold.
- `optim_options::Optim.Options`: Convergence tolerances and tracing.

# Returns
- `X_opt::Vector{Matrix{Float64}}`: Optimized per-site orthogonal matrices.
- `res`: Optim result object (convergence info, trace).
- `final_doping::Union{Float64, Nothing}`: Achieved doping, or `nothing`.

"""
function optimize_X(Xs::AbstractVector{<:AbstractMatrix}, loss_fct::Function, doping_fct::Function;
    doping_kwargs::DopingSettings=DopingSettings(),
    optim_alg_options::Union{Optim.LBFGS, Optim.BFGS},
    optim_options::Optim.Options)

    # ── Pack Vector{Matrix} → 3D Array (n × n × Nsites) for Optim ──────
    Nsites = length(Xs)
    n = size(Xs[1], 1)
    X3d = Array{Float64, 3}(undef, n, n, Nsites)
    for s in 1:Nsites
        X3d[:, :, s] .= Xs[s]
    end

    # Wrap a Vector{Matrix}-accepting closure into one that accepts a 3D array.
    _slices(X3d) = [@view X3d[:, :, s] for s in axes(X3d, 3)]
    loss3d(X3d)    = loss_fct(_slices(X3d))
    doping3d(X3d)  = doping_fct(_slices(X3d))

    # No density constraint: minimize the pure energy objective on the Stiefel manifold.
    if !doping_kwargs.enforce_density
        grad_energy!(G, x) = copyto!(G, first(Zygote.gradient(loss3d, x)))
        function energy_fvg!(G, x)
            val, back = Zygote.pullback(loss3d, x)
            copyto!(G, first(back(one(val))))
            return val
        end

        optimized_loss = Optim.OnceDifferentiable(loss3d, grad_energy!, energy_fvg!, X3d)
        res = Optim.optimize(optimized_loss, X3d, optim_alg_options, optim_options)
        X_opt_3d = Optim.minimizer(res)
        X_opt = [X_opt_3d[:, :, s] for s in 1:Nsites]
        return X_opt, res, nothing
    end

    # Density constraint: use a penalty term to enforce the density constraint while minimizing the energy.

    # Set up augmented-Lagrangian variables.
    η = 0.0
    λ = doping_kwargs.λ

    # run density optimization loop, where in each iteration we minimize the augmented Lagrangian with the current penalty and multiplier, then update those based on the constraint violation.
    X_current = X3d
    last_res = nothing
    last_doping = doping3d(X_current)
    total_iters = 0
    total_trace = []
    for _ in 1:max(doping_kwargs.density_opt_iters, 1) # usually only a few iterations are needed
        η_local = η
        λ_local = λ

        # new loss function with penalty term for density constraint
        loss_augmented(x) = begin
            dens = doping3d(x)
            constraint = dens - doping_kwargs.δ
            return loss3d(x) + η_local * constraint + 0.5 * λ_local * constraint^2
        end

        # Combined value-and-gradient via Zygote pullback (avoids redundant forward pass)
        grad_aug!(G, x) = copyto!(G, first(Zygote.gradient(loss_augmented, x)))
        function aug_fvg!(G, x)
            val, back = Zygote.pullback(loss_augmented, x)
            copyto!(G, first(back(one(val))))
            return val
        end

        optimized_loss = Optim.OnceDifferentiable(loss_augmented, grad_aug!, aug_fvg!, X_current)
        res = Optim.optimize(optimized_loss, X_current, optim_alg_options, optim_options)
        total_iters += res.iterations
        total_trace = vcat(total_trace, res.trace)

        # update augmented Lagrangian parameters based on constraint violation
        last_res = res
        X_current = Optim.minimizer(res)
        last_doping = doping3d(X_current)
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

    X_opt = [X_current[:, :, s] for s in 1:Nsites]
    return X_opt, last_res, last_doping
end

"""
    GaugeFixedStiefel <: Optim.Manifold

Custom manifold that combines Stiefel tangent projection with gauge projection.
Uses CholQR retraction (faster than SVD) and projects out the U(N) gauge
redundancy from tangent vectors.

Since the loss depends only on Γ = Xᵀ J X, two matrices X₁ and X₂ that differ
by a U(N) transformation (the centralizer of J in O(2N)) give identical Γ.
This creates a gauge redundancy comprising ~50% of the optimization dimensions.

The gauge projection decomposes each Lie-algebra element A ∈ so(2N) into:
  A_phys = (A + Γ A Γ) / 2   (physical: changes Γ)
  A_gauge = (A - Γ A Γ) / 2   (gauge: leaves Γ invariant)

By projecting out the gauge component from both the gradient and the L-BFGS
search direction, the Hessian approximation is not polluted by flat gauge
directions, improving convergence.

# Fields
- `J::Matrix{Float64}`: The symplectic matrix (2N × 2N), precomputed once.
- `_XG`, `_Γ`, `_ΓA`, `_ΓAΓ`: Pre-allocated scratch buffers (2N × 2N) to avoid
  allocations in the `project_tangent!` hot path.
"""
struct GaugeFixedStiefel <: Optim.Manifold
    J::Matrix{Float64}
    _XG::Matrix{Float64}
    _Γ::Matrix{Float64}
    _ΓA::Matrix{Float64}
    _ΓAΓ::Matrix{Float64}
end

"""
    GaugeFixedStiefel(J::Matrix{Float64})

Construct a `GaugeFixedStiefel` manifold from the symplectic matrix `J`.
All scratch buffers are allocated automatically.
"""
function GaugeFixedStiefel(J::Matrix{Float64})
    GaugeFixedStiefel(J, similar(J), similar(J), similar(J), similar(J))
end

"""
    Optim.retract!(m::GaugeFixedStiefel, X)

CholQR retraction: project X back to the Stiefel manifold via Cholesky-based
orthogonalization. Faster than SVD retraction (~18× for 20×20 matrices) and
numerically stable for the small steps typical of L-BFGS line searches.
"""
function Optim.retract!(m::GaugeFixedStiefel, X)
    overlap = X'X + I * 1e-15 # add small regularization for numerical stability
    X .= X / cholesky(overlap).U
end

"""
    Optim.project_tangent!(m::GaugeFixedStiefel, G, X)

Project the ambient gradient/direction `G` onto the tangent space of the Stiefel
manifold at `X`, then further project out gauge (centralizer) directions.

Steps:
1. Stiefel projection: G ← G - X · sym(XᵀG)
2. Extract Lie-algebra element: A = XᵀG (guaranteed skew-symmetric)
3. Compute covariance matrix: Γ = XᵀJX
4. Gauge projection: A_phys = (A + ΓAΓ)/2
5. Reconstruct: G ← X · A_phys
"""
function Optim.project_tangent!(m::GaugeFixedStiefel, G, X)
    # Step 1: Stiefel tangent projection
    # G_tangent = G - X · sym(X'G), where sym(M) = (M + M')/2
    mul!(m._XG, transpose(X), G)                      # _XG = X'G
    @inbounds for j in axes(m._Γ, 2), i in axes(m._Γ, 1)
        m._Γ[i,j] = (m._XG[i,j] + m._XG[j,i]) / 2   # _Γ = sym(X'G)
    end
    mul!(G, X, m._Γ, -1.0, 1.0)                       # G -= X · sym(X'G)

    # Step 2: Extract Lie-algebra element A = X'G (skew-symmetric after step 1)
    mul!(m._XG, transpose(X), G)                       # _XG = A

    # Step 3: Compute covariance matrix Γ = X'JX
    mul!(m._ΓA, m.J, X)                                # _ΓA = J·X (temp)
    mul!(m._Γ, transpose(X), m._ΓA)                    # _Γ = X'JX = Γ

    # Step 4: Gauge projection — keep only the physical component
    # A_phys = (A + Γ·A·Γ) / 2
    mul!(m._ΓA, m._Γ, m._XG)                           # _ΓA = Γ·A
    mul!(m._ΓAΓ, m._ΓA, m._Γ)                          # _ΓAΓ = Γ·A·Γ
    @. m._XG = (m._XG + m._ΓAΓ) / 2                   # _XG = A_phys

    # Step 5: Reconstruct gradient in ambient space
    mul!(G, X, m._XG)                                   # G = X · A_phys

    return G
end

"""
    Optim.retract!(m::GaugeFixedStiefel, X3d::AbstractArray{<:Real, 3})

CholQR retraction applied slice-wise to a 3D array of per-site X matrices
(product Stiefel manifold).  Each `X3d[:,:,s]` is retracted independently.
"""
function Optim.retract!(m::GaugeFixedStiefel, X3d::AbstractArray{<:Real, 3})
    for s in axes(X3d, 3)
        Xs = @view X3d[:, :, s]
        overlap = Xs'Xs + I * 1e-15
        Xs .= Xs / cholesky(overlap).U
    end
    return X3d
end

"""
    Optim.project_tangent!(m::GaugeFixedStiefel, G3d::AbstractArray{<:Real, 3}, X3d::AbstractArray{<:Real, 3})

Gauge-fixed Stiefel tangent projection applied slice-wise to 3D arrays of
per-site gradient / direction and position matrices.  Scratch buffers are
reused across sites (sequential loop).
"""
function Optim.project_tangent!(m::GaugeFixedStiefel, G3d::AbstractArray{<:Real, 3}, X3d::AbstractArray{<:Real, 3})
    for s in axes(X3d, 3)
        G = @view G3d[:, :, s]
        X = @view X3d[:, :, s]

        # Step 1: Stiefel tangent projection
        mul!(m._XG, transpose(X), G)
        @inbounds for j in axes(m._Γ, 2), i in axes(m._Γ, 1)
            m._Γ[i,j] = (m._XG[i,j] + m._XG[j,i]) / 2
        end
        mul!(G, X, m._Γ, -1.0, 1.0)

        # Step 2: Extract Lie-algebra element A = X'G
        mul!(m._XG, transpose(X), G)

        # Step 3: Compute covariance matrix Γ = X'JX
        mul!(m._ΓA, m.J, X)
        mul!(m._Γ, transpose(X), m._ΓA)

        # Step 4: Gauge projection — keep only the physical component
        mul!(m._ΓA, m._Γ, m._XG)
        mul!(m._ΓAΓ, m._ΓA, m._Γ)
        @. m._XG = (m._XG + m._ΓAΓ) / 2

        # Step 5: Reconstruct gradient in ambient space
        mul!(G, X, m._XG)
    end
    return G3d
end

"""
    with_manifold(alg, manifold)

Create a copy of the Optim algorithm `alg` with its `manifold` field replaced.
Preserves all other algorithm settings (memory size, line search, etc.).
"""
function with_manifold(alg::Optim.LBFGS, manifold::Optim.Manifold)
    Optim.LBFGS(alg.m, alg.alphaguess!, alg.linesearch!, alg.P, alg.precondprep!, manifold, alg.scaleinvH0)
end
function with_manifold(alg::Optim.BFGS, manifold::Optim.Manifold)
    Optim.BFGS(alg.alphaguess!, alg.linesearch!, alg.initial_invH, alg.initial_stepnorm, manifold)
end