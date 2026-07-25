#= 
    Gutzwiller projector functions.

    TODO: add also the projector that projects out empty + double occupied states
    TODO: add option to choose Backend. TensorMap for iPEPS and iTensors for finitePEPS?
=#

"""
    gutzwiller_projector(::Hubbard, z::Float64)

Constructs the Gutzwiller projector matrix `P` that projects out double occupancies and compensates the changes in the average particle density by a fugacity factor `z`. 
This form is usually used in the Hubbard model with strong Coulomb repulsion.

```
P = z^((1-n↑ - n↓)/2) * (1 - n↑ n↓)
```

# Keyword Arguments
- `z::Float64`: The fugacity factor that adjusts the weight of the empty state to compensate for the change in particle density after projection.

# Returns
- `P::TensorMap`: The Gutzwiller projector as a `TensorMap`.

"""
function gutzwiller_projector(::Val{:Hubbard}, z::Float64)
    V_hub = hub.hubbard_space(Trivial, Trivial)
    V_tJ = tJ.tj_space(Trivial, Trivial)
    P = zeros(Float64, V_hub → V_tJ) # from hubbard (Vect[FermionParity](0=>2, 1=>2)) to tJ (Vect[FermionParity](0=>1, 1=>2)
    S = FermionParity
    P[(S(0), S(0))][1, 1] = sqrt(z) # |0> -> sqrt(z) |0>
    # P[(S(0), S(0))][1, 2] = 0     # |0> -> 0 |↑↓>
    P[(S(1), S(1))][1, 1] = 1.0     # |↑> -> |↑>
    P[(S(1), S(1))][2, 2] = 1.0     # |↓> -> |↓>
    return P
end

"""
    gutzwiller_projector(z::Float64; type="Hubbard")

Routing function for different types of Gutzwiller projectors.
    
"""
function gutzwiller_projector(z::Float64; type="Hubbard") 
    lowercase(type)=="hubbard" && return gutzwiller_projector(Val(:Hubbard), z)
end

#= 
    Functions that apply the Gutzwiller projector to a PEPS
=#

"""
    gutzwiller_project(z::AbstractArray{<:Real}, peps::InfinitePEPS; type="Hubbard")

Apply the Gutzwiller projection to a (spin-1/2 / Nf=2) PEPS, with the fugacity of each site
in the unit cell given by the corresponding entry of `z`.

# Keyword Arguments
- `z::AbstractArray{<:Real}`: The fugacity layout, same shape as the unit cell: `z[r, c]` is the fugacity of the site at position `(r, c)`. A single fugacity (`[z1]`) is applied to every site.
- `peps::InfinitePEPS`: The input PEPS to which the Gutzwiller projection will be applied.

"""
function gutzwiller_project(z::AbstractArray{<:Real}, peps::InfinitePEPS; type="Hubbard")
    @assert length(z) == 1 || size(z) == size(peps.A) "Size of fugacity layout z must match the unit cell size of peps (or be a single fugacity)."
    zs = length(z) == 1 ? fill(only(z), size(peps.A)) : z

    # build a new PEPS rather than writing into peps.A: the projector maps the Hubbard
    # physical space to the tJ one, so the result does not fit back into the input array,
    # and mutating would compound the projection when called repeatedly from a solver.
    pepsGW = InfinitePEPS(map((zi, A) -> gutzwiller_projector(Float64(zi); type=type) * A, zs, peps.A))
    return PEPSKit.peps_normalize(pepsGW)
end

"""
    solve_for_fugacity(peps::InfinitePEPS, build_env::Function, δ_target::Real; z_range::NTuple{2, Float64} = (0.0, 1.0), initial_env::Union{Nothing,CTMRGEnv}=nothing)

Find the fugacity `z` in the Gutzwiller projector such that the doping after projection of `peps` matches the target doping `δ_target`.

# Keyword Arguments
- `peps::InfinitePEPS`: The input PEPS to which the Gutzwiller projection will be applied.
- `δ_target::Real`: The target doping level that we want to achieve after applying the Gutzwiller projection.
- `χ_env_max`: The bond dimension of the environment tensors.
- `atol::Float64`: The absolute tolerance for the root-finding algorithm when matching the doping level.
- `z_initial::Union{Nothing, Float64}`: An optional initial guess for the fugacity `z` to speed up convergence. If `nothing`, the Gutzwiller approximation `z=2δ/(1+δ)`.
- `env_init::Union{Nothing, CTMRGEnv}`: An optional initial environment to speed up convergence. If `nothing`, a new environment will be built from the projected PEPS.

# Returns
- `z::Float64`: The fugacity factor that achieves the target doping level after projection

"""
function solve_for_fugacity(
        peps::InfinitePEPS,
        δ_target::Real;
        χ_env_max::Int = 20,
        atol::Float64=1e-5,
        z_initial::Union{Nothing, Float64}=nothing,
        env_init::Union{Nothing, CTMRGEnv}=nothing
    )

    # build initial environment
    z_init = isnothing(z_initial) ? 2δ_target/(1+δ_target) : z_initial
    peps_projected = gutzwiller_project([z_init], peps)
    env_init = _fugacity_env(peps_projected, χ_env_max; env_init=env_init)

    function mismatch(z)
        peps_projected = gutzwiller_project([z], peps)
        env_init = _fugacity_env(peps_projected, χ_env_max; env_init=env_init)
        δ_projected, _ = doping_pepsGW(peps_projected, env_init)
        return δ_target - δ_projected
    end

    return find_zero(mismatch, z_init; atol=atol), env_init
end

"""
    _fugacity_env(peps, χ_env_max; env_init=nothing)

CTMRG environment used by the fugacity solvers. Identity initialization is deterministic,
and `truncrank` lets CTMRG pick the environment spaces itself, which matters here because
the projected PEPS changes on every solver step and a fixed space would pin the sector
split found for the first `z` onto all later ones.

`tol=1e-6` rather than `1e-8`: this environment is rebuilt on every solver step, and the
looser inner tolerance cuts the scalar solve from ~250 s to ~80 s while moving the
resulting `z` by only ~4e-6. Warm-starting from the previous step helps on top of that.
"""
function _fugacity_env(
        peps::InfinitePEPS, χ_env_max::Int;
        env_init::Union{Nothing, CTMRGEnv}=nothing
    )
    env0 = isnothing(env_init) ?
        initialize_ctmrg_environment(peps, IdentityInitialization()) : env_init
    # Plain rank cut here, deliberately. A value cut (`trunctol`) is the more robust choice
    # for the *unprojected* PEPS, but the Gutzwiller-projected spectrum is flat enough that
    # a relative cut keeps states up to the cap: measured ~50 s per environment and no
    # convergence, against ~2-7 s and err ~1e-6 with truncrank at the same χ.
    env, = leading_boundary(
        env0, peps; tol = 1e-6, maxiter = 100, trunc = truncrank(χ_env_max)
    )
    return env
end

"""
    solve_for_fugacity(peps::InfinitePEPS, δ_target::AbstractMatrix; kwargs...)

Site-resolved version of [`solve_for_fugacity`](@ref): find a fugacity *layout* `z` such
that the doping of the Gutzwiller-projected `peps` matches `δ_target` site by site, rather
than only on average.

Sites that share a target doping share a fugacity, so a target layout `[δ1 δ2; δ2 δ1]`
is solved with two unknowns and yields `[z1 z2; z2 z1]`. This keeps the number of CTMRG
environments per solver step down to the number of *distinct* target values.

# Returns
- `z::Matrix{Float64}`: fugacity layout, same shape as the unit cell
- `env::CTMRGEnv`: converged environment of the projected PEPS at the solution
"""
function solve_for_fugacity(
        peps::InfinitePEPS,
        δ_target::AbstractMatrix{<:Real};
        χ_env_max::Int = 20,
        atol::Float64=1e-5,
        z_initial::Union{Nothing, AbstractMatrix{<:Real}}=nothing,
        env_init::Union{Nothing, CTMRGEnv}=nothing
    )
    @assert size(δ_target) == size(peps.A) "Size of target doping layout must match the unit cell size of peps."

    # one unknown per distinct target doping; `group[r, c]` indexes into that unknown vector
    δ_distinct = unique(vec(δ_target))
    group = map(δ -> findfirst(==(δ), δ_distinct), δ_target)
    expand(zs) = map(g -> zs[g], group)

    z0 = isnothing(z_initial) ? [2δ/(1+δ) for δ in δ_distinct] :
        [z_initial[findfirst(==(k), group)] for k in eachindex(δ_distinct)]

    env = env_init
    function residual!(F, zs)
        peps_projected = gutzwiller_project(expand(zs), peps)
        env = _fugacity_env(peps_projected, χ_env_max; env_init=env)
        _, layout = doping_pepsGW(peps_projected, env)
        for k in eachindex(δ_distinct)
            F[k] = mean(layout[group .== k]) - δ_distinct[k]
        end
        return F
    end

    sol = nlsolve(residual!, z0; ftol=atol)
    sol.f_converged || sol.x_converged ||
        @warn "solve_for_fugacity did not converge" residual_norm=sol.residual_norm

    return expand(sol.zero), env
end