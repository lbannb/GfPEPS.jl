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
    gutzwiller_project(z::Float64, peps::InfinitePEPS; type="Hubbard")

Apply the Gutzwiller projection to a PEPS to every site in the unit cell.
"""
function gutzwiller_project(z::Float64, peps::InfinitePEPS; type="Hubbard")
    P = gutzwiller_projector(z; type=type)
    pepsGW = InfinitePEPS(collect(P * t for t in peps.A))
    return PEPSKit.peps_normalize(pepsGW)
end

"""
    gutzwiller_project(z::Matrix{Float64}, peps::InfinitePEPS)

Apply different Gutzwiller projection to a (spin-1/2 / Nf=2) PEPS to the corresponding site in the unit cell determined by `z::Matrix{Float64}`.

# Keyword Arguments
- `z::Matrix{Float64}`: The matrix of fugacity factors to be applied to each site in the unit cell. Each element `z[r, c]` corresponds to the fugacity factor for the site at position `(r, c)` in the unit cell of the PEPS.
- `peps::InfinitePEPS`: The input PEPS to which the Gutzwiller projection will be applied.

"""
function gutzwiller_project(z::Matrix{Float64}, peps::InfinitePEPS)
    @assert size(z) == size(peps.A) "Size of fugacity matrix z must match the unit cell size of peps."

    for r in 1:size(peps.A, 1), c in 1:size(peps.A, 2)
        P = gutzwiller_projector(z[r, c]; type="Hubbard")
        peps.A[r, c] = P * peps.A[r, c]
    end

    return PEPSKit.peps_normalize(peps)
end

"""
    solve_for_fugacity(peps::InfinitePEPS, build_env::Function, δ_target::Real; z_range::NTuple{2, Float64} = (0.0, 1.0), initial_env::Union{Nothing,CTMRGEnv}=nothing)

Find the fugacity `z` in the Gutzwiller projector such that the doping after projection of `peps` matches the target doping `δ_target`.

# Keyword Arguments
- `peps::InfinitePEPS`: The input PEPS to which the Gutzwiller projection will be applied.
- `δ_target::Real`: The target doping level that we want to achieve after applying the Gutzwiller projection.
- `χ_env_max`: The maximum bond dimension of the environment tensors.
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

    function get_env(peps::InfinitePEPS; env_init::Union{Nothing, CTMRGEnv}=nothing)
        boundary_alg = (; tol = 1e-8, maxiter = 500, alg = :simultaneous)
        Espace = Vect[FermionParity](0 => χ_env_max / 2, 1 => χ_env_max / 2)

        if env_init === nothing
            χ0 = min(6, χ_env_max)
            env = init_ctmrg_env(peps);
            env, _ = grow_env(peps, env, χ0, χ_env_max; boundary_alg...);

            return env;
        else
            Espace = Vect[FermionParity](0 => χ_env_max ÷ 2, 1 => χ_env_max ÷ 2) 
            env, = leading_boundary(env_init, peps; boundary_alg..., trunc = truncspace(Espace));
            return env;
        end
    end

    # build initial environment
    z_init = isnothing(z_initial) ? 2δ_target/(1+δ_target) : z_initial
    peps_projected = gutzwiller_project(z_init, peps)
    env_init = get_env(peps_projected; env_init=env_init)

    function mismatch(z)
        peps_projected = gutzwiller_project(z, peps)
        env_init = get_env(peps_projected; env_init=env_init)
        δ_projected, _ = doping_pepsGW(peps_projected, env_init)
        return δ_target - δ_projected
    end

    return find_zero(mismatch, z_init; atol=atol), env_init
end