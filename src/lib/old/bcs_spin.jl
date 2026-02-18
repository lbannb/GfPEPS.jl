"""
BCS spin-1/2 Hamiltonian with singlet pairing terms on square lattice
```
    H = -t ∑_{i,v} (c†_{iα} c_{i+v,α} + h.c.) - μ ∑_i c†_{iα} c_{iα}
        + ∑_{i,v} (Δv ϵ_{αβ} c†_{iα} c†_{i+v,β} + h.c.)
```
where v sums over the basis vectors e_x, e_y. 

- s-wave state: Δy = Δx.
- d-wave state: Δy = -Δx.
- (p+ip) state: Δy = i Δx.
"""
function BCS_spin_hamiltonian(
        T::Type{<:Number}, lattice::InfiniteSquare; pairing_type::String="d_wave", t::Float64 = 1.0,
        Δ_0::Float64 = 0.5, μ::Float64 = 0.0
    )
    Δx = Δ_0
    if pairing_type == "s_wave"
        Δy = Δx
    elseif pairing_type == "d_wave"
        Δy = -Δx
    elseif pairing_type == "p_ip_wave"
        Δy = im*Δx
    end

    pspace = hub.hubbard_space(Trivial, Trivial)
    pspaces = fill(pspace, (lattice.Nrows, lattice.Ncols))
    num = hub.e_num(T, Trivial, Trivial)

    unit = TensorKit.id(T, pspace)
    hopping = (-t) * hub.e_hopping(T, Trivial, Trivial) -
        (μ / 4) * (num ⊗ unit + unit ⊗ num)
    pairing = sqrt(2) * hub.singlet_plus(T, Trivial, Trivial)
    pairing += pairing'
    return LocalOperator(
        pspaces,
        map(nearest_neighbours(lattice)) do bond
            return bond => hopping + pairing * (_is_xbond(bond) ? Δx : Δy)
        end...
    )
end
BCS_spin_hamiltonian(lattice; pairing_type="d_wave", Δ_0=1.0, μ=0.0) = BCS_spin_hamiltonian(ComplexF64, lattice; pairing_type=pairing_type, Δ_0=Δ_0, μ=μ)

"""
Check if a 2-site bond is a nearest neighbor x-bond
"""
function _is_xbond(bond)
    return bond[2] - bond[1] == CartesianIndex(0, 1)
end

#= Energy functions when diagonalised via Fourier transform =#
"""
    ξ(k::AbstractVector{<:Real},t::Real,μ::Real)

Returns:
```
    -2t * (cos(k_x) + cos(k_y)) - μ
```
"""
# ξ(k::AbstractVector{<:Real},t::Real,μ::Real) = -2t * (cos(k[1]) + cos(k[2])) - μ
ξ(k::AbstractVector{<:Real}, params::BCS) = -2 * params.t * (cos(k[1]) + cos(k[2])) - params.μ

"""
    Δ(pairing_type::String, kwargs...)

Returns pairing amplitude:
```
    ● d_wave: 2*Δ_0*(cos(k_x) - cos(k_y))
    ● s_wave: 2*Δ_0
    ● p+ip_wave: 2*Δ_0*(sin(k_x) - im*sin(k_y))
    ● kitaev_vortex_free: J*(e^{ik_x} + e^{ik_y})
```
"""
function Δ(k::AbstractVector{<:Real}, params::BCS)
    if params.pairing_type == "d_wave"
        return Δ(Val(:d_wave), k, params.Δ_0)
    elseif params.pairing_type == "s_wave"
        return Δ(Val(:s_wave), k, params.Δ_0)
    elseif params.pairing_type == "p_ip_wave"
        return Δ(Val(:p_ip_wave), k, params.Δ_0)
    elseif params.pairing_type == "s_d_wave"
        return Δ(Val(:s_d_wave), k, params.Δ_0, params.Δ_02)
    else
        throw(ArgumentError("Unsupported pairing_type $(params.pairing_type) for BCS parameters"))
    end
end

Δ(::Val{:d_wave},k::AbstractVector{<:Real},Δ_0) = 2*Δ_0*(cos(k[1]) - cos(k[2]))
Δ(::Val{:s_wave},k::AbstractVector{<:Real},Δ_0::Real) = 2*Δ_0 * (cos(k[1]) + cos(k[2]))
Δ(::Val{:p_ip_wave},k::AbstractVector{<:Real},Δ_0::Real) = 2*Δ_0*(sin(k[1]) + im*sin(k[2]))
Δ(::Val{:s_d_wave},k::AbstractVector{<:Real},Δ_d::Real,Δ_s::Real) = Δ(Val(:d_wave), k, Δ_d) + Δ(Val(:s_wave), k, Δ_s)

function E(k::AbstractVector{<:Real}, params::BCS)
    return sqrt(ξ(k, params)^2 + abs(Δ(k, params))^2)
end

"""
    exact_energy_BCS_k(bz::BrillouinZone2D, t::Real, μ::Real, Δ_kwargs...)

Returns the exact ground state energy per site of a BCS mean field Hamiltonian over the Brillouin zone `bz`.
"""
function exact_energy(params::BCS, bz::BrillouinZone2D)
    return mean(map(eachcol(bz.kvals)) do k
        ξ(k,params) - E(k, params)
    end)
end

"""
    has_dirac_points(bz::BrillouinZone2D, t::Real, μ::Real, pairing_type::String, Δ_kwargs...)

Checks if there are Dirac points (zero-energy modes) in the energy spectrum over the Brillouin zone `bz`.
"""
function has_dirac_points(bz::BrillouinZone2D, params::BCS)
    dirac_point_found = false
    for k in eachcol(bz.kvals)
        if isapprox(E(k, params), 0.0; atol = 1e-6)
            @warn ("Dirac point found at k = $k. This may lead to convergence issues during optimization.")
            dirac_point_found = true
        end
    end
    return dirac_point_found
end

"""
The energy of a Gaussian fPEPS evaluated from 
the fiducial state correlation matrix `G`.
"""
function energy_CM(Γ_fiducial::AbstractMatrix, bz::BrillouinZone2D, Nf::Int, params::BCS)
    A = Γ_fiducial[1:2*Nf, 1:2*Nf]
    B = Γ_fiducial[1:2*Nf, 2*Nf+1:end]
    D = Γ_fiducial[2*Nf+1:end, 2*Nf+1:end]

    χ = div(size(Γ_fiducial, 1) - 2 * Nf, 8)
    return mean(
        map(eachcol(bz.kvals)) do k
            G_in = G_in_single_k(k, χ)
            Gf = A + B * inv(D + G_in) * transpose(B)
            # qq ordering of Majorana modes: (c_1, c_2, ..., c_(2(4Nv + Nf)))
            # return real(
            #     ξ(k,t,mu) * (2 - Gf[1, 2] - Gf[3, 4]) / 2 +
            #         Δ(pairing_type, k, Δ_0) * (Gf[1, 4] + Gf[2, 3] + 1.0im * (Gf[2, 4] - Gf[1, 3])) / 2 
            # )
            return real(
                ξ(k, params) * (2 - Gf[1, 2] - Gf[3, 4]) / 2 +
                    Δ(k, params) * (Gf[1, 4] + Gf[2, 3] + 1.0im * (Gf[2, 4] - Gf[1, 3])) / 2
            )
        end
    )
end

#======================================================================================
Functions to solve μ from hole density
======================================================================================#
function exact_doping(bz::BrillouinZone2D, t::Real, μ::Real, pairing_type::String, Δ_0::Real)
    bcs_params = BCS(t, μ, pairing_type, Δ_0)

    return mean(map(eachcol(bz.kvals)) do k
        ξ(k,bcs_params) / E(k, bcs_params)
    end)
end

function solve_for_mu(bz::BrillouinZone2D, δ::Real, t::Real, pairing_type::String, Δ_0::Real; μ_range::NTuple{2, Float64} = (-5.0, 5.0))
    μ = find_zero(x -> δ - exact_doping(bz, t, x, pairing_type, Δ_0), μ_range)
    return μ
end

"""
    doping_bcs(Γ::AbstractMatrix, bz::BrillouinZone, Nf::Int)

The average doping `δ = 1 - (1/N) ∑_k ⟨f†_{kσ} f_{kσ}⟩`
evaluated from the fiducial state correlation matrix `Γ`.

Note:   `⟨f†_{k↑} f_{k↑}⟩ = 1/2 * (1 - Gf[1,2])`
        `⟨f†_{k↓} f_{k↓}⟩ = 1/2 * (1 - Gf[3,4])`

"""
function doping_bcs(Γ::AbstractMatrix, bz::BrillouinZone2D, Nf::Int)
    A, B, D = get_Γ_blocks(Γ, Nf)
    Nv = div(size(Γ, 1) - 2 * Nf, 8)
    return mean(
        map(eachcol(bz.kvals)) do k
            G_in_k = G_in_single_k(k, Nv)
            Gf = GaussianMap_single_k(A, B, D, G_in_k)
            return real(Gf[1, 2] + Gf[3, 4]) / 2
        end
    )
end

"""
    doping_bcs(X::AbstractMatrix, bz::BrillouinZone2D, Nf::Int, Nv::Int)

The average doping `δ = 1 - (1/N) ∑_i ⟨f†_{iσ} f_{iσ}⟩`
evaluated from the matrix `X` from which the fiducial state correlation matrix `Γ` is built.
"""
function doping_bcs(X::AbstractMatrix, bz::BrillouinZone2D, Nf::Int, Nv::Int)
    Γ = Γ_fiducial(X, Nv, Nf)
    return doping_bcs(Γ, bz, Nf)
end

"""
Gutzwiller projector from Hubbard (spin-1/2) 
to the no-double-occupancy tJ space.

In Gutzwiller approximation, z = 2δ/(1+δ), 
where δ is doping before projection.
"""
function gutzwiller_projector(z::Float64)
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
Apply Gutzwiller projection to (spin-1/2) PEPS to every site in the unit cell.
"""
function gutzwiller_project(z::Float64, peps::InfinitePEPS)
    P = gutzwiller_projector(z)
    pepsGW = InfinitePEPS(collect(P * t for t in peps.A))
    return PEPSKit.peps_normalize(pepsGW)
end

"""
Apply custom Gutzwiller projection to (spin-1/2) PEPS to the corresponding site in the unit cell determined by `z::Matrix{Float64}`.
`z` is a matrix of size equal to the unit cell size of `peps`,
where each element is the fugacity to be applied to the corresponding site in the unit cell.
"""
function gutzwiller_project(z::Matrix{Float64}, peps::InfinitePEPS)
    @assert size(z) == size(peps.A) "Size of fugacity matrix z must match the unit cell size of peps."

    for r in 1:size(peps.A, 1), c in 1:size(peps.A, 2)
        P = gutzwiller_projector(z[r, c])
        peps.A[r, c] = P * peps.A[r, c]
    end

    return PEPSKit.peps_normalize(peps)
    # P = gutzwiller_projector(z)
    # pepsGW = InfinitePEPS(collect(P * t for t in peps.A))
    # return PEPSKit.peps_normalize(pepsGW)
end

"""
    doping_peps(peps::InfinitePEPS, env::CTMRGEnv)

The average doping `δ = 1 - (1/N) ∑_i ⟨f†_{iσ} f_{iσ}⟩`
evaluated from the GfPEPS iPEPS tensor.
"""
function doping_peps(peps::InfinitePEPS, env::CTMRGEnv)
    # Get unit cell dimensions
    Nx, Ny = size(peps.A)

    # Initialize total density accumulator
    total_density = 0.0
    
    # Loop over every site in the unit cell
    density_distribution = zeros(Float64, Nx, Ny)
    for r in 1:Nx, c in 1:Ny
        # Construct the operator specifically for site (r, c)
        O = LocalOperator(space.(peps.A, 1), ((r, c),) => hub.e_num(Trivial, Trivial))
        exp_val = real(expectation_value(peps, O, env))

        density_distribution[r, c] = 1 - exp_val
        # Accumulate the expectation value
        total_density += exp_val
    end

    # Average density = Sum / Number of sites
    avg_density = total_density / (Nx * Ny)

    return 1 - avg_density, density_distribution
end

"""
    doping_pepsGW(peps::InfinitePEPS, env::CTMRGEnv)

The average doping `δ = 1 - (1/N) ∑_i ⟨f†_{iσ} f_{iσ}⟩`
evaluated from the Gutzwiller projected GfPEPS tensor.
"""
function doping_pepsGW(peps::InfinitePEPS, env::CTMRGEnv)
    V = Vect[FermionParity](0 => 1, 1 => 2)

    # Number operator in Gutzwiller projected space
    function e_num_GW(V)
        t = zeros(ComplexF64, V ← V)
        I = sectortype(t)
        t[(I(1), I(1))][1, 1] = 1
        t[(I(1), I(1))][2, 2] = 1
        return t
    end

    # Get unit cell dimensions
    Nx, Ny = size(peps.A)

    # Initialize total density accumulator
    total_density = 0.0
    
    # Loop over every site in the unit cell
    density_distribution = zeros(Float64, Nx, Ny)
    for r in 1:Nx, c in 1:Ny
        # Construct the operator specifically for site (r, c)
        lattice_site_space = space(peps.A[r, c], 1) 
        O = LocalOperator(space.(peps.A, 1), ((r, c),) => e_num_GW(lattice_site_space))
        exp_val = real(expectation_value(peps, O, env))

        density_distribution[r, c] = 1 - exp_val
        # Accumulate the expectation value
        total_density += exp_val
    end

    # Average density = Sum / Number of sites
    avg_density = total_density / (Nx * Ny)

    return 1 - avg_density, density_distribution
end

"""
    solve_for_fugacity(peps::InfinitePEPS, build_env::Function, δ_target::Real; z_range::NTuple{2, Float64} = (0.0, 1.0), initial_env::Union{Nothing,CTMRGEnv}=nothing)

Find the fugacity `z` in the Gutzwiller projector such that the doping after projection of `peps`
matches the target doping `δ_target`.
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
    # return find_zero(mismatch, (0.0, 1.0); atol=atol), env_init
end

""" 
    flip_spin(mat, peps)
Flip the spin on the sites where `mat[r, c] != 0`.
"""
function flip_spins_hubbard(mat, peps)
    @assert size(mat) == size(peps.A)

    for r in 1:size(peps.A, 1), c in 1:size(peps.A, 2)
        if mat[r, c] == 0
            continue
        end

        T = peps.A[r, c]
        P = codomain(T)[1] 
        U = TensorMap(zeros, eltype(T), P, P)
        
        # identity block for even parity sector
        b_even = block(U, FermionParity(0))
        b_even[1,1] = 1.0
        b_even[2,2] = 1.0

        # swap block for odd parity sector
        b_odd = block(U, FermionParity(1))
        b_odd[1,2] = 1.0
        b_odd[2,1] = 1.0

        peps.A[r, c] = U * T

        # S_flip = hub.S_x(Trivial, Trivial)
        # peps.A[r, c] = S_flip * peps.A[r, c]
    end

    return PEPSKit.peps_normalize(peps)
end