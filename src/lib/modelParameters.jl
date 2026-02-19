abstract type AbstractBdGHamiltonian end

#= 
    Distinguish between real and momentum space BdG parameters.
=#
"""
    RealSpaceBdGHamiltonian(hopping_mat::Matrix{ComplexF64}, pairing_mat::Matrix{ComplexF64}, μ::Real)

Stores the parameters in real space to construct the Hamiltonian matrix ``H_BdG`` in Boguliubov de Gennes (BdG) form:

``
H_BdG = [hopping_mat  pairing_mat; 
       pairing_mat'  -hopping_matᵀ]
``

# Fields
- `hopping::Union{Union{Matrix{ComplexF64}, Matrix{Real}}, Union{Vector{ComplexF64}, Vector{Real}}}`: Matrix or vector of hopping amplitudes
    * `hopping::Union{Matrix{ComplexF64}, Matrix{Real}}`: complete matrix with elements t_ij
    * `hopping::Union{Vector{ComplexF64}, Vector{Real}}`: [t_1] (NN), [t_1, t_2] (NN + NNN), etc. where t_n is the hopping amplitude for n-th neighbor
- `pairing::Union{Union{Matrix{ComplexF64}, Matrix{Real}}, Union{Vector{ComplexF64}, Vector{Real}}}`: Matrix or vector of pairing amplitudes
    * `pairing::Union{Matrix{ComplexF64}, Matrix{Real}}`: complete matrix with elements Δ_ij
    * `pairing::Union{Vector{ComplexF64}, Vector{Real}}`: [Δ_1] (NN), [Δ_1, Δ_2] (NN + NNN), etc. where Δ_n is the pairing amplitude for n-th neighbor
- `μ::Real`: Chemical potential (can be updated when solve_μ_from_δ = true in DopingSettings)

"""
mutable struct RealSpaceBdGHamiltonian <: AbstractBdGHamiltonian
    hopping::Union{Union{Matrix{ComplexF64}, Matrix{Real}}, Union{Vector{ComplexF64}, Vector{Real}}}
    pairing::Union{Union{Matrix{ComplexF64}, Matrix{Real}}, Union{Vector{ComplexF64}, Vector{Real}}}
    μ::Real # chemical potential -> This can be changed when solve_μ_from_δ = true in DopingSettings

    function RealSpaceBdGHamiltonian(
        hopping::Union{Union{Matrix{ComplexF64}, Matrix{Real}}, Union{Vector{ComplexF64}, Vector{Real}}}, 
        pairing::Union{Union{Matrix{ComplexF64}, Matrix{Real}}, Union{Vector{ComplexF64}, Vector{Real}}}, 
        μ::Real)

        size(hopping) != size(pairing) && throw(ArgumentError("Hopping and pairing matrices must have the same dimensions"))
        return new(hopping, pairing, μ)
    end
end

#= 
    TODO: add specific models here for finite implementation
=#

"""
    MomentumSpaceBdGHamiltonian(Nf::Int, hopping_mat::Matrix{ComplexF64}, pairing_mat::Matrix{ComplexF64}, μ::Real)

Stores the parameters in momentum space to construct the Hamiltonian matrix ``H_BdG_k`` in BCS form for a specific momentum `k`:

``
H_BdG_k = [hopping_mat(k)  pairing_mat(k); 
       pairing_mat(k)'  -hopping_matᵀ(k)]
``

# Fields
- `Nf::Int`: Number of Abrikosov fermions in the unit cell
- `hopping::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Real}}`: where the dict entries represent the hopping amplitude on the corresponding connection:
    * (x, y) => t_ij, where x and y follow the lattice geometry
    * (Example for square lattice x=±1, y=±1): (1,0) => t_(1,x), (-1,0) => t_(1,-x), (0,1) => t_(1,y), (0,-1) => t_(1,-y), (1,1) => t_(2,xy) etc.
- `pairing::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Real}}`: where the dict entries represent the pairing amplitude on the corresponding connection:
    * (x, y) => Δ_ij, where x and y follow the lattice geometry
    * (Example for square lattice x=±1, y=±1): (1,0) => Δ_(1,x), (-1,0) => Δ_(1,-x), (0,1) => Δ_(1,y), (0,-1) => Δ_(1,-y), (1,1) => Δ_(2,xy) etc.
- `μ::Real`: Chemical potential (can be updated when solve_μ_from_δ = true in DopingSettings)
- `ξ_fct::Function`: Function to compute ξ(k, hopping, μ)
- `Δ_fct::Function`: Function to compute Δ(k, pairing)

"""
mutable struct MomentumSpaceBdGHamiltonian <: AbstractBdGHamiltonian
    Nf::Int # number of Abrikosov fermions in the unit cell
    hopping::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Real}}
    pairing::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Real}}
    μ::Real # chemical potential -> This can be changed when solve_μ_from_δ = true in DopingSettings

    ξ_fct::Function # function to compute ξ(k, hopping, μ)
    Δ_fct::Function # function to compute Δ(k, pairing)

    function MomentumSpaceBdGHamiltonian(Nf::Int, hopping::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Real}}, pairing::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Real}}, μ::Real, ξ_fct::Function, Δ_fct::Function)
        return new(Nf, hopping, pairing, μ, ξ_fct, Δ_fct)
    end
end

#= 
    Some example BCS Hamiltonians in momentum space for translation invariant systems
=#

function default_BCS_hamiltonian(
    hopping::Union{Vector{ComplexF64}, Vector{Real}},
    pairing::Union{Vector{ComplexF64}, Vector{Real}},
    μ::Real,
    lattice::AbstractInfiniteLattice;
    pairing_type::String = "d_wave")

    function ξ_fct(k::AbstractVector{<:Real}, hopping::Union{Vector{ComplexF64}, Vector{Real}}, μ::Real)
        # TODO: add more lattice types here
        if lattice isa AbstractRectangularInfiniteLattice
            # NN hopping
            length(hopping) == 1 && return -2 * hopping[1] * (cos(k[1]) + cos(k[2])) - μ

            # NN + NNN hopping
            length(hopping) == 2 && return -2 * hopping[1] * (cos(k[1]) + cos(k[2])) - 4 * hopping[2] * cos(k[1]) * cos(k[2]) - μ

            # TODO: add more scenarios here
        end 
    end


end

"""
    MomentumSpaceBdGHamiltonian(::Val{:NN}, Nf::Int, k::AbstractVector, t::Real, μ::Real, pairing_type::String, Δ_0::Real)

Constructs the Hamiltonian matrix `H_BdG_k` in the Nambu basis for momentum `k` for a standard BCS model with nearest-neighbor hopping and pairing on a square lattice.

# Keyword arguments
- `Nf::Int`: Number of Abrikosov fermions
- `k::AbstractVector`: Momentum vector (k_x, k_y) for which to construct the `H_BdG_k`
- `t::Real`: Hopping amplitude
- `μ::Real`: Chemical potential
- `pairing_type::String`: Type of pairing symmetry, e.g. "d_wave", "s_wave", "p_ip_wave"
- `Δ_0::Real`: Pairing amplitude

# Returns
- `H_BdG_k::MomentumSpaceBdGHamiltonian`: The Hamiltonian matrix in momentum space for the given parameters.


TODO: make t::Vector{Real} and then depending on the size choose NN, NNN etc.
"""
function MomentumSpaceBdGHamiltonian(::Val{:NN}, Nf::Int, k::AbstractVector{<:Real}, t::Real, μ::Real, pairing_type::String, Δ_0::Real)
    return MomentumSpaceBdGHamiltonian(Nf, get_ξ_mat_k_NN(ξ, Nf, k, t, μ), get_Δ_mat_k_NN(Δ, Nf, k, pairing_type, Δ_0), μ)
end

function MomentumSpaceBdGHamiltonian(::Val{:kitaev_HC_square_vortex_free}, Nf::Int, k::AbstractVector{<:Real}, Jx::Real, Jy::Real, Jz::Real)
    return MomentumSpaceBdGHamiltonian(Nf, get_ξ_mat_k_kitaev_HC_square_vortex_free(Nf, k, Jx, Jy, Jz), get_Δ_mat_k_kitaev_HC_square_vortex_free(Nf, k, Jx, Jy), 0.0)
end

#= 
    Standard BCS formulas in momentum space with NN hopping and pairing on a square lattice.
=#
"""
    ξ(k::AbstractVector{<:Real},t::Real,μ::Real)

Returns:
```
    -2t * (cos(k_x) + cos(k_y)) - μ
```
"""
ξ(k::AbstractVector{<:Real}, t::Real, μ::Real) = -2 * t * (cos(k[1]) + cos(k[2])) - μ

"""
    Δ(k::AbstractVector{<:Real}, pairing_type::String, Δ_0::Real)

Returns pairing amplitude:
```
    ● d_wave: 2*Δ_0*(cos(k_x) - cos(k_y))
    ● s_wave: 2*Δ_0
    ● p+ip_wave: 2*Δ_0*(sin(k_x) - im*sin(k_y))
```
"""
function Δ(k::AbstractVector{<:Real}, pairing_type::String, Δ_0::Real)
    pairing_type === "d_wave" && return Δ(Val(:d_wave), k, Δ_0)
    pairing_type === "s_wave" && return Δ(Val(:s_wave), k, Δ_0)
    pairing_type === "p_ip_wave" && return Δ(Val(:p_ip_wave), k, Δ_0)
    throw(ArgumentError("Unsupported pairing_type $pairing_type for BCS parameters"))
end
Δ(::Val{:d_wave},k::AbstractVector{<:Real},Δ_0) = 2*Δ_0*(cos(k[1]) - cos(k[2]))
Δ(::Val{:s_wave},k::AbstractVector{<:Real},Δ_0::Real) = 2*Δ_0 * (cos(k[1]) + cos(k[2]))
Δ(::Val{:p_ip_wave},k::AbstractVector{<:Real},Δ_0::Real) = 2*Δ_0*(sin(k[1]) + im*sin(k[2]))

#= 
    Functions to construct specific hopping and pairing matrices.

    Note: Add specific functions for different lattice geometries and hopping/pairing ranges here.
=#
"""
    get_ξ_mat_k_NN(Nf::Int, k::AbstractVector, t::Real, μ::Real)

Returns the standard Fourier transformed hopping matrix ξ(k) in the Nambu basis for nearest-neighbor (NN) hopping on a translation invariant square lattice.

"""
function get_ξ_mat_k_NN(ξ_fct::Function, Nf::Int, k::AbstractVector, kwargs...)
    return Diagonal([ξ_fct(k, kwargs...) for i in 1:Nf])
end

"""
    get_Δ_mat_k_NN(Nf::Int, k::AbstractVector, pairing_type::String, Δ_0::Real)

Returns the standard Fourier transformed pairing matrix Δ(k) in the Nambu basis for nearest-neighbor (NN) pairing on a translation invariant square lattice.

"""
function get_Δ_mat_k_NN(Δ_fct::Function, Nf::Int, k::AbstractVector, kwargs...)
    Δ_mat = zeros(ComplexF64, Nf, Nf)
    for i in 1:div(Nf, 2)
        Δ_mat[i, Nf - i + 1] = Δ_fct(k, kwargs...)
        Δ_mat[Nf - i + 1, i] = -Δ_fct(-k, kwargs...)
    end
    return Δ_mat
end

"""
    get_ξ_mat_k_kitaev_HC_square_vortex_free(Nf::Int, k::AbstractVector, Jx::Real, Jy::Real, Jz::Real)

Returns the Fourier transformed hopping matrix ξ(k) in the Nambu basis for the vortex free sector of the Kitaev honeycomb model, which is mapped to a square lattice of Dirac fermions.

The hopping has the form ``ξ(k) = 2*(Jz - Jx*cos(k_x) - Jy*cos(k_y))`` for the vortex free configuration.
"""
function get_ξ_mat_k_kitaev_HC_square_vortex_free(Nf::Int, k::AbstractVector, Jx::Real, Jy::Real, Jz::Real)
    ξ_kitaev_square(k::AbstractVector{<:Real}, Jx::Real, Jy::Real, Jz::Real) = 2 * (Jz - Jx * cos(k[1]) - Jy * cos(k[2]))
    return get_ξ_mat_k_NN(ξ_kitaev_square, Nf, k, Jx, Jy, Jz)
end

"""
    get_Δ_mat_k_kitaev_HC_square_vortex_free(Nf::Int, k::AbstractVector, Jx::Real, Jy::Real)

Returns the Fourier transformed pairing matrix Δ(k) in the Nambu basis for the vortex free sector of the Kitaev honeycomb model, which is mapped to a square lattice of Dirac fermions.

The pairing has the form ``Δ(k) = 2*im*(Jx*sin(k_x) + Jy*sin(k_y))`` for the vortex free configuration.
"""
function get_Δ_mat_k_kitaev_HC_square_vortex_free(Nf::Int, k::AbstractVector, Jx::Real, Jy::Real)
    Δ_kitaev_square(k::AbstractVector{<:Real}, Jx::Real, Jy::Real) = 2im * (Jx * sin(k[1]) + Jy * sin(k[2]))
    return get_Δ_mat_k_NN(Δ_kitaev_square, Nf, k, Jx, Jy)
end