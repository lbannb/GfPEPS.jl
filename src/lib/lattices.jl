#= 
    Finite and infinite lattice types
=#
abstract type AbstractLattice end
abstract type AbstractInfiniteLattice end

#= 
    Rectangular lattice types
=#
abstract type AbstractRectangularLattice <: AbstractLattice end
abstract type AbstractInfiniteRectangularLattice <: AbstractInfiniteLattice end

#= 
    ToDo: add more lattice types (triangular, honeycomb, kagome, etc.)
=#

"""
    SquareLattice(Lx::Int, Ly::Int)

Represents a finite rectangular lattice with size ```Lx * Ly```.
"""
struct RectLattice <: AbstractRectangularLattice 
    Lx::Int
    Ly::Int
end

"""
    InfiniteRectLattice(Lx::Int, Ly::Int; N_kx::Int = 48, N_ky::Int = 48, bc::Tuple{Symbol, Symbol} = (:APBC, :PBC))

Represents a unit cell of size ```Lx * Ly``` which is repeated over the infinite lattice.
Also contains the allowed momentum values ```kvals::Matrix{Float64}``` for the given unit cell.

# Keyword arguments
- `Lx::Int`: Number of sites in the x-direction of the unit cell
- `Ly::Int`: Number of sites in the y-direction of the unit cell

# Optional keyword arguments:
- `N_kx::Int = 48`: Number of k-points in the x-direction
- `N_ky::Int = 48`: Number of k-points in the y-direction
- `bc::Tuple{Symbol, Symbol} = (:APBC, :PBC)`: Tuple specifying boundary conditions for x and y directions, e.g. `(:PBC, :APBC)`
- `shift_x::Float64 = 0.0`: Shift in the x-direction of the k-grid
- `shift_y::Float64 = 0.0`: Shift in the y-direction of the k-grid
"""
struct InfiniteRectLattice <: AbstractInfiniteRectangularLattice 
    Lx::Int
    Ly::Int
    uc_layout::Matrix{Int} # Lx x Ly matrix specifying the layout of sites in the unit cell

    kvals::Matrix{Float64} # 2 x (N_kx * N_ky) matrix of k-points in the Brillouin zone
    N_kx::Int
    N_ky::Int
    bc::Tuple{Symbol, Symbol}
    shift_x::Float64
    shift_y::Float64

    function InfiniteRectLattice(Lx::Int, Ly::Int; 
        uc_layout::Matrix{Int} = fill(1, Lx, Ly),
        N_kx::Int = 48, 
        N_ky::Int = 48,
        bc::Tuple{Symbol, Symbol} = (:APBC, :PBC),
        shift_x::Float64 = 0.0,
        shift_y::Float64 = 0.0)

        allowed_bcs = (:PBC, :APBC)
        if !(bc[1] in allowed_bcs && bc[2] in allowed_bcs)
            throw(ArgumentError("Boundary conditions must be :APBC or :PBC. Got: $bc"))
        end

        @assert size(uc_layout) == (Lx, Ly) "uc_layout must be of size Lx x Ly"

        new(Lx, Ly, uc_layout, get_2D_k_grid(Lx, Ly, N_kx, N_ky; x_bc=Val(bc[1]), shift_x=shift_x, y_bc=Val(bc[2]), shift_y=shift_y), N_kx, N_ky, bc, shift_x, shift_y)
    end
end

#= 
    Functions for momentum pairs
=#
"""
    get_kvals(::Val{:PBC}, L::Int, N_k::Int)

Returns `N_k` uniformly spaced momentum values for a 1D Brillouin zone of width `2π/L`
with periodic boundary conditions (PBC).

# Arguments
- `L`: Brillouin zone width
- `N_k`: Number of momentum points to generate

# Returns
- `Vector{Float64}`: Allowed momentum values 2π*m/(N_k*L) where:
  - If N_k is even: m ∈ {-(N_k-2)/2, ..., N_k/2}
  - If N_k is odd: m ∈ {-(N_k-1)/2, ..., (N_k-1)/2}
"""
function get_kvals(::Val{:PBC},L::Int,N_k::Int)
    if iseven(N_k)
        return [2π*m/N_k for m in (-(N_k-2)/2):N_k/2] ./ L
    else
        return [2π*m/N_k for m in (-(N_k-1)/2):(N_k-1)/2] ./ L
    end
end

"""
    get_kvals(::Val{:APBC}, L::Int, N_k::Int)

Returns `N_k` uniformly spaced momentum values for a 1D Brillouin zone of width `2π/L`
with anti-periodic boundary conditions (APBC).

# Arguments
- `L`: Brillouin zone width
- `N_k`: Number of momentum points to generate

# Returns
- `Vector{Float64}`: Allowed momentum values (2m-1)π/(N_k*L) where:
  - If N_k is even: m ∈ {1, ..., N_k/2}, returns both ±k values
  - If N_k is odd: m ∈ {1, ..., (N_k-1)/2}, returns ±k values plus π
"""
function get_kvals(::Val{:APBC},L::Int,N_k::Int)
    if iseven(N_k)
        kvals = [(2*m-1)*π/N_k for m in 1:N_k/2] 
		return vcat(-kvals,kvals)./ L
    else
        kvals = [(2*m-1)*π/N_k for m in 1:(N_k-1)/2] 
		return vcat(-kvals,kvals,pi)./ L
    end
end

"""
        get_2D_k_grid(Lx, Ly; x_bc=Val(:APBC), shift_x=0.0, y_bc=Val(:PBC),  shift_y=0.0)

Create the 2D momentum grid from 1D k-values (with optional offsets) and
return a meshgrid of the form: transpose([[kx_1, ky_1]; 
                                [kx_2, ky_1]; 
                                    ... 
                                [kx_Lx, ky_1];
                                [kx_1, ky_2];
                                [kx_2, ky_2];
                                    ...
                                [kx_Lx, ky_2];
                                    ...
                                [kx_Lx, ky_Ly]]);

Returns
- Matrix of size 2x(N_kx*N_ky) where:
    - row 1 = kx vals
    - row 2 = ky vals

Notes
- set the offsets, such that zero modes are avoided as those make the optimization of Γ harder.
"""
function get_2D_k_grid(Lx::Int, Ly::Int, N_kx::Int, N_ky::Int; 
    x_bc::Union{Val{:APBC}, Val{:PBC}} = Val(:APBC),
    shift_x::Float64 = pi/2,
    y_bc::Union{Val{:APBC}, Val{:PBC}} = Val(:PBC),
    shift_y::Float64 = pi/2)

    # TODO: test with correct kvals but first take from paper to compare
    k_vals_x = sort(get_kvals(x_bc, Lx, N_kx) .+ shift_x)
    k_vals_y = sort(get_kvals(y_bc, Ly, N_ky) .+ shift_y)

    # create full Cartesian-product meshgrid
    KX = repeat(collect(k_vals_x), N_ky)
    KY = collect(Iterators.flatten(map(k_vals_y) do ky
        repeat([ky], N_kx)
    end))

    return hcat(KX,KY)'
end

#= 
    Functions for number of majorana modes in the unit cell
=#
function get_Nf_in_uc(Nf::Int, lattice::Union{AbstractInfiniteRectangularLattice, AbstractRectangularLattice})
    return Nf * lattice.Lx * lattice.Ly
end

function get_Λ_in_uc(Λ::Int, lattice::Union{AbstractInfiniteRectangularLattice, AbstractRectangularLattice})
    return 2Λ*(lattice.Lx + lattice.Ly)
end

function get_Λ_in_uc_x_dir(Λ::Int, lattice::Union{AbstractInfiniteRectangularLattice, AbstractRectangularLattice})
    return 2Λ*lattice.Lx
end

function get_Λ_in_uc_y_dir(Λ::Int, lattice::Union{AbstractInfiniteRectangularLattice, AbstractRectangularLattice})
    return 2Λ*lattice.Ly
end

function get_number_of_modes(Nf::Int, Λ::Int, lattice::Union{AbstractLattice, AbstractInfiniteLattice})
    return get_Nf_in_uc(Nf, lattice) + get_Λ_in_uc(Λ, lattice)
end

#= 
    Functions for number of sites in the lattice
=#
function get_number_of_sites(lattice::Union{AbstractInfiniteRectangularLattice, AbstractRectangularLattice})
    return lattice.Lx * lattice.Ly
end