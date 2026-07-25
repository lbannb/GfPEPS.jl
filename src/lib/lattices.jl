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

Lattice orientation:
```
    ⟶ x
    ↓
    y
```

# Keyword arguments
- `Lx::Int`: Number of sites in the x-direction of the unit cell
- `Ly::Int`: Number of sites in the y-direction of the unit cell

# Optional keyword arguments:
- `uc_layout::Matrix{Int} = fill(1, Ly, Lx)`: Ly x Lx matrix specifying the layout of sites in the unit cell, e.g. for a 2x2 unit cell with 4 different sites: [1 2; 3 4]
    - dim 1: y-direction
    - dim 2: x-direction

- `N_kx::Int = 48`: Number of k-points in the x-direction
- `N_ky::Int = 48`: Number of k-points in the y-direction
- `bc::Tuple{Symbol, Symbol} = (:APBC, :PBC)`: Tuple specifying boundary conditions for x and y directions, e.g. `(:PBC, :APBC)`
- `shift_x::Float64 = 0.0`: Shift in the x-direction of the k-grid
- `shift_y::Float64 = 0.0`: Shift in the y-direction of the k-grid
"""
struct InfiniteRectLattice <: AbstractInfiniteRectangularLattice 
    Lx::Int
    Ly::Int
    uc_layout::Matrix{Int}
    kvals::Matrix{Float64} # 2 x (N_kx * N_ky) matrix of k-points in the Brillouin zone
    N_kx::Int
    N_ky::Int
    bc::Tuple{Symbol, Symbol}
    shift_x::Float64
    shift_y::Float64

    function InfiniteRectLattice(Lx::Int, Ly::Int; 
        uc_layout::Matrix{Int} = fill(1, Ly, Lx),
        N_kx::Int = 48, 
        N_ky::Int = 48,
        bc::Tuple{Symbol, Symbol} = (:APBC, :PBC),
        shift_x::Float64 = 0.0,
        shift_y::Float64 = 0.0)

        allowed_bcs = (:PBC, :APBC)
        if !(bc[1] in allowed_bcs && bc[2] in allowed_bcs)
            throw(ArgumentError("Boundary conditions must be :APBC or :PBC. Got: $bc"))
        end

        @assert size(uc_layout) == (Ly, Lx) "uc_layout must be of size Ly x Lx"

        new(Lx, Ly, uc_layout, get_2D_k_grid(Lx, Ly, N_kx, N_ky; x_bc=Val(bc[1]), shift_x=shift_x, y_bc=Val(bc[2]), shift_y=shift_y), N_kx, N_ky, bc, shift_x, shift_y)
    end
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
function get_number_of_distinct_sites_in_uc(lattice::AbstractInfiniteLattice)
    return length(unique(vec(lattice.uc_layout)))
end

function get_number_of_sites(lattice::Union{AbstractInfiniteRectangularLattice, AbstractRectangularLattice})
    return lattice.Lx * lattice.Ly
end

"""
        get_site_index(x::Number, y::Number, lattice::AbstractInfiniteLattice)

Returns the site index corresponding to the coordinates (x, y) on the given lattice. 
The indexing convention is column-major, meaning that sites are indexed first along the x-direction and then along the y-direction.

Example: [(1,1)=>1 (2,1)=>2; (1,2)=>3 (2,2)=>4] for a 2x2 unit cell.

# Keyword Arguments
- `x::Union{Int, Float64}`: x-coordinate of the site
- `y::Union{Int, Float64}`: y-coordinate of the site
- `lattice::AbstractInfiniteLattice`: The lattice on which the site is located. The function currently supports `AbstractInfiniteRectangularLattice` and will throw an error for unsupported lattice types.

# Returns
- `site_index::Number`: The corresponding site index in column-major order.

"""
function get_site_index(x::Number, y::Number, lattice::AbstractInfiniteLattice)
    @assert 1 <= x <= lattice.Lx && 1 <= y <= lattice.Ly "Site coordinates out of bounds for the given lattice dimensions."

    if lattice isa AbstractInfiniteRectangularLattice
        return Int(x + (y - 1) * lattice.Lx)
    else
        throw(ArgumentError("Unsupported lattice type."))
    end
end