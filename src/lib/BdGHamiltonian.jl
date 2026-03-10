abstract type AbstractBdGHamiltonian end

#= 
    Distinguish between real and momentum space BdG parameters.
=#
"""
    RealSpaceBdGHamiltonian(
        hopping::Union{Union{Matrix{ComplexF64}, Matrix{Float64}}, Union{Vector{ComplexF64}, Vector{Float64}}}, 
        pairing::Union{Union{Matrix{ComplexF64}, Matrix{Float64}}, Union{Vector{ComplexF64}, Vector{Float64}}}, 
        μ::Real
    )

Stores the parameters in real space to construct the Hamiltonian matrix ``H_BdG`` in Boguliubov de Gennes (BdG) form:

``
H_BdG = [hopping_mat  pairing_mat; 
       pairing_mat'  -hopping_matᵀ]
``

# Fields
- `hopping::Union{Union{Matrix{ComplexF64}, Matrix{Float64}}, Union{Vector{ComplexF64}, Vector{Float64}}}`: Matrix or vector of hopping amplitudes
    * `hopping::Union{Matrix{ComplexF64}, Matrix{Float64}}`: complete matrix with elements t_ij
    * `hopping::Union{Vector{ComplexF64}, Vector{Float64}}`: [t_1] (NN), [t_1, t_2] (NN + NNN), etc. where t_n is the hopping amplitude for n-th neighbor
- `pairing::Union{Union{Matrix{ComplexF64}, Matrix{Float64}}, Union{Vector{ComplexF64}, Vector{Float64}}}`: Matrix or vector of pairing amplitudes
    * `pairing::Union{Matrix{ComplexF64}, Matrix{Float64}}`: complete matrix with elements Δ_ij
    * `pairing::Union{Vector{ComplexF64}, Vector{Float64}}`: [Δ_1] (NN), [Δ_1, Δ_2] (NN + NNN), etc. where Δ_n is the pairing amplitude for n-th neighbor
- `μ::Real`: Chemical potential (can be updated when solve_μ_from_δ = true in DopingSettings)

"""
mutable struct RealSpaceBdGHamiltonian <: AbstractBdGHamiltonian
    hopping::Union{Union{Matrix{ComplexF64}, Matrix{Float64}}, Union{Vector{ComplexF64}, Vector{Float64}}}
    pairing::Union{Union{Matrix{ComplexF64}, Matrix{Float64}}, Union{Vector{ComplexF64}, Vector{Float64}}}
    μ::Real # chemical potential -> This can be changed when solve_μ_from_δ = true in DopingSettings

    function RealSpaceBdGHamiltonian(
        hopping::Union{Union{Matrix{ComplexF64}, Matrix{Float64}}, Union{Vector{ComplexF64}, Vector{Float64}}}, 
        pairing::Union{Union{Matrix{ComplexF64}, Matrix{Float64}}, Union{Vector{ComplexF64}, Vector{Float64}}}, 
        μ::Real
    )

        size(hopping) != size(pairing) && throw(ArgumentError("Hopping and pairing matrices must have the same dimensions"))
        return new(hopping, pairing, μ)
    end
end

#= 
    TODO: add specific models here for finite implementation
=#

"""
    MomentumSpaceBdGHamiltonian(
        hopping::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}}, 
        pairing::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}}, 
        μ::Real, 
        ξ_mat::Function, 
        Δ_mat::Function,
        E_shift::Function = (k, ξ_k, Δ_k, μ) -> 0.0
    )

Stores the parameters in momentum space to construct the Hamiltonian matrix ``H_BdG_k`` in BCS form for a specific momentum `k`:

``
H_BdG_k = [ξ_mat(k)  Δ_mat(k); 
       Δ_mat(k)†  -ξ_mat(-k)^T]
``

# Fields
- `hopping::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}}`: where the dict entries represent the hopping amplitude on the corresponding connection:
    * (x, y) => t_ij, where x and y follow the lattice geometry
    * (Example for square lattice x=±1, y=±1): (1,0) => t_(1,x), (-1,0) => t_(1,-x), (0,1) => t_(1,y), (0,-1) => t_(1,-y), (1,1) => t_(2,xy) etc.
- `pairing::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}}`: where the dict entries represent the pairing amplitude on the corresponding connection:
    * (x, y) => Δ_ij, where x and y follow the lattice geometry
    * (Example for square lattice x=±1, y=±1): (1,0) => Δ_(1,x), (-1,0) => Δ_(1,-x), (0,1) => Δ_(1,y), (0,-1) => Δ_(1,-y), (1,1) => Δ_(2,xy) etc.
- `μ::Real`: Chemical potential (can be updated when solve_μ_from_δ = true in DopingSettings)
- `ξ_mat_k::Function`: Function to compute hopping matrix ξ_mat(k) from the hopping dict for a given momentum k
- `Δ_mat_k::Function`: Function to compute pairing matrix Δ_mat(k) from the pairing dict for a given momentum k
- `E_shift::Function`: Function to implement arbitrary energy shifts.

"""
mutable struct MomentumSpaceBdGHamiltonian <: AbstractBdGHamiltonian
    hopping::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}}
    pairing::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}}
    μ::Real # chemical potential -> This can be changed when solve_μ_from_δ = true in DopingSettings
    ξ_mat_k::Function # Function to compute hopping matrix ξ_mat(k)
    Δ_mat_k::Function # Function to compute pairing matrix Δ_mat(k)
    E_shift::Function # Function to implement arbitrary energy shifts
    interaction_type::Vector{String} # ["NN"], ["NNN"], ["NN","NNN"] 

    function MomentumSpaceBdGHamiltonian(
        hopping::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}}, 
        pairing::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}}, 
        μ::Real, 
        ξ_mat_k::Function, 
        Δ_mat_k::Function,
        interaction_type::Vector{String};
        E_shift::Function = (k, ξ_k, Δ_k, μ) -> 0.0
    )
        return new(hopping, pairing, μ, ξ_mat_k, Δ_mat_k, E_shift, interaction_type)
    end
end

#= 
    Some example BCS Hamiltonians in momentum space for translation invariant systems
=#
"""
    default_BCS_hamiltonian(
        hopping::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}}, 
        pairing::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}}, 
        μ::Real, lattice::AbstractInfiniteLattice; 
        interaction_type::Vector{String} = ["NN"], 
        pairing_type::String = "d_wave")

Returns a `MomentumSpaceBdGHamiltonian` which follows the standard BCS form (Nf=2):
```
    H = ∑_k_σ ξ(k) c_k,σ^† c_k,σ + ( Δ(k) c_k,↑^† c_-k,↓^† + h.c. )
´´´

# Keyword Arguments
- `hopping::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}}`: where the dict entries represent the hopping amplitude on the corresponding connection.
- `pairing::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}}`: where the dict entries represent the pairing amplitude on the corresponding connection.
- `μ::Real`: Chemical potential
- `lattice::AbstractInfiniteLattice`: The lattice on which the BCS model is defined
- `h::Real = 0.0`: External field
- `interaction_type::Vector{String}=["NN"]`: Type of hopping interactions to include. Options: 
    * "NN" (Nearest neighbor)
    * "NNN" (Next nearest neighbor)

# Returns
- `MomentumSpaceBdGHamiltonian`: A `MomentumSpaceBdGHamiltonian` object with the specified parameters and functions to compute ξ(k) and Δ(k) for the given lattice and interaction types.

"""
function default_BCS_hamiltonian(
    hopping::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}},
    pairing::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}},
    μ::Real,
    lattice::AbstractInfiniteLattice;
    Nf::Int = 2,
    E_shift::Function = (k, ξ_mat_k, Δ_mat_k, μ) -> 0.0,
    interaction_type::Vector{String} = ["NN"])

    ξ_mat_k = get_ξ_mat_k(lattice, Nf, hopping, μ)
    Δ_mat_k = get_Δ_mat_k(lattice, Nf, pairing)

    # function ξ_k(k::AbstractVector{<:Real}, hopping::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}}, μ::Real)
    #     # TODO: add more lattice types here

    #     NN(k, hopping) = begin
    #         if lattice isa AbstractInfiniteRectangularLattice
    #             -2 * (hopping[(1,0)] * cos(k[1]) + hopping[(0,1)] * cos(k[2]))
    #         else
    #             throw(ArgumentError("Unsupported lattice type for ξ_k."))
    #         end
    #     end

    #     NNN(k, hopping) = begin
    #         if lattice isa AbstractInfiniteRectangularLattice
    #             - 2 * hopping[(1,1)] * cos(k[1]+k[2]) - 2 * hopping[(1,-1)] * cos(k[1]-k[2])
    #         else
    #             throw(ArgumentError("Unsupported lattice type for ξ_k."))
    #         end
    #     end

    #     # recall (t_x = t_-x and t_y = t_-y because of hermiticity)
    #     # NN hopping 
    #     interaction_type == ["NN"] && return NN(k, hopping) - μ

    #     # NNN hopping
    #     interaction_type == ["NNN"] && return NNN(k, hopping) - μ

    #     # NN + NNN hopping
    #     interaction_type == ["NN","NNN"] && return NN(k, hopping) + NNN(k, hopping) - μ

    #     # TODO: add more scenarios here
    #     throw(ArgumentError("Unsupported interaction_type $interaction_type."))
    # end

    # function Δ_k(k::AbstractVector{<:Real}, pairing::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}})
    #     # TODO: add more lattice types here

    #     NN(k, pairing) = begin
    #         if lattice isa AbstractInfiniteRectangularLattice
    #             pairing[(1,0)]*cis(k[1]) + pairing[(-1,0)] * cis(-k[1]) + pairing[(0,1)] * cis(k[2]) + pairing[(0,-1)] * cis(-k[2])
    #         else
    #             throw(ArgumentError("Unsupported lattice type for Δ_k."))
    #         end
    #     end

    #     NNN(k, pairing) = begin
    #         if lattice isa AbstractInfiniteRectangularLattice
    #             pairing[(1,1)] * cis(k[1]+k[2]) + pairing[(-1,-1)] * conj(cis(k[1]+k[2])) + pairing[(1,-1)] * cis(k[1]-k[2]) + pairing[(-1,1)] * conj(cis(k[1]-k[2]))
    #         else
    #             throw(ArgumentError("Unsupported lattice type for Δ_k."))
    #         end
    #     end

    #     # NN hopping 
    #     interaction_type == ["NN"] && return NN(k, pairing)
        
    #     # NNN hopping
    #     interaction_type == ["NNN"] && return NNN(k, pairing)

    #     # NN + NNN hopping
    #     interaction_type == ["NN","NNN"] && return NN(k, pairing) + NNN(k, pairing)

    #     # TODO: add more scenarios here
    #     throw(ArgumentError("Unsupported interaction_type $interaction_type."))
    # end

    return MomentumSpaceBdGHamiltonian(hopping, pairing, μ, ξ_mat_k, Δ_mat_k, interaction_type; E_shift=E_shift)
end

"""
    kitaev_BCS_hamiltonian(
        Jx::Real,
        Jy::Real,
        Jz::Real,
        lattice::AbstractInfiniteLattice;
        # TODO: add external field?
        interaction_type::Vector{String} = ["NN"])

Returns a `MomentumSpaceBdGHamiltonian` for the Kitaev model in the vortex free configuration.
TODO: Distinguish between different vortex configurations in the future

# Keyword Arguments
- `Jx::Real`: Coupling constant for x-bonds
- `Jy::Real`: Coupling constant for y-bonds
- `Jz::Real`: Coupling constant for z-bonds
- `lattice::AbstractInfiniteLattice`: The lattice on which the Kitaev model is defined 
- `h::Real = 0.0`: External field
- `interaction_type::Vector{String} = ["NN"]`: Type of interactions to include. Options:
    * "NN" (Nearest neighbor)
    * "NNN" (Next nearest neighbor)

# Returns
- `MomentumSpaceBdGHamiltonian`: A `MomentumSpaceBdGHamiltonian` object representing the Kitaev model in the vortex free configuration, with functions to compute ξ(k) and Δ(k) based on the specified couplings and lattice geometry.

"""

function kitaev_BCS_hamiltonian(
    Jx::Real,
    Jy::Real,
    Jz::Real,
    lattice::AbstractInfiniteLattice;
    interaction_type::Vector{String} = ["NN"])

    # TODO: add Honeycomb lattice
    if lattice isa AbstractInfiniteRectangularLattice
        μ = -2Jz
        E_shift = (k, ξ_k, Δ_k, μ) -> -Jz # Z2 background gauge field

        # TODO: add more interaction types here if needed
        if interaction_type == ["NN"]
             hopping = get_anisotropic_coupling_dict(lattice, [[Jx, Jx, Jy, Jy]], interaction_type=["NN"])
             pairing = get_anisotropic_coupling_dict(lattice, [[Jx,-Jx,Jy,-Jy]], interaction_type=["NN"])

             return default_BCS_hamiltonian(hopping, pairing, μ, lattice; E_shift=E_shift, interaction_type=["NN"], Nf=1)
        end
    else
        throw(ArgumentError("Unsupported lattice type for Kitaev BCS Hamiltonian."))
    end
end

"""
    get_isotropic_coupling_dict(couplings::Union{Vector{ComplexF64}, Vector{Float64}}; interaction_type::Vector{String} = ["NN"])

Returns a dictionary of isotropic couplings for the specified interaction type.

# Keyword Arguments
- `lattice::AbstractInfiniteLattice`: The lattice object for which the coupling dictionary is being constructed.
- `couplings::Vector{Union{Vector{ComplexF64}, Vector{Float64}}}`: Vector of coupling values. Example: [t_1 (NN), t_2 (NNN), etc.]
- `interaction_type::Vector{String}=["NN"]`: Vector of interaction types to include. Options:
    * "NN" (Nearest neighbor)
    * "NNN" (Next nearest neighbor)

# Returns
- `Dict{Tuple{Float64, Float64}, eltype(couplings)}`: Dictionary where the keys are tuples representing the lattice connections and the values are the corresponding coupling constants.
    * For "NN": (1,0), (-1,0), (0,1), (0,-1) => coupling value
    * For "NNN": (1,1), (-1,-1), (1,-1), (-1,1) => coupling value
"""
function get_isotropic_coupling_dict(lattice::AbstractInfiniteLattice, couplings::Union{Vector{ComplexF64}, Vector{Float64}}; interaction_type::Vector{String} = ["NN"])
    length(couplings) != length(interaction_type) && throw(ArgumentError("Length of couplings vector must match the number of interaction types specified in interaction_type."))

    coupling_dict = Dict{Tuple{Float64, Float64}, eltype(couplings)}()
    valid_interaction_type = false

    # TODO: add more lattice types here
    if "NN" in interaction_type
        if lattice isa AbstractInfiniteRectangularLattice
            coupling_dict[(1,0)] = couplings[1]
            coupling_dict[(-1,0)] = couplings[1]
            coupling_dict[(0,1)] = couplings[1]
            coupling_dict[(0,-1)] = couplings[1]
            valid_interaction_type = true
        end
    end
    if "NNN" in interaction_type
        if lattice isa AbstractInfiniteRectangularLattice
            coupling_dict[(1,1)] = couplings[1]
            coupling_dict[(-1,-1)] = couplings[1]
            coupling_dict[(1,-1)] = couplings[1]
            coupling_dict[(-1,1)] = couplings[1]
            valid_interaction_type = true
        end
    end

    !valid_interaction_type && throw(ArgumentError("Unsupported interaction_type $interaction_type."))
    return coupling_dict
end

"""
        get_anisotropic_coupling_dict(lattice::AbstractInfiniteLattice, couplings::Vector{Union{Vector{ComplexF64}, Vector{Float64}}}; interaction_type::Vector{String} = ["NN"])

Returns a dictionary of anisotropic couplings for the specified interaction type.

# Keyword Arguments
- `lattice::AbstractInfiniteLattice`: The lattice object for which the coupling dictionary is being constructed.
- `couplings::Vector{Union{Vector{ComplexF64}, Vector{Float64}}}`: Every Vector contains the couplings for the given interaction type. Example: [ [t_1_(1,0), t_1_(-1,0), ...], [t_2_(1,1), t_2_(-1,1),...], ... ] for a square lattice
- `interaction_type::Vector{String}=["NN"]`: Vector of interaction types to include. Options:
    * "NN" (Nearest neighbor)
    * "NNN" (Next nearest neighbor)

# Returns
- `Dict{Tuple{Float64, Float64}, eltype(couplings)}`
    Dictionary where the keys are tuples representing the lattice connections and the values are the corresponding coupling constants.
    * For "NN": (1,0) => t_x, (-1,0) => t_-x, (0,1) => t_y, (0,-1) => t_-y
    * For "NNN": (1,1) => t_(xy), (-1,-1) => t_(-xy), (1,-1) => t_(x-y), (-1,1) => t_(-x+y)
"""
function get_anisotropic_coupling_dict(lattice::AbstractInfiniteLattice, couplings::Union{Vector{Vector{ComplexF64}}, Vector{Vector{Float64}}}; interaction_type::Vector{String} = ["NN"])
    length(couplings) != length(interaction_type) && throw(ArgumentError("Length of couplings vector must match the number of interaction types specified in interaction_type."))

    coupling_dict = Dict{Tuple{Float64, Float64}, eltype(couplings[1])}()
    valid_interaction_type = false

    if "NN" in interaction_type
        if lattice isa AbstractInfiniteRectangularLattice
            coupling_dict[(1,0)] = couplings[1][1]
            coupling_dict[(-1,0)] = couplings[1][2]
            coupling_dict[(0,1)] = couplings[1][3]
            coupling_dict[(0,-1)] = couplings[1][4]
            valid_interaction_type = true
        end
    end
    if "NNN" in interaction_type
        if lattice isa AbstractInfiniteRectangularLattice
            coupling_dict[(1,1)] = couplings[2][1]
            coupling_dict[(-1,-1)] = couplings[2][2]
            coupling_dict[(1,-1)] = couplings[2][3]
            coupling_dict[(-1,1)] = couplings[2][4]
            valid_interaction_type = true
        end
    end

    !valid_interaction_type && throw(ArgumentError("Unsupported interaction_type $interaction_type."))
    return coupling_dict
end

#= 
    Functions to create hopping and pairing matrices for different lattice geometries
=#

"""
        get_site_index(x::Union{Int, Float64}, y::Union{Int, Float64}, lattice::AbstractInfiniteLattice)

Returns the site index corresponding to the coordinates (x, y) on the given lattice. 
The indexing convention is column-major, meaning that sites are indexed first along the x-direction and then along the y-direction.

Example: [(1,1)=>1 (2,1)=>2; (1,2)=>3 (2,2)=>4] for a 2x2 unit cell.

# Keyword Arguments
- `x::Union{Int, Float64}`: x-coordinate of the site
- `y::Union{Int, Float64}`: y-coordinate of the site
- `lattice::AbstractInfiniteLattice`: The lattice on which the site is located. The function currently supports `AbstractInfiniteRectangularLattice` and will throw an error for unsupported lattice types.

# Returns
- `site_index::Int`: The corresponding site index in column-major order.

"""
function get_site_index(x::Union{Int, Float64}, y::Union{Int, Float64}, lattice::AbstractInfiniteLattice)
    @assert 1 <= x <= lattice.Lx && 1 <= y <= lattice.Ly "Site coordinates out of bounds for the given lattice dimensions."

    if lattice isa AbstractInfiniteRectangularLattice
        return Int(x + (y - 1) * lattice.Lx)
    else
        throw(ArgumentError("Unsupported lattice type."))
    end
end

"""
    wrap_site_index(x::Int, y::Int, dx::Float64, dy::Float64, lattice::AbstractInfiniteLattice)

Returns the wrapped destination coordinates (xdst, ydst) and the number of crossed supercells (Tx, Ty) when applying a displacement (dx, dy) to a site at coordinates (x, y) on the given lattice. 
The function handles periodic boundary conditions by wrapping the destination coordinates back into the unit cell and calculating how many times the displacement crosses the boundaries of the unit cell.

# Keyword Arguments
- `x::Int`: x-coordinate of the original site
- `y::Int`: y-coordinate of the original site
- `dx::Float64`: x-component of the displacement
- `dy::Float64`: y-component of the displacement
- `lattice::AbstractInfiniteLattice`: The lattice on which the site is located. The function currently supports `AbstractInfiniteRectangularLattice` and will throw an error for unsupported lattice types.

# Returns
- `wrapped_x::Float64`: x-coordinate of the wrapped destination site within the unit cell
- `wrapped_y::Float64`: y-coordinate of the wrapped destination site within the unit cell
- `Tx::Int`: Number of times the displacement crosses the boundary in the x-direction (number of supercells crossed)
- `Ty::Int`: Number of times the displacement crosses the boundary in the y-direction (number of supercells crossed)

"""

function wrap_site_index(x::Int, y::Int, dx::Float64, dy::Float64, lattice::AbstractInfiniteLattice)
    Lx, Ly = lattice.Lx, lattice.Ly

    if lattice isa AbstractInfiniteRectangularLattice
        # wrapped destination inside the unit cell
        wrapped_x = mod1(x + dx, Lx)
        wrapped_y = mod1(y + dy, Ly)

        # number of crossed supercells
        Tx = fld(x + dx - 1, Lx)
        Ty = fld(y + dy - 1, Ly)

        return wrapped_x, wrapped_y, Tx, Ty
    else
        throw(ArgumentError("Unsupported lattice type."))
    end
end

"""
    get_k_matrix_kernel(k::AbstractVector{<:Real}, coupling_dict::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}}, lattice::AbstractInfiniteLattice)

Returns the kernel matrix in momentum space for a given momentum `k`, coupling dictionary, and lattice geometry. 
The kernel matrix is constructed by summing over the contributions from all couplings specified in the `coupling_dict`, where each coupling corresponds to a specific displacement (dx, dy) on the lattice. 
The function handles the wrapping of site indices according to the lattice geometry and applies the appropriate phase factors based on the momentum `k` and the supercell translations.

# Keyword Arguments
- `k::AbstractVector{<:Real}`: Momentum vector for which to compute the kernel matrix
- `coupling_dict::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}}`: Dictionary where keys are tuples representing displacements (dx, dy) and values are the corresponding coupling amplitudes (hopping or pairing)
- `lattice::AbstractInfiniteLattice`: The lattice geometry which determines how site indices are wrapped and how the kernel matrix is constructed. The function currently supports `AbstractInfiniteRectangularLattice` and will throw an error for unsupported lattice types.

# Returns
- `mat_kernel::Matrix{ComplexF64}`: The resulting kernel matrix in momentum space for the given momentum `k`.

"""
function get_k_matrix_kernel(k::AbstractVector{<:Real}, coupling_dict::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}}, lattice::AbstractInfiniteLattice)
    Lx, Ly = lattice.Lx, lattice.Ly
    Nsites = get_number_of_sites(lattice)
    mat_kernel = zeros(ComplexF64, Nsites, Nsites)

    for (bond, amplitude) in coupling_dict
        dx, dy = bond
        for x in 1:Lx, y in 1:Ly
            src = get_site_index(x, y, lattice)

            xdst, ydst, Tx, Ty = wrap_site_index(x, y, dx, dy, lattice)
            dst = get_site_index(xdst, ydst, lattice)

            # supercell translation in microscopic lattice units
            Tdx = Tx * Lx
            Tdy = Ty * Ly

            mat_kernel[src, dst] += amplitude * cis(k[1] * Tdx + k[2] * Tdy)
        end
    end
    return mat_kernel
end

"""
    get_ξ_mat_k(lattice::AbstractInfiniteLattice, Nf::Int, k::AbstractVector{<:Real}, H_bdg_k::MomentumSpaceBdGHamiltonian)

Returns the hopping matrix ξ(k) in momentum space for a given momentum `k`, lattice geometry, and momentum-space BdG Hamiltonian parameters.

# Keyword Arguments
- `lattice::AbstractInfiniteLattice`: The lattice geometry which determines how site indices are wrapped and how the kernel matrix is constructed.
- `Nf::Int`: Number of Abrikosov fermions
- `k::AbstractVector{<:Real}`: Momentum vector for which to compute the hopping matrix
- `H_bdg_k::MomentumSpaceBdGHamiltonian`: The momentum-space BdG Hamiltonian containing the hopping parameters and chemical potential

# Returns
- `ξ_mat_k::Function`: The function returning the hopping matrix ξ(k) in momentum space for the given momentum `k`.

"""
function get_ξ_mat_k(lattice::AbstractInfiniteLattice, Nf::Int, hopping_dict::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}}, μ::Real)
    function ξ_mat_k(k::AbstractVector{<:Real})
        Nsites = get_number_of_sites(lattice)
        mat = get_k_matrix_kernel(k, hopping_dict, lattice)

        # add chemical potential on the diagonal
        for i in 1:Nsites
            mat[i, i] -= μ
        end

        # same forall fermion flavors
        return kron(mat, I(Nf))
    end

    return ξ_mat_k
end

"""
    get_Δ_mat_k(lattice::AbstractInfiniteLattice, Nf::Int, k::AbstractVector{<:Real}, H_bdg_k::MomentumSpaceBdGHamiltonian)

Returns the pairing matrix Δ(k) in momentum space for a given momentum `k`, lattice geometry, and momentum-space BdG Hamiltonian parameters.

# Keyword Arguments
- `lattice::AbstractInfiniteLattice`: The lattice geometry which determines how site indices are wrapped and how the kernel matrix is constructed.
- `Nf::Int`: Number of Abrikosov fermions
- `k::AbstractVector{<:Real}`: Momentum vector for which to compute the pairing matrix
- `H_bdg_k::MomentumSpaceBdGHamiltonian`: The momentum-space BdG Hamiltonian containing the pairing parameters

# Returns
- `Δ_mat_k::Function`: The function returning the pairing matrix Δ(k) in momentum space for the given momentum `k`.

"""
function get_Δ_mat_k(lattice::AbstractInfiniteLattice, Nf::Int, pairing_dict::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}})
    function Δ_mat_k(k::AbstractVector{<:Real})
        Nsites = get_number_of_sites(lattice)
        Δ_sites_k = get_k_matrix_kernel(k, pairing_dict, lattice)
        Δ_full = zeros(ComplexF64, Nf * Nsites, Nf * Nsites)

        if Nf == 1 # spinless fermions
            Δ_full = Δ_sites_k
        elseif Nf == 2  # standard BCS form with spin up and down flavors
            Δ_sites_minus_k = get_k_matrix_kernel(-k, pairing_dict, lattice)

            for src_site in 1:Nsites, dst_site in 1:Nsites
                src_up = Nf * src_site - 1
                src_dn = Nf * src_site
                dst_up = Nf * dst_site - 1
                dst_dn = Nf * dst_site

                Δ_full[src_up, dst_dn] = Δ_sites_k[src_site, dst_site]
                Δ_full[src_dn, dst_up] = -Δ_sites_minus_k[dst_site, src_site]
            end
        else
            throw(ArgumentError("Momentum-space BdG Hamiltonians currently supports Nf = 1 or Nf = 2. Got Nf = $Nf."))
        end
        return Δ_full
    end
end

#= 
    Energy functions for BCS Hamiltonians
=#

# function get_ξ_mat_k(Nf::Int, k::AbstractVector{<:Real}, H_bdg_k::MomentumSpaceBdGHamiltonian)
#     return kron(H_bdg_k.ξ_k(k, H_bdg_k.μ), I(Nf))
# end

# function get_Δ_mat_k(Nf::Int, k::AbstractVector{<:Real}, H_bdg_k::MomentumSpaceBdGHamiltonian)
#     Δ_mat_k = H_bdg_k.Δ_k(k)

#     if Nf == 1
#         return Δ_mat_k
#     elseif Nf == 2
#         Δ_minus_k_mat = H_bdg_k.Δ_k(-k)
#         Nsites = size(Δ_mat_k, 1)
#         Δ_full = zeros(ComplexF64, 2 * Nsites, 2 * Nsites)

#         for src_site in 1:Nsites, dst_site in 1:Nsites
#             src_up = 2src_site - 1
#             src_dn = 2src_site
#             dst_up = 2dst_site - 1
#             dst_dn = 2dst_site

#             Δ_full[src_up, dst_dn] = Δ_mat_k[src_site, dst_site]
#             Δ_full[src_dn, dst_up] = -Δ_minus_k_mat[dst_site, src_site]
#         end

#         return Δ_full
#     end

#     throw(ArgumentError("Momentum-space BdG Hamiltonians currently support Nf = 1 or Nf = 2. Got Nf = $Nf."))
# end

"""
    E(k::AbstractVector{<:Real}, H_bdg::MomentumSpaceBdGHamiltonian)

Returns the positive quasiparticle energies for a given momentum `k` based on the BdG Hamiltonian parameters.

# Keyword Arguments
- `k::AbstractVector{<:Real}`: Momentum vector for which to compute the energy.
- `H_bdg::MomentumSpaceBdGHamiltonian`: The BdG Hamiltonian object containing the parameters and functions to compute ξ(k) and Δ(k).

"""
function E(k::AbstractVector{<:Real}, H_bdg::MomentumSpaceBdGHamiltonian)
    H_bdG_k = [H_bdg.ξ_mat_k(k)  H_bdg.Δ_mat_k(k); 
               H_bdg.Δ_mat_k(k)' -transpose(H_bdg.ξ_mat_k(-k))]

    eigenvals = eigen(Hermitian(H_bdG_k)).values
    return eigenvals[eigenvals .> 1e-10]
end

"""
    exact_energy(kvals::AbstractMatrix, H_bdg::MomentumSpaceBdGHamiltonian, Nf::Int)

Returns the exact ground state energy per site of a BCS mean field Hamiltonian.

"""
function exact_energy(lattice::AbstractInfiniteLattice, H_bdg::MomentumSpaceBdGHamiltonian)
    Nsites = get_number_of_sites(lattice)

    return mean(map(eachcol(lattice.kvals)) do k
        E_k = E(k, H_bdg)
        sum_E = sum(E_k)

        0.5 * (real(tr(H_bdg.ξ_mat_k(k))) - sum_E) / Nsites +
            H_bdg.E_shift(k, H_bdg.ξ_mat_k, H_bdg.Δ_mat_k, H_bdg.μ)
        
        # ξ_mat_k = H_bdg.ξ_k(k, H_bdg.μ)
        # Δ_mat_k = H_bdg.Δ_k(k)
        # H_BdG_k_mat = get_BdG_k_matrix(lattice, Nf, k, H_bdg)
        # sum_E = sum(get_positive_quasiparticle_energies(H_BdG_k_mat))

        # (0.5 * (Nf * real(tr(ξ_mat_k)) - sum_E) / Ns) + H_bdg.E_shift(k, ξ_mat_k, Δ_mat_k, H_bdg.μ)
    end)
end

"""
    has_dirac_points(kvals::AbstractMatrix, H_bdg::MomentumSpaceBdGHamiltonian)

Checks if there are Dirac points (zero-energy modes) in the quasiparticle energy spectrum over a given momentum set `kvals`.

"""
function has_dirac_points(kvals::AbstractMatrix, H_bdg::MomentumSpaceBdGHamiltonian)
    dirac_point_found = false
    for k in eachcol(kvals)
        if isapprox(minimum(E(k, H_bdg)), 0.0; atol = 1e-6)
            @warn ("Dirac point found at k = $k. This may lead to convergence issues during optimization.")
            dirac_point_found = true
        end
    end
    return dirac_point_found
end

#= 
    Doping functions for BCS Hamiltonians
=#

"""
        avg_density(lattice::AbstractInfiniteLattice, H_bdg::MomentumSpaceBdGHamiltonian, Nf::Int)

Returns the average particle density per site for a BCS mean field Hamiltonian.
Using the Bogoliubov coefficients u_k and v_k: `|u_k|^2 + |v_k|^2 = 1` the BCS state can be written as:
```
    |BCS⟩ = ∏_k (u_k + v_k c_k,↑^† c_-k,↓^†) |0⟩
```

We get for a single band Hamiltonian:
```
    <c_k,↑† c_k,↑> = <c_k,↓† c_k,↓> = |v_k|^2
```

The average density per site for an arbitrary number of bands m is given by:
```
    <n> = (1/Nsites) (1/N_k) ∑_k ∑_{m : E_m(k) > 0} |v_m(k)|^2
```

# Keyword Arguments
- `lattice::AbstractInfiniteLattice`: The lattice object containing the momentum vectors over which to compute the average density.
- `Nf::Int`: Number of Abrikosov fermions
- `H_bdg::MomentumSpaceBdGHamiltonian`: The BdG Hamiltonian object containing the parameters and functions to compute ξ(k) and Δ(k).

# Returns
- `avg_density::Float64`: The average particle density per site for the BCS mean field Hamiltonian.

"""
function avg_density(lattice::AbstractInfiniteLattice, H_bdg::MomentumSpaceBdGHamiltonian)
    Nsites = get_number_of_sites(lattice)

    return mean(map(eachcol(lattice.kvals)) do k
        H_bdG_k = [H_bdg.ξ_mat_k(k)  H_bdg.Δ_mat_k(k); 
               H_bdg.Δ_mat_k(k)' -transpose(H_bdg.ξ_mat_k(-k))]
        
        Np = size(H_bdg.ξ_mat_k(k), 1) # number of particle modes = number of sites * number of flavors

        F = eigen(Hermitian(H_bdG_k))
        pos_inds = F.values .> 1e-10
        v_m_k = F.vectors[Np+1:end, pos_inds]

        sum(abs2, v_m_k) / Nsites
    end)
end

"""
    exact_doping(lattice::AbstractInfiniteLattice, H_bdg::MomentumSpaceBdGHamiltonian)

Returns the exact doping level `δ = 1 - <n>` of a BCS mean field Hamiltonian.

# Keyword Arguments
- `lattice::AbstractInfiniteLattice`: The lattice object containing the momentum vectors over which to compute the doping.
- `H_bdg::MomentumSpaceBdGHamiltonian`: The BdG Hamiltonian object containing the parameters and functions to compute ξ(k) and Δ(k).

# Returns
- `δ::Float64`: The exact doping level for the BCS mean field Hamiltonian, calculated as `δ = 1 - <n>`, where `<n>` is the average particle density per site.

"""
function exact_doping(lattice::AbstractInfiniteLattice, H_bdg::MomentumSpaceBdGHamiltonian)
    return 1.0 - avg_density(lattice, H_bdg)
end

"""
    solve_for_mu(lattice::AbstractInfiniteLattice, δ::Real, H_bdg::MomentumSpaceBdGHamiltonian; μ_range::NTuple{2, Float64} = (-5.0, 5.0))

Finds the chemical potential `μ` that corresponds to a given doping level `δ` for a BCS mean field Hamiltonian by solving the number equation.

# Keyword Arguments
- `lattice::AbstractInfiniteLattice`: The lattice object containing the momentum vectors over which to compute the doping.
- `δ::Real`: Target doping level to solve for.
- `H_bdg::MomentumSpaceBdGHamiltonian`: The BdG Hamiltonian object containing the parameters and functions to compute ξ(k) and Δ(k).

# Optional Arguments
- `μ_range::NTuple{2, Float64} = (-5.0, 5.0)`: Range of chemical potential values to search for the solution.

# Returns
- `μ::Real`: The chemical potential that corresponds to the target doping level `δ`.

"""
function solve_for_mu(lattice::AbstractInfiniteLattice, δ::Real, H_bdg::MomentumSpaceBdGHamiltonian; μ_range::NTuple{2, Float64} = (-5.0, 5.0))
    μ_initial = H_bdg.μ
    objective(x) = begin
        H_bdg.μ = x
        return δ - exact_doping(lattice, H_bdg)
    end

    μ = find_zero(objective, μ_range)
    H_bdg.μ = μ_initial

    return μ
end

