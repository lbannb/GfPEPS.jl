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
        hopping::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Float64}}, 
        pairing::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Float64}}, 
        μ::Real, 
        ξ_fct::Function, 
        Δ_fct::Function,
        E_shift::Function = (k, ξ_k, Δ_k, μ) -> 0.0
    )

Stores the parameters in momentum space to construct the Hamiltonian matrix ``H_BdG_k`` in BCS form for a specific momentum `k`:

``
H_BdG_k = [hopping_mat(k)  pairing_mat(k); 
       pairing_mat(k)'  -hopping_matᵀ(k)]
``

# Fields
- `hopping::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Float64}}`: where the dict entries represent the hopping amplitude on the corresponding connection:
    * (x, y) => t_ij, where x and y follow the lattice geometry
    * (Example for square lattice x=±1, y=±1): (1,0) => t_(1,x), (-1,0) => t_(1,-x), (0,1) => t_(1,y), (0,-1) => t_(1,-y), (1,1) => t_(2,xy) etc.
- `pairing::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Float64}}`: where the dict entries represent the pairing amplitude on the corresponding connection:
    * (x, y) => Δ_ij, where x and y follow the lattice geometry
    * (Example for square lattice x=±1, y=±1): (1,0) => Δ_(1,x), (-1,0) => Δ_(1,-x), (0,1) => Δ_(1,y), (0,-1) => Δ_(1,-y), (1,1) => Δ_(2,xy) etc.
- `μ::Real`: Chemical potential (can be updated when solve_μ_from_δ = true in DopingSettings)
- `ξ_fct::Function`: Function to compute ξ(k, hopping, μ)
- `Δ_fct::Function`: Function to compute Δ(k, pairing)
- `E_shift::Function`: Function to implement arbitrary energy shifts.

"""
mutable struct MomentumSpaceBdGHamiltonian <: AbstractBdGHamiltonian
    hopping::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Float64}}
    pairing::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Float64}}
    μ::Real # chemical potential -> This can be changed when solve_μ_from_δ = true in DopingSettings

    ξ_fct::Function     # function to compute ξ(k, hopping, μ)
    Δ_fct::Function     # function to compute Δ(k, pairing)
    E_shift::Function   # function to implement arbitrary energy shifts

    function MomentumSpaceBdGHamiltonian(
        hopping::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Float64}}, 
        pairing::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Float64}}, 
        μ::Real, 
        ξ_fct::Function, 
        Δ_fct::Function;
        E_shift::Function = (k, ξ_k, Δ_k, μ) -> 0.0
    )
        return new(hopping, pairing, μ, ξ_fct, Δ_fct, E_shift)
    end
end

#= 
    Some example BCS Hamiltonians in momentum space for translation invariant systems
=#
"""
    default_BCS_hamiltonian(
        hopping::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Float64}}, 
        pairing::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Float64}}, 
        μ::Real, lattice::AbstractInfiniteLattice; 
        interaction_type::Vector{String} = ["NN"], 
        pairing_type::String = "d_wave")

Returns a `MomentumSpaceBdGHamiltonian` which follows the standard BCS form:
```
    H = ∑_k ξ(k) c_k^† c_k + ( Δ(k) c_k^† c_-k^† + h.c. )
´´´

# Keyword Arguments
- `hopping::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Float64}}`: where the dict entries represent the hopping amplitude on the corresponding connection.
- `pairing::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Float64}}`: where the dict entries represent the pairing amplitude on the corresponding connection.
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
    hopping::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Float64}},
    pairing::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Float64}},
    μ::Real,
    lattice::AbstractInfiniteLattice;
    E_shift::Function = (k, ξ_k, Δ_k, μ) -> 0.0,
    interaction_type::Vector{String} = ["NN"])

    function ξ_fct(k::AbstractVector{<:Real}, hopping::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Float64}}, μ::Real)
        # TODO: add more lattice types here
        if lattice isa AbstractRectangularInfiniteLattice
            #=  
                collection of standard interaction types which can be combined
                TODO: add higher neighbor interactions here if needed
            =#
            NN(k, hopping) = -2 * (hopping[(1,0)] * cos(k[1]) + hopping[(0,1)] * cos(k[2]))
            NNN(k, hopping) = - 2 * hopping[(1,1)] * cos(k[1]+k[2]) - 2 * hopping[(1,-1)] * cos(k[1]-k[2])

            # recall (t_x = t_-x and t_y = t_-y because of hermiticity)
            # NN hopping 
            interaction_type == ["NN"] && return NN(k, hopping) - μ

            # NNN hopping
            interaction_type == ["NNN"] && return NNN(k, hopping) - μ

            # NN + NNN hopping
            interaction_type == ["NN","NNN"] && return NN(k, hopping) + NNN(k, hopping) - μ

            # TODO: add more scenarios here
            throw(ArgumentError("Unsupported interaction_type $interaction_type."))
        end 
    end

    function Δ_fct(k::AbstractVector{<:Real}, pairing::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Float64}})
        # TODO: add more lattice types here
        if lattice isa AbstractRectangularInfiniteLattice
            #= 
                collection of standard pairing symmetries which can be combined
                TODO: add higher neighbor pairings here if needed
            =#
            # (This form allows for all pairing types depending on the choice of pairing)
            NN(k, pairing) = pairing[(1,0)]*cis(k[1]) + pairing[(-1,0)] * cis(-k[1]) + pairing[(0,1)] * cis(k[2]) + pairing[(0,-1)] * cis(-k[2])
            NNN(k, pairing) = pairing[(1,1)] * cis(k[1]+k[2]) + pairing[(-1,-1)] * conj(cis(k[1]+k[2])) + pairing[(1,-1)] * cis(k[1]-k[2]) + pairing[(-1,1)] * conj(cis(k[1]-k[2]))

            # NN hopping 
            interaction_type == ["NN"] && return NN(k, pairing)
            
            # NNN hopping
            interaction_type == ["NNN"] && return NNN(k, pairing)

            # NN + NNN hopping
            interaction_type == ["NN","NNN"] && return NN(k, pairing) + NNN(k, pairing)

            # TODO: add more scenarios here
            throw(ArgumentError("Unsupported interaction_type $interaction_type."))
        end 
    end

    return MomentumSpaceBdGHamiltonian(hopping, pairing, μ, ξ_fct, Δ_fct; E_shift=E_shift)
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
    if lattice isa AbstractRectangularInfiniteLattice
        μ = -2Jz
        E_shift = (k, ξ_k, Δ_k, μ) -> -Jz # Z2 background gauge field

        # TODO: add more interaction types here if needed
        if interaction_type == ["NN"]
             hopping = get_anisotropic_coupling_dict(lattice, [[Jx, Jx, Jy, Jy]], interaction_type=["NN"])
             pairing = get_anisotropic_coupling_dict(lattice, [[Jx,-Jx,Jy,-Jy]], interaction_type=["NN"])

             return default_BCS_hamiltonian(hopping, pairing, μ, lattice; interaction_type=["NN"], E_shift=E_shift)
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
- `Dict{Tuple{Int64, Int64}, eltype(couplings)}`: Dictionary where the keys are tuples representing the lattice connections and the values are the corresponding coupling constants.
    * For "NN": (1,0), (-1,0), (0,1), (0,-1) => coupling value
    * For "NNN": (1,1), (-1,-1), (1,-1), (-1,1) => coupling value
"""
function get_isotropic_coupling_dict(lattice::AbstractInfiniteLattice, couplings::Union{Vector{ComplexF64}, Vector{Float64}}; interaction_type::Vector{String} = ["NN"])
    length(couplings) != length(interaction_type) && throw(ArgumentError("Length of couplings vector must match the number of interaction types specified in interaction_type."))

    coupling_dict = Dict{Tuple{Int64, Int64}, eltype(couplings)}()
    valid_interaction_type = false

    # TODO: add more lattice types here
    if "NN" in interaction_type
        if lattice isa AbstractRectangularInfiniteLattice
            coupling_dict[(1,0)] = couplings[1]
            coupling_dict[(-1,0)] = couplings[1]
            coupling_dict[(0,1)] = couplings[1]
            coupling_dict[(0,-1)] = couplings[1]
            valid_interaction_type = true
        end
    end
    if "NNN" in interaction_type
        if lattice isa AbstractRectangularInfiniteLattice
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
- `Dict{Tuple{Int64, Int64}, eltype(couplings)}`
    Dictionary where the keys are tuples representing the lattice connections and the values are the corresponding coupling constants.
    * For "NN": (1,0) => t_x, (-1,0) => t_-x, (0,1) => t_y, (0,-1) => t_-y
    * For "NNN": (1,1) => t_(xy), (-1,-1) => t_(-xy), (1,-1) => t_(x-y), (-1,1) => t_(-x+y)
"""
function get_anisotropic_coupling_dict(lattice::AbstractInfiniteLattice, couplings::Union{Vector{Vector{ComplexF64}}, Vector{Vector{Float64}}}; interaction_type::Vector{String} = ["NN"])
    length(couplings) != length(interaction_type) && throw(ArgumentError("Length of couplings vector must match the number of interaction types specified in interaction_type."))

    coupling_dict = Dict{Tuple{Int64, Int64}, eltype(couplings[1])}()
    valid_interaction_type = false

    if "NN" in interaction_type
        if lattice isa AbstractRectangularInfiniteLattice
            coupling_dict[(1,0)] = couplings[1][1]
            coupling_dict[(-1,0)] = couplings[1][2]
            coupling_dict[(0,1)] = couplings[1][3]
            coupling_dict[(0,-1)] = couplings[1][4]
            valid_interaction_type = true
        end
    end
    if "NNN" in interaction_type
        if lattice isa AbstractRectangularInfiniteLattice
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