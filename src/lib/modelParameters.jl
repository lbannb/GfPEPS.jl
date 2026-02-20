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

"""
    default_BCS_hamiltonian(
        hopping::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Real}}, 
        pairing::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Real}}, 
        μ::Real, lattice::AbstractInfiniteLattice; 
        interaction_type::Vector{String} = ["NN"], 
        pairing_type::String = "d_wave")

Returns a `MomentumSpaceBdGHamiltonian` which follows the standard BCS form.

# Keyword Arguments
- `hopping::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Real}}`: where the dict entries represent the hopping amplitude on the corresponding connection.
- `pairing::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Real}}`: where the dict entries represent the pairing amplitude on the corresponding connection.
- `μ::Real`: Chemical potential
- `interaction_type::Vector{String}=["NN"]`: Type of hopping interactions to include. Options: 
    * "NN" (Nearest neighbor)
    * "NNN" (Next nearest neighbor)

"""
function default_BCS_hamiltonian(
    hopping::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Real}},
    pairing::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Real}},
    μ::Real,
    lattice::AbstractInfiniteLattice;
    interaction_type::Vector{String} = ["NN"])

    function ξ_fct(k::AbstractVector{<:Real}, hopping::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Real}}, μ::Real)
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

    function Δ_fct(k::AbstractVector{<:Real}, pairing::Union{Dict{Tuple{Int64, Int64}, ComplexF64}, Dict{Tuple{Int64, Int64}, Real}})
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
            interaction_type == "NN" && return NN(k, pairing)
            
            # NNN hopping
            interaction_type == "NNN" && return NNN(k, pairing)

            # NN + NNN hopping
            interaction_type == "NN+NNN" && return NN(k, pairing) + NNN(k, pairing)

            # TODO: add more scenarios here
            throw(ArgumentError("Unsupported interaction_type $interaction_type."))
        end 
    end

    return MomentumSpaceBdGHamiltonian(lattice.Nf, hopping, pairing, μ, ξ_fct, Δ_fct)
end

"""
    get_isotropic_coupling_dict(couplings::Union{Vector{ComplexF64}, Vector{Real}}; interaction_type::Vector{String} = ["NN"])

Returns a dictionary of isotropic couplings for the specified interaction type.

# Keyword Arguments
- `couplings::Union{Vector{ComplexF64}, Vector{Real}}`: Vector of coupling values. Example: [t_1 (NN), t_2 (NNN), etc.]
- `interaction_type::Vector{String}=["NN"]`: Vector of interaction types to include. Options:
    * "NN" (Nearest neighbor)
    * "NNN" (Next nearest neighbor)

# Returns
- `Dict{Tuple{Int64, Int64}, eltype(couplings)}`: Dictionary where the keys are tuples representing the lattice connections and the values are the corresponding coupling constants.
    * For "NN": (1,0), (-1,0), (0,1), (0,-1) => coupling value
    * For "NNN": (1,1), (-1,-1), (1,-1), (-1,1) => coupling value
"""
function get_isotropic_coupling_dict(couplings::Union{Vector{ComplexF64}, Vector{Real}}; interaction_type::Vector{String} = ["NN"])
    length(couplings) != length(interaction_type) && throw(ArgumentError("Length of couplings vector must match the number of interaction types specified in interaction_type."))

    coupling_dict = Dict{Tuple{Int64, Int64}, eltype(couplings)}()
    valid_interaction_type = false

    if "NN" in interaction_type
        coupling_dict[(1,0)] = couplings[1]
        coupling_dict[(-1,0)] = couplings[1]
        coupling_dict[(0,1)] = couplings[1]
        coupling_dict[(0,-1)] = couplings[1]
        valid_interaction_type = true
    end
    if "NNN" in interaction_type
        coupling_dict[(1,1)] = couplings[1]
        coupling_dict[(-1,-1)] = couplings[1]
        coupling_dict[(1,-1)] = couplings[1]
        coupling_dict[(-1,1)] = couplings[1]
        valid_interaction_type = true
    end

    !valid_interaction_type && throw(ArgumentError("Unsupported interaction_type $interaction_type."))
    return coupling_dict
end


function get_anisotropic_coupling_dict(couplings::Union{Vector{ComplexF64}, Vector{Real}}; interaction_type::Vector{String} = ["NN"])
    # length(couplings) != length(interaction_type) && throw(ArgumentError("Length of couplings vector must match the number of interaction types specified in interaction_type."))

    # coupling_dict = Dict{Tuple{Int64, Int64}, eltype(couplings)}()
    # valid_interaction_type = false

    # # recall 
    # if "NN" in interaction_type
    #     coupling_dict[(1,0)] = couplings[1]
    #     coupling_dict[(-1,0)] = couplings[2]
    #     coupling_dict[(0,1)] = couplings[3]
    #     coupling_dict[(0,-1)] = couplings[4]
    #     valid_interaction_type = true
    # end
    # if "NNN" in interaction_type
    #     coupling_dict[(1,1)] = couplings[1]
    #     coupling_dict[(-1,-1)] = couplings[2]
    #     coupling_dict[(1,-1)] = couplings[3]
    #     coupling_dict[(-1,1)] = couplings[4]
    #     valid_interaction_type = true
    # end

    # !valid_interaction_type && throw(ArgumentError("Unsupported interaction_type $interaction_type."))
    # return coupling_dict
end

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