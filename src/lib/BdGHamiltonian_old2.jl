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
        ξ_k::Function, 
        Δ_k::Function,
        E_shift::Function = (k, ξ_k, Δ_k, μ) -> 0.0
    )

Stores the parameters in momentum space to construct the Hamiltonian matrix ``H_BdG_k`` in BCS form for a specific momentum `k`:

``
H_BdG_k = [ξ_k(k)  Δ_k(k); 
    Δ_k(k)'  -ξ_k(k)]
``

# Fields
- `hopping::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}}`: where the dict entries represent the hopping amplitude on the corresponding connection:
    * (x, y) => t_ij, where x and y follow the lattice geometry
    * (Example for square lattice x=±1, y=±1): (1,0) => t_(1,x), (-1,0) => t_(1,-x), (0,1) => t_(1,y), (0,-1) => t_(1,-y), (1,1) => t_(2,xy) etc.
- `pairing::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}}`: where the dict entries represent the pairing amplitude on the corresponding connection:
    * (x, y) => Δ_ij, where x and y follow the lattice geometry
    * (Example for square lattice x=±1, y=±1): (1,0) => Δ_(1,x), (-1,0) => Δ_(1,-x), (0,1) => Δ_(1,y), (0,-1) => Δ_(1,-y), (1,1) => Δ_(2,xy) etc.
- `μ::Real`: Chemical potential (can be updated when solve_μ_from_δ = true in DopingSettings)
- `ξ_k::Function`: Function to compute the matrix-valued hopping kernel ξ(k, μ) in the unit-cell basis.
- `Δ_k::Function`: Function to compute the matrix-valued pairing kernel Δ(k) in the unit-cell basis.
- `E_shift::Function`: Function to implement arbitrary energy shifts.

"""
mutable struct MomentumSpaceBdGHamiltonian <: AbstractBdGHamiltonian
    hopping::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}}
    pairing::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}}
    μ::Real # chemical potential -> This can be changed when solve_μ_from_δ = true in DopingSettings
    ξ_k::Function # function to compute the hopping matrix ξ(k)
    Δ_k::Function # function to compute the pairing matrix Δ(k)
    E_shift::Function # function to implement arbitrary energy shifts
    interaction_type::Vector{String} # ["NN"], ["NNN"], ["NN","NNN"] 

    function MomentumSpaceBdGHamiltonian(
        hopping::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}}, 
        pairing::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}}, 
        μ::Real, 
        ξ_k::Function, 
        Δ_k::Function,
        interaction_type::Vector{String};
        E_shift::Function = (k, ξ_k, Δ_k, μ) -> 0.0
    )
        return new(hopping, pairing, μ, ξ_k, Δ_k, E_shift, interaction_type)
    end
end

@inline function _rectangular_site_index(ix::Int, iy::Int, lattice::AbstractInfiniteRectangularLattice)
    return ix + (iy - 1) * lattice.Lx
end

@inline function _wrap_rectangular_site(ix::Int, iy::Int, dx::Int, dy::Int, lattice::AbstractInfiniteRectangularLattice)
    raw_x = ix + dx
    raw_y = iy + dy

    wrapped_x = mod1(raw_x, lattice.Lx)
    wrapped_y = mod1(raw_y, lattice.Ly)

    shift_x = fld(raw_x - wrapped_x, lattice.Lx)
    shift_y = fld(raw_y - wrapped_y, lattice.Ly)

    return wrapped_x, wrapped_y, shift_x, shift_y
end

@inline function _rectangular_interaction_label(dx::Int, dy::Int)
    if (abs(dx) == 1 && dy == 0) || (dx == 0 && abs(dy) == 1)
        return "NN"
    elseif abs(dx) == 1 && abs(dy) == 1
        return "NNN"
    end

    return nothing
end

@inline function _integer_displacement(displacement::Tuple{Float64, Float64})
    dx = round(Int, displacement[1])
    dy = round(Int, displacement[2])

    isapprox(displacement[1], dx; atol=1e-12) || throw(ArgumentError("Only integer rectangular lattice displacements are supported. Got $(displacement[1])."))
    isapprox(displacement[2], dy; atol=1e-12) || throw(ArgumentError("Only integer rectangular lattice displacements are supported. Got $(displacement[2])."))

    return dx, dy
end

function _rectangular_kernel_matrix(
    k::AbstractVector{<:Real},
    couplings::Union{Dict{Tuple{Float64, Float64}, ComplexF64}, Dict{Tuple{Float64, Float64}, Float64}},
    lattice::AbstractInfiniteRectangularLattice,
    interaction_type::Vector{String};
    prefactor::Number,
    μ::Real = 0.0,
    include_chemical_potential::Bool = false)

    Nsites = get_number_of_sites(lattice)
    kernel = zeros(ComplexF64, Nsites, Nsites)

    for (displacement, amplitude) in couplings
        dx, dy = _integer_displacement(displacement)
        label = _rectangular_interaction_label(dx, dy)

        isnothing(label) && throw(ArgumentError("Unsupported interaction displacement $(displacement) for rectangular lattice."))
        !(label in interaction_type) && continue

        for iy in 1:lattice.Ly, ix in 1:lattice.Lx
            src = _rectangular_site_index(ix, iy, lattice)
            wrapped_x, wrapped_y, shift_x, shift_y = _wrap_rectangular_site(ix, iy, dx, dy, lattice)
            dst = _rectangular_site_index(wrapped_x, wrapped_y, lattice)
            phase = cis(k[1] * dx + k[2] * dy)

            kernel[src, dst] += prefactor * amplitude * phase
        end
    end

    if include_chemical_potential
        @inbounds for site in 1:Nsites
            kernel[site, site] -= μ
        end
    end

    return kernel
end

function _normal_flavor_structure(Nf::Int)
    return Matrix{ComplexF64}(I, Nf, Nf)
end

function get_ξ_k_matrix(lattice::AbstractInfiniteLattice, Nf::Int, k::AbstractVector{<:Real}, H_bdg_k::MomentumSpaceBdGHamiltonian)
    return kron(H_bdg_k.ξ_k(k, H_bdg_k.μ), _normal_flavor_structure(Nf))
end

function get_Δ_k_matrix(lattice::AbstractInfiniteLattice, Nf::Int, k::AbstractVector{<:Real}, H_bdg_k::MomentumSpaceBdGHamiltonian)
    Δ_k_mat = H_bdg_k.Δ_k(k)

    if Nf == 1
        return Δ_k_mat
    elseif Nf == 2
        Δ_minus_k_mat = H_bdg_k.Δ_k(-k)
        Nsites = size(Δ_k_mat, 1)
        Δ_full = zeros(ComplexF64, 2 * Nsites, 2 * Nsites)

        for src_site in 1:Nsites, dst_site in 1:Nsites
            src_up = 2src_site - 1
            src_dn = 2src_site
            dst_up = 2dst_site - 1
            dst_dn = 2dst_site

            Δ_full[src_up, dst_dn] = Δ_k_mat[src_site, dst_site]
            Δ_full[src_dn, dst_up] = -Δ_minus_k_mat[dst_site, src_site]
        end

        return Δ_full
    end

    throw(ArgumentError("Momentum-space BdG Hamiltonians currently support Nf = 1 or Nf = 2. Got Nf = $Nf."))
end

function get_reduced_BdG_k_matrix(k::AbstractVector{<:Real}, H_bdg_k::MomentumSpaceBdGHamiltonian)
    ξ_k_mat = H_bdg_k.ξ_k(k, H_bdg_k.μ)
    Δ_k_mat = H_bdg_k.Δ_k(k)
    ξ_minus_k_mat = H_bdg_k.ξ_k(-k, H_bdg_k.μ)

    return [ξ_k_mat Δ_k_mat; adjoint(Δ_k_mat) -transpose(ξ_minus_k_mat)]
end

function get_positive_quasiparticle_energies(H_BdG_k_mat::AbstractMatrix)
    evals = eigvals(Hermitian(H_BdG_k_mat))
    return [real(e) for e in evals if real(e) > 1e-10]
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
    E_shift::Function = (k, ξ_k, Δ_k, μ) -> 0.0,
    interaction_type::Vector{String} = ["NN"])

    if lattice isa AbstractInfiniteRectangularLattice
        ξ_k = (k::AbstractVector{<:Real}, μ_local::Real) -> _rectangular_kernel_matrix(
            k,
            hopping,
            lattice,
            interaction_type;
            prefactor=-1,
            μ=μ_local,
            include_chemical_potential=true)

        Δ_k = (k::AbstractVector{<:Real}) -> _rectangular_kernel_matrix(
            k,
            pairing,
            lattice,
            interaction_type;
            prefactor=1,
            include_chemical_potential=false)

        return MomentumSpaceBdGHamiltonian(hopping, pairing, μ, ξ_k, Δ_k, interaction_type; E_shift=E_shift)
    end

    throw(ArgumentError("Unsupported lattice type for default_BCS_hamiltonian."))
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

             return default_BCS_hamiltonian(hopping, pairing, μ, lattice; E_shift=E_shift, interaction_type=["NN"])
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
    Energy functions for BCS Hamiltonians
=#

function get_BdG_k_matrix(lattice::AbstractInfiniteLattice, Nf::Int, k::AbstractVector{<:Real}, H_bdg_k::MomentumSpaceBdGHamiltonian)
    ξ_mat = get_ξ_k_matrix(lattice, Nf, k, H_bdg_k)
    Δ_mat = get_Δ_k_matrix(lattice, Nf, k, H_bdg_k)
    ξ_minus_k_mat = get_ξ_k_matrix(lattice, Nf, -k, H_bdg_k)

    H_BdG_k_mat = [ξ_mat Δ_mat; adjoint(Δ_mat) -transpose(ξ_minus_k_mat)]
    return H_BdG_k_mat
end

"""
    E(k::AbstractVector{<:Real}, H_bdg::MomentumSpaceBdGHamiltonian)

Returns the energy of the Bogoliubov quasiparticles for a given momentum `k` based on the BdG Hamiltonian parameters.

# Keyword Arguments
- `k::AbstractVector{<:Real}`: Momentum vector for which to compute the energy.
- `H_bdg::MomentumSpaceBdGHamiltonian`: The BdG Hamiltonian object containing the parameters and functions to compute ξ(k) and Δ(k).

"""
function E(k::AbstractVector{<:Real}, H_bdg::MomentumSpaceBdGHamiltonian)
    return minimum(get_positive_quasiparticle_energies(get_reduced_BdG_k_matrix(k, H_bdg)))
end

"""
    exact_energy(kvals::AbstractMatrix, H_bdg::MomentumSpaceBdGHamiltonian, Nf::Int)

Returns the exact ground state energy per site of a BCS mean field Hamiltonian.

"""
function exact_energy(lattice::AbstractInfiniteLattice, H_bdg::MomentumSpaceBdGHamiltonian, Nf::Int)
    Ns = get_number_of_sites(lattice)

    return mean(map(eachcol(lattice.kvals)) do k
        ξ_k_mat = H_bdg.ξ_k(k, H_bdg.μ)
        Δ_k_mat = H_bdg.Δ_k(k)
        H_BdG_k_mat = get_BdG_k_matrix(lattice, Nf, k, H_bdg)
        sum_E = sum(get_positive_quasiparticle_energies(H_BdG_k_mat))

        (0.5 * (Nf * real(tr(ξ_k_mat)) - sum_E) / Ns) + H_bdg.E_shift(k, ξ_k_mat, Δ_k_mat, H_bdg.μ)
    end)
end

"""
    has_dirac_points(kvals::AbstractMatrix, H_bdg::MomentumSpaceBdGHamiltonian)

Checks if there are Dirac points (zero-energy modes) in the quasiparticle energy spectrum over a given momentum set `kvals`.

"""
function has_dirac_points(kvals::AbstractMatrix, H_bdg::MomentumSpaceBdGHamiltonian)
    dirac_point_found = false
    for k in eachcol(kvals)
        if isapprox(E(k, H_bdg), 0.0; atol = 1e-6)
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
    exact_doping(kvals::AbstractMatrix, H_bdg::MomentumSpaceBdGHamiltonian)

Returns the exact doping level of a BCS mean field Hamiltonian.

"""
function exact_doping(kvals::AbstractMatrix, H_bdg::MomentumSpaceBdGHamiltonian)
    Ns = size(H_bdg.ξ_k(kvals[:, 1], H_bdg.μ), 1)

    return mean(map(eachcol(kvals)) do k
        evals, evecs = eigen(Hermitian(get_reduced_BdG_k_matrix(k, H_bdg)))
        negative_modes = findall(evals .< -1e-10)
        projector = evecs[:, negative_modes] * adjoint(evecs[:, negative_modes])
        occupation_k = real(tr(projector[1:Ns, 1:Ns]))

        1.0 - 2.0 * occupation_k / Ns
    end)
end

"""
    solve_for_mu(kvals::AbstractMatrix, δ::Real, H_bdg::MomentumSpaceBdGHamiltonian; μ_range::NTuple{2, Float64} = (-5.0, 5.0))

Finds the chemical potential `μ` that corresponds to a given doping level `δ` for a BCS mean field Hamiltonian by solving the number equation.

# Keyword Arguments
- `kvals::AbstractMatrix`: Matrix of momentum vectors over which to compute the doping.
- `δ::Real`: Target doping level to solve for.
- `H_bdg::MomentumSpaceBdGHamiltonian`: The BdG Hamiltonian object containing the parameters and functions to compute ξ(k) and Δ(k).

# Optional Arguments
- `μ_range::NTuple{2, Float64} = (-5.0, 5.0)`: Range of chemical potential values to search for the solution.

# Returns
- `μ::Real`: The chemical potential that corresponds to the target doping level `δ`.

"""
function solve_for_mu(kvals::AbstractMatrix, δ::Real, H_bdg::MomentumSpaceBdGHamiltonian; μ_range::NTuple{2, Float64} = (-5.0, 5.0))
    μ_initial = H_bdg.μ
    objective(x) = begin
        H_bdg.μ = x
        return δ - exact_doping(kvals, H_bdg)
    end

    μ = find_zero(objective, μ_range)
    H_bdg.μ = μ_initial

    return μ
end

