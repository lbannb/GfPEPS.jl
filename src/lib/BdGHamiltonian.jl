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
        hopping::Dict{Int, <:Dict{<:Tuple{<:Number, <:Number}, <:Number}}, 
        pairing::Dict{Int, <:Dict{<:Tuple{<:Number, <:Number}, <:Number}}, 
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
- `hopping::Dict{Int, <:Dict{<:Tuple{<:Number, <:Number}, <:Number}}`: where the dict entries represent the hopping amplitude on the corresponding connection:
    * (x, y) => t_ij, where x and y follow the lattice geometry
    * (Example for square lattice x=±1, y=±1): (1,0) => t_(1,x), (-1,0) => t_(1,-x), (0,1) => t_(1,y), (0,-1) => t_(1,-y), (1,1) => t_(2,xy) etc.
- `pairing::Dict{Int, <:Dict{<:Tuple{<:Number, <:Number}, <:Number}}`: where the dict entries represent the pairing amplitude on the corresponding connection:
    * (x, y) => Δ_ij, where x and y follow the lattice geometry
    * (Example for square lattice x=±1, y=±1): (1,0) => Δ_(1,x), (-1,0) => Δ_(1,-x), (0,1) => Δ_(1,y), (0,-1) => Δ_(1,-y), (1,1) => Δ_(2,xy) etc.
- `μ::Real`:   Side dependent chemical potentials [μ₁, μ₂, ...] (can be updated when solve_μ_from_δ = true in DopingSettings)
- `ξ_mat_k::Function`: Function to compute hopping matrix ξ_mat(k) from the hopping dict for a given momentum k
- `Δ_mat_k::Function`: Function to compute pairing matrix Δ_mat(k) from the pairing dict for a given momentum k
- `E_shift::Function`: Function to implement arbitrary energy shifts.

"""
mutable struct MomentumSpaceBdGHamiltonian <: AbstractBdGHamiltonian
    hopping::Dict{Int, <:Dict{<:Tuple{<:Number, <:Number}, <:Number}}
    pairing::Dict{Int, <:Dict{<:Tuple{<:Number, <:Number}, <:Number}}
    μ::Real   # chemical potential -> This can be changed when solve_μ_from_δ = true in DopingSettings
    ξ_mat_k::Function # Function to compute hopping matrix ξ_mat(k)
    Δ_mat_k::Function # Function to compute pairing matrix Δ_mat(k)
    E_shift::Function # Function to implement arbitrary energy shifts

    function MomentumSpaceBdGHamiltonian(
        hopping::Dict{Int, <:Dict{<:Tuple{<:Number, <:Number}, <:Number}}, 
        pairing::Dict{Int, <:Dict{<:Tuple{<:Number, <:Number}, <:Number}}, 
        μ::Real, 
        ξ_mat_k::Function, 
        Δ_mat_k::Function,
        E_shift::Function = (k, ξ_k, Δ_k, μ) -> 0.0
    )
        return new(hopping, pairing, μ, ξ_mat_k, Δ_mat_k, E_shift)
    end
end

#= 
    Some example BCS Hamiltonians in momentum space for translation invariant systems
=#
"""
    default_BCS_hamiltonian(
        hopping::Dict{Int, <:Dict{<:Tuple{<:Number, <:Number}, <:Number}}, 
        pairing::Dict{Int, <:Dict{<:Tuple{<:Number, <:Number}, <:Number}}, 
        μ::Real, lattice::AbstractInfiniteLattice; 
        interaction_type::Vector{String} = ["NN"], 
        pairing_type::String = "d_wave")

Returns a `MomentumSpaceBdGHamiltonian` which follows the standard BCS form (Nf=2):
```
    H = ∑_k_σ ξ(k) c_k,σ^† c_k,σ + ( Δ(k) c_k,↑^† c_-k,↓^† + h.c. )
´´´

# Keyword Arguments
- `hopping::Dict{Int, <:Dict{<:Tuple{<:Number, <:Number}, <:Number}}`: where the dict entries represent the hopping amplitude on the corresponding connection.
- `pairing::Dict{Int, <:Dict{<:Tuple{<:Number, <:Number}, <:Number}}`: where the dict entries represent the pairing amplitude on the corresponding connection.
- `μ::Real`:   Side dependent chemical potentials [μ₁, μ₂, ...] (can be updated when solve_μ_from_δ = true in DopingSettings)
- `lattice::AbstractInfiniteLattice`: The lattice on which the BCS model is defined
- `h::Real = 0.0`: External field

# Returns
- `MomentumSpaceBdGHamiltonian`: A `MomentumSpaceBdGHamiltonian` object with the specified parameters and functions to compute ξ(k) and Δ(k) for the given lattice and interaction types.

"""
function default_BCS_hamiltonian(
    hopping::Dict{Int, <:Dict{<:Tuple{<:Number, <:Number}, <:Number}},
    pairing::Dict{Int, <:Dict{<:Tuple{<:Number, <:Number}, <:Number}},
    μ::Real,
    lattice::AbstractInfiniteLattice;
    Nf::Int = 2,
    E_shift::Function = (k, ξ_mat_k, Δ_mat_k, μ) -> 0.0)

    ξ_mat_k = get_ξ_mat_k(lattice, Nf, hopping, μ)
    Δ_mat_k = get_Δ_mat_k(lattice, Nf, pairing)

    return MomentumSpaceBdGHamiltonian(hopping, pairing, μ, ξ_mat_k, Δ_mat_k, E_shift)
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

    lattice isa AbstractInfiniteRectangularLattice ||
        throw(ArgumentError("Unsupported lattice type for Kitaev BCS Hamiltonian."))
    interaction_type == ["NN"] ||
        throw(ArgumentError("Only NN interactions are implemented for the Kitaev BCS Hamiltonian."))

    μ = -2Jz
    # Z2 background gauge field (vortex-free sector): constant shift of -Jz per site in the unit cell
    E_shift = (k, ξ_mat_k, Δ_mat_k, μ) -> -Jz * get_number_of_sites(lattice)

    # vortex-free sector: uniform gauge, so every (distinct) site in the unit cell carries the same couplings
    site_labels = unique(vec(lattice.uc_layout))
    hopping = Dict(s => Dict((1,0) => -Jx, (-1,0) => -Jx, (0,1) => -Jy, (0,-1) => -Jy) for s in site_labels)
    pairing = Dict(s => Dict((1,0) => Jx, (-1,0) => -Jx, (0,1) => Jy, (0,-1) => -Jy) for s in site_labels)

    return default_BCS_hamiltonian(hopping, pairing, μ, lattice; E_shift=E_shift, Nf=1)
end

#= 
    Functions to create hopping and pairing matrices for different lattice geometries
=#

"""
    wrap_site_index(x::Number, y::Number, dx::Number, dy::Number, lattice::AbstractInfiniteLattice)

Returns the wrapped destination coordinates (xdst, ydst) and the number of crossed supercells (Tx, Ty) when applying a displacement (dx, dy) to a site at coordinates (x, y) on the given lattice. 
The function handles periodic boundary conditions by wrapping the destination coordinates back into the unit cell and calculating how many times the displacement crosses the boundaries of the unit cell.

# Keyword Arguments
- `x::Number`: x-coordinate of the original site
- `y::Number`: y-coordinate of the original site
- `dx::Number`: x-component of the displacement
- `dy::Number`: y-component of the displacement
- `lattice::AbstractInfiniteLattice`: The lattice on which the site is located. The function currently supports `AbstractInfiniteRectangularLattice` and will throw an error for unsupported lattice types.

# Returns
- `wrapped_x::Float64`: x-coordinate of the wrapped destination site within the unit cell
- `wrapped_y::Float64`: y-coordinate of the wrapped destination site within the unit cell
- `Tx::Number`: Number of times the displacement crosses the boundary in the x-direction (number of supercells crossed)
- `Ty::Number`: Number of times the displacement crosses the boundary in the y-direction (number of supercells crossed)

"""
function wrap_site_index(x::Number, y::Number, dx::Number, dy::Number, lattice::AbstractInfiniteLattice)
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
    get_k_matrix_kernel(k::AbstractVector{<:Real}, coupling_dict::Dict{Int, <:Dict{<:Tuple{<:Number, <:Number}, <:Number}}, lattice::AbstractInfiniteLattice)

Returns the kernel matrix in momentum space for a given momentum `k`, coupling dictionary, and lattice geometry. 
The kernel matrix is constructed by summing over the contributions from all couplings specified in the `coupling_dict`, where each coupling corresponds to a specific displacement (dx, dy) on the lattice. 
The function handles the wrapping of site indices according to the lattice geometry and applies the appropriate phase factors based on the momentum `k` and the supercell translations.

# Keyword Arguments
- `k::AbstractVector{<:Real}`: Momentum vector for which to compute the kernel matrix
- `coupling_dict::Dict{Int, <:Dict{<:Tuple{<:Number, <:Number}, <:Number}}`: Dictionary where keys are tuples representing displacements (dx, dy) and values are the corresponding coupling amplitudes (hopping or pairing)
- `lattice::AbstractInfiniteLattice`: The lattice geometry which determines how site indices are wrapped and how the kernel matrix is constructed. The function currently supports `AbstractInfiniteRectangularLattice` and will throw an error for unsupported lattice types.

# Returns
- `mat_kernel::Matrix{ComplexF64}`: The resulting kernel matrix in momentum space for the given momentum `k`.

"""
function get_k_matrix_kernel(k::AbstractVector{<:Real}, coupling_dict::Dict{Int, <:Dict{<:Tuple{<:Number, <:Number}, <:Number}}, lattice::AbstractInfiniteLattice)
    Lx, Ly = lattice.Lx, lattice.Ly
    Nsites = get_number_of_sites(lattice)
    mat_kernel = zeros(ComplexF64, Nsites, Nsites)

    for x in 1:Lx, y in 1:Ly
        site_label = lattice.uc_layout[y, x] # site label in the unit cell

        # Get the valid bonds for this specific site type
        site_bonds = get(coupling_dict, site_label, Dict())

        for (bond, amplitude) in site_bonds
            dx, dy = bond
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
    get_ξ_mat_k(lattice::AbstractInfiniteLattice, Nf::Int, hopping_dict::Dict{Int, <:Dict{<:Tuple{<:Number, <:Number}, <:Number}}, μ_init::Real)

Returns the hopping matrix ξ(k) in momentum space for a given momentum `k`, lattice geometry, and momentum-space BdG Hamiltonian parameters.

# Keyword Arguments
- `lattice::AbstractInfiniteLattice`: The lattice geometry which determines how site indices are wrapped and how the kernel matrix is constructed.
- `Nf::Int`: Number of Abrikosov fermions
- `hopping_dict::Dict{Int, <:Dict{<:Tuple{<:Number, <:Number}, <:Number}}`: Dictionary where keys are tuples representing displacements (dx, dy) and values are the corresponding hopping amplitudes
- `μ_init::Real`: Initial side dependent chemical potentials [μ₁, μ₂, ...] for H_BdG (Can be updated when solve_μ_from_δ = true in DopingSettings)

# Returns
- `ξ_mat_k::Function`: The function returning the hopping matrix ξ(k) in momentum space for the given momentum `k`.

"""
function get_ξ_mat_k(lattice::AbstractInfiniteLattice, Nf::Int, hopping_dict::Dict{Int, <:Dict{<:Tuple{<:Number, <:Number}, <:Number}}, μ_init::Real)
    function ξ_mat_k(k::AbstractVector{<:Real}; μ::Real = μ_init)
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
function get_Δ_mat_k(lattice::AbstractInfiniteLattice, Nf::Int, pairing_dict::Dict{Int, <:Dict{<:Tuple{<:Number, <:Number}, <:Number}})
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

"""
    E(k::AbstractVector{<:Real}, H_bdg::MomentumSpaceBdGHamiltonian)

Returns the positive quasiparticle energies for a given momentum `k` based on the BdG Hamiltonian parameters.

# Keyword Arguments
- `k::AbstractVector{<:Real}`: Momentum vector for which to compute the energy.
- `H_bdg::MomentumSpaceBdGHamiltonian`: The BdG Hamiltonian object containing the parameters and functions to compute ξ(k) and Δ(k).

"""
function E(k::AbstractVector{<:Real}, H_bdg::MomentumSpaceBdGHamiltonian)
    H_bdG_k = [H_bdg.ξ_mat_k(k; μ = H_bdg.μ)  H_bdg.Δ_mat_k(k); 
               H_bdg.Δ_mat_k(k)' -transpose(H_bdg.ξ_mat_k(-k; μ = H_bdg.μ))]

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

        0.5 * (real(tr(H_bdg.ξ_mat_k(k; μ = H_bdg.μ))) - sum_E) +
            H_bdg.E_shift(k, H_bdg.ξ_mat_k, H_bdg.Δ_mat_k, H_bdg.μ)
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

# Optional Arguments
- `j::Union{Nothing, Int} = nothing`: If specified, compute the average density at site index `j` instead of the average density over all sites.

# Returns
- `avg_density::Float64`: The average particle density per site for the BCS mean field Hamiltonian.

"""
function avg_density(lattice::AbstractInfiniteLattice, H_bdg::MomentumSpaceBdGHamiltonian; j::Union{Nothing, Vector{Int}} = nothing)
    Nsites = get_number_of_sites(lattice)
    Np = size(H_bdg.ξ_mat_k([0.0, 0.0]; μ = H_bdg.μ), 1) # This is Nsites * Nf
    Nf = Np ÷ Nsites

    # If site index j is specified, compute the density at site j, otherwise compute the average density over all sites.
    j = isnothing(j) ? UnitRange(1, Np) : reduce(vcat, [Nf*(s-1)+1 : Nf*s for s in j]) 
    
    return mean(map(eachcol(lattice.kvals)) do k
        H_bdG_k = [H_bdg.ξ_mat_k(k; μ = H_bdg.μ)  H_bdg.Δ_mat_k(k); 
               H_bdg.Δ_mat_k(k)' -transpose(H_bdg.ξ_mat_k(-k; μ = H_bdg.μ))]
        
        Np = size(H_bdg.ξ_mat_k(k; μ = H_bdg.μ), 1) # number of particle modes = number of sites * number of flavors

        F = eigen(Hermitian(H_bdG_k))
        pos_inds = F.values .> 1e-10
        v_m_k = F.vectors[Np+1:end, pos_inds]

        sum(abs2, v_m_k[j, :]) / Nsites
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
function exact_doping(lattice::AbstractInfiniteLattice, H_bdg::MomentumSpaceBdGHamiltonian; j::Union{Nothing, Vector{Int}} = nothing)
    return 1.0 - avg_density(lattice, H_bdg; j=j)
end

"""
    solve_for_mu(lattice::AbstractInfiniteLattice, doping_layout::Matrix{Float64}, H_bdg::MomentumSpaceBdGHamiltonian; μ_range::NTuple{2, Float64} = (-5.0, 5.0))

Finds the chemical potential `μ` that corresponds to a given doping level `δ` for a BCS mean field Hamiltonian by solving the number equation.

# Keyword Arguments
- `lattice::AbstractInfiniteLattice`: The lattice object containing the momentum vectors over which to compute the doping.
- `doping_layout::Matrix{Float64}`: Matrix specifying the target doping for each site in the unit cell.
- `H_bdg::MomentumSpaceBdGHamiltonian`: The BdG Hamiltonian object containing the parameters and functions to compute ξ(k) and Δ(k).

# Optional Arguments
- `μ_range::NTuple{2, Float64} = (-5.0, 5.0)`: Range of chemical potential values to search for the solution.

# Returns
- `μ::Real`: The chemical potential that corresponds to the target doping level `δ`.

"""
function solve_for_mu(lattice::AbstractInfiniteLattice, δ::Real, H_bdg::MomentumSpaceBdGHamiltonian; μ_range::NTuple{2, Float64} = (-5.0, 5.0))
    objective(x) = begin
        H_bdg.μ = x
        return δ - exact_doping(lattice, H_bdg)
    end
    μ = find_zero(objective, μ_range)
    H_bdg.μ = μ

    return μ
end

function get_doping_layout(Nf::Int, Λ::Int, lattice::AbstractInfiniteLattice, X_vec::AbstractArray)
    # divide by number of k-points
    Nk = size(lattice.kvals, 2)
    invN = 1.0 / Nk # actually faster when precomputed, because multiplication is faster than division

    Nsites = get_number_of_sites(lattice)
    N_distinct = get_number_of_distinct_sites_in_uc(lattice)
    
    # Construct the symplectic form (2Nf × 2Nf × Nk) (column-major order for all k values, to avoid allocations in the inner loop)
    # occupation in the majorana basis
    J0 = [0 1; -1 0]
    
    # Precompute a separate batched symplectic form for each distinct site
    J_batches = Vector{Array{Float64, 3}}(undef, N_distinct)
    dim_J = 2 * Nf * Nsites

    for s in 1:N_distinct
        J_s = zeros(Float64, dim_J, dim_J)

        N_sitetype = length(findall(x -> x == s, vec(lattice.uc_layout)))
        # Find all absolute site indices in the unit cell belonging to distinct site type `s`
        for x in 1:lattice.Lx, y in 1:lattice.Ly
            if lattice.uc_layout[y, x] == s
                site_i = get_site_index(x, y, lattice)
                
                # Each spatial site has 2*Nf Majorana modes
                idx_start = (site_i - 1) * 2 * Nf + 1
                idx_end   = site_i * 2 * Nf
                
                # Set the corresponding block in J_s to J0
                J_s[idx_start:idx_end, idx_start:idx_end] = kron(I(Nf), J0)
            end
        end
        
        # Divide by number of same sites in the unit cell to get the *average* for this site type
        J_s ./= N_sitetype
        
        # Batch over all k-points
        J_batched = Array{Float64, 3}(undef, dim_J, dim_J, Nk)
        for k in 1:Nk
            J_batched[:, :, k] = J_s
        end
        J_batches[s] = J_batched
    end

    CM_out = CM_out_X(X_vec, Nf, Λ, lattice)

    # Fast Trace Formula: Tr(J * CM) = sum(J .* CM^T) = - sum(J .* CM) = - dot(J, CM)
    # Recall: δ = 1 - <n>, <n> = 0.25 * Tr(J * CM) + 0.5*Nf
    # Note: I am not sure about the last part + 0.5*Nf -> true for S=1/2 but check for general S.
    δ_arr = map(J -> real(0.25 * dot(J, CM_out) * invN), J_batches)

    return [δ_arr[lattice.uc_layout[r, c]] for c in 1:lattice.Lx, r in 1:lattice.Ly]
end

