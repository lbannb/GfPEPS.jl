"""
    energy_loss(Nf::Int, H_BdG::MomentumSpaceBdGHamiltonian, lattice::AbstractInfiniteLattice)

Returns a function energy(CM_out) that computes the mean energy for BCS Hamiltonians with parameters `params`.
This funciton is used for the optimization of the covariance matrix of the PEPS ansatz.
This function should be highly optimized as it is called many times during the optimization, so we precompute as much as possible and avoid allocations in the inner loop.
"""
function energy_loss(Nf::Int, H_BdG::MomentumSpaceBdGHamiltonian, lattice::AbstractInfiniteLattice)
    kvals = lattice.kvals

    E_shift_summed = sum(map(eachcol(kvals)) do k
        ξ_mat_k = H_BdG.ξ_mat_k(k; μ = H_BdG.μ)
        Δ_mat_k = H_BdG.Δ_mat_k(k)
        return 0.5 * real(tr(ξ_mat_k)) + H_BdG.E_shift(k, ξ_mat_k, Δ_mat_k, H_BdG.μ)
    end)

    # divide by number of k-points
    # actually faster when precomputed, because multiplication is faster than division
    Nk = size(kvals, 2)
    invN = 1.0 / Nk
    
    # Construct the Hamiltonian tensor (2Nf × 2Nf × Nk) (column-major order for all k values, to avoid allocations in the inner loop)
    # we need the adjoint here because dot(H, CM_out) = sum(H' .* CM_out))
    H_BdG_batched = stack(map(k -> conj(H_BdG_majorana_k(Nf, k, H_BdG, lattice)), eachcol(kvals)))
    # H_BdG_batched = stack(map(k -> H_BdG_majorana_k(Nf, k, H_BdG, lattice), eachcol(kvals)))

    function energy(CM_out::AbstractArray)
        # Fast Trace Formula: Tr(H * CM) = sum(H .* CM^T) = - sum(H .* CM) = - dot(conj(H), CM)
        # Since input is already CM^T (see GaussianMap), this is just a dot product.

        # return real((E_shift_summed - 0.25 * sum(H_BdG_batched .* CM_out)) * invN)
        return real((E_shift_summed - 0.25 * dot(H_BdG_batched, CM_out)) * invN)
    end

    return energy
end

"""
    doping_loss(Nf::Int, lattice::AbstractInfiniteLattice)

Returns a function doping(CM_out) that computes the mean doping for a given covariance matrix `CM_out` of the PEPS ansatz.
This function is used for the optimization of the covariance matrix of the PEPS ansatz, when we want to enforce a certain doping level.
This function should be highly optimized as it is called many times during the optimization, so we precompute as much as possible and avoid allocations in the inner loop.
"""
function doping_loss(Nf::Int, lattice::AbstractInfiniteLattice)
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

    function doping(CM_out::AbstractArray)
        # Fast Trace Formula: Tr(J * CM) = sum(J .* CM^T) = - sum(J .* CM) = - dot(J, CM)
        # Recall: δ = 1 - <n>, <n> = 0.25 * Tr(J * CM) + 0.5*Nf
        # Note: I am not sure about the last part + 0.5*Nf -> true for S=1/2 but check for general S.
        return map(J -> real(0.25 * dot(J, CM_out) * invN), J_batches)
    end
end

"""
    energy_loss_X(lattice::AbstractInfiniteLattice, Nf::Int, Λ::Int, H_BdG::MomentumSpaceBdGHamiltonian)

Returns the energy from the CM_out as a function of the orthogonal matrices `Xvec` of the unit cell, using the Gaussian map.
"""
function energy_loss_X(lattice::AbstractInfiniteLattice, Nf::Int, Λ::Int, H_BdG::MomentumSpaceBdGHamiltonian)
    G_in = G_in_Fourier(Λ, lattice)
    energy = energy_loss(Nf, H_BdG, lattice)
    function loss(Xvec::AbstractVector)
        A, B, D = get_Γ_blocks(Xvec, Nf, Λ, lattice)
        return real(energy(GaussianMap(A, B, D, G_in)))
    end
    return loss
end

"""
    doping_loss_X(lattice::AbstractInfiniteLattice, Nf::Int, Λ::Int)

Returns the doping from the CM_out as a function of the orthogonal matrices `Xvec` of the unit cell, using the Gaussian map.
"""
function doping_loss_X(lattice::AbstractInfiniteLattice, Nf::Int, Λ::Int)
    G_in = G_in_Fourier(Λ, lattice)
    doping = doping_loss(Nf, lattice)
    function loss(Xvec::AbstractVector)
        A, B, D = get_Γ_blocks(Xvec, Nf, Λ, lattice)
        return real(doping(GaussianMap(A, B, D, G_in)))
    end
    return loss
end

"""
    doping_loss_X_persite(lattice, Nf, Λ)

Returns the doping from `CM_out` as a function of a vector of per-site
orthogonal matrices `Xvec`, using the per-site Gaussian map.
"""


#= 
    Functions to compute the energy + doping from the covariance matrix of the fiducial state.
=#

"""
    energy_CM(Γ_fiducial::AbstractMatrix, Nf::Int, H_BdG::MomentumSpaceBdGHamiltonian, lattice::AbstractInfiniteLattice)

The energy of a Gaussian fPEPS evaluated from the fiducial state correlation matrix `Γ_fiducial`.
"""
function energy_CM(X_vec::AbstractArray{<:AbstractMatrix}, Nf::Int, Λ::Int, H_BdG::MomentumSpaceBdGHamiltonian, lattice::AbstractInfiniteLattice)
    G_in = G_in_Fourier(Λ, lattice)
    A, B, D = get_Γ_blocks(X_vec, Nf, Λ, lattice)
    
    return real(energy_loss(Nf, H_BdG, lattice)(GaussianMap(A, B, D, G_in)))
end

"""
    doping_CM(Γ::AbstractMatrix, Nf::Int, lattice::AbstractInfiniteLattice)

The average doping `δ = 1 - (1/N) ∑_k ⟨f†_{kσ} f_{kσ}⟩`
evaluated from the fiducial state correlation matrix `Γ`.

For trivial unit cell and Nf=2:     `⟨f†_{k↑} f_{k↑}⟩ = 1/2 * (1 - Gf[1,2])`
                                    `⟨f†_{k↓} f_{k↓}⟩ = 1/2 * (1 - Gf[3,4])`

"""
function doping_CM(X_vec::AbstractArray{<:AbstractMatrix}, Nf::Int, Λ::Int, lattice::AbstractInfiniteLattice)
    G_in = G_in_Fourier(Λ, lattice)
    A, B, D = get_Γ_blocks(X_vec, Nf, Λ, lattice)
    
    return real(doping_loss(Nf, lattice)(GaussianMap(A, B, D, G_in)))
end

"""
    doping_CM(Γ::AbstractMatrix, Nf::Int, lattice::AbstractInfiniteLattice)

The average doping `δ = 1 - (1/N) ∑_k ⟨f†_{kσ} f_{kσ}⟩` for the whole unit cell
evaluated from the fiducial state correlation matrix `Γ`.

For trivial unit cell and Nf=2:     `⟨f†_{k↑} f_{k↑}⟩ = 1/2 * (1 - Gf[1,2])`
                                    `⟨f†_{k↓} f_{k↓}⟩ = 1/2 * (1 - Gf[3,4])`

"""
function avg_doping_CM(X_vec::AbstractArray{<:AbstractMatrix}, Nf::Int, Λ::Int, lattice::AbstractInfiniteLattice)
    # divide by number of k-points
    Nk = size(lattice.kvals, 2)
    invN = 1.0 / Nk # actually faster when precomputed, because multiplication is faster than division
    invN /= get_number_of_sites(lattice) # also divide by number of sites in the unit cell to get the average per site

    # Construct the symplectic form (2Nf × 2Nf × Nk) (column-major order for all k values, to avoid allocations in the inner loop)
    # occupation in the majorana basis
    J0 = [0 1; -1 0]
    J = kron(I(get_Nf_in_uc(Nf,lattice)), J0)

    # repeat for all k-points
    J_batched = Array{eltype(J)}(undef, size(J,1), size(J,2), Nk)
    @inbounds for k in 1:Nk
        J_batched[:, :, k] = J
    end

    function doping(CM_out::AbstractArray)
        # Fast Trace Formula: Tr(J * CM) = sum(J .* CM^T) = - sum(J .* CM) = - dot(J, CM)
        # Recall: δ = 1 - <n>, <n> = 0.25 * Tr(J * CM) + 0.5*Nf
        # Note: I am not sure about the last part + 0.5*Nf -> true for S=1/2 but check for general S.
        return real(0.25 * dot(J_batched, CM_out) * invN)
    end

    G_in = G_in_Fourier(Λ, lattice)
    A, B, D = get_Γ_blocks(X_vec, Nf, Λ, lattice)
    return real(doping((GaussianMap(A, B, D, G_in))))
end