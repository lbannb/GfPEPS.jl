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
end

"""
    energy_loss_X(kvals::AbstractMatrix, Nf::Int, Λ::Int, params::BCS)

Returns the energy from the CM_out as a function of the orthogonal matrix X, using the Gaussian map.
"""
function energy_loss_X(lattice::Union{AbstractLattice, AbstractInfiniteLattice}, Nf::Int, Λ::Int, H_BdG::MomentumSpaceBdGHamiltonian)
    G_in = G_in_Fourier(Λ, lattice)
    energy = energy_loss(Nf, H_BdG, lattice)
    function loss(Xs::AbstractVector)
        return real(energy(GaussianMap(get_Γ_blocks(Γ_fiducial(X, Nf, Λ, lattice), Nf, lattice)..., G_in)))
    end
    return loss
end

"""
    energy_loss_X_persite(lattice, Nf, Λ, H_BdG)

Returns the energy from `CM_out` as a function of a vector of orthogonal matrices `Xs`, using the Gaussian map for every orthogonal matrix `X`.
The number of orthogonal matrices in `Xs` matches the number of distinct sites in the unit cell.

The returned closure accepts an `AbstractVector{<:AbstractMatrix}`, each of size `n × n` where `n = 2(Nf + 4Λ)`.
"""
function energy_loss_X_persite(lattice::Union{AbstractLattice, AbstractInfiniteLattice}, Nf::Int, Λ::Int, H_BdG::MomentumSpaceBdGHamiltonian)
    G_in = G_in_Fourier_persite(Λ, lattice)
    energy = energy_loss(Nf, H_BdG, lattice)
    function loss(Xs::AbstractVector)
        A, B, D = get_Γ_blocks(Xs, Nf, Λ, lattice)
        return real(energy(GaussianMap(A, B, D, G_in)))
    end
    return loss
end

"""
    doping_loss_X(lattice::Union{AbstractLattice, AbstractInfiniteLattice}, Nf::Int, Λ::Int)

Returns the doping from the CM_out as a function of the orthogonal matrix X, using the Gaussian map.
"""
function doping_loss_X(lattice::Union{AbstractLattice, AbstractInfiniteLattice}, Nf::Int, Λ::Int)
    G_in = G_in_Fourier(Λ, lattice)
    doping = doping_loss(Nf, lattice)
    function loss(X)
        return real(doping(GaussianMap(get_Γ_blocks(Γ_fiducial(X, Nf, Λ, lattice), Nf, lattice)..., G_in)))
    end
    return loss
end

"""
    doping_loss_X_persite(lattice, Nf, Λ)

Returns the doping from `CM_out` as a function of a vector of per-site
orthogonal matrices `Xs`, using the per-site Gaussian map.
"""
function doping_loss_X_persite(lattice::Union{AbstractLattice, AbstractInfiniteLattice}, Nf::Int, Λ::Int)
    G_in = G_in_Fourier_persite(Λ, lattice)
    doping = doping_loss(Nf, lattice)
    function loss(Xs::AbstractVector)
        A, B, D = get_Γ_blocks(Xs, Nf, Λ, lattice)
        return real(doping(GaussianMap(A, B, D, G_in)))
    end
    return loss
end

#= 
    Functions to compute the energy + doping from the covariance matrix of the fiducial state.
=#

"""
    energy_CM(Γ_fiducial::AbstractMatrix, Nf::Int, H_BdG::MomentumSpaceBdGHamiltonian, lattice::Union{AbstractLattice, AbstractInfiniteLattice})

The energy of a Gaussian fPEPS evaluated from the fiducial state correlation matrix `Γ_fiducial`.
"""
function energy_CM(X_vec::AbstractArray{<:AbstractMatrix}, Nf::Int, Λ::Int, H_BdG::MomentumSpaceBdGHamiltonian, lattice::Union{AbstractLattice, AbstractInfiniteLattice})
    G_in = G_in_Fourier_persite(Λ, lattice)
    A, B, D = get_Γ_blocks(X_vec, Nf, Λ, lattice)
    
    return real(energy_loss(Nf, H_BdG, lattice)(GaussianMap(A, B, D, G_in)))
end

"""
    doping_CM(Γ::AbstractMatrix, Nf::Int, lattice::Union{AbstractLattice, AbstractInfiniteLattice})

The average doping `δ = 1 - (1/N) ∑_k ⟨f†_{kσ} f_{kσ}⟩`
evaluated from the fiducial state correlation matrix `Γ`.

For trivial unit cell and Nf=2:     `⟨f†_{k↑} f_{k↑}⟩ = 1/2 * (1 - Gf[1,2])`
                                    `⟨f†_{k↓} f_{k↓}⟩ = 1/2 * (1 - Gf[3,4])`

"""
function doping_CM(X_vec::AbstractArray{<:AbstractMatrix}, Nf::Int, Λ::Int, lattice::AbstractInfiniteLattice)
    G_in = G_in_Fourier_persite(Λ, lattice)
    A, B, D = get_Γ_blocks(X_vec, Nf, Λ, lattice)
    
    return real(doping_loss(Nf, lattice)(GaussianMap(A, B, D, G_in)))
end