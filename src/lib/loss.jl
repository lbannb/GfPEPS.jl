"""
    energy_loss(params::BCS, kvals::AbstractMatrix, Nf::Int)

Returns a function energy(CM_out) that computes the mean energy for BCS Hamiltonians with parameters `params`.
This funciton is used for the optimization of the covariance matrix of the PEPS ansatz.
This function should be highly optimized as it is called many times during the optimization, so we precompute as much as possible and avoid allocations in the inner loop.
"""
function energy_loss(params::BCS, kvals::AbstractMatrix, Nf::Int)
    ξk_batched_summed = sum(map(k -> ξ(k, params), eachcol(kvals)))

    # divide by number of k-points
    Nk = size(kvals, 2)
    invN = 1.0 / Nk # actually faster when precomputed, because multiplication is faster than division
    
    # Construct the Hamiltonian tensor (2Nf × 2Nf × Nk) (column-major order for all k values, to avoid allocations in the inner loop)
    H_BdG_batched = stack(map(k -> H_BdG_majorana_k(Nf, k, params), eachcol(kvals)))

    function energy(CM_out::AbstractArray)
        # Fast Trace Formula: Tr(H * CM) = sum(H .* CM^T)
        # Since input is already CM^T (see GaussianMap), this is just a dot product.
        return real((ξk_batched_summed + 0.25 * sum(H_BdG_batched .* CM_out)) * invN)
    end

    return energy
end

function doping_loss(Nf::Int, lattice::Union{AbstractLattice, AbstractInfiniteLattice})
    # divide by number of k-points
    Nk = size(lattice.kvals, 2)
    invN = 1.0 / Nk # actually faster when precomputed, because multiplication is faster than division

    # Construct the symplectic form (2Nf × 2Nf × Nk) (column-major order for all k values, to avoid allocations in the inner loop)
    # occupation in the majorana basis
    J0 = [0 1; -1 0]
    J = kron(I(get_Nf_in_uc(Nf,lattice)), J0)

    function doping(CM_out::AbstractArray)
        # Fast Trace Formula: Tr(J * CM) = sum(J .* CM^T)
        # Since input is already CM^T (see GaussianMap), this is just a dot product.
        return real((0.5 * Nf + 0.25*sum(J .* CM_out)) * invN)
    end
end

"""
    energy_loss_X(kvals::AbstractMatrix, Nf::Int, Λ::Int, params::BCS)

Returns the energy from the CM_out as a function of the orthogonal matrix X, using the Gaussian map.
"""
function energy_loss_X(lattice::Union{AbstractLattice, AbstractInfiniteLattice}, kvals::AbstractMatrix, Nf::Int, Λ::Int, params::BCS)
    G_in = G_in_Fourier(kvals, Λ, lattice)
    energy = energy_loss(params, kvals, Nf)
    function loss(X)
        return real(energy(GaussianMap(get_Γ_blocks(Γ_fiducial(X, Nf, Λ, lattice), Nf, lattice)..., G_in)))
    end
    return loss
end

function doping_loss_X(lattice::Union{AbstractLattice, AbstractInfiniteLattice}, Nf::Int, Λ::Int)
    G_in = G_in_Fourier(lattice.kvals, Λ, lattice)
    doping = doping_loss(Nf, lattice)
    function loss(X)
        return real(doping(GaussianMap(get_Γ_blocks(Γ_fiducial(X, Nf, Λ, lattice), Nf, lattice)..., G_in)))
    end
    return loss
end