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

    function energy(CM_out::AbstractArray)
        # Fast Trace Formula: Tr(H * CM) = sum(H .* CM^T) = - sum(H .* CM) = - dot(conj(H), CM)
        # Since input is already CM^T (see GaussianMap), this is just a dot product.
        return real((E_shift_summed - 0.25 * dot(H_BdG_batched, CM_out)) * invN)
    end

    return energy
end

"""
    doping_loss(Nf::Int, lattice::AbstractInfiniteLattice)

Returns a function doping(CM_out) that computes the mean doping per site for a given covariance matrix `CM_out` of the PEPS ansatz.
This function is used for the optimization of the covariance matrix of the PEPS ansatz, when we want to enforce a certain doping level.
This function should be highly optimized as it is called many times during the optimization, so we precompute as much as possible and avoid allocations in the inner loop.
"""
function doping_loss(Nf::Int, lattice::AbstractInfiniteLattice)
    # divide by number of k-points and sites in the unit cell (doping per site)
    Nk = size(lattice.kvals, 2)
    invN = 1.0 / (Nk * get_number_of_sites(lattice)) # actually faster when precomputed, because multiplication is faster than division

    # Construct the symplectic form (2Nf × 2Nf × Nk) (column-major order for all k values, to avoid allocations in the inner loop)
    # occupation in the majorana basis
    J0 = [0 1; -1 0]
    J = kron(I(get_Nf_in_uc(Nf, lattice)), J0)

    # repeat for all k-points
    J_batched = Array{eltype(J)}(undef, size(J, 1), size(J, 2), Nk)
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
    CM_out_X(X_vec, Nf, Λ, lattice, gm_inputs=gaussian_map_inputs(Λ, lattice))

Covariance matrix of the physical state (batched over k) for the per-site orthogonal
matrices `X_vec`, computed with the blocked Gaussian map: the intra-unit-cell bonds are
contracted once via a k-independent Schur complement (`contract_inner_modes`), then the
wrap bonds are contracted per k-point (`GaussianMap`). Pass a precomputed
`gm_inputs = gaussian_map_inputs(Λ, lattice)` inside optimization loops.
"""
function CM_out_X(X_vec::AbstractArray{<:AbstractMatrix}, Nf::Int, Λ::Int, lattice::AbstractInfiniteLattice,
                  gm_inputs=gaussian_map_inputs(Λ, lattice))
    A, B, D = get_Γ_blocks(X_vec, Nf, Λ, lattice)
    A_eff, B_eff, D_eff = contract_inner_modes(A, B, D, gm_inputs.G_intra, gm_inputs.inner, gm_inputs.bdry)
    return GaussianMap(A_eff, B_eff, D_eff, gm_inputs.G_wrap)
end

"""
    energy_loss_X(lattice::AbstractInfiniteLattice, Nf::Int, Λ::Int, H_BdG::MomentumSpaceBdGHamiltonian)

Returns the energy from the CM_out as a function of the orthogonal matrices `X_vec` of the unit cell, using the blocked Gaussian map.
This is the unified loss function for all quadratic Hamiltonians expressed as a `MomentumSpaceBdGHamiltonian`
(BCS as well as Kitaev models) and for arbitrary rectangular unit cells.
"""
function energy_loss_X(lattice::AbstractInfiniteLattice, Nf::Int, Λ::Int, H_BdG::MomentumSpaceBdGHamiltonian)
    gm_inputs = gaussian_map_inputs(Λ, lattice)
    energy = energy_loss(Nf, H_BdG, lattice)
    function loss(X_vec::AbstractVector)
        return real(energy(CM_out_X(X_vec, Nf, Λ, lattice, gm_inputs)))
    end
    return loss
end

"""
    doping_loss_X(lattice::AbstractInfiniteLattice, Nf::Int, Λ::Int)

Returns the doping from the CM_out as a function of the orthogonal matrices `X_vec` of the unit cell, using the blocked Gaussian map.
"""
function doping_loss_X(lattice::AbstractInfiniteLattice, Nf::Int, Λ::Int)
    gm_inputs = gaussian_map_inputs(Λ, lattice)
    doping = doping_loss(Nf, lattice)
    function loss(X_vec::AbstractVector)
        return real(doping(CM_out_X(X_vec, Nf, Λ, lattice, gm_inputs)))
    end
    return loss
end

#=
    Functions to compute the energy + doping from the covariance matrix of the fiducial state.
=#

"""
    energy_CM(X_vec::AbstractArray{<:AbstractMatrix}, Nf::Int, Λ::Int, H_BdG::MomentumSpaceBdGHamiltonian, lattice::AbstractInfiniteLattice)

The energy of a Gaussian fPEPS evaluated from the per-site orthogonal matrices `X_vec` of the fiducial state.
"""
function energy_CM(X_vec::AbstractArray{<:AbstractMatrix}, Nf::Int, Λ::Int, H_BdG::MomentumSpaceBdGHamiltonian, lattice::AbstractInfiniteLattice)
    return real(energy_loss(Nf, H_BdG, lattice)(CM_out_X(X_vec, Nf, Λ, lattice)))
end

"""
    doping_CM(X_vec::AbstractArray{<:AbstractMatrix}, Nf::Int, Λ::Int, lattice::AbstractInfiniteLattice)

The average doping per site `δ = 1 - (1/N) ∑_k ⟨f†_{kσ} f_{kσ}⟩`
evaluated from the per-site orthogonal matrices `X_vec` of the fiducial state.

For trivial unit cell and Nf=2:     `⟨f†_{k↑} f_{k↑}⟩ = 1/2 * (1 - Gf[1,2])`
                                    `⟨f†_{k↓} f_{k↓}⟩ = 1/2 * (1 - Gf[3,4])`
"""
function doping_CM(X_vec::AbstractArray{<:AbstractMatrix}, Nf::Int, Λ::Int, lattice::AbstractInfiniteLattice)
    return real(doping_loss(Nf, lattice)(CM_out_X(X_vec, Nf, Λ, lattice)))
end
