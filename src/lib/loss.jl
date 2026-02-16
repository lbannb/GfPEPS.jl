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

    H_BdG_batched = Vector{Matrix{ComplexF64}}()
    for k in eachcol(kvals)
        push!(H_BdG_batched, H_BdG_majorana_k(Nf, k, params))
    end

    # custom trace function, avoiding allocations
    @inline function trAB_slice_no_alloc(A::Matrix{ComplexF64}, CM_out::AbstractArray, kidx::Int)
        res = 0.0
        @inbounds for j in axes(A, 2), i in axes(A, 1)
            res += real(A[i, j] * CM_out[kidx, j, i])
        end
        return res
    end

    function energy(CM_out::AbstractArray)
        E = ξk_batched_summed
        @inbounds for i in 1:Nk
            E += -0.25 * trAB_slice_no_alloc(H_BdG_batched[i], CM_out, i)
        end
        return real(E * invN)
    end

    return energy
end

# """
#     energy_loss(params::Kitaev, bz::BrillouinZone2D)

# Returns a function energy(CM_out) computing the mean energy density for Kitaev Hamiltonian with parameters `params`.
# Note: only for Nf=1 Kitaev Hamiltonian (after transforming Hamiltonian to 1x1 square lattice unit cell)
# """
# function energy_loss(params::Kitaev, bz::BrillouinZone2D)
#     k_vals = bz.kvals

#     ξk_batched = map(k -> ξ(k, params), eachcol(k_vals))
#     Δk_batched = imag.(map(k -> Δ(k, params), eachcol(k_vals)))
#     ξk_batched_summed = sum(ξk_batched)

#     # divide by number of k-points
#     N = size(k_vals, 2)
#     invN = 1.0 / size(k_vals, 2)

#     function energy(CM_out::AbstractArray)
#         #= 
#             qq-ordering of Majorana modes: (c_1, c_2, ..., c_(2(4Nv + Nf)))
#         =#
#         @inbounds E = 0.5 * (ξk_batched_summed - dot(ξk_batched, real.(CM_out[:, 1, 2]))) - 0.5 * dot(Δk_batched, imag.(CM_out[:, 1, 2])) - params.Jz * N
#         return real(E  * invN)
#     end

#     return energy
# end

"""
    energy_loss_X(kvals::AbstractMatrix, Nf::Int, Λ::Int, params::BCS)

Returns the energy from the CM_out as a function of the orthogonal matrix X, using the Gaussian map.
"""
function energy_loss_X(lattice::Union{AbstractLattice, AbstractInfiniteLattice}, kvals::AbstractMatrix, Nf::Int, Λ::Int, params::BCS)
    Nf_in_uc = get_Nf_in_uc(Nf, lattice)
    Λ_in_uc = get_Λ_in_uc(Λ, lattice)

    G_in = G_in_Fourier(kvals, Λ)
    energy = energy_loss(params, kvals, Nf)
    function loss(X)
        return real(energy(GaussianMap(get_Γ_blocks(Γ_fiducial(X, Λ, Nf), Nf)..., G_in)))
    end
    return loss
end