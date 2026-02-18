"""
    energy_loss(params::BCS, bz::BrillouinZone2D)

Returns a function energy(CM_out) computing the mean energy density for BCS Hamiltonian with parameters `params`.
Note: only for Nf=2 (Spinful) BCS Hamiltonian
"""
function energy_loss(params::BCS, bz::BrillouinZone2D)
    k_vals = bz.kvals

    ξk_batched = map(k -> ξ(k, params), eachcol(k_vals))
    Δk_batched = map(k -> Δ(k, params), eachcol(k_vals))
    ξk_batched_summed = sum(ξk_batched)

    # divide by number of k-points
    invN = 1.0 / size(k_vals, 2) # actually faster when precomputed, because multiplication is faster than division

    function energy(CM_out::AbstractArray)
        #= 
            qq-ordering of Majorana modes: (c_1, c_2, ..., c_(2(4Nv + Nf)))
        =#
        η = 0.25 .* (CM_out[:, 1, 4] .+ CM_out[:, 2, 3] .+ im .* (CM_out[:, 2, 4] .- CM_out[:, 1, 3]))
        @inbounds E = ξk_batched_summed - 0.5*(dot(ξk_batched, CM_out[:, 1, 2]) + dot(ξk_batched, CM_out[:, 3, 4])) + 2*real(dot(Δk_batched, η))
        return real(E * invN)
    end

    return energy
end

"""
    energy_loss(params::Kitaev, bz::BrillouinZone2D)

Returns a function energy(CM_out) computing the mean energy density for Kitaev Hamiltonian with parameters `params`.
Note: only for Nf=1 Kitaev Hamiltonian (after transforming Hamiltonian to 1x1 square lattice unit cell)
"""
function energy_loss(params::Kitaev, bz::BrillouinZone2D)
    k_vals = bz.kvals

    ξk_batched = map(k -> ξ(k, params), eachcol(k_vals))
    Δk_batched = imag.(map(k -> Δ(k, params), eachcol(k_vals)))
    ξk_batched_summed = sum(ξk_batched)

    # divide by number of k-points
    N = size(k_vals, 2)
    invN = 1.0 / size(k_vals, 2)

    function energy(CM_out::AbstractArray)
        #= 
            qq-ordering of Majorana modes: (c_1, c_2, ..., c_(2(4Nv + Nf)))
        =#
        @inbounds E = 0.5 * (ξk_batched_summed - dot(ξk_batched, real.(CM_out[:, 1, 2]))) - 0.5 * dot(Δk_batched, imag.(CM_out[:, 1, 2])) - params.Jz * N
        return real(E  * invN)
    end

    return energy
end

"""
    optimize_loss(t::Real, Δ_x::Real, Δ_y::Real, μ::Real, Lx::Int, Ly::Int, Nf::Int, Nv::Int)

Returns the energy from the CM_out as a function of the orthogonal matrix X, obtained from the minimization, using the Gaussian map.
"""
function optimize_loss(bz::BrillouinZone2D, Nf::Int, Nv::Int, params::Union{BCS,Kitaev})
    G_in = G_in_Fourier(bz, Nv)
    energy = energy_loss(params, bz)
    function loss(X)
        return real(energy(GaussianMap(get_Γ_blocks(Γ_fiducial(X, Nv, Nf), Nf)..., G_in)))
    end
    return loss
end