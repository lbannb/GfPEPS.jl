"""
    ⊕(A::AbstractMatrix, n::Integer)

repeat A ⊕ A ⊕ ... (n times) via kron

"""
function ⊕(A::AbstractMatrix, n::Integer)
    @assert n >= 1

    Id = Matrix{eltype(A)}(I, n, n)
    return kron(Id, A)
end

"""
    rand_orth(n::Int)

Generate a random real orthogonal matrix via QR decomposition.
"""
function rand_orth(n::Int)
    M = randn(Float64, (n, n))
    F = qr(M)
    Q = Matrix(F.Q)
    R = F.R
    # absorb signs of diag(R) into Q
    λ = diag(R) ./ abs.(diag(R))
    Q .= Q .* λ'
    return Q
end

"""
    rand_CM(Nf::Int, Λ::Int; parity::Int = 1)

Generate a random covariance matrix `Γ` of a pure Gaussian state with given parity in Majorana representation.

Returns `Γ`, `X` where `X` is the orthogonal matrix used to construct `Γ`.
"""
function rand_CM(Nf::Int, Λ::Int; parity::Int = 1)
    N = Nf + 4Λ
    info = parity == 1 ? "even" : "odd"

    while true
        X = rand_orth(2N)
        Γ = Γ_fiducial(X, Nf, Λ)

        if pfaffian(Γ) ≈ parity # for pure BCS state, parity = Pf(Γ) = +1 (even) / -1 (odd)
            @info "Created initial covariance matrix with $info parity"
            return Γ, X
        end
    end
end

#=
    PEPSKit helper functions

    Environment initialization lives upstream now:
        env0 = initialize_ctmrg_environment(peps, IdentityInitialization())
    `IdentityInitialization` is the identity between the ket and bra virtual spaces, i.e.
    exactly what `init_ctmrg_env` used to build, and CTMRG grows the environment spaces from
    trivial up to `trunc`, which is what `grow_env` used to do by hand.
=#
