"""get_bogoliubov_blocks(M::AbstractMatrix)
Extract the `U` and `V` blocks from the Bogoliubov transformation matrix `M = [U conj(V); V conj(U)]`.
"""
function get_bogoliubov_blocks(M::AbstractMatrix)
    N = div(size(M, 1), 2)
    U = M[1:N, 1:N]
    V = M[N+1:end, 1:N]
    return U, V
end

"""Return the spectrum and canonical transform that diagonalize the fermionic quadratic Hamiltonian `H`.

The Bogoliubov matrix `M = [U conj(V); V conj(U)]` satisfies `M' * H * M == diagm(vcat(E, -E))`, and the
returned vector `E` contains the positive eigenvalues in descending order.
"""
function bogoliubov(H::Hermitian)
    N = div(size(H, 1), 2)
    E, M0 = eigen(H; sortby = (x -> -real(x)))

    U = M0[1:N, 1:N]
    V = M0[N+1:end, 1:N]

    # bring to correct form
    M = similar(M0)
    M[1:N, 1:N] = U
    M[N+1:end, 1:N] = V
    M[1:N, N+1:end] = conj.(V)
    M[N+1:end, N+1:end] = conj.(U)

    # check canonical constraints
    @assert M' * M ≈ I
    @assert U'U + V'V ≈ I
    @assert transpose(U) * V ≈ - transpose(V) * U
    
    return E, M
end

"""skew_canonical_form(P::AbstractMatrix)

Return a pair `(S, X)` where `X = transpose(S)*P*S` is the canonical form for `P` (See: https://doi.org/10.1007/BF02906230).
"""
function skew_canonical_form(P::AbstractMatrix)
    # Check skew-symmetry
    @assert transpose(P) ≈ -P

    W = P'P
    @assert ishermitian(W)

    E, Φ = eigen(Hermitian(W); sortby = (x -> -real(x)))
    alphas = sqrt.(abs.(E))
    tol = 1e-7

    # sort indices by magnitude descending to make pairing stable
    idx_sorted = sortperm(alphas, rev = true)
    nonzero_idx = [i for i in idx_sorted if !isapprox(alphas[i], 0.0; atol=tol)]

    # ensure we have an even number of nonzero modes (otherwise pairing impossible)
    if isodd(length(nonzero_idx))
        error("skew_canonical_form: odd number of nonzero canonical values (check tolerance).")
    end

    # build orthogonal eigenvectors
    Φ_prime = similar(Φ)
    for j in nonzero_idx
        Φ_prime[:, j] = (P' * conj.(Φ[:, j])) / alphas[j]
    end

    # build S
    S = similar(P)
    n_zeros = 0
    for j in eachindex(alphas)
        if isapprox(alphas[j], 0.0; atol=tol)
            S[:, end-n_zeros] = Φ[:, j]
            n_zeros += 1
        else
            if isodd(j)
                S[:, j] = Φ[:, j]
            else
                S[:, j] = Φ_prime[:, j-1]
            end
        end
    end

    X = S'*P*conj(S)

    # permutation to have positive elements in the upper-right of each 2x2 block
    perm_mat = canonical_skew_permutation(X)
    X = perm_mat' * X * perm_mat
    S = S * perm_mat

    # absorb phases into S to have X is real
    S, X = absorb_phases(S, X)

    X[abs.(X) .< tol] .= 0.0
    return S,X
end

"""absorb_phases(S::AbstractMatrix, X::AbstractMatrix)

Adjust phases of the paired columns in `S` so that the corresponding canonical matrix `X` becomes real with
positive entries in its upper-right elements. Returns the modified `(S2, X2)` pair.
"""
function absorb_phases(S::AbstractMatrix, X::AbstractMatrix)
    S2 = copy(S)
    X2 = copy(X)
    tol_absorb = 1e-10

    n = size(X2,1)
    i = 1
    while i <= n-1
        x = X2[i, i+1]
        y = X2[i+1, i]
        # If already real with nonnegative value, skip
        if !(abs(imag(x)) ≈ 0 && real(x) >= 0)
            φ = angle(x)
            d = exp(1im * φ/2)          # uniform phase for the pair
            @views S2[:, i  ] .*= d
            @views S2[:, i+1] .*= d
            # Block transforms by conj(d)^2
            # After this, value becomes real ≈ |x|
            X2[i, i+1] = abs(x)
            X2[i+1, i] = -abs(x)
        end
        i += 2
    end
    max_imag = maximum(abs, imag(X2))
    if max_imag > tol_absorb
        @warn "absorb_phases: residual imaginary part in X2 exceeds tolerance" max_imag=max_imag tol=tol_absorb
    end
    # @assert isapprox(imag(X2), zeros(ComplexF64, n,n), atol=1e-10) "X2 should be real after phase absorption"

    return S2, real(X2)
end

"""canonical_skew_permutation(P::AbstractMatrix)

Return a permutation matrix that reorders 2×2 skew blocks so their upper-right elements have a nonnegative real part.
"""
function canonical_skew_permutation(P::AbstractMatrix)
    n = size(P,1)
    perm = collect(1:n)
    i = 1
    while i < n
        a = P[perm[i], perm[i+1]]
        b = P[perm[i+1], perm[i]]
        # Detect a 2×2 skew block (nonzero pair with b ≈ -a)
        if abs(a) > 0 && isapprox(b, -a; atol=1e-14, rtol=1e-10)
            # If real part of upper-right entry is < 0, swap the two indices
            if real(a) < 0
                perm[i], perm[i+1] = perm[i+1], perm[i]
            end
            i += 2
        else
            i += 1
        end
    end
    S = Matrix{eltype(P)}(I, n, n)
    return S[:, perm]
end

"""bloch_messiah_decomposition(M::AbstractMatrix)

Compute the Bloch–Messiah decomposition of the Bogoliubov transformation `M` and return the
left (`Dmat`), middle (`UV_mat`), and right (`Cmat`) blocks such that
`M ≈ Dmat * UV_mat * Cmat`.
"""
function bloch_messiah_decomposition(M::AbstractMatrix)
    N = div(size(M, 1), 2)

    U,V = GfPEPS.get_bogoliubov_blocks(M)

    Q = conj.(V) * transpose(V)
    @assert Q' ≈ Q "Q should be Hermitian"
    Q = Hermitian(Q) # enforce exact Hermiticity
    P = conj.(V) * transpose(U)
    @assert transpose(P) ≈ -P "P should be skew-symmetric"
    P = (P - transpose(P)) / 2 # enforce exact skew-symmetry
    @assert Q*P ≈ P*conj.(Q)

    _, B = eigen(Q; sortby = (x -> -real(x)))
    # Q_bar = real(B'*Q*B)
    P_bar = B'*P*conj.(B)
    @assert P_bar ≈ - transpose(P_bar) "P_bar should be skew-symmetric"

    # Bring P_bar to canonical form
    S, _ = skew_canonical_form(P_bar)
    P_canonical = S' * P_bar * conj.(S)

    A = permute_zero_cols_to_end(P_canonical)

    D = B * S * A

    @assert D'*P*conj(D) ≈ D'*conj(V)*transpose(U)*conj(D)
    
    F = MatrixFactorizations.rq(D' * U)
    R = Matrix(F.R)
    Q = Matrix(F.Q)

    # Fix phases so diagonal of R becomes positive real.
    d = diag(R)
    ph = similar(d)
    for i in eachindex(d)
        ph[i] = (abs(d[i]) > 0) ? d[i]/abs(d[i]) : one(d[i])   # unit-modulus (or 1 if zero)
    end
    Φ  = Diagonal(conj.(ph))            # multiply R on right by Φ to remove phases
    Rpos = R * Φ                        # now diagonal(Rpos) = abs.(d) ≥ 0 (real)
    Qnew = Φ' * Q                       # keep A invariant: (R Φ)(Φ' Q) = R Q
    Ubar = Rpos                         # Ū with positive diagonal
    C    = Qnew

    Ubar = R
    C = Q

    Vbar = transpose(D) * V * C'

    # @assert D'*conj(V)*transpose(U)*conj(D) ≈ D'*conj(V)*transpose(C)*conj(C)*transpose(U)*conj(D)
    # @assert D'*conj(V)*transpose(Q)*conj(Q)*transpose(U)*conj(D) ≈ D'*conj(V)*transpose(Q)*R

    @assert C'C ≈ I
    @assert Q'Q ≈ I

    @assert U ≈ D*Ubar*C "Something went wrong with Bloch-Messiah decomposition for U"
    @assert V ≈ conj.(D)*Vbar*C "Something went wrong with Bloch-Messiah decomposition for V"

    # remove numerical noise
    Ubar = real(Ubar)
    Ubar[abs.(Ubar) .< 1e-12] .= 0.0
    Vbar = real(Vbar)
    Vbar[abs.(Vbar) .< 1e-12] .= 0.0

    Dmat = [D zeros(N,N); zeros(N,N) conj.(D)]
    UV_mat = [Ubar Vbar; Vbar Ubar]
    Cmat = [C zeros(N,N); zeros(N,N) conj.(C)]

    @assert M ≈ Dmat * UV_mat * Cmat "Bloch-Messiah decomposition failed to reconstruct Bogoliubov transformation M"

    return Dmat, UV_mat, Cmat
end

"""permute_zero_cols_to_end(P::AbstractMatrix)

Return a permutation matrix that shifts zero-valued columns of `P` to the end while preserving the order of the others.
"""
function permute_zero_cols_to_end(P::AbstractMatrix)
    n = size(P,1)
    perm = collect(1:n)
    i = 1
    j = n
    while i < j
        if all(iszero, P[:, perm[i]])
            perm[i], perm[j] = perm[j], perm[i]
            j -= 1
        else
            i += 1
        end
    end
    A = Matrix{eltype(P)}(I, n, n)
    return A[:, perm]
end

"""get_mats_from_bloch_messiah(Dmat, UVmat, Cmat)

Extract the `D`, `Ubar`, `Vbar`, and `C` matrices from the doubled Bloch–Messiah blocks.
"""
function get_mats_from_bloch_messiah(Dmat, UVmat, Cmat)
    N = div(size(UVmat, 1), 2)

    Ubar = UVmat[1:div(size(UVmat, 1), 2), 1:div(size(UVmat, 2), 2)]
    Vbar = UVmat[div(size(UVmat, 1), 2)+1:end, 1:div(size(UVmat, 2), 2)]
    C = Cmat[1:div(size(Cmat, 1), 2), 1:div(size(Cmat, 2), 2)]
    D = Dmat[1:div(size(Dmat, 1), 2), 1:div(size(Dmat, 2), 2)]

    return D,Ubar,Vbar,C
end

"""truncated_bloch_messiah(Dmat,UVmat,Cmat)

Return a truncated decomposition that removes zero columns from the `Vbar` block, keeping compatible block structure.
"""
function truncated_bloch_messiah(Dmat,UVmat,Cmat)
    D,Ubar,Vbar,C = get_mats_from_bloch_messiah(Dmat, UVmat, Cmat)

    # discard zero columns
    zero_ind = findfirst(col -> all(iszero, col), eachcol(Vbar))

    if zero_ind === nothing
        return Dmat, UVmat, Cmat
    end
    
    D_prime = D[:, 1:zero_ind-1]
    Vbar_prime = Vbar[1:zero_ind-1, 1:zero_ind-1]
    Ubar_prime = Ubar[1:zero_ind-1, 1:zero_ind-1]
    C_prime = C[1:zero_ind-1, :]

    Dmat_prime = [D_prime zeros(size(D_prime)); zeros(size(D_prime)) conj.(D_prime)]
    UVmat_prime = [Ubar_prime Vbar_prime; Vbar_prime Ubar_prime]
    Cmat_prime = [C_prime zeros(size(C_prime)); zeros(size(C_prime)) conj.(C_prime)]

    return Dmat_prime, UVmat_prime, Cmat_prime
end