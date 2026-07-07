#=
    Bogoliubov transformation and Bloch-Messiah decomposition.

    Ported from the robust reference implementation in
    QuantumNaturalfPEPS.jl (features branch, src/TrialStates/GaussianState.jl):
    handles degenerate spectra, zero modes and platform-dependent LAPACK
    behaviour, which the previous eigen-based implementation did not.
=#

"""
    get_bogoliubov_blocks(M::AbstractMatrix)

Extract the `U` and `V` blocks from the Bogoliubov transformation matrix `M = [U conj(V); V conj(U)]`.
"""
function get_bogoliubov_blocks(M::AbstractMatrix)
    N = div(size(M, 1), 2)
    U = M[1:N, 1:N]
    V = M[N+1:end, 1:N]
    return U, V
end

"""
    bogoliubov(H::Hermitian; tol=1e-8)

Return the spectrum and canonical transform that diagonalize the fermionic quadratic Hamiltonian `H`.

The Bogoliubov matrix `M = [U conj(V); V conj(U)]` satisfies `M' * H * M == diagm(vcat(E, -E))`.

Degenerate spectra are handled robustly by splitting the modes into two groups:

- **Nonzero modes** (`|E| > zero_tol`). Their `+E` / `-E` eigenspaces are distinct, but a near-degenerate
  `±E` pair at very small `|E|` is mixed by the eigensolver and would break the particle-hole pairing.
  We therefore orthogonalize `[X  C(X)]` through its SVD (polar) factor, which absorbs that mixing as a
  subspace rotation and restores the canonical structure.
- **Exact zero modes** (`|E| <= zero_tol`). Here `+E` and `-E` coincide, so `[X  C(X)]` becomes rank
  deficient and the SVD step is ill-defined. The zero-mode subspace is invariant under particle-hole
  conjugation, so we instead rebuild a particle-hole symmetric (Majorana) basis of it and pair the
  Majoranas into fermions.

This guarantees the canonical (anti)commutation relations by construction, so the resulting `M` is always a
valid Bogoliubov transformation and is well-conditioned for the subsequent Bloch-Messiah decomposition.
"""
function bogoliubov(H::Hermitian; tol=1e-8)
    N = div(size(H, 1), 2)

    # Particle-hole conjugation C: [X_u; X_v] -> [conj(X_v); conj(X_u)]. C is the antiunitary
    # symmetry of the BdG Hamiltonian (C H C⁻¹ = -H): it maps an eigenvector at energy +E to
    # one at -E. Hence M = [X  C(X)], with X the N quasiparticle (creation) vectors built from
    # the non-negative part of the spectrum.
    _ph_conj(X) = vcat(conj.(X[N+1:end, :]), conj.(X[1:N, :]))

    E0, M0 = eigen(H)
    zero_tol = zero_mode_threshold(E0)

    # Split the spectrum into strictly positive modes and exact zero modes (|E| <= zero_tol).
    pos_idx = findall(>(zero_tol), E0)
    pos_idx = pos_idx[sortperm(E0[pos_idx]; rev=true)] # descending energy
    zero_idx = findall(x -> abs(x) <= zero_tol, E0)

    n_zero_pairs = N - length(pos_idx)
    @assert 2 * n_zero_pairs == length(zero_idx) "Zero-mode subspace has odd dimension ($(length(zero_idx))); adjust zero_tol=$zero_tol."

    # Nonzero modes: orthogonalize [X  C(X)] via its SVD (polar) factor. This is full rank away
    # from exact zero modes and robustly repairs near-degenerate ±E pairs that the eigensolver mixed.
    X = M0[:, pos_idx]
    if !isempty(pos_idx)
        F = svd(hcat(X, _ph_conj(X)))
        X = (F.U * F.V')[:, 1:length(pos_idx)]
    end

    if n_zero_pairs > 0
        Dblock = @view H[1:N, N+1:end]
        if maximum(abs, Dblock) < 1e-7 * max(maximum(abs, H), one(real(eltype(H))))
            # Number-conserving Hamiltonian (no pairing): the zero modes are particle-hole *pairs*
            # ([u;0] particle / [0;v] hole), not genuine Majoranas. Treating them as Majoranas would mix
            # particle and hole and give a non-integer ⟨N⟩. Instead we keep them as ordinary single-particle
            # orbitals: take the particle u-vectors (an orthonormal basis of the upper block of the zero
            # subspace = ker(T)) and fill exactly half of the degenerate Fermi-level orbitals — the lower
            # half stay empty (particle modes), the upper half are occupied (their hole conjugates) — so the
            # Bogoliubov vacuum is the definite-N, particle-hole-symmetric half-filled ground state.
            Z = M0[:, zero_idx]
            u_basis = svd(Z[1:N, :]).U[:, 1:n_zero_pairs] # orthonormal particle orbitals
            particles = vcat(u_basis, zeros(eltype(u_basis), N, n_zero_pairs))
            m = n_zero_pairs ÷ 2
            X = hcat(X, particles[:, 1:m], _ph_conj(particles[:, m+1:end]))
        else
            # Exact zero modes: the E = 0 eigenspace is mapped onto itself by C and makes [X  C(X)] rank
            # deficient, so we rebuild a particle-hole symmetric (Majorana) basis and pair them into fermions.
            X = hcat(X, _zero_mode_fermions(M0[:, zero_idx], _ph_conj, n_zero_pairs; tol=zero_tol))
        end
    end

    M = hcat(X, _ph_conj(X))

    U = M[1:N, 1:N]
    V = M[N+1:end, 1:N]

    # E = diag(M' H M) so that E[k] = -E[k+N] exactly.
    E = real.(diag(M' * H * M))

    @assert isapprox(M' * M, I, atol=tol) "Bogoliubov M is not unitary."
    @assert isapprox(U'U + V'V, I, atol=tol) "Bogoliubov blocks violate U'U + V'V = I."
    @assert isapprox(transpose(U) * V + transpose(V) * U, zeros(N, N), atol=tol) "Bogoliubov blocks violate UᵀV + VᵀU = 0."

    return E, M
end

"""
    zero_mode_threshold(E; rel_floor=1e-11, rel_ceiling=1e-4)

Choose, from the spectrum `E` of a BdG Hamiltonian, the energy below which a mode is treated as a zero
mode. A fixed absolute tolerance is not generic and we use the meaningful scale: `‖H‖ ≈ maximum(|E|)`.
"""
function zero_mode_threshold(E::AbstractVector; rel_floor=1e-11, rel_ceiling=1e-4)
    scale = maximum(abs, E)
    scale == 0 && return float(one(scale))          # H ≡ 0: handled by the caller's even-count check
    a = sort(abs.(E))
    k = count(<(rel_ceiling * scale), a)            # number of near-zero candidates
    k == 0 && return rel_floor * scale              # gapped: only machine-zeros are zero modes
    bottom = k < length(a) ? a[k+1] : scale         # bottom of the bulk spectrum
    return sqrt(a[k] * bottom)                      # cut in the middle of the gap (log scale)
end

"""
    _zero_mode_fermions(Z, _ph_conj, n_pairs; tol=1e-7)

Build `n_pairs` fermionic zero-mode creation vectors from the `2*n_pairs` orthonormal zero-energy
eigenvectors stored as columns of `Z`. The zero-mode subspace is invariant under particle-hole
conjugation `C`, so we first construct a particle-hole symmetric (Majorana) basis `ω` satisfying
`C(ω) = ω`, then pair the Majoranas into complex fermions `c† = (γ₁ + i γ₂)/√2`.
"""
function _zero_mode_fermions(Z::AbstractMatrix, _ph_conj, n_pairs::Int; tol=1e-7)
    # Re-orthonormalize the zero-mode eigenvectors: for a degenerate cluster LAPACK may return
    # vectors that span the right subspace but are not mutually orthonormal on all platforms.
    Z = Matrix(qr(Z).Q)
    # Particle-hole operator restricted to the zero-mode subspace (in the Z basis): a vector
    # ω with coordinates w is Majorana (C-real) iff A * conj(w) = w. A is symmetric unitary,
    # so T(w) = A * conj(w) is an antiunitary involution (T² = I).
    A = Z' * _ph_conj(Z)
    A = (A + transpose(A)) / 2 # enforce the exact symmetry expected of a PH involution
    @assert isapprox(A' * A, I, atol=tol) "Particle-hole operator is not unitary on the zero-mode subspace."

    dim = size(A, 1) # = 2 * n_pairs
    # Real representation of T on (Re w, Im w): its +1 eigenspace is the orthonormal Majorana basis.
    Ar, Ai = real.(A), imag.(A)
    Tmat = Symmetric([Ar Ai; Ai -Ar])
    F = eigen(Tmat)
    plus = findall(>(0), F.values) # eigenvalues are ±1; keep the C-real (+1) subspace
    @assert length(plus) == dim "Zero-mode real structure has wrong +1 multiplicity ($(length(plus)) vs $dim)."
    R = F.vectors[:, plus] # (2·dim) × dim, orthonormal real columns

    # Pair a "real-type" Majorana (top block dominant) with an "imaginary-type" one (bottom block
    # dominant) so that c† = (γ₁ + iγ₂)/√2 stays real for a real Hamiltonian, avoiding spurious
    # per-configuration phases in the overlaps.
    top = vec(sum(abs2, @view(R[1:dim, :]); dims=1))
    bot = vec(sum(abs2, @view(R[dim+1:end, :]); dims=1))
    real_cols = [j for j in axes(R, 2) if top[j] >= bot[j]]
    imag_cols = [j for j in axes(R, 2) if top[j] <  bot[j]]
    order = Int[]
    for k in 1:max(length(real_cols), length(imag_cols))
        k <= length(real_cols) && push!(order, real_cols[k])
        k <= length(imag_cols) && push!(order, imag_cols[k])
    end
    R = R[:, order]
    B = R[1:dim, :] .+ im .* R[dim+1:end, :] # orthonormal Majorana modes (columns, in Z basis)

    Zm = Z * B # Majorana zero modes in the original 2N-dimensional space

    X0 = Matrix{ComplexF64}(undef, size(Z, 1), n_pairs)
    for j in 1:n_pairs
        @views X0[:, j] .= (Zm[:, 2j-1] .+ im .* Zm[:, 2j]) ./ sqrt(2)
    end
    return X0
end

"""
    skew_canonical_form(P::AbstractMatrix)

Return a pair `(S, X)` where `X = transpose(S)*P*S` is the canonical form for `P` (See: https://doi.org/10.1007/BF02906230).
"""
function skew_canonical_form(P::AbstractMatrix)
    # Check skew-symmetry
    @assert isapprox(transpose(P), -P; atol=1e-10) "P should be skew-symmetric"

    W = P'P
    @assert ishermitian(W)

    E, Φ = eigen(Hermitian(W); sortby = (x -> -real(x)))
    alphas = sqrt.(abs.(E))
    tol = 1e-7

    # sort indices by magnitude descending to make pairing stable
    idx_sorted = sortperm(alphas, rev = true)
    nonzero_idx = [i for i in idx_sorted if !isapprox(alphas[i], 0.0; atol=tol)]
    zero_idx = [i for i in idx_sorted if isapprox(alphas[i], 0.0; atol=tol)]

    # ensure we have an even number of nonzero modes (otherwise pairing impossible)
    if isodd(length(nonzero_idx))
        error("skew_canonical_form: odd number of nonzero canonical values (check tolerance).")
    end

    # Initialize with zeros to safely allow projections
    S = zeros(eltype(P), size(P))
    pos = 1

    # build paired columns using Gram-Schmidt to safely handle degeneracies
    for i in nonzero_idx
        v = copy(Φ[:, i])

        # Project out all previously established basis vectors in S
        for prev in 1:(pos-1)
            v -= S[:, prev] * (S[:, prev]' * v)
        end

        # If the vector is fully spanned by previous pairs, skip it
        if norm(v) < tol
            continue
        end

        v1 = v / norm(v)
        v2 = (P' * conj(v1)) / alphas[i]

        S[:, pos]   = v1
        S[:, pos+1] = v2
        pos += 2
    end

    # append nullspace vectors safely
    for idx in zero_idx
        v = copy(Φ[:, idx])
        for prev in 1:(pos-1)
            v -= S[:, prev] * (S[:, prev]' * v)
        end
        if norm(v) < tol
            continue
        end
        S[:, pos] = v / norm(v)
        pos += 1
    end

    # enforce orthonormality (more stable with qr)
    Q = qr(S).Q
    S = Matrix(Q)

    # create canonical transformation X and zero small entries
    X = S' * P * conj(S)

    # permutation to have positive elements in the upper-right of each 2x2 block
    perm_mat = canonical_skew_permutation(X)
    X = perm_mat' * X * perm_mat
    S = S * perm_mat

    X[abs.(X) .< tol] .= 0.0

    return S, X
end

"""
    absorb_phases(S::AbstractMatrix, X::AbstractMatrix)

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
        # If already real with nonnegative value, skip
        if !(abs(imag(x)) ≈ 0 && real(x) >= 0)
            φ = angle(x)
            d = exp(1im * φ/2)          # uniform phase for the pair
            @views S2[:, i  ] .*= d
            @views S2[:, i+1] .*= d
            # Block transforms by conj(d)^2; after this, value becomes real ≈ |x|
            X2[i, i+1] = abs(x)
            X2[i+1, i] = -abs(x)
        end
        i += 2
    end
    max_imag = maximum(abs, imag(X2))
    if max_imag > tol_absorb
        @warn "absorb_phases: residual imaginary part in X2 exceeds tolerance" max_imag=max_imag tol=tol_absorb
    end

    return S2, real(X2)
end

"""
    canonical_skew_permutation(P::AbstractMatrix)

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

"""
    bloch_messiah_decomposition(M::AbstractMatrix)

Compute the Bloch–Messiah decomposition of the Bogoliubov transformation `M` and return the
left (`Dmat`), middle (`UV_mat`), and right (`Cmat`) blocks such that `M ≈ Dmat * UV_mat * Cmat`.
"""
function bloch_messiah_decomposition(M::AbstractMatrix)
    N = div(size(M, 1), 2)

    U, V = get_bogoliubov_blocks(M)

    Q = conj.(V) * transpose(V)
    @assert Q' ≈ Q "Q should be Hermitian"
    Q = Hermitian(Q) # enforce exact Hermiticity
    P = conj.(V) * transpose(U)

    @assert isapprox(transpose(P), -P; atol=1e-10) "P should be skew-symmetric"
    P = (P - transpose(P)) / 2 # enforce exact skew-symmetry
    @assert isapprox(Q*P, P*conj.(Q); atol=1e-10) "Q*P != P*conj.(Q)"

    E_Q, B = eigen(Q; sortby = (x -> -real(x)))
    @assert norm(B' * B - I, Inf) < 1e-10
    @assert norm(B * B' - I, Inf) < 1e-10
    P_bar = B'*P*conj.(B)
    @assert isapprox(P_bar, -transpose(P_bar); atol=1e-10) "P_bar should be skew-symmetric"
    P_bar = (P_bar - transpose(P_bar)) / 2 # enforce exact skew-symmetry

    # Bring P_bar to canonical form by block-diagonalizing within degenerate subspaces of Q to
    # avoid mixing. `degeneracy_atol` groups the eigenvalues of Q into degenerate subspaces;
    # 1e-8 sits safely between LAPACK's degeneracy resolution (~1e-10) and physical splittings (≳1e-5).
    degeneracy_atol = 1e-8

    # Partition the already-sorted Q spectrum into disjoint contiguous blocks.
    q_blocks = UnitRange{Int}[]
    i = firstindex(E_Q)
    while i <= lastindex(E_Q)
        j = i
        while j < lastindex(E_Q) && isapprox(E_Q[j + 1], E_Q[j]; atol=degeneracy_atol, rtol=0)
            j += 1
        end
        push!(q_blocks, i:j)
        i = j + 1
    end

    S = zeros(ComplexF64, size(P_bar))
    for idx in q_blocks
        P_sub = P_bar[idx, idx] # Extract the sub-block corresponding to the eigenvalue
        if norm(P_sub, Inf) < 1e-10
            # P carries no useful pairing information in this degenerate block (empty or fully
            # occupied Slater block). The gauge is then completely underdetermined; choose S_sub = I.
            S_sub = Matrix{ComplexF64}(I, length(idx), length(idx))
        else
            # P has structure, use it to fix the gauge.
            S_sub, X_sub = skew_canonical_form(P_sub) # Canonical form for this block
            S_sub, _ = absorb_phases(S_sub, X_sub)  # makes canonical blocks real
            @assert (norm(S_sub' * S_sub - I(length(idx)), Inf) < 1e-10) "Gauge fixing failed in a degenerate block."
        end
        S[idx, idx] = S_sub # Place the canonical transformation in the correct block of S
    end

    @assert norm(S' * S - I, Inf) < 1e-10
    @assert norm(S * S' - I, Inf) < 1e-10

    P_canonical = S' * P_bar * conj.(S)

    A = permute_zero_cols_to_end(P_canonical)
    @assert norm(A' * A - I, Inf) < 1e-10
    @assert norm(A * A' - I, Inf) < 1e-10

    D = B * S * A
    @assert D' * D ≈ I "D should be unitary"

    @assert isapprox(D'*P*conj(D), D'*conj(V)*transpose(U)*conj(D); atol=1e-10)

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
    Ubar = R * Φ                        # now diagonal(Ubar) = abs.(d) ≥ 0 (real)
    C    = Φ' * Q                       # keep the product invariant: (R Φ)(Φ' Q) = R Q

    Vbar = transpose(D) * V * C'

    # Canonicalize fully occupied Slater blocks (|v| = 1, hence u = 0). The RQ factorization
    # above is driven by U, which vanishes on these columns, so it leaves their gauge
    # undetermined and Vbar generically complex there. We identify the occupied columns (zero
    # Ubar column) and bring their Vbar sub-block to a real form via an SVD, absorbing the left
    # and right unitaries into D and C so that M = Dmat * UV_mat * Cmat stays invariant.
    occ = findall(j -> norm(@view Ubar[:, j]) < 1e-9, axes(Ubar, 2))
    if !isempty(occ) && maximum(abs, imag(@view Vbar[:, occ])) > 1e-9
        row_support = findall(i -> norm(@view Vbar[i, occ]) > 1e-9, axes(Vbar, 1))
        @assert length(row_support) == length(occ) "Occupied Slater block is not square; cannot canonicalize."
        Vblk = Vbar[row_support, occ]
        @assert isapprox(Vblk' * Vblk, I, atol=1e-8) "Occupied Slater block is not unitary."

        Fo = svd(Vblk) # Vblk = Fo.U * Diagonal(Fo.S) * Fo.V'
        D[:, row_support] = D[:, row_support] * conj(Fo.U) # rotate Vbar rows by Fo.U'
        C[occ, :] = Fo.V' * C[occ, :]                      # rotate Vbar columns by Fo.V
        Ubar = D' * U * C'
        Vbar = transpose(D) * V * C'
    end

    # Fix phases on identity block of Vbar
    diagV = diag(Vbar)
    id_cols = findall(x -> isapprox(abs(x), 1.0; atol=1e-10), diagV)
    if !isempty(id_cols)
        phase = ones(eltype(Vbar), size(Vbar, 2))
        for j in id_cols
            phase[j] = exp(-1im * angle(diagV[j]))
        end
        Phi = Diagonal(phase)

        #=
            Absorb these phases into Ubar and C to keep the overall transformation invariant:
                Ubar*C = (Ubar*Phi)*(Phi'*C)
                Vbar*C = (Vbar*Phi)*(Phi'*C)
        =#
        Ubar = Ubar * Phi
        Vbar = Vbar * Phi
        C = Phi' * C
    end

    @assert C'C ≈ I
    @assert Q'Q ≈ I

    @assert U ≈ D*Ubar*C "Something went wrong with Bloch-Messiah decomposition for U"
    @assert V ≈ conj.(D)*Vbar*C "Something went wrong with Bloch-Messiah decomposition for V"

    # Remove numerical noise. Ubar and Vbar are real by construction; bound the residual
    # imaginary part with the max-entry norm (a Frobenius-norm isapprox grows with size).
    real_tol = 1e-8
    @assert maximum(abs, imag(Ubar)) < real_tol "Ubar should be real (max imaginary entry $(maximum(abs, imag(Ubar))))"
    Ubar = real(Ubar)
    Ubar[abs.(Ubar) .< 1e-12] .= 0.0
    @assert maximum(abs, imag(Vbar)) < real_tol "Vbar should be real (max imaginary entry $(maximum(abs, imag(Vbar))))"
    Vbar = real(Vbar)
    Vbar[abs.(Vbar) .< 1e-12] .= 0.0

    Dmat = [D zeros(N,N); zeros(N,N) conj.(D)]
    UV_mat = [Ubar Vbar; Vbar Ubar]
    Cmat = [C zeros(N,N); zeros(N,N) conj.(C)]

    @assert isapprox(M, Dmat * UV_mat * Cmat; atol=1e-10) "Bloch-Messiah decomposition failed to reconstruct Bogoliubov transformation M"

    return Dmat, UV_mat, Cmat
end

"""
    permute_zero_cols_to_end(P::AbstractMatrix)

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

"""
    get_mats_from_bloch_messiah(Dmat, UVmat, Cmat)

Extract the `D`, `Ubar`, `Vbar`, and `C` matrices from the doubled Bloch–Messiah blocks.
"""
function get_mats_from_bloch_messiah(Dmat, UVmat, Cmat)
    Ubar = UVmat[1:div(size(UVmat, 1), 2), 1:div(size(UVmat, 2), 2)]
    Vbar = UVmat[div(size(UVmat, 1), 2)+1:end, 1:div(size(UVmat, 2), 2)]
    C = Cmat[1:div(size(Cmat, 1), 2), 1:div(size(Cmat, 2), 2)]
    D = Dmat[1:div(size(Dmat, 1), 2), 1:div(size(Dmat, 2), 2)]

    return D, Ubar, Vbar, C
end

"""
    truncated_bloch_messiah(Dmat, UVmat, Cmat)

Return a truncated decomposition that removes zero columns from the `Vbar` block, keeping compatible block structure.
"""
function truncated_bloch_messiah(Dmat, UVmat, Cmat)
    D, Ubar, Vbar, C = get_mats_from_bloch_messiah(Dmat, UVmat, Cmat)

    # discard numerically zero columns
    tol = 1e-10
    zero_ind = findfirst(col -> maximum(abs.(col)) < tol, eachcol(Vbar))

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
