"""
    helper(k::Real, lattice::Union{AbstractLattice, AbstractInfiniteLattice})

Helper function to construct G_in_single_k(k::AbstractVector{<:Real}, Λ::Integer) where k is either kx or ky.

Example (Trivial unit cell):
helper(k, lattice) = [  0   -e^{i k} * σ_x
                    e^{-i k} * σ_x   0  ]

"""
function helper(k::Real, Nrc::Int)
    σ_x = [0 1; 1 0]
    σ_xN = kron(I(Nrc), σ_x)

    return [zeros(2*Nrc,2*Nrc) -cis(k)*σ_xN;
            conj(cis(k))*σ_xN zeros(2*Nrc,2*Nrc)] # Hackenbroich 2010 (lrud)
end

"""
    build_J(Λ::Int, Nf::Int, lattice::Union{AbstractLattice, AbstractInfiniteLattice})

Construct the symplectic matrix J for `N = get_number_of_modes()` modes.
Example: For a trivial unit cell: N = Nf + 4Λ

"""
function build_J(Λ::Int, Nf::Int)
    return ⊕([0.0 1.0; -1.0 0.0], Nf + 4Λ)
end
Zygote.@nograd build_J # constructing J is not something we need gradients through

"""
    Γ_fiducial(X::AbstractMatrix, Λ::Int, Nf::Int, lattice::Union{AbstractLattice, AbstractInfiniteLattice})

Construct the covariance matrix for the fiducial state Q from the orthogonal matrix X in the Majorana representation.
We choose Γ to be qq-ordered.

Γ_fiducial = [A B; -B' D]

Where A ∈ ℝ^(2Nf x 2Nf), B ∈ ℝ^(2Nf x 8Λ), D ∈ ℝ^(8Λ x 8Λ) for a trivial unit cell for example.
A and D are antisymmetric.

The modes of the A block are qq-ordered as: (c_1, c_2, ..., c_(2Nf))
The modes of the D block have the same ordering (lrud) as G_in_single_k, i.e., (c_l1^1, c_l1^2, c_r1^1, c_r1^2, ..., c_lΛ^1, c_lΛ^2, c_rΛ^1, c_rΛ^2, c_u1^1, c_u1^2, c_d1^1, c_d1^2, ..., c_uΛ^1, c_uΛ^2, c_dΛ^1, c_dΛ^2) 
The modes of the B block are ordered as above.

Note:
- X must be an orthogonal matrix: X * X' = I 
- Γ_fiducial is either given in Fourier space or real space, depending on X
"""
function Γ_fiducial(X::AbstractMatrix, Nf::Int, Λ::Int)
    Γ = transpose(X) * build_J(Λ, Nf) * X

    return (Γ - transpose(Γ)) / 2 # ensure exact antisymmetry
end

"""
    get_Γ_blocks(X_vec, Nf, Λ)

Assemble the block-diagonal `A`, `B`, `D` blocks from a collection of per-site
orthogonal matrices `X_vec` (one per unit-cell site).

The physical modes of all sites are concatenated first, followed by the virtual
modes of all sites (same ordering convention as `G_in_single_k_persite`).

# Returns
- `A_total  ::Matrix{Float64}`    (2 Nf Nsites) x (2 Nf Nsites),  block-diagonal
- `B_total  ::Matrix{Float64}`    (2 Nf Nsites) x (8Λ Nsites),    block-diagonal
- `D_total  ::Matrix{Float64}`    (8Λ Nsites)   x (8Λ Nsites),    block-diagonal
"""
function get_Γ_blocks(X_vec::AbstractArray{<:AbstractMatrix}, Nf::Int, Λ::Int, lattice::AbstractInfiniteLattice)
    # Extend X_vec to size: lattice.Lx * lattice.Ly, by repeating the X_vec entries according to lattice.uc_layout
    X_vec = vec([X_vec[lattice.uc_layout[r, c]] for c in 1:lattice.Lx, r in 1:lattice.Ly])
    # Compute per-site Γ matrices, then extract A/B/D blocks separately.
    # Using separate map calls avoids Zygote's Tangent-vs-Tuple issue that
    # arises when map returns tuples.
    Γs = map(X -> Γ_fiducial(X, Nf, Λ), X_vec)
    As = map(Γ -> Γ[1:2Nf, 1:2Nf], Γs)
    Bs = map(Γ -> Γ[1:2Nf, 2Nf+1:end], Γs)
    Ds = map(Γ -> Γ[2Nf+1:end, 2Nf+1:end], Γs)

    # Build block-diagonal matrices with fewer intermediate arrays than nested
    # hcat/vcat constructions.
    A_total = Matrix(BlockDiagonal(As))
    B_total = Matrix(BlockDiagonal(Bs))
    D_total = Matrix(BlockDiagonal(Ds))

    return A_total, B_total, D_total
end

"""
    G_in_single_k(k::AbstractVector{<:Real}, Λ::Integer, lattice::AbstractInfiniteLattice)

Returns the Fourier transformed covariance matrix fof the virtual bonds for one k-value:
```
    G_in_single_k(k, Λ, lattice) = [⊕_{i=1}^{Λ} G_in_single_k(kx)] ⊕ [⊕_{i=1}^{Λ} G_in_single_k(ky)]
```

Each site `(ix, iy)` in the `Lx × Ly` unit cell carries `4Λ` virtual fermions
(= `8Λ` Majorana modes), ordered as (lrud) within each site:

    [l₁, l₁', r₁, r₁', …, lΛ, lΛ', rΛ, rΛ',
     u₁, u₁', d₁, d₁', …, uΛ, uΛ', dΛ, dΛ']

Sites are linearly indexed in column-major order: `s = ix + (iy-1)*Lx`.
The resulting matrix has size `(8Λ Lx Ly) × (8Λ Lx Ly)`.

Bond contraction direction (left ← right, down ← up) use:
- Phase `e^{ikx}` / `e^{iky}` for inter-unit-cell bonds (wrapping around the unit cell boundary).
- Phase `1` for intra-unit-cell bonds.
"""
function G_in_single_k(k::AbstractVector{<:Real}, Λ::Integer, lattice::AbstractInfiniteLattice)
    σ_x = [0 1; 1 0]
    Lx, Ly = lattice.Lx, lattice.Ly
    Λ_per_site = 8Λ

    G = zeros(ComplexF64, Λ_per_site * Lx * Ly, Λ_per_site * Lx * Ly)

    for iy in 1:Ly, ix in 1:Lx
        s = ix + (iy - 1) * Lx   # column-major linear index of current site

        # --- x-direction bond: right(ix,iy) ← left(ix+1,iy) ---
        s_next_x = mod1(ix + 1, Lx) + (iy - 1) * Lx # column-major linear index of next site in x direction
        phase_x  = (ix == Lx) ? k[1] * Lx : 0.0          # only phase for inter-unit-cell (ix == Lx)

        for α in 1:Λ
            # right modes of source  (within LR block, bond α)
            r0 = Λ_per_site * (s - 1) + 4(α - 1) + 3
            # left modes of target   (within LR block, bond α)
            l0 = Λ_per_site * (s_next_x - 1) + 4(α - 1) + 1

            # fill in the 2×2 block for this bond
            for a in 0:1, b in 0:1
                v = σ_x[a + 1, b + 1]
                G[l0 + a, r0 + b] = -cis(phase_x)  * v      # G[left, right]  = -e^{iφ} σ_x
                G[r0 + b, l0 + a] = conj(cis(phase_x)) * v  # G[right, left]  =  e^{-iφ} σ_x
            end
        end

        # --- y-direction bond: down(ix,iy) → up(ix,iy+1) ---
        s_next_y = ix + (mod1(iy + 1, Ly) - 1) * Lx # column-major linear index of next site in y direction
        phase_y  = (iy == Ly) ? k[2] * Ly : 0.0          # only phase for inter-unit-cell (iy == Ly)

        for α in 1:Λ
            # down modes of source  (within UD block, bond α)
            d0 = Λ_per_site * (s - 1) + 4Λ + 4(α - 1) + 3
            # up modes of target    (within UD block, bond α)
            u0 = Λ_per_site * (s_next_y - 1) + 4Λ + 4(α - 1) + 1

            for a in 0:1, b in 0:1
                v = σ_x[a + 1, b + 1]
                G[u0 + a, d0 + b] = -cis(phase_y)  * v
                G[d0 + b, u0 + a] = conj(cis(phase_y)) * v
            end
        end
    end

    return G
end

"""
    G_in_Fourier(Λ::Int, lattice::AbstractInfiniteLattice)

Returns the Fourier transformed covariance matrix of the virtual bonds for all k-points in the Brillouin zone, using `G_in_single_k` for each k.
It has dimensions `(Nk × d × d)` where Nk is the number of k-points and `d = 8Λ Lx Ly` is the number of virtual Majorana modes in the unit cell.

"""
function G_in_Fourier(Λ::Int, lattice::AbstractInfiniteLattice)
    kvals  = lattice.kvals
    n_virt = 8Λ * lattice.Lx * lattice.Ly

    res = Array{ComplexF64, 3}(undef, size(kvals, 2), n_virt, n_virt)
    for (i, k) in enumerate(eachcol(kvals))
        res[i, :, :] = G_in_single_k(k, Λ, lattice)
    end
    return res
end
#=
    Blocked Gaussian map: smart mode ordering (inner / boundary) for larger unit cells.

    The virtual modes of the unit cell are split into
    - *inner* modes: modes on intra-unit-cell bonds. Their bond CM entries carry no
      k-dependent phase (the phase only appears on bonds wrapping around the unit cell).
    - *boundary* modes: l-modes of the first column, r-modes of the last column,
      u-modes of the first row and d-modes of the last row. All wrap bonds (and hence
      all k-dependence) live entirely inside this block.

    Contracting the inner modes is therefore a k-independent Schur complement that is
    computed once per loss evaluation and yields effective (A_eff, B_eff, D_eff) blocks
    of the unit-cell fiducial state Γ_uc (cf. Hackenbroich et al., PRB 101, 115134).
    The per-k inversion in `GaussianMap` then only runs over the 4Λ(Lx+Ly) boundary
    modes instead of all 8Λ·Lx·Ly virtual modes.
=#

"""
    virtual_mode_partition(Λ::Int, lattice::AbstractInfiniteLattice)

Return `(inner, boundary)`: the global Majorana indices of the inner (intra-cell bond) and
boundary (wrap bond) virtual modes, in the same global ordering as `G_in_single_k`.
For a 1x1 unit cell every mode is a boundary mode and `inner` is empty.
"""
function virtual_mode_partition(Λ::Int, lattice::AbstractInfiniteLattice)
    Lx, Ly = lattice.Lx, lattice.Ly
    Λ_per_site = 8Λ
    boundary = Int[]
    for iy in 1:Ly, ix in 1:Lx
        base = Λ_per_site * (get_site_index(ix, iy, lattice) - 1)
        for α in 1:Λ
            ix == 1  && append!(boundary, base .+ (4(α-1)+1 : 4(α-1)+2))       # l modes
            ix == Lx && append!(boundary, base .+ (4(α-1)+3 : 4(α-1)+4))       # r modes
            iy == 1  && append!(boundary, base .+ 4Λ .+ (4(α-1)+1 : 4(α-1)+2)) # u modes
            iy == Ly && append!(boundary, base .+ 4Λ .+ (4(α-1)+3 : 4(α-1)+4)) # d modes
        end
    end
    sort!(boundary)
    inner = setdiff(1:Λ_per_site*Lx*Ly, boundary)
    return inner, boundary
end
Zygote.@nograd virtual_mode_partition

"""
    gaussian_map_inputs(Λ::Int, lattice::AbstractInfiniteLattice)

Precompute all k-independent inputs of the blocked Gaussian map:

- `inner`, `boundary`: virtual mode partition (see `virtual_mode_partition`),
- `G_intra`: real CM of the intra-cell bonds (k-independent, zero on the boundary block),
- `G_wrap`: batched CM of the wrap bonds, `(Nk × d_b × d_b)` with `d_b = length(boundary)`.

For a 1x1 unit cell `inner` is empty, `G_intra` is zero and `G_wrap` equals the full
`G_in_Fourier`, so the blocked map reduces exactly to the previous implementation.
"""
function gaussian_map_inputs(Λ::Int, lattice::AbstractInfiniteLattice)
    inner, boundary = virtual_mode_partition(Λ, lattice)

    # intra-cell bonds carry phase 1 (they never wrap), so G at any k restricted to
    # non-boundary couplings is the k-independent intra-cell CM
    G_intra = real(G_in_single_k(zeros(2), Λ, lattice))
    G_intra[boundary, boundary] .= 0.0 # remove wrap couplings (they live entirely in the boundary block)

    # batched wrap-bond CM: all k-dependence of G_in lives in the boundary block
    # ponytail: allocates Nk × d_b² complex; build per-k inside GaussianMap if memory ever matters
    kvals = lattice.kvals
    d_b = length(boundary)
    G_wrap = Array{ComplexF64, 3}(undef, size(kvals, 2), d_b, d_b)
    for (i, k) in enumerate(eachcol(kvals))
        G_wrap[i, :, :] = G_in_single_k(k, Λ, lattice)[boundary, boundary]
    end

    return (; inner, boundary, G_intra, G_wrap)
end
Zygote.@nograd gaussian_map_inputs

"""
    contract_inner_modes(A, B, D, G_intra, inner, boundary)

Contract the intra-unit-cell virtual bonds via the (k-independent) Schur complement over
the inner modes, returning the effective blocks `(A_eff, B_eff, D_eff)` of the unit-cell
fiducial state Γ_uc restricted to physical + boundary modes:

    A_eff = A + B_I M⁻¹ B_Iᵀ,   B_eff = B_∂ - B_I M⁻¹ D_I∂,   D_eff = D_∂∂ + D_I∂ᵀ M⁻¹ D_I∂,

with `M = D_II + G_intra[inner, inner]` (using `D[boundary, inner] = -D[inner, boundary]ᵀ`).
All operations are Zygote-differentiable; this runs once per loss evaluation.
"""
function contract_inner_modes(A::AbstractMatrix, B::AbstractMatrix, D::AbstractMatrix,
                              G_intra::AbstractMatrix, inner::Vector{Int}, boundary::Vector{Int})
    isempty(inner) && return A, B, D

    M_II = D[inner, inner] + G_intra[inner, inner]
    B_I  = B[:, inner]
    D_Ib = D[inner, boundary]

    Y = M_II \ transpose(B_I)   # d_i × n
    W = M_II \ D_Ib             # d_i × d_b

    A_eff = A + B_I * Y
    B_eff = B[:, boundary] - B_I * W
    D_eff = D[boundary, boundary] + transpose(D_Ib) * W

    return A_eff, B_eff, D_eff
end

"""
    GaussianMap(A::AbstractMatrix, B::AbstractMatrix, D::AbstractMatrix, CM_in::AbstractArray)

Returns the Gaussian map: CM_out = B * inv(D + CM_in) * B' + A.
This contracts the virtual bonds and only the physical modes remain.
The computation is parallelized over k-points using threads and uses pre-allocated output to avoid `map`/`stack` overhead.

# Keyword Arguments:
- `A`, `B`, `D` are the blocks of the covariance matrix of the fiducial state
- `CM_in` is the covariance matrix of the virtual bonds (Nk × d × d)

# Returns:
- `CM_out` of shape (n × n × Nk)
"""
function GaussianMap(A::AbstractMatrix, B::AbstractMatrix, D::AbstractMatrix, CM_in::AbstractArray)
    Nk = size(CM_in, 1)
    n = size(A, 1)
    d = size(D, 1)
    T = promote_type(eltype(A), eltype(B), eltype(D), eltype(CM_in))
    Γ_out = Array{T}(undef, n, n, Nk)

    # Pre-convert to promoted element type so all mul!/ldiv! dispatch to BLAS
    B_T  = convert(Matrix{T}, B)
    Bt_T = Matrix{T}(transpose(B))

    # Per-thread scratch buffers (reused across k iterations within each thread)
    nt = Threads.nthreads()
    S_bufs = [Matrix{T}(undef, d, n) for _ in 1:nt]

    # Always dense: with the blocked map (contract_inner_modes) D is a dense Schur
    # complement over the boundary modes only, so sparse factorization never pays off.
    Dmat = convert(Matrix{T}, D)

    Threads.@threads for k in 1:Nk
        @inbounds begin
            tid = Threads.threadid()
            G_k = @view CM_in[k, :, :]

            F = lu(Dmat + Matrix(G_k))

            # ldiv! API is not consistently in-place across factor types,
            # so assign into thread-local dense buffer.
            S = S_bufs[tid]
            S .= F \ Bt_T

            out_k = @view Γ_out[:, :, k]
            mul!(out_k, B_T, S)
            out_k .+= A
        end
    end

    return Γ_out
end

"""
    rrule(::typeof(GaussianMap), A, B, D, CM_in)

Custom reverse-mode differentiation rule for `GaussianMap`.
Stores LU factorizations from the forward pass and reuses them in the backward
pass (adjoint solves via `F' \\ v`), cutting the number of factorizations in half.
Both forward and backward passes are parallelized over k-points.
"""
function rrule(::typeof(GaussianMap), A::AbstractMatrix, B::AbstractMatrix, D::AbstractMatrix, CM_in::AbstractArray)
    Nk = size(CM_in, 1)
    d = size(D, 1)
    n = size(A, 1)
    T = promote_type(eltype(A), eltype(B), eltype(D), eltype(CM_in))

    # Pre-convert to promoted element type so all mul!/ldiv! dispatch to BLAS
    B_T  = convert(Matrix{T}, B)
    Bt_T = Matrix{T}(transpose(B))

    # Forward pass: compute output and store LU factorizations + solutions for reuse in backward pass
    Γ_out = Array{T}(undef, n, n, Nk)

    # Always dense (see GaussianMap)
    Dmat = convert(Matrix{T}, D)

    factors = Vector{LU{T, Matrix{T}, Vector{Int}}}(undef, Nk)
    S_all = Array{T}(undef, d, n, Nk)

    Threads.@threads for k in 1:Nk
        @inbounds begin
            G_k = @view CM_in[k, :, :]

            F_k = lu(Dmat + Matrix(G_k))
            factors[k] = F_k

            S_k = @view S_all[:, :, k]
            S_k .= F_k \ Bt_T

            out_k = @view Γ_out[:, :, k]
            mul!(out_k, B_T, S_k)
            out_k .+= A
        end
    end

    # Projectors to ensure gradients match the input types (real for A, B, D)
    project_A = ProjectTo(A)
    project_B = ProjectTo(B)
    project_D = ProjectTo(D)

    function GaussianMap_pullback(Δ_raw)
        Δ = unthunk(Δ_raw)

        # Thread-local accumulators (zero-initialized) to avoid race conditions
        nt = Threads.nthreads()
        dA_local = [zeros(T, n, n) for _ in 1:nt]
        dB_local = [zeros(T, size(B)...) for _ in 1:nt]
        dD_local = [zeros(T, d, d) for _ in 1:nt]

        # Thread-local scratch buffers for in-place matrix operations
        W_bufs  = [Matrix{T}(undef, d, n) for _ in 1:nt]
        tmp_dd  = [Matrix{T}(undef, d, d) for _ in 1:nt]  # for W_k * S_kᴴ

        dCM = similar(CM_in)

        Threads.@threads for k in 1:Nk
            @inbounds begin
                tid = Threads.threadid()
                Δ_k = @view Δ[:, :, k]
                S_k = @view S_all[:, :, k]

                # ∂L/∂A += Δ_k
                dA_local[tid] .+= Δ_k

                # ∂L/∂B (contribution 1): dB += Δ_k * S_kᴴ  (in-place accumulate)
                mul!(dB_local[tid], Δ_k, adjoint(S_k), one(T), one(T))

                # W_k = M_kᴴ \ (Bᵀ * Δ_k)  (adjoint solve, reusing stored LU)
                mul!(W_bufs[tid], Bt_T, Δ_k)
                W_bufs[tid] .= adjoint(factors[k]) \ W_bufs[tid]

                # ∂L/∂B (contribution 2): W_kᵀ  →  (d,n)ᵀ = (n,d)
                dB_local[tid] .+= transpose(W_bufs[tid])

                # ∂L/∂M_k = -W_k * S_kᴴ  →  (d,n)*(n,d) = (d,d)
                mul!(tmp_dd[tid], W_bufs[tid], adjoint(S_k), -one(T), zero(T))

                # ∂L/∂D += ∂L/∂M_k ,  ∂L/∂G_k = ∂L/∂M_k
                dD_local[tid] .+= tmp_dd[tid]
                @view(dCM[k, :, :]) .= tmp_dd[tid]
            end
        end

        # Reduce thread-local accumulators in-place (no extra allocation)
        for i in 2:nt
            dA_local[1] .+= dA_local[i]
            dB_local[1] .+= dB_local[i]
            dD_local[1] .+= dD_local[i]
        end

        return NoTangent(), project_A(dA_local[1]), project_B(dB_local[1]), project_D(dD_local[1]), dCM
    end

    return Γ_out, GaussianMap_pullback
end