const USE_DENS_DIM = 64 # heuristic cutoff for when to use dense vs sparse matrices in GaussianMap

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

"""
    build_J(Λ::Int, Nf::Int, lattice::Union{AbstractLattice, AbstractInfiniteLattice})

Overload that accepts a lattice argument.  For the per-site ansatz every site
carries the *same* single-site symplectic matrix `J` of size `2(Nf + 4Λ)`.
"""
function build_J(Λ::Int, Nf::Int, ::Union{AbstractLattice, AbstractInfiniteLattice})
    return build_J(Λ, Nf)
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
    Γ_fiducial(X::AbstractMatrix, Nf::Int, Λ::Int, lattice)

Backward-compatible overload that ignores the lattice argument for a single
(coarse-grained) X matrix.
"""
function Γ_fiducial(X::AbstractMatrix, Nf::Int, Λ::Int, ::Union{AbstractLattice, AbstractInfiniteLattice})
    return Γ_fiducial(X, Nf, Λ)
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
    Nsites = get_number_of_sites(lattice)

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

function X_matrix_form(X_vec::AbstractVector{<:AbstractMatrix}, lattice::Union{AbstractLattice, AbstractInfiniteLattice})
    # Extend X_vec to size: lattice.Lx * lattice.Ly, by repeating the X_vec entries according to lattice.uc_layout
    return [X_vec[lattice.uc_layout[r, c]] for c in 1:lattice.Lx, r in 1:lattice.Ly]
end

# """
#     get_Γ_blocks(Γ::AbstractMatrix, Nf::Int)

# Helper function to extract the A, B, D blocks from the covariance matrix Γ of the fiducial state.
# """
# function get_Γ_blocks(Γ::AbstractMatrix, Nf::Int)
#     A = @view Γ[1:2*Nf, 1:2*Nf]
#     B = @view Γ[1:2*Nf, 2*Nf+1:end]
#     D = @view Γ[2*Nf+1:end, 2*Nf+1:end]
#     return A,B,D
# end

# """
#     get_Γ_blocks(Γ::AbstractMatrix, Nf::Int, lattice)

# Extract A, B, D blocks from the covariance matrix of a coarse-grained fiducial
# state with `Nf_in_uc = Nf * Lx * Ly` physical fermions.
# """
# function get_Γ_blocks(Γ::AbstractMatrix, Nf::Int, lattice::Union{AbstractLattice, AbstractInfiniteLattice})
#     Nf_in_uc = get_Nf_in_uc(Nf, lattice)
#     A = @view Γ[1:2*Nf_in_uc, 1:2*Nf_in_uc]
#     B = @view Γ[1:2*Nf_in_uc, 2*Nf_in_uc+1:end]
#     D = @view Γ[2*Nf_in_uc+1:end, 2*Nf_in_uc+1:end]
#     return A, B, D
# end

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
#= old single site implementation =#
# function G_in_single_k(k::AbstractVector{<:Real}, Λ::Integer, lattice::Union{AbstractLattice, AbstractInfiniteLattice})
#     return Matrix(BlockDiagonal([⊕(helper(k[1], lattice.Lx), Λ),⊕(helper(k[2], lattice.Ly), Λ)]))
# end


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
#= old single site implementation =#
# function G_in_Fourier(Λ::Int, lattice::Union{AbstractLattice, AbstractInfiniteLattice})
#     kvals = lattice.kvals
#     Λ_in_uc = get_Λ_in_uc(Λ, lattice)

#     res = Array{ComplexF64,3}(undef, size(kvals,2), 2*Λ_in_uc, 2*Λ_in_uc)
#     for (i, col) in enumerate(eachcol(kvals))
#         res[i, :, :] = G_in_single_k(col, Λ, lattice)
#     end
#     return res
# end

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

    # Choose dense route for small unit cells (e.g. 1x1) to avoid sparse overhead;
    # otherwise use sparse matrices for larger d.
    use_dense = d < USE_DENS_DIM

    Dmat = use_dense ? convert(Matrix{T}, D) : sparse(convert(Matrix{T}, D))

    Threads.@threads for k in 1:Nk
        @inbounds begin
            tid = Threads.threadid()
            G_k = @view CM_in[k, :, :]

            M = use_dense ? Dmat + Matrix(G_k) : Dmat + sparse(G_k)
            F = lu(M)

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

    # Choose dense route for small unit cells (e.g. 1x1) to avoid sparse overhead;
    # otherwise use sparse matrices for larger d.
    use_dense = d < USE_DENS_DIM

    Dmat = use_dense ? convert(Matrix{T}, D) : sparse(convert(Matrix{T}, D))

    factors = Vector{Any}(undef, Nk)
    S_all = Array{T}(undef, d, n, Nk)

    Threads.@threads for k in 1:Nk
        @inbounds begin
            tid = Threads.threadid()
            G_k = @view CM_in[k, :, :]

            M_k = use_dense ? Dmat + Matrix(G_k) : Dmat + sparse(G_k)

            F_k = lu(M_k)
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