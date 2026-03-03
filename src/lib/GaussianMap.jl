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
    G_in_single_k(k::AbstractVector{<:Real}, Λ::Integer, lattice::Union{AbstractLattice, AbstractInfiniteLattice})

Returns the Fourier transformed covariance matrix for one k value.

The ordering of the majorana modes is (lrud) for each virtual fermion, i.e., (c_l1^1, c_l1^2, c_r1^1, c_r1^2, ..., c_lΛ^1, c_lΛ^2, c_rΛ^1, c_rΛ^2, c_u1^1, c_u1^2, c_d1^1, c_d1^2, ..., c_uΛ^1, c_uΛ^2, c_dΛ^1, c_dΛ^2) 

G_in_single_k(k, Λ, lattice) = [⊕_{i=1}^{Λ} G_in_single_k(kx)] ⊕ [⊕_{i=1}^{Λ} G_in_single_k(ky)]

"""
function G_in_single_k(k::AbstractVector{<:Real}, Λ::Integer, lattice::Union{AbstractLattice, AbstractInfiniteLattice})
    return Matrix(BlockDiagonal([⊕(helper(k[1], lattice.Lx), Λ),⊕(helper(k[2], lattice.Ly), Λ)]))
end

"""
    G_in_Fourier(kvals::AbstractMatrix, Λ::Int)

Returns the Fourier transformed (F) covariance matrix of all the virtual bonds: G_in = F Γ_in F†

"""
function G_in_Fourier(Λ::Int, lattice::Union{AbstractLattice, AbstractInfiniteLattice})
    kvals = lattice.kvals
    Λ_in_uc = get_Λ_in_uc(Λ, lattice)

    res = Array{ComplexF64,3}(undef, size(kvals,2), 2*Λ_in_uc, 2*Λ_in_uc)
    for (i, col) in enumerate(eachcol(kvals))
        res[i, :, :] = G_in_single_k(col, Λ, lattice)
    end
    return res
end

"""
    build_J(Λ::Int, Nf::Int, lattice::Union{AbstractLattice, AbstractInfiniteLattice})

Construct the symplectic matrix J for `N = get_number_of_modes()` modes.
Example: For a trivial unit cell: N = Nf + 4Λ

"""
function build_J(Λ::Int, Nf::Int, lattice::Union{AbstractLattice, AbstractInfiniteLattice})
    return ⊕([0.0 1.0; -1.0 0.0], get_number_of_modes(Nf, Λ, lattice))
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
function Γ_fiducial(X::AbstractMatrix, Nf::Int, Λ::Int, lattice::Union{AbstractLattice, AbstractInfiniteLattice})
    Γ = transpose(X) * build_J(Λ, Nf, lattice) * X

    return (Γ - transpose(Γ)) / 2 # ensure exact antisymmetry
end

"""
    get_Γ_blocks(Γ::AbstractMatrix, Nf::Int, lattice::Union{AbstractLattice, AbstractInfiniteLattice})

Helper function to extract the A, B, D blocks from the covariance matrix Γ of the fiducial state.
"""
function get_Γ_blocks(Γ::AbstractMatrix, Nf::Int, lattice::Union{AbstractLattice, AbstractInfiniteLattice})
    NF_in_uc = get_Nf_in_uc(Nf, lattice)

    A = @view Γ[1:2*NF_in_uc, 1:2*NF_in_uc]
    B = @view Γ[1:2*NF_in_uc, 2*NF_in_uc+1:end]
    D = @view Γ[2*NF_in_uc+1:end, 2*NF_in_uc+1:end]
    return A,B,D
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
    M_bufs = [Matrix{T}(undef, d, d) for _ in 1:nt]
    S_bufs = [Matrix{T}(undef, d, n) for _ in 1:nt]

    Threads.@threads for k in 1:Nk
        @inbounds begin
            tid = Threads.threadid()
            G_k = @view CM_in[k, :, :]

            # M = D + G_k  (fused broadcast into thread-local buffer)
            M = M_bufs[tid]
            @. M = D + G_k

            # In-place LU factorization (no copy)
            F = lu!(M)

            # S = F \ Bᵀ  (in-place solve into thread-local buffer)
            ldiv!(S_bufs[tid], F, Bt_T)

            # Γ_out[:,:,k] = B * S + A  (in-place multiply + add)
            out_k = @view Γ_out[:, :, k]
            mul!(out_k, B_T, S_bufs[tid])
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
    factors = Vector{LU{T, Matrix{T}, Vector{Int}}}(undef, Nk)
    S_all = Array{T}(undef, d, n, Nk)

    Threads.@threads for k in 1:Nk
        @inbounds begin
            G_k = @view CM_in[k, :, :]

            # Fresh alloc needed per k: lu! stores reference and we keep factors[k]
            M_k = D .+ G_k
            F_k = lu!(M_k)              # in-place LU (saves copy vs lu())
            factors[k] = F_k

            # S_k = F_k \ Bᵀ  (in-place solve directly into S_all view)
            S_k = @view S_all[:, :, k]
            ldiv!(S_k, F_k, Bt_T)

            # Γ_out[:,:,k] = B * S_k + A  (in-place multiply + add)
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
                ldiv!(adjoint(factors[k]), W_bufs[tid])

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

function GaussianMap_single_k(A::AbstractMatrix, B::AbstractMatrix, D::AbstractMatrix, CM_in::AbstractMatrix)
    return B * ((D + CM_in) \ transpose(B)) .+ A
end
