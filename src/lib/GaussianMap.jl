"""
    helper(k::Real)

Helper function to construct G_in_single_k(k::AbstractVector{<:Real}, Λ::Integer) where k is either kx or ky.

helper(k) = [  0   e^{i k} * σ_x
                    -e^{-i k} * σ_x   0  ]

"""
function helper(k::Real)
    σ_x = [0 1; 1 0]

    return [zeros(2,2) -cis(k)*σ_x;
            conj(cis(k))*σ_x zeros(2,2)] # Hackenbroich 2010 (lrud) (lrdu)
end

"""
    G_in_single_k(k::AbstractVector{<:Real}, Λ::Integer)

Returns the Fourier transformed covariance matrix for one k value.

The ordering of the majorana modes is (lrud) for each virtual fermion, i.e., (c_l1^1, c_l1^2, c_r1^1, c_r1^2, ..., c_lΛ^1, c_lΛ^2, c_rΛ^1, c_rΛ^2, c_u1^1, c_u1^2, c_d1^1, c_d1^2, ..., c_uΛ^1, c_uΛ^2, c_dΛ^1, c_dΛ^2) 

G_in_single_k(k, Λ) = [⊕_{i=1}^{Λ} G_in_single_k(kx)] ⊕ [⊕_{i=1}^{Λ} G_in_single_k(ky)]

"""
function G_in_single_k(k::AbstractVector{<:Real}, Λ::Integer)
    return Matrix(BlockDiagonal([⊕(helper(k[1]), Λ),⊕(helper(k[2]), Λ)]))
end

"""
    G_in_Fourier(kvals::AbstractMatrix, Λ::Int)

Returns the Fourier transformed (F) covariance matrix of all the virtual bonds: G_in = F Γ_in F†

"""
function G_in_Fourier(kvals::AbstractMatrix, Λ::Int)
    kvals = kvals

    res = Array{ComplexF64,3}(undef, size(kvals,2), 8*Λ, 8*Λ)
    for (i, col) in enumerate(eachcol(kvals))
        res[i, :, :] = G_in_single_k(col, Λ)
    end
    return res
end

"""
    build_J(Λ::Int)

Construct the symplectic matrix J for 4*Λ+Nf modes.
"""
function build_J(Λ::Int,Nf::Int)
    return ⊕([0.0 1.0; -1.0 0.0], 4*Λ+Nf)
end
Zygote.@nograd build_J # constructing J is not something we need gradients through

"""
    Γ_fiducial(X::AbstractMatrix, Λ::Int, Nf::Int)

Construct the covariance matrix for the fiducial state A from orthogonal matrix X in the Majorana representation.
We choose Γ to be qq-ordered.

Γ_fiducial = [A B; -B' D]

Where A ∈ ℝ^(2Nf x 2Nf), B ∈ ℝ^(2Nf x 8Λ), D ∈ ℝ^(8Λ x 8Λ).
A and D are antisymmetric.

The modes of the A block are qq-ordered as: (c_1, c_2, ..., c_(2Nf))
The modes of the D block have the same ordering (lrud) as G_in_single_k, i.e., (c_l1^1, c_l1^2, c_r1^1, c_r1^2, ..., c_lΛ^1, c_lΛ^2, c_rΛ^1, c_rΛ^2, c_u1^1, c_u1^2, c_d1^1, c_d1^2, ..., c_uΛ^1, c_uΛ^2, c_dΛ^1, c_dΛ^2) 
The modes of the B block are ordered as above.

Note:
- X must be an orthogonal matrix: X * X' = I 
- Γ_fiducial is either given in Fourier space or real space, depending on X
"""
function Γ_fiducial(X::AbstractMatrix, Λ::Int, Nf::Int)
    Γ = transpose(X) * build_J(Λ,Nf) * X

    return (Γ - transpose(Γ)) / 2 # ensure exact antisymmetry
end

function get_Γ_blocks(Γ::AbstractMatrix, Nf::Int)
    A = Γ[1:2*Nf, 1:2*Nf]
    B = Γ[1:2*Nf, 2*Nf+1:end]
    D = Γ[2*Nf+1:end, 2*Nf+1:end]
    return A,B,D
end

"""
    GaussianMap(CM_out::AbstractMatrix, CM_in::AbstractArray, Nf::Int, Λ::Int)

Returns the Gaussian map: CM_out = B * inv(D + CM_in) * B' + A.
This contracts the virtual bonds and only the physical modes remain.

Keyword arguments:
- `CM_out::AbstractMatrix`: The covariance matrix of the fiducial state / the covariance matrix dual to the Gaussian map
- `CM_in::AbstractMatrix`: The covariance matrix of the virtual bonds
- `Nf::Int`: number of physical fermions
- `Λ::Int`: number of virtual fermions per bond

Note:
- CM_out must be a real antisymmetric matrix, i.e., CM_out² = -I
- The covariance matrices are currently only Fourier transformed TODO: also do for real space / no translation inv systems

"""
function GaussianMap(A::AbstractMatrix, B::AbstractMatrix, D::AbstractMatrix, CM_in::AbstractArray)
    Bt = transpose(B)

    # Gaussian map for each (kx,ky)
    # mats = map(s -> B * ((D .- s) \ transpose(B)) .+ A, eachslice(CM_in; dims=1)) # Kraus thesis
    mats = map(s -> B * ((D .+ s) \ Bt) .+ A, eachslice(CM_in; dims=1)) # Hong hao paper
    return permutedims(stack(mats, dims=3), (3,1,2))
end

function GaussianMap_single_k(A::AbstractMatrix, B::AbstractMatrix, D::AbstractMatrix, CM_in::AbstractMatrix)
    return B * ((D + CM_in) \ transpose(B)) .+ A
end
