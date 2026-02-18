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

Construct the covariance matrix for the fiducial state A from orthogonal matrix X in the Majorana representation.
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

    A = Γ[1:2*NF_in_uc, 1:2*NF_in_uc]
    B = Γ[1:2*NF_in_uc, 2*NF_in_uc+1:end]
    D = Γ[2*NF_in_uc+1:end, 2*NF_in_uc+1:end]
    return A,B,D
end

"""
    GaussianMap(A::AbstractMatrix, B::AbstractMatrix, D::AbstractMatrix, CM_in::AbstractArray)

Returns the Gaussian map: CM_out = B * inv(D + CM_in) * B' + A.
This contracts the virtual bonds and only the physical modes remain.

Keyword arguments:
- `A`, `B`, `D` are the blocks of the covariance matrix of the fiducial state
- `CM_in` is the covariance matrix of the virtual bonds 

"""
# function GaussianMap(A::AbstractMatrix, B::AbstractMatrix, D::AbstractMatrix, CM_in::AbstractArray)
#     Bt = transpose(B)

#     # Gaussian map for each (kx,ky)
#     # mats = map(s -> B * ((D .- s) \ transpose(B)) .+ A, eachslice(CM_in; dims=1)) # Kraus thesis
#     mats = map(s -> B * ((D .+ s) \ Bt) .+ A, eachslice(CM_in; dims=1)) # Hong hao paper
#     return permutedims(stack(mats, dims=3), (3,1,2))
# end
function GaussianMap(A::AbstractMatrix, B::AbstractMatrix, D::AbstractMatrix, CM_in::AbstractArray)
    Bt = transpose(B)

    # Gaussian map for each (kx,ky)
    # mats = map(s -> B * ((D .- s) \ transpose(B)) .+ A, eachslice(CM_in; dims=1)) # Kraus thesis
    mats = map(s -> B * ((D .+ s) \ Bt) .+ A, eachslice(CM_in; dims=1)) # Hong hao paper

    # Stack into a 3D tensor [i, j, k]
    # return stack(mats)
    return permutedims(stack(mats, dims=3), (2,1,3))
end


function GaussianMap_single_k(A::AbstractMatrix, B::AbstractMatrix, D::AbstractMatrix, CM_in::AbstractMatrix)
    return B * ((D + CM_in) \ transpose(B)) .+ A
end
