"""
    H_BdG_majorana_k(Nf::Int, k::AbstractVector, params::BCS)

Constructs the Hamiltonian matrix `H_BdG_k` in the Majorana basis (qq-ordered) for momentum `k`.

Bogoliubov-de-Gennes Hamiltonian in Nambu basis (Dirac fermions aᵢ) is given by:

    hat{H} =  1/2 α† H_BdG α, with α = (a₁, a₂, ..., a_Nf, a₁†, a₂†, ..., a_Nf†)ᵀ being the Nambu spinor and H_BdG being the Bogoliubov-de-Gennes matrix.

In Majorana basis (Majorana fermions cᵢ), the Hamiltonian becomes:

    hat{H} = i/4 rᵀ Ω H_BdG Ω† r, with r = (c₁, c₃, ..., c_2Nf-1, c₂, c₄, ...,  c_Nf)ᵀ being the vector of Majorana operators (qp-ordered).


Returns i/2 * (Ω H_BdG_k Ω†) for momentum `k` in qq-ordering.
"""
function H_BdG_majorana_k(Nf::Int, k::AbstractVector, params::BCS)
    # 1. Construct H_BdG in Nambu basis (Dirac fermions qp-ordered)
    #= 
        The Nambu spinor for momentum k is α† = (a†ₖ a-ₖ)
    =#
    ξ_mat = Diagonal([ξ(k, params) for i in 1:Nf])

    # flipped diagonal for pairing with our choice of nambu spinor
    Δ_mat = zeros(ComplexF64, Nf, Nf)
    for i in 1:div(Nf, 2)
        Δ_mat[i, Nf - i + 1] = Δ(k, params)
        Δ_mat[Nf - i + 1, i] = -Δ(-k, params)
    end

    H_BdG_sep = [ξ_mat Δ_mat; Δ_mat' -ξ_mat]

    # 2. Permute to qq-ordering: (a₁, a₂, ..., a_Nf, a†₁, a†₂, ..., a†_Nf) -> (a₁, a†₁, a₂, a†₂, ...)
    p = zeros(Int, 2*Nf)
    for i in 1:Nf
        p[2*i - 1] = i          # aᵢ
        p[2*i]     = Nf + i     # a†ᵢ
    end
    H_BdG_k = H_BdG_sep[p, p]

    # 3. Transform to Majorana basis (qq-ordered) with unitary Ω
    Ω0 = [1  1; im  -im]
    Ω = kron(I(Nf), Ω0)

    return im * 0.5 .* Ω * H_BdG_k * Ω'
end

"""
    get_parent_hamiltonian(Γ::AbstractMatrix)

Return the parent Hamiltonian in Dirac representation (qp-ordering) of the fiducial state correlation matrix `Γ` in Majorana representation (qq-ordering).
"""
function get_parent_hamiltonian(Γ::AbstractMatrix, Nf::Int, Λ::Int)
    N = div(size(Γ, 1), 2)

    # Transform to Dirac fermions (qq-ordering)
    Ω0 = [1  1; im  -im]
    Ω = kron(I(N), Ω0)
    Γ_fiducial_dirac = 1/4 .* Ω' * Γ * Ω
    #= Now has the following ordering (qq)
        (f_1,f_1†, ..., f_Nf, f_Nf†, l_1, l_1†, r_1, r_1†, ..., l_Λ, l_Λ†, r_Λ, r_Λ†, d_1, d_1†, u_1, u_1†, ..., d_Λ, d_Λ†, u_Λ, u_Λ†)
    =#

    # bring to qp-ordering
    perm = vcat(1:2:(2N), 2:2:(2N))
    Γ_fiducial_dirac = Γ_fiducial_dirac[perm, perm]
    #= Now has the following ordering (qp)
        (f_1, ..., f_Nf, l_1, r_1, ..., l_Λ, r_Λ, d_1, u_1, ..., d_Λ, u_Λ, f_1†, ..., f_Nf†, l_1†, r_1†, ..., l_Λ†, r_Λ†, d_1†, u_1†, ..., d_Λ†, u_Λ†)
    =#

    # group virtual fermions as (l1,...,lΛ,r1,...,rΛ,d1,...,dΛ,u1,...,uΛ)
    L = collect(1:2:2Λ)    # l1, l2, ...
    R = collect(2:2:2Λ)    # r1, r2, ...
    D = collect(2Λ+1:2:4Λ)  # d1, d2, ...
    U = collect(2Λ+2:2:4Λ)  # u1, u2, ...
    perm_virtual = vcat(L, R, D, U)
    
    perm_total = vcat(
        1:Nf,                       # physical (already fine)
        Nf .+ perm_virtual,         # reorder virtuals
        (Nf+4Λ) .+ (1:Nf),         # f†
        (2Nf+4Λ) .+ perm_virtual    # reordered virtual†
    )
    Γ_fiducial_dirac = Γ_fiducial_dirac[perm_total, perm_total]

    # now reorder to (f,u,r,d,l)
    L = collect(Nf+1:Nf+Λ)    # l1, l2, ...
    R = collect(Nf+Λ+1:Nf+2Λ)   # r1, r2, ...
    D = collect(Nf+2Λ+1:Nf+3Λ)  # d1, d2, ...
    U = collect(Nf+3Λ+1:Nf+4Λ)  # u1, u2, ...
    perm_virtual = vcat(U, R, D, L)

    perm_reorder = vcat(1:Nf, 
        perm_virtual,
        (Nf+4Λ) .+ (1:Nf), # f†
        (Nf+4Λ) .+ perm_virtual # virtual†
    )
    Γ_fiducial_dirac = Γ_fiducial_dirac[perm_reorder, perm_reorder]

    @assert Γ_fiducial_dirac' ≈ -Γ_fiducial_dirac "Fiducial state CM in Dirac representation must be anti-hermitian"
    @assert Γ_fiducial_dirac*Γ_fiducial_dirac' ≈ I / 4 "Fiducial state CM in Dirac representation must be pure"

    return Hermitian(-2im .* Γ_fiducial_dirac)
end

""" 
    get_empty_peps_tensor(Nf::Int, Λ::Int)

Create an empty fPEPS tensor with the correct dimensions and spaces for given number of physical (Nf) and virtual (Λ) fermions.
"""
function get_empty_fpeps_tensor(Nf::Int, Λ::Int)
    physical_spaces = Vect[fℤ₂](0 => 2^Nf / 2, 1 => 2^Nf / 2)
    V_bonds = Vect[fℤ₂](0 => 2^Λ / 2, 1 => 2^Λ / 2)
    virtual_spaces = V_bonds ⊗ V_bonds ⊗ V_bonds ⊗ V_bonds

    codomain_spaces = reduce(⊗, [physical_spaces, virtual_spaces])
    domain_space = ProductSpace{GradedSpace{FermionParity, Tuple{Int64, Int64}}, 0}()

    T = zeros(ComplexF64, dim(physical_spaces), dim(virtual_spaces))
    T = reshape(T, (2^Nf, 2^Λ, 2^Λ, 2^Λ, 2^Λ))

    return T, codomain_spaces, domain_space
end

"""
    translate(X::AbstractMatrix, Nf::Int, Λ::Int, lattice::Union{AbstractInfiniteLattice, AbstractFiniteLattice}; tol=1e-10)

Get PEPS tensor by contracting virtual axes of ⟨ω|F⟩,
where |ω⟩, |F⟩ are the virtual and the fiducial states.
```
            -2
            ↓
            ω
            ↑
            1  -1
            ↑ ↗
    -5  --←-F-→- 2 -→-ω-←- -3
            ↓
            -4
```
Input axis order
```
        5  1                2
        ↑ ↗                 ↑
    2-←-F-→-3   1-←-ω-→-2   ω
        ↓                   ↓
        4                   1
```
"""
function translate(X::AbstractMatrix, Nf::Int, Λ::Int, lattice::Union{AbstractLattice, AbstractInfiniteLattice}; tol=1e-10)
    Γ_fiduc = Γ_fiducial(X, Nf, Λ, lattice)

    H = get_parent_hamiltonian(Γ_fiduc, Nf, Λ)
    _, M = bogoliubov(H)

    # Bloch Messiah decomposition
    Dmat,UVmat,Cmat = bloch_messiah_decomposition(M)
    Dmat_prime,UVmat_prime,Cmat_prime = truncated_bloch_messiah(Dmat, UVmat, Cmat)

    D, Ubar, Vbar, C = get_mats_from_bloch_messiah(Dmat_prime, UVmat_prime, Cmat_prime)

    M_A = size(Vbar, 2)
    parity = mod(size(Vbar, 1), 2)
    v_prod = prod([abs(Vbar[i-1, i]) for i in 2:2:M_A])

    # compute full matrices for overlap
    R_mat_full = D*Vbar # has the same ordering as H
    Q_mat = Ubar*Vbar # has the same ordering as H

    # @assert Q_mat ≈ - transpose(Q_mat)
    Q_mat = (Q_mat - transpose(Q_mat)) / 2 # enforce exact skew-symmetry

    states_f = 0:(2^Nf - 1)
    states_v = 0:(2^Λ - 1)

    # Cartesian product; store as tuples
    states = [(f,u,r,d,l) for f in states_f for u in states_v for r in states_v
                                   for d in states_v for l in states_v]

    ind_f_dict = translate_occ_to_TM_dict(Nf)
    ind_v_dict = translate_occ_to_TM_dict(Λ)

    T, codomain_space, domain_space = get_empty_fpeps_tensor(Nf, Λ)

    # get tensor elements with overlap formula from 10.1103/PhysRevB.107.125128
    Threads.@threads for state in states
        f_occ, u_occ, r_occ, d_occ, l_occ = state

        # convert occ to bitstrings
        f = (digits(f_occ, base=2, pad=Nf))
        u = (digits(u_occ, base=2, pad=Λ))
        l = (digits(l_occ, base=2, pad=Λ))
        d = (digits(d_occ, base=2, pad=Λ))
        r = (digits(r_occ, base=2, pad=Λ))

        # Boolean occupation vector to select rows from R_mat_full (true if occupied)
        occ_bool = vcat(f, u, r, d, l) .== 1
        M_prime = sum(occ_bool)

        parity_f = mod(sum(f), 2)
        parity_v = mod(sum(l) + sum(u) + sum(r) + sum(d), 2)

        if mod(M_prime,2) != parity || parity_f != parity_v # skip if parity doesn't match
            continue
        end

        if M_prime!=0  
            # build R_mat
            R_mat = R_mat_full[occ_bool,:]
            fsign = isodd((M_prime * (M_prime - 1)) ÷ 2) ? -1 : 1 # fermionic sign from reordering
            pf = pfaffian([zeros(M_prime,M_prime) R_mat; -transpose(R_mat) Q_mat])
            T[ind_f_dict[f], ind_v_dict[u], ind_v_dict[r], ind_v_dict[d], ind_v_dict[l]] = fsign * pf / v_prod
        else # all unoccupied
            T[1,1,1,1,1] = pfaffian(Q_mat) / v_prod
        end
    end
    # remove numerical noise for stability
    T[abs.(T) .< tol] .= 0.0

    fiducial_state = TensorMap(T, codomain_space ← domain_space)
    ω = virtual_bond_state(Λ)

    V = fermion_space()
    fuser_virtual = isomorphism(Int, fuse(fill(V, Λ)...), reduce(⊗, fill(V, Λ)))
    # The maximally entangled bond state ω is in the full tensor product basis of the two virtual fermions (Λ flavors).
    # We now transform ω to to the explicit tensor product basis of |l> ⊗ |r> ( or |d> ⊗ |u> ).
    ω = (fuser_virtual ⊗ fuser_virtual) * ω

    @tensor A[-1; -2 -3 -4 -5] := conj(ω[1 -2]) * conj(ω[2 -3]) * fiducial_state[-1 1 2 -4 -5]

    # normalize as projecting the virtual bonds needs normalization afterwards
    return PEPSKit.peps_normalize(InfinitePEPS(A; unitcell = (lattice.Lx, lattice.Ly)))
end

function translate_occ_to_TM_dict(N)
    nstates = 2^N
    even = []
    odd  = []
    for x in 0: nstates-1
        d = (digits(x, base=2, pad=N))
        if isodd(sum(d))
            push!(odd, d)
        else
            push!(even, d)
        end
    end
    mapping = Dict{Vector{Int}, Int}()
    for (i,x) in enumerate((even..., odd...))
        mapping[x] = i
    end
    return mapping
end