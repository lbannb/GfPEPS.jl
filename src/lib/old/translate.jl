"""
    get_parent_hamiltonian(Γ::AbstractMatrix)

Return the parent Hamiltonian in Dirac representation (qp-ordering) of the fiducial state correlation matrix `Γ` in Majorana representation (qq-ordering).
"""
function get_parent_hamiltonian(Γ::AbstractMatrix, Nf::Int, Nv::Int)
    N = div(size(Γ, 1), 2)

    # Transform to Dirac fermions (qq-ordering)
    Ω0 = [1  1; im  -im]
    Ω = kron(I(N), Ω0)
    Γ_fiducial_dirac = 1/4 .* Ω' * Γ * Ω
    #= Now has the following ordering (qq)
        (f_1,f_1†, ..., f_Nf, f_Nf†, l_1, l_1†, r_1, r_1†, ..., l_Nv, l_Nv†, r_Nv, r_Nv†, d_1, d_1†, u_1, u_1†, ..., d_Nv, d_Nv†, u_Nv, u_Nv†)
    =#

    # bring to qp-ordering
    perm = vcat(1:2:(2N), 2:2:(2N))
    Γ_fiducial_dirac = Γ_fiducial_dirac[perm, perm]
    #= Now has the following ordering (qp)
        (f_1, ..., f_Nf, l_1, r_1, ..., l_Nv, r_Nv, d_1, u_1, ..., d_Nv, u_Nv, f_1†, ..., f_Nf†, l_1†, r_1†, ..., l_Nv†, r_Nv†, d_1†, u_1†, ..., d_Nv†, u_Nv†)
    =#

    # group virtual fermions as (l1,...,lNv,r1,...,rNv,d1,...,dNv,u1,...,uNv)
    L = collect(1:2:2Nv)    # l1, l2, ...
    R = collect(2:2:2Nv)    # r1, r2, ...
    D = collect(2Nv+1:2:4Nv)  # d1, d2, ...
    U = collect(2Nv+2:2:4Nv)  # u1, u2, ...
    perm_virtual = vcat(L, R, D, U)
    
    perm_total = vcat(
        1:Nf,                       # physical (already fine)
        Nf .+ perm_virtual,         # reorder virtuals
        (Nf+4Nv) .+ (1:Nf),         # f†
        (2Nf+4Nv) .+ perm_virtual    # reordered virtual†
    )
    Γ_fiducial_dirac = Γ_fiducial_dirac[perm_total, perm_total]

    # now reorder to (f,u,r,d,l)
    L = collect(Nf+1:Nf+Nv)    # l1, l2, ...
    R = collect(Nf+Nv+1:Nf+2Nv)   # r1, r2, ...
    D = collect(Nf+2Nv+1:Nf+3Nv)  # d1, d2, ...
    U = collect(Nf+3Nv+1:Nf+4Nv)  # u1, u2, ...
    perm_virtual = vcat(U, R, D, L)

    perm_reorder = vcat(1:Nf, 
        perm_virtual,
        (Nf+4Nv) .+ (1:Nf), # f†
        (Nf+4Nv) .+ perm_virtual # virtual†
    )
    Γ_fiducial_dirac = Γ_fiducial_dirac[perm_reorder, perm_reorder]

    @assert Γ_fiducial_dirac' ≈ -Γ_fiducial_dirac "Fiducial state CM in Dirac representation must be anti-hermitian"
    @assert Γ_fiducial_dirac*Γ_fiducial_dirac' ≈ I / 4 "Fiducial state CM in Dirac representation must be pure"

    return Hermitian(-2im .* Γ_fiducial_dirac)
end

""" 
    get_empty_peps_tensor(Nf::Int, Nv::Int)

Create an empty fPEPS tensor with the correct dimensions and spaces for given number of physical (Nf) and virtual (Nv) fermions.
"""
function get_empty_fpeps_tensor(Nf::Int, Nv::Int)
    physical_spaces = Vect[fℤ₂](0 => 2^Nf / 2, 1 => 2^Nf / 2)
    V_bonds = Vect[fℤ₂](0 => 2^Nv / 2, 1 => 2^Nv / 2)
    virtual_spaces = V_bonds ⊗ V_bonds ⊗ V_bonds ⊗ V_bonds

    codomain_spaces = reduce(⊗, [physical_spaces, virtual_spaces])
    domain_space = ProductSpace{GradedSpace{FermionParity, Tuple{Int64, Int64}}, 0}()

    T = zeros(ComplexF64, dim(physical_spaces), dim(virtual_spaces))
    T = reshape(T, (2^Nf, 2^Nv, 2^Nv, 2^Nv, 2^Nv))

    return T, codomain_spaces, domain_space
end

"""
    translate(X::AbstractMatrix, Nf::Int, Nv::Int; tol=1e-10)

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
function translate(X::AbstractMatrix, Nf::Int, Nv::Int; tol=1e-10, unitcell = (1,1))
    Γ_fiduc = Γ_fiducial(X, Nv, Nf)

    H = get_parent_hamiltonian(Γ_fiduc, Nf, Nv)
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
    states_v = 0:(2^Nv - 1)

    # Cartesian product; store as tuples
    states = [(f,u,r,d,l) for f in states_f for u in states_v for r in states_v
                                   for d in states_v for l in states_v]

    ind_f_dict = translate_occ_to_TM_dict(Nf)
    ind_v_dict = translate_occ_to_TM_dict(Nv)

    T, codomain_space, domain_space = get_empty_fpeps_tensor(Nf, Nv)

    # get tensor elements with overlap formula from 10.1103/PhysRevB.107.125128
    Threads.@threads for state in states
        f_occ, u_occ, r_occ, d_occ, l_occ = state

        # convert occ to bitstrings
        f = (digits(f_occ, base=2, pad=Nf))
        u = (digits(u_occ, base=2, pad=Nv))
        l = (digits(l_occ, base=2, pad=Nv))
        d = (digits(d_occ, base=2, pad=Nv))
        r = (digits(r_occ, base=2, pad=Nv))

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
    ω = virtual_bond_state(Nv)

    V = fermion_space()
    fuser_virtual = isomorphism(Int, fuse(fill(V, Nv)...), reduce(⊗, fill(V, Nv)))
    # The maximally entangled bond state ω is in the full tensor product basis of the two virtual fermions (Λ flavors).
    # We now transform ω to to the explicit tensor product basis of |l> ⊗ |r> ( or |d> ⊗ |u> ).
    ω = (fuser_virtual ⊗ fuser_virtual) * ω

    @tensor A[-1; -2 -3 -4 -5] := conj(ω[1 -2]) * conj(ω[2 -3]) * fiducial_state[-1 1 2 -4 -5]

    # normalize as projecting the virtual bonds needs normalization afterwards
    return PEPSKit.peps_normalize(InfinitePEPS(A; unitcell = unitcell))
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