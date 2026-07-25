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
    rand_CM(Nf::Int, Nv::Int; parity::Int = 1)

Generate a random covariance matrix `Γ` of a pure Gaussian state with given parity in Majorana representation.

Returns `Γ`, `X` where `X` is the orthogonal matrix used to construct `Γ`.
"""
function rand_CM(Nf::Int, Nv::Int; parity::Int = 1)
    N = Nf + 4Nv
    info = parity == 1 ? "even" : "odd"

    while true
        X = rand_orth(2N)
        Γ = Γ_fiducial(X, Nv, Nf)

        if pfaffian(Γ) ≈ parity # for pure BCS state, parity = Pf(Γ) = +1 (even) / -1 (odd)
            @info "Created initial covariance matrix with $info parity"
            return Γ, X
        end
    end
end

#= PEPSKit helper =#
function initialize_ctmrg_env_old(peps::InfinitePEPS, χ0::Int, χ::Int; kwargs...)
    Espace = Vect[FermionParity](0 => χ0 ÷ 2, 1 => χ0 ÷ 2) 
    env = CTMRGEnv(rand, ComplexF64, peps, Espace) 

    χ_eff_array = begin
        arr = [χ0]
        while arr[end] < χ
            push!(arr, min(arr[end] * 2, χ))
        end

        arr
    end

    info = nothing
    for χ_eff in χ_eff_array 
        @info "Growing environment to χ_eff = $χ_eff"
        env, info = leading_boundary( 
            env, peps; tol=1e-5, maxiter=500, alg= :simultaneous, trunc = truncdim(χ_eff) 
        ) 
    end

    # do last step with fixed space truncation
    Espace = Vect[FermionParity](0 => χ÷2, 1 => χ÷2) 
    env, info = leading_boundary( 
        env, peps; kwargs..., trunc = truncspace(Espace) 
    )

    return env, info
end

function init_ctmrg_env(peps)
    corner_space = oneunit(space(peps.A[1],1)) # Vect[FermionParity](0 => 1)  

    rows, cols = size(peps.A)
    # store corner and edge tensors as in PEPSKit
    C_type = tensormaptype(spacetype(peps.A[1]), 1, 1, ComplexF64)
    corners = Array{C_type}(undef, 4, rows, cols)
    T_type = tensormaptype(spacetype(peps.A[1]), 3, 1, ComplexF64)
    edges = Array{T_type}(undef, 4, rows, cols)

    #= Init corners as identities =#
    for r in 1:rows, c in 1:cols
        for dir in 1:4
            corners[dir, r, c] = TensorMap([1.0 + 0.0*im], corner_space ← corner_space)
        end
    end

    for i in eachindex(peps.A)
        r, c = Tuple(CartesianIndices(peps.A)[i])

        # get vector spaces V of virtual links
        space_u = domain(peps.A[i])[1]
        space_r = domain(peps.A[i])[2]
        space_d = domain(peps.A[i])[3]
        space_l = domain(peps.A[i])[4]
        
        #= Edge tensors as identities =#
        # We want the edge tensor to be the identity on the virtual bonds of the ket-bra layer.
        # The physical space of the edge tensor is V' ⊗ V (dual of the network bond V ⊗ V').
        # We construct the state |I> in V' ⊗ V corresponding to the identity operator.
        I_u = permute(id(space_u), ((1, 2), ())) # space_u ⊗ space_u' ← One
        I_r = permute(id(space_r), ((1, 2), ()))
        I_d = permute(id(space_d), ((1, 2), ()))
        I_l = permute(id(space_l), ((1, 2), ()))

        I_c = id(corner_space) # corner ← corner

        Tr_u = I_c ⊗ I_u # corner ⊗ space_u' ⊗ space_u ← corner
        Tr_r = I_c ⊗ I_r
        Tr_d = I_c ⊗ I_d
        Tr_l = I_c ⊗ I_l

        # normalize
        edges[1, r, c] = Tr_u / norm(Tr_u)
        edges[2, r, c] = Tr_r / norm(Tr_r)
        edges[3, r, c] = Tr_d / norm(Tr_d)
        edges[4, r, c] = Tr_l / norm(Tr_l)
    end

    return CTMRGEnv(corners, edges)
end

function grow_env(peps, env, χ_0, χ; kwargs...)
    χ_eff_array = begin
        arr = [χ_0]
        while arr[end] < χ
            push!(arr, min(arr[end] * 2, χ))
        end

        arr
    end

    info = nothing
    for χ_eff in χ_eff_array 
        @info "Growing environment to χ_eff = $χ_eff"
        env, info = leading_boundary( 
            env, peps; tol=1e-5, maxiter=500, alg= :simultaneous, trunc = truncdim(χ_eff) 
        ) 
    end

    # do last step with fixed space truncation
    Espace = Vect[FermionParity](0 => χ÷2, 1 => χ÷2) 
    env, info = leading_boundary( 
        env, peps; kwargs..., trunc = truncspace(Espace) 
    )

    return env, info
end