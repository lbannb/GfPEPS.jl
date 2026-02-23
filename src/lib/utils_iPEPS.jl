#= 
    PEPSKit helper functions
=#

"""
    init_ctmrg_env(peps::InfinitePEPS)

Initializes the CTMRG environment for a given infinite PEPS. 
The corner and edge tensors are initialized as identities on the virtual bonds of the ket-bra layer.

# Arguments
- `peps::InfinitePEPS`: The infinite PEPS for which to initialize the CTMRG environment.

# Returns
- `CTMRGEnv`: The initialized CTMRG environment containing the corner and edge tensors in the PEPSKit.jl type.

"""
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

"""
    grow_env(peps, env, χ_0, χ; kwargs...)

Grows the CTMRG environment from an initial bond dimension `χ_0` to a target bond dimension `χ` by iteratively applying the leading boundary algorithm with increasing bond dimensions.

# Keyword Arguments
- `peps`: The infinite PEPS for which to grow the environment.
- `env`: The initial CTMRG environment to start from.
- `χ_0`: The initial bond dimension to start the growth from.
- `χ`: The target bond dimension to grow the environment to.
- `kwargs...`: Additional keyword arguments to pass to the leading boundary algorithm, such as convergence criteria and truncation method.

# Returns
- `env`: The grown CTMRG environment with bond dimension `χ`.
- `info`: Information about the convergence of the leading boundary algorithm at each step of the growth process.

"""
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

