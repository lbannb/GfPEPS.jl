"""
    get_kvals(::Val{:PBC}, L::Int, N_k::Int)

Returns `N_k` uniformly spaced momentum values for a 1D Brillouin zone of width `2π/L`
with periodic boundary conditions (PBC).

# Arguments
- `L`: Brillouin zone width
- `N_k`: Number of momentum points to generate

# Returns
- `Vector{Float64}`: Allowed momentum values 2π*m/(N_k*L) where:
  - If N_k is even: m ∈ {-(N_k-2)/2, ..., N_k/2}
  - If N_k is odd: m ∈ {-(N_k-1)/2, ..., (N_k-1)/2}
"""
function get_kvals(::Val{:PBC},L::Int,N_k::Int)
    if iseven(N_k)
        return [2π*m/N_k for m in (-(N_k-2)/2):N_k/2] ./ L
    else
        return [2π*m/N_k for m in (-(N_k-1)/2):(N_k-1)/2] ./ L
    end
end

"""
    get_kvals(::Val{:APBC}, L::Int, N_k::Int)

Returns `N_k` uniformly spaced momentum values for a 1D Brillouin zone of width `2π/L`
with anti-periodic boundary conditions (APBC).

# Arguments
- `L`: Brillouin zone width
- `N_k`: Number of momentum points to generate

# Returns
- `Vector{Float64}`: Allowed momentum values (2m-1)π/(N_k*L) where:
  - If N_k is even: m ∈ {1, ..., N_k/2}, returns both ±k values
  - If N_k is odd: m ∈ {1, ..., (N_k-1)/2}, returns ±k values plus π
"""
function get_kvals(::Val{:APBC},L::Int,N_k::Int)
    if iseven(N_k)
        kvals = [(2*m-1)*π/N_k for m in 1:N_k/2] 
		return vcat(-kvals,kvals)./ L
    else
        kvals = [(2*m-1)*π/N_k for m in 1:(N_k-1)/2] 
		return vcat(-kvals,kvals,pi)./ L
    end
end

"""
        get_2D_k_grid(Lx, Ly; x_bc=Val(:APBC), shift_x=0.0, y_bc=Val(:PBC),  shift_y=0.0)

Create the 2D momentum grid from 1D k-values (with optional offsets) and
return a meshgrid of the form: transpose([[kx_1, ky_1]; 
                                [kx_2, ky_1]; 
                                    ... 
                                [kx_Lx, ky_1];
                                [kx_1, ky_2];
                                [kx_2, ky_2];
                                    ...
                                [kx_Lx, ky_2];
                                    ...
                                [kx_Lx, ky_Ly]]);

Returns
- Matrix of size 2x(N_kx*N_ky) where:
    - row 1 = kx vals
    - row 2 = ky vals

Notes
- set the offsets, such that zero modes are avoided as those make the optimization of Γ harder.
"""
function get_2D_k_grid(Lx::Int, Ly::Int, N_kx::Int, N_ky::Int; 
    x_bc::Union{Val{:APBC}, Val{:PBC}} = Val(:APBC),
    shift_x::Float64 = pi/2,
    y_bc::Union{Val{:APBC}, Val{:PBC}} = Val(:PBC),
    shift_y::Float64 = pi/2)

    # TODO: test with correct kvals but first take from paper to compare
    k_vals_x = sort(get_kvals(x_bc, Lx, N_kx) .+ shift_x)
    k_vals_y = sort(get_kvals(y_bc, Ly, N_ky) .+ shift_y)

    # create full Cartesian-product meshgrid
    KX = repeat(collect(k_vals_x), N_ky)
    KY = collect(Iterators.flatten(map(k_vals_y) do ky
        repeat([ky], N_kx)
    end))

    return hcat(KX,KY)'
end