"""
    BrillouinZone2D(Lx::Int,Ly::Int,bc::Tuple{Symbol,Symbol}; shift_x=0.0, shift_y=0.0) 

    Constructs the 1st Brillouin zone for a 2D lattice of size Lx×Ly with specified boundary conditions.
"""
struct BrillouinZone2D
    Lx::Int
    Ly::Int
    kvals::Matrix{Float64} # col1 = x col2 = y

    function BrillouinZone2D(Lx::Int,Ly::Int,bc::Tuple{Symbol,Symbol}; shift_x=0.0, shift_y=0.0) 
        return new(Lx,Ly, get_2D_k_grid(Lx,Ly; x_periodic=Val(bc[1]), shift_x=shift_x, y_periodic=Val(bc[2]), shift_y=shift_y))
    end
end

"""
    get_kvals(::Val{:PBC}, L)

Returns the allowed momentum values for a 1D chain with periodic boundary conditions (PBC).

# Arguments
- `L`: System size (number of sites)

# Returns
- `Vector{Float64}`: Allowed momentum values 2π*m/L where:
  - If L is even: m ∈ {-(L-2)/2, ..., L/2}
  - If L is odd: m ∈ {-(L-1)/2, ..., (L-1)/2}
"""
function get_kvals(::Val{:PBC},L)
    if iseven(L)
        return [2π*m/L for m in (-(L-2)/2):L/2] 
    else
        return [2π*m/L for m in (-(L-1)/2):(L-1)/2] 
    end
end

"""
    get_kvals(::Val{:APBC}, L)

Returns the allowed momentum values for a 1D chain with anti-periodic boundary conditions (APBC).

# Arguments
- `L`: System size (number of sites)

# Returns
- `Vector{Float64}`: Allowed momentum values (2m-1)π/L where:
  - If L is even: m ∈ {1, ..., L/2}, returns both ±k values
  - If L is odd: m ∈ {1, ..., (L-1)/2}, returns ±k values plus π
"""
function get_kvals(::Val{:APBC},L)
    if iseven(L)
        kvals = [(2*m-1)*π/L for m in 1:L/2] 
		return vcat(-kvals,kvals)
    else
        kvals = [(2*m-1)*π/L for m in 1:(L-1)/2] 
		return vcat(-kvals,kvals,pi)
    end
end

"""
        get_2D_k_grid(Lx, Ly; x_periodic=Val(:APBC), shift_x=0.0, y_periodic=Val(:PBC),  shift_y=0.0)

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
- Matrix of size 2x(Lx*Ly) where:
    - row 1 = kx vals
    - row 2 = ky vals

Notes
- set the offsets, such that zero modes are avoided as those make the optimization of Γ harder.
"""
function get_2D_k_grid(Lx::Int, Ly::Int; 
    x_periodic::Union{Val{:APBC}, Val{:PBC}} = Val(:APBC),
    shift_x::Float64 = pi/2,
    y_periodic::Union{Val{:APBC}, Val{:PBC}} = Val(:PBC),
    shift_y::Float64 = pi/2)

    # TODO: test with correct kvals but first take from paper to compare
    k_vals_x = sort(get_kvals(x_periodic, Lx) .+ shift_x)
    k_vals_y = sort(get_kvals(y_periodic, Ly) .+ shift_y)

    # create meshgrid
    KX = repeat([kx for kx in k_vals_x], Ly)
    KY = collect(Iterators.flatten(map(k_vals_y) do ky
        repeat([ky],Lx)
    end))

    return hcat(KX,KY)'
end