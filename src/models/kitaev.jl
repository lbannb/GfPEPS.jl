"""
Kitaev Hamiltonian mapped to Dirac fermions on a square lattice
```
     H =  ∑_{r} [ J_x (c†_{r} + c_{r})*(c†_{r+r_1} - c_{r+r_1})
        + J_y (c†_{r} + c_{r})*(c†_{r+r_2} - c_{r+r_2})
        + J_z (2 c†_{r} c_{r} - 1) ]
```
where r sums over the lattice vectors and `r₁ = (0, 1)` and `r₂ = (1, 0)` correspond to the
nearest-neighbour bonds of the infinite square lattice.
"""
function Kitaev_Hamiltonian(
        T::Type{<:Number}, lattice::InfiniteSquare; gauge_field::String="vortex_free",
        Jx::Real = 1.0, Jy::Real = 1.0, Jz::Real = 1.0
    )
    gauge_field == "vortex_free" || throw(ArgumentError("Only vortex_free gauge field is implemented."))

    pspace = fermion_space()
    spaces = fill(pspace, (lattice.Nrows, lattice.Ncols))

    num = FO.f_num(T)
    id_site = TensorKit.id(T, pspace)
    onsite = Jz * (2 * num - id_site)
    
    pp = FO.f_plus_f_plus(T)
    pm = FO.f_plus_f_min(T)
    mp = FO.f_min_f_plus(T)
    mm = FO.f_min_f_min(T)
    base_link = pp - pm + mp - mm
    op_x = Jx * base_link
    op_y = Jy * base_link

    bonds_x = Tuple{CartesianIndex, CartesianIndex}[]
    bonds_y = Tuple{CartesianIndex, CartesianIndex}[]
    for (a, b) in nearest_neighbours(lattice)
        δ = b - a
        if δ == CartesianIndex(0, 1)
            push!(bonds_x, (a, b))
        elseif δ == CartesianIndex(1, 0)
            push!(bonds_y, (b, a))
        else
            throw(ArgumentError("Unexpected bond displacement $δ for InfiniteSquare lattice."))
        end
    end

    return LocalOperator(
        spaces,
        ((site,) => onsite for site in vertices(lattice))...,
        (bond => op_x for bond in bonds_x)...,
        (bond => op_y for bond in bonds_y)...,
    )
end
Kitaev_Hamiltonian(lattice::InfiniteSquare; gauge_field::String="vortex_free", Jx::Real = 1.0, Jy::Real = 1.0, Jz::Real = 1.0) = Kitaev_Hamiltonian(ComplexF64, lattice; gauge_field=gauge_field, Jx=Jx, Jy=Jy, Jz=Jz)

function kitaev_two_point_correlator(r::Int, peps::InfinitePEPS, env::CTMRGEnv)
    pm = FO.f_plus_f_min(ComplexF64)

    corrh = correlator(
        peps, pm, CartesianIndex(1, 1), CartesianIndex(1, 2):CartesianIndex(1, r), env
    )

    corrv = correlator(
        peps, pm, CartesianIndex(1, 1), CartesianIndex(2, 1):CartesianIndex(r, 1), env
    )

    return corrh,corrv
end