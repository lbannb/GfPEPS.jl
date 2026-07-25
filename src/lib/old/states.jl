fermion_space() = Vect[fℤ₂](0 => 1, 1 => 1) # single fermion / Z2 graded space

"""
    single_site_operator(T::Type{<:Number} = ComplexF64)

Create the sceleton for single-site fermionic operators
"""
function single_site_operator(T)
    V = fermion_space()
    return zeros(T, V ← V)
end

"""
    two_site_operator(T::Type{<:Number} = ComplexF64)

Create the sceleton for two-site fermionic operators
"""
function two_site_operator(T::Type{<:Number} = ComplexF64)
    V = fermion_space()
    return zeros(T, V ⊗ V ← V ⊗ V)
end

"""
    f_dag_f_dag(T::Type{<:Number} = ComplexF64)

Creates the two-body operator f1† f2†
"""
function f_dag_f_dag(T::Type{<:Number} = ComplexF64)
    op = two_site_operator(T)
    I = sectortype(op)
    # only non-zero matrix element is ⟨1,1| f1† f2† |0,0⟩ = 1
    op[(I(1), I(1), dual(I(0)), dual(I(0)))] .= 1
    return op
end

"""
    vacuum_state(T::Type{<:Number}, n::Int)

Create the vacuum state for `n` spinless fermions
"""
function vacuum_state(T::Type{<:Number}, n::Int)
    vac = zeros(T, fermion_space())
    vac.data[1] = 1.0
    return (n > 1) ? reduce(⊗, fill(vac, n)) : vac
end
vacuum_state(n::Int) = vacuum_state(ComplexF64, n)

"""
    virtual_bond_state(T::Type{<:Number}, Nv::Int)

Construct the maximally entangled state (MES) on virtual bonds
for Nv flavours of virtual fermions `(a1_i, a2_i)` (i = 1, ..., Nv)
```
    |ω⟩ = ∏_{α=1}^Nv 1/sqrt(2) (1 + a1†_α a2†_α) |0⟩

    For horizontal bond: a1†_α=r†_iα, a2†_α=l†_(i+̂x)α
    For vertical bond:   a1†_α=d†_iα, a2†_u=d†_(i+̂y)α
```
"""
function virtual_bond_state(T::Type{<:Number}, Nv::Int)
    ff = f_dag_f_dag(T)
    vac = vacuum_state(T, 2)
    # MES for one pair of (a1_i, a2_i) on the bond
    # the resulting fermion order is (a1_1, a2_1, ..., a1_Λ, a2_Λ)
    ω = (1 / sqrt(2)) * (unit ⊗ unit + ff) * vac
    # ω = (unit ⊗ unit + ff) * vac
    if Nv > 1
        # reorder fermions to (a1_1, ..., a1_Λ, a2_1, ..., a2_Λ)
        ω = reduce(⊗, fill(ω, Nv))
        perm = Tuple(vcat(1:2:(2*Nv), 2:2:(2*Nv)))
        ω = permute(ω, (perm, ()))
    end
    return ω
end
virtual_bond_state(Nv) = virtual_bond_state(ComplexF64, Nv)