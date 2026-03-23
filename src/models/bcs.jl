"""
    BCS_spin_hamiltonian(T::Type{<:Number}, lattice::InfiniteSquare, H_BdG::MomentumSpaceBdGHamiltonian)

Returns a BCS spin-1/2 (Nf=2) Hamiltonian as a TensorMap object on the infinite square lattice.
```
    H = -t ∑_{i,v} (c†_{iα} c_{i+v,α} + h.c.) - μ ∑_i c†_{iα} c_{iα}
        + ∑_{i,v} (Δv ϵ_{αβ} c†_{iα} c†_{i+v,β} + h.c.)
```
where v sums over the basis vectors e_x, e_y. 

- s-wave state: Δy = Δx.
- d-wave state: Δy = -Δx.
- (p+ip) state: Δy = i Δx.
"""
function BCS_spin_hamiltonian(T::Type{<:Number}, lattice::InfiniteSquare, H_BdG::MomentumSpaceBdGHamiltonian, uc_layout::AbstractMatrix)
    pspace = hub.hubbard_space(Trivial, Trivial)
    pspaces = fill(pspace, (lattice.Nrows, lattice.Ncols))
    num = hub.e_num(T, Trivial, Trivial)

    ham_terms = []

    # Map each site in the unit cell
    for y in 1:lattice.Nrows, x in 1:lattice.Ncols
        src_pos = CartesianIndex(y, x)
        site_label = uc_layout[y, x]

        # On-site chemical potential term
        push!(ham_terms, (src_pos,) => - (H_BdG.μ[site_label]) * num)

        # Get the valid bonds for site_label
        site_hoppings = get(H_BdG.hopping, site_label, Dict())
        site_pairings = get(H_BdG.pairing, site_label, Dict())

        unique_bonds = Set{Tuple{Int, Int}}() # set only contains unique elements
        for (dx, dy) in keys(site_hoppings)
            if dx > 0 || (dx == 0 && dy > 0)
                push!(unique_bonds, (Int(dx), Int(dy)))
            end
        end
        for (dx, dy) in keys(site_pairings)
            if dx > 0 || (dx == 0 && dy > 0)
                push!(unique_bonds, (Int(dx), Int(dy)))
            end
        end

        for bond in unique_bonds
            dx, dy = bond
            hopping_amplitude = get(site_hoppings, bond, 0.0)
            pairing_amplitude = get(site_pairings, bond, 0.0)

            dst_pos = CartesianIndex(y + dy, x + dx)
            # hopping
            bond_term = hopping_amplitude * hub.e_hopping(T, Trivial, Trivial)
            # pairing
            pairing_term = pairing_amplitude * sqrt(2) * hub.singlet_plus(T, Trivial, Trivial)
            pairing_term += pairing_term' # add hermitian conjugate
            bond_term += pairing_term

            push!(ham_terms, (src_pos, dst_pos) => bond_term)
        end
    end

    return LocalOperator(
        pspaces,
        ham_terms...
    )
end
BCS_spin_hamiltonian(lattice, H_BdG, uc_layout) = BCS_spin_hamiltonian(ComplexF64, lattice, H_BdG, uc_layout)

"""
Check if a 2-site bond is a nearest neighbor x-bond
"""
function _is_xbond(bond)
    return bond[2] - bond[1] == CartesianIndex(0, 1)
end


"""
    doping_peps(peps::InfinitePEPS, env::CTMRGEnv)

The average doping `δ = 1 - (1/N) ∑_i ⟨f†_{iσ} f_{iσ}⟩`
evaluated from the GfPEPS iPEPS tensor.
"""
function doping_peps(peps::InfinitePEPS, env::CTMRGEnv)
    # Get unit cell dimensions
    Nx, Ny = size(peps.A)

    # Initialize total density accumulator
    total_density = 0.0
    
    # Loop over every site in the unit cell
    density_distribution = zeros(Float64, Nx, Ny)
    for r in 1:Nx, c in 1:Ny
        # Construct the operator specifically for site (r, c)
        O = LocalOperator(space.(peps.A, 1), ((r, c),) => hub.e_num(Trivial, Trivial))
        exp_val = real(expectation_value(peps, O, env))

        density_distribution[r, c] = 1 - exp_val
        # Accumulate the expectation value
        total_density += exp_val
    end

    # Average density = Sum / Number of sites
    avg_density = total_density / (Nx * Ny)

    return 1 - avg_density, density_distribution
end

"""
    doping_pepsGW(peps::InfinitePEPS, env::CTMRGEnv)

The average doping `δ = 1 - (1/N) ∑_i ⟨f†_{iσ} f_{iσ}⟩`
evaluated from the Gutzwiller projected GfPEPS tensor.
"""
function doping_pepsGW(peps::InfinitePEPS, env::CTMRGEnv)
    V = Vect[FermionParity](0 => 1, 1 => 2)

    # Number operator in Gutzwiller projected space
    function e_num_GW(V)
        t = zeros(ComplexF64, V ← V)
        I = sectortype(t)
        t[(I(1), I(1))][1, 1] = 1
        t[(I(1), I(1))][2, 2] = 1
        return t
    end

    # Get unit cell dimensions
    Nx, Ny = size(peps.A)

    # Initialize total density accumulator
    total_density = 0.0
    
    # Loop over every site in the unit cell
    density_distribution = zeros(Float64, Nx, Ny)
    for r in 1:Nx, c in 1:Ny
        # Construct the operator specifically for site (r, c)
        lattice_site_space = space(peps.A[r, c], 1) 
        O = LocalOperator(space.(peps.A, 1), ((r, c),) => e_num_GW(lattice_site_space))
        exp_val = real(expectation_value(peps, O, env))

        density_distribution[r, c] = 1 - exp_val
        # Accumulate the expectation value
        total_density += exp_val
    end

    # Average density = Sum / Number of sites
    avg_density = total_density / (Nx * Ny)

    return 1 - avg_density, density_distribution
end


""" 
    flip_spin(mat, peps)
Flip the spin on the sites where `mat[r, c] != 0`.
"""
function flip_spins_hubbard(mat, peps)
    @assert size(mat) == size(peps.A)

    for r in 1:size(peps.A, 1), c in 1:size(peps.A, 2)
        if mat[r, c] == 0
            continue
        end

        T = peps.A[r, c]
        P = codomain(T)[1] 
        U = TensorMap(zeros, eltype(T), P, P)
        
        # identity block for even parity sector
        b_even = block(U, FermionParity(0))
        b_even[1,1] = 1.0
        b_even[2,2] = 1.0

        # swap block for odd parity sector
        b_odd = block(U, FermionParity(1))
        b_odd[1,2] = 1.0
        b_odd[2,1] = 1.0

        peps.A[r, c] = U * T

        # S_flip = hub.S_x(Trivial, Trivial)
        # peps.A[r, c] = S_flip * peps.A[r, c]
    end

    return PEPSKit.peps_normalize(peps)
end