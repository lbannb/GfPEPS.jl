function tj_model_NNN(
    particle_symmetry::Type{<:Sector},
    spin_symmetry::Type{<:Sector},
    lattice::InfiniteSquare;
    t1=1.0,
    t2=0.5,
    J1=1.0,
    J2=0.5,
    slave_fermion::Bool=false,
)
    hopping =
        TJOperators.e_plusmin(particle_symmetry, spin_symmetry; slave_fermion) +
        TJOperators.e_minplus(particle_symmetry, spin_symmetry; slave_fermion)
    num = TJOperators.e_number(particle_symmetry, spin_symmetry; slave_fermion)
    heis =
        TJOperators.S_exchange(particle_symmetry, spin_symmetry; slave_fermion) -
        (1 / 4) * (num ⊗ num)
        
    spaces = fill(domain(term_AA)[1], (lattice.Nrows, lattice.Ncols))

    h_NN = (-t1) * hopping + J1 * heis
    h_NNN = (-t2) * hopping + J2 * heis

    return LocalOperator(
        spaces,
        (neighbor => h_NN for neighbor in PEPSKit.nearest_neighbours(lattice))...,
        (neighbor => h_NNN for neighbor in PEPSKit.next_nearest_neighbours(lattice))...,
    )
end

function tj_model_with_doping(
        T::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector},
        lattice::InfiniteSquare;
        t = 2.5, J = 1.0, mu = 0.0, slave_fermion::Bool = false, n::Integer =0,
    )
    hopping =
        TJOperators.e_plusmin(particle_symmetry, spin_symmetry; slave_fermion) +
        TJOperators.e_minplus(particle_symmetry, spin_symmetry; slave_fermion)
    num = TJOperators.e_number(particle_symmetry, spin_symmetry; slave_fermion)
    heis =
        TJOperators.S_exchange(particle_symmetry, spin_symmetry; slave_fermion) -
        (1 / 4) * (num ⊗ num)
    pspace = space(num, 1)
    unit = TensorKit.id(pspace)
    h = (-t) * hopping + J * heis - (mu / 4) * (num ⊗ unit + unit ⊗ num)
    if T <: Real
        h = real(h)
    end
    spaces = fill(pspace, (lattice.Nrows, lattice.Ncols))
    H = PEPSKit.nearest_neighbour_hamiltonian(spaces, h)

    if particle_symmetry === Trivial
        iszero(n) || throw(ArgumentError("imposing particle number requires `U₁` symmetry"))
    elseif particle_symmetry === U1Irrep
        full_charge_sector = fℤ₂(0) ⊠ U1Irrep(n)

        H = MPSKit.add_physical_charge(H, fill(full_charge_sector, size(spaces)...))
    else
        throw(ArgumentError("symmetry not implemented"))
    end

    return H
end

"""
    find_optimal_Δ(t::Real, J::Real, δ_target::Real; pairing_type::String="d_wave")

Returns the optimal gap parameter Δ for the t-J model at given hopping amplitude `t`, exchange interaction `J`, and target doping `δ_target`, assuming a specified pairing symmetry.:

# Keyword arguments
- `t::Real`: Hopping amplitude
- `J::Real`: Exchange interaction
- `δ_target::Real`: Target doping level
- `bz::BrillouinZone2D`: Brillouin zone

# Optional arguments
- `pairing_type::String="d_wave"`: Choose from:
    - d_wave
    - s_wave
"""
function find_optimal_Δ(t::Real, J::Real, δ_target::Real, bz::BrillouinZone2D; pairing_type::String="d_wave")
    # Gutzwiller approximation factors
    gt = 2*δ_target/(1+δ_target)
    gs = 4/(1+δ_target)^2

    t = gt * t

    function loss(x)
        Δ_eff = gs*J*x[1]

        μ = solve_for_mu(bz, δ_target, t, pairing_type, Δ_eff; μ_range=(-10.0, 10.0))

        params = BCS(t, μ, pairing_type, Δ_eff)

        return exact_energy(params,bz) + 2*gs*J*x[1]^2
    end

    Δ_min = 1e-10
    Δ_max = 10.0
    # res = Optim.optimize(loss, [Δ_min, Δ_max], Optim.NelderMead(), Optim.Options(
            # iterations = 100,
            # g_tol = 1e-6,
            # show_trace = true,
            # f_reltol = 1e-6))
    res = Optim.optimize(loss, Δ_min, Δ_max, Brent())

    return Optim.minimizer(res)[1]
end

function measure_Sz(peps, env)
    # Get unit cell dimensions
    Nx, Ny = size(peps.A)
    
    magn_distribution = zeros(Float64, Nx, Ny)
    total_magn = 0.0
    for r in 1:Nx, c in 1:Ny
        # Construct the operator specifically for site (r, c)
        O = LocalOperator(space.(peps.A, 1), ((r, c),) => (tJ.u_num(Trivial, Trivial) - tJ.d_num(Trivial, Trivial)))
        exp_val = real(expectation_value(peps, O, env))

        magn_distribution[r, c] = exp_val / 2
        # Accumulate the expectation value
        total_magn += exp_val / 2
    end

    return total_magn, magn_distribution
end

""" 
    flip_spin(mat, peps)
Flip the spin on the sites where `mat[r, c] != 0`.
"""
function flip_spins_tJ(mat, peps)
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

        # swap block for odd parity sector
        b_odd = block(U, FermionParity(1))
        b_odd[1,2] = 1.0
        b_odd[2,1] = 1.0

        peps.A[r, c] = U * T
    end

    return PEPSKit.peps_normalize(peps)
end