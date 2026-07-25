using GfPEPS
using TensorKit
using PEPSKit
using Printf, Random, Statistics

#=
    GfPEPS.jl — d-wave BCS state on a 2x2 unit cell, at fixed hole density.

    The package workflow is always the same four steps:

      1. lattice = InfiniteRectLattice(...)               geometry + momentum grid
      2. H_BdG   = default_BCS_hamiltonian(...)           the quadratic Hamiltonian
      3. Ψ       = Gaussian_fPEPS(Nf, Λ, lattice, H_BdG)  optimise the covariance matrix
      4. Ψ.peps                                           an InfinitePEPS for PEPSKit.jl

    Everything the optimiser needs lives in the covariance matrix, so steps 1-3 never build
    a tensor network. Step 4 translates the result into an iPEPS, after which observables
    are ordinary PEPSKit expectation values — computed here and checked against the
    covariance-matrix values they must reproduce.
=#

Random.seed!(1234)   # the initial X of the covariance-matrix optimisation is random

Nf = 2               # spinful fermions per site (spin up + down)
Λ = 2                # virtual fermions per bond → PEPS bond dimension D = 2^Λ
χ_E = 32             # CTMRG environment bond dimension: accuracy/cost of step 4
N_k = 24             # momentum-space grid for the BdG Hamiltonian
BC = (:PBC, :APBC)   # boundary conditions for the BdG Hamilton

# 2x2 unit cell with two distinct sublattices in a checkerboard pattern. `uc_layout = [1 2; 2 1]`
# labels the sites; sites sharing a label share one optimisation variable X.
lattice = InfiniteRectLattice(2, 2; N_kx = N_k, N_ky = N_k, bc = BC, uc_layout = [1 2; 2 1])

# BCS parameters, different on the two sublattices so the doping comes out non-uniform.
# d-wave pairing means Δ_y = -Δ_x.
t1, t2 = 1.0, -2.0
Δ1_x, Δ1_y = 1.0, -1.0
Δ2_x, Δ2_y = -2.0, 2.0
μ = 1.0

# Couplings are keyed by sublattice label, then by bond vector. The displacements are (2, 0)
# rather than (1, 0) because sites of one sublattice sit two lattice sites apart.
hopping = Dict(
    1 => Dict((2, 0) => t1, (-2, 0) => t1, (0, 2) => t1, (0, -2) => t1), # sublattice 1
    2 => Dict((2, 0) => t2, (-2, 0) => t2, (0, 2) => t2, (0, -2) => t2), # sublattice 2
)
pairing = Dict(
    1 => Dict((2, 0) => Δ1_x, (-2, 0) => Δ1_x, (0, 2) => Δ1_y, (0, -2) => Δ1_y), # sublattice 1
    2 => Dict((2, 0) => Δ2_x, (-2, 0) => Δ2_x, (0, 2) => Δ2_y, (0, -2) => Δ2_y), # sublattice 2
)
H_BdG = default_BCS_hamiltonian(hopping, pairing, μ, lattice)

# Target hole density. With `enforce_density`, the constructor first solves for the chemical
# potential that gives δ, then keeps δ fixed during the optimisation via an augmented
# Lagrangian — so μ above is only a starting value.
δ_target = 0.10
doping_kwargs = DopingSettings(; enforce_density = true, δ = δ_target)

# Construct the Gaussian_fPEPS
Ψ = Gaussian_fPEPS(Nf, Λ, lattice, H_BdG; doping_kwargs = doping_kwargs)

# Energy and site-resolved hole density, straight from the covariance matrix.
# (Use `energy_CM`, not `Ψ.optim_energy`: with `enforce_density` the latter is the minimum
# of the augmented Lagrangian and still carries the density-penalty term.)
E_CM = GfPEPS.energy_CM(Ψ.X_opt, Nf, Λ, H_BdG, lattice)
doping_CM = GfPEPS.get_doping_layout(Nf, Λ, lattice, Ψ.X_opt)

# ── PEPSKit side ──────────────────────────────────────────────────────────────
# Contract the iPEPS with CTMRG. `IdentityInitialization` starts from an identity
# environment, which converges far more reliably than random tensors;
# `truncrank(χ_E)` is what actually sets the environment bond dimension.
# `trunctol` is a truncation scheme that keeps the environment tensors' singular values above a relative tolerance. 
# The two are combined with `&` so the truncation is the intersection of the two criteria.

boundary_alg = (; tol = 1.0e-8, maxiter = 100, alg = :SimultaneousCTMRG, trunc = trunctol(; rtol=1e-4) & truncrank(χ_E))
env0 = initialize_ctmrg_environment(Ψ.peps, IdentityInitialization())
env, info = leading_boundary(env0, Ψ.peps; boundary_alg...)
info.converged || @warn "CTMRG did not converge" info.convergence_error

# The same two observables, now as PEPSKit expectation values. Both must reproduce the
# covariance-matrix values; the residual difference is the CTMRG truncation error, and
# shrinks as χ_E grows.
ham = GfPEPS.BCS_spin_hamiltonian(ComplexF64, InfiniteSquare(lattice.Lx, lattice.Ly), H_BdG, lattice.uc_layout) # there are some Hamiltonians in the /model folder. They must be constructed within the PEPSKit framework
E_PEPS = real(expectation_value(Ψ.peps, ham, env))
δ_PEPS, doping_PEPS = GfPEPS.doping_peps(Ψ.peps, env)

# ── Comparison ────────────────────────────────────────────────────────────────
println()
@printf("%-24s %14s %14s %12s\n", "", "covariance mat", "iPEPS (χ_E=$χ_E)", "difference")
println("─"^68)
@printf("%-24s %14.6f %14.6f %12.2e\n", "energy per unit cell", E_CM, E_PEPS, E_PEPS - E_CM)
@printf("%-24s %14.6f %14.6f %12.2e\n", "hole density δ", mean(doping_CM), δ_PEPS, δ_PEPS - mean(doping_CM))
for I in CartesianIndices(doping_CM)
    y, x = Tuple(I)
    @printf("%-24s %14.6f %14.6f %12.2e\n", "  δ at site ($y,$x)",
        doping_CM[I], doping_PEPS[I], doping_PEPS[I] - doping_CM[I])
end
println("─"^68)
@printf("%-24s %14.6f   (exact BCS ground state)\n", "reference energy", Ψ.exact_energy)
@printf("%-24s %14.6f   (enforced by DopingSettings)\n", "target δ", δ_target)
@info "The energy can be improved by increasing the number of virtual fermions Λ. This is however costly and will take longer than in this example."