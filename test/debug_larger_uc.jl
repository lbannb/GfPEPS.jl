using GfPEPS, Random, LinearAlgebra, Statistics

Random.seed!(1234)

Nf = 2
Λ = 2
t = 1.0
Δ = 1.0

# 1×1 unit cell reference
lattice1x1 = InfiniteRectLattice(1,1; N_kx=6, N_ky=6, bc=(:PBC, :APBC))
hopping1 = get_isotropic_coupling_dict(lattice1x1, [t]; interaction_type=["NN"])
pairing1 = get_isotropic_coupling_dict(lattice1x1, [Δ]; interaction_type=["NN"])
H_BdG1 = default_BCS_hamiltonian(hopping1, pairing1, 0.0, lattice1x1; interaction_type=["NN"])
E_exact_1x1 = GfPEPS.exact_energy(lattice1x1, H_BdG1, Nf)
println("Exact energy (1×1): ", E_exact_1x1)

# 2×2 unit cell
lattice2x2 = InfiniteRectLattice(2,2; N_kx=6, N_ky=6, bc=(:PBC, :APBC))
hopping2 = get_isotropic_coupling_dict(lattice2x2, [t]; interaction_type=["NN"])
pairing2 = get_isotropic_coupling_dict(lattice2x2, [Δ]; interaction_type=["NN"])
H_BdG2 = default_BCS_hamiltonian(hopping2, pairing2, 0.0, lattice2x2; interaction_type=["NN"])
E_exact_2x2 = GfPEPS.exact_energy(lattice2x2, H_BdG2, Nf)
println("Exact energy (2×2): ", E_exact_2x2)

# Compare
println("Difference: ", abs(E_exact_1x1 - E_exact_2x2))

# Also check the commented-out formula (matrix diagonalization approach)
println("\n--- Matrix diagonalization approach for 2×2 ---")
E_matrix = Statistics.mean(map(eachcol(lattice2x2.kvals)) do k
    H_BdG_k_mat = GfPEPS.get_BdG_k_matrix(lattice2x2, Nf, k, H_BdG2)
    ξ_mat = H_BdG_k_mat[1:cld(size(H_BdG_k_mat, 1), 2), 1:cld(size(H_BdG_k_mat, 2), 2)]

    evals = eigvals(Hermitian(H_BdG_k_mat))
    sum_E = sum(e for e in evals if e > 0)
    Ns = GfPEPS.get_Nf_in_uc(Nf, lattice2x2) / Nf

    (0.5 * (real(tr(ξ_mat)) - sum_E) / Ns) + H_BdG2.E_shift(k, H_BdG2.ξ_k(k, H_BdG2.hopping, H_BdG2.μ), H_BdG2.Δ_k(k, H_BdG2.pairing), H_BdG2.μ)
end)
println("Matrix diag energy (2×2): ", E_matrix)

# Also try the optimization energy using the loss function directly
println("\n--- Loss function check ---")
# check E_shift_summed
kvals = lattice2x2.kvals
E_shift_summed = sum(map(eachcol(kvals)) do k
    ξ_k = H_BdG2.ξ_fct(k, H_BdG2.hopping, H_BdG2.μ)
    Δ_k = H_BdG2.Δ_fct(k, H_BdG2.pairing)
    return Nf * 0.5 * ξ_k + H_BdG2.E_shift(k, ξ_k, Δ_k, H_BdG2.μ)
end)
println("E_shift_summed: ", E_shift_summed)
Nsites = GfPEPS.get_number_of_sites(lattice2x2)
println("Number of sites: ", Nsites)

# Run the actual optimization for 2×2
println("\n--- Running optimization for 2×2 ---")
using Optim
Ψ_trial = Gaussian_fPEPS(Nf, Λ, lattice2x2, H_BdG2;
    optim_options = Optim.Options(; iterations=2000, g_tol=1e-8, f_reltol=1e-10, successive_f_tol = 10, show_trace=false, extended_trace=false, store_trace=true)
)
println("Exact energy:  ", Ψ_trial.exact_energy)
println("Optim energy:  ", Ψ_trial.optim_energy)
println("Difference:    ", abs(Ψ_trial.exact_energy - Ψ_trial.optim_energy))

# Also run 1×1 for comparison
println("\n--- Running optimization for 1×1 ---")
Ψ_trial_1x1 = Gaussian_fPEPS(Nf, Λ, lattice1x1, H_BdG1;
    optim_options = Optim.Options(; iterations=2000, g_tol=1e-8, f_reltol=1e-10, successive_f_tol = 10, show_trace=false, extended_trace=false, store_trace=true)
)
println("Exact energy:  ", Ψ_trial_1x1.exact_energy)
println("Optim energy:  ", Ψ_trial_1x1.optim_energy)
println("Difference:    ", abs(Ψ_trial_1x1.exact_energy - Ψ_trial_1x1.optim_energy))

# Cross-check: evaluate the 2×2 loss function with 4 copies of the 1×1 optimal X
println("\n--- Cross-check: 2×2 loss with 1×1 optimal X ---")
X_1x1 = Ψ_trial_1x1.X_opt[1]
Xs_replicated = [copy(X_1x1) for _ in 1:4]
loss_2x2 = GfPEPS.energy_loss_X_persite(lattice2x2, Nf, Λ, H_BdG2)
E_cross = loss_2x2(Xs_replicated)
println("2×2 loss with replicated 1×1 X: ", E_cross)
println("This tells us if the 2×2 ansatz can represent the 1×1 state")
