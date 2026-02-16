using Revise
using Test
using GfPEPS
using TensorKit
using PEPSKit
using Random 
using LinearAlgebra
using JSON: parsefile
using BenchmarkTools

config = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_larger_uc.json"))

Nf = config["params"]["N_physical_fermions_on_site"]
Nv = config["params"]["N_virtual_fermions_on_bond"]
config["params"]["N_virtual_fermions_on_bond"] = 2
Nv = config["params"]["N_virtual_fermions_on_bond"]
N = (Nf + 4*Nv)

X_opt, optim_energy, E_exact, _ = GfPEPS.get_X_opt(;conf=config);
Γ_fiducial = GfPEPS.Γ_fiducial(X_opt, Nv, Nf)

t = config["hamiltonian"]["t"]
pairing_type = config["hamiltonian"]["pairing_type"]
Δ_0 = config["hamiltonian"]["Δ_0"]
μ = config["hamiltonian"]["μ"]
params = GfPEPS.BCS(
    t,
    μ,
    pairing_type,
    Δ_0,
)

Lx = config["system_params"]["Lx"]
Ly = config["system_params"]["Ly"]
bz = BrillouinZone2D(Lx, Ly, (:APBC, :PBC))
energy = GfPEPS.energy_CM(Γ_fiducial, bz, Nf, params)

# G_in = GfPEPS.G_in_Fourier(bz, Nv)
# energy = GfPEPS.energy_loss(params, bz)
# energy2 = real(energy(GfPEPS.GaussianMap(GfPEPS.get_Γ_blocks(GfPEPS.Γ_fiducial(X_opt, Nv, Nf), Nf)..., G_in)))

Γ_psi = GfPEPS.GaussianMap(GfPEPS.get_Γ_blocks(Γ_fiducial, Nf)..., GfPEPS.G_in_Fourier(BrillouinZone2D(Lx,Ly,(:APBC,:PBC)), Nv))

E = 0
for (i,k) in enumerate(eachcol(bz.kvals))
    H_BdG_k = GfPEPS.H_BdG_majorana_k(Nf, k, params)
    global E += -0.25 * (tr(H_BdG_k * Γ_psi[i,:,:])) # = -1/4 * Tr(A * Γ) = 1/4 * Tr(Aᵀ * Γ)
end

ξk_batched = map(k -> GfPEPS.ξ(k, params), eachcol(bz.kvals))
E += sum(ξk_batched)
E /= length(eachcol(bz.kvals))

@show optim_energy
@show energy
@show E

E_test_fn = GfPEPS.energy_loss(params, bz.kvals, Nf)
E_test2_fn = GfPEPS.energy_loss(params, bz)

@benchmark E_test_fn(Γ_psi)
@benchmark E_test2_fn(Γ_psi)


@btime E_test_fn(Γ_psi)
@btime E_test2_fn(Γ_psi)

@test E ≈ optim_energy atol=1e-5
# Nf = 2
# Δ_mat = zeros(ComplexF64, Nf, Nf)
# for i in 1:div(Nf, 2)
#     Δ_mat[i, Nf - i + 1] = GfPEPS.Δ([1,0], params)
#     Δ_mat[Nf - i + 1, i] = -GfPEPS.Δ([-1,0], params)
# end
# display(Δ_mat)

# GfPEPS.Δ([1,0], params)