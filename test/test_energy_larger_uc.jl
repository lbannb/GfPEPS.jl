using Revise
using Test
using GfPEPS
using TensorKit
using PEPSKit
using Random 
using LinearAlgebra
using JSON: parsefile

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

Lx = 6
Ly = 6
bz = BrillouinZone2D(Lx, Ly, (:APBC, :PBC))
energy = GfPEPS.energy_CM(Γ_fiducial, bz, Nf, params)

# G_in = GfPEPS.G_in_Fourier(bz, Nv)
# energy = GfPEPS.energy_loss(params, bz)
# energy2 = real(energy(GfPEPS.GaussianMap(GfPEPS.get_Γ_blocks(GfPEPS.Γ_fiducial(X_opt, Nv, Nf), Nf)..., G_in)))

@show optim_energy
@show energy

Γ_psi = GfPEPS.GaussianMap(GfPEPS.get_Γ_blocks(Γ_fiducial, Nf)..., GfPEPS.G_in_Fourier(BrillouinZone2D(Lx,Ly,(:APBC,:PBC)), Nv))

E = 0
for (i,k) in enumerate(eachcol(bz.kvals))
    H_BdG_k = GfPEPS.H_BdG_majorana_k(Nf, k, params)

    E += -0.5 * (tr(H_BdG_k * Γ_psi[i,:,:]))
end

ξk_batched = map(k -> GfPEPS.ξ(k, params), eachcol(bz.kvals))
E += 0.5*sum(ξk_batched)
E /= length(eachcol(bz.kvals))

@show E