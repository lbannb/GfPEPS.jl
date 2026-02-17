using Revise
using Test
using GfPEPS
using TensorKit
using PEPSKit
using Random 
using LinearAlgebra
using JSON: parsefile
using BenchmarkTools
using Optim
using LineSearches

Random.seed!(1234)

config = parsefile(joinpath(GfPEPS.test_config_path, "conf_test_BCS_larger_uc.json"))

Nf = config["params"]["N_physical_fermions_on_site"]
Nv = config["params"]["N_virtual_fermions_on_bond"]
config["params"]["N_virtual_fermions_on_bond"] = 4
Nv = config["params"]["N_virtual_fermions_on_bond"]
N = (Nf + 4*Nv)

t = config["hamiltonian"]["t"]
pairing_type = "s_wave"
Δ_0 = config["hamiltonian"]["Δ_0"]
μ = config["hamiltonian"]["μ"]
params = GfPEPS.BCS(
    t,
    μ,
    pairing_type,
    Δ_0,
)

lattice = GfPEPS.InfiniteRectLattice(1,1;N_kx=24, N_ky=24)

dopingSet = GfPEPS.DopingSettings(; δ=0.16, enforce_density=false)
optopt = Optim.Options(; iterations=1000, g_tol=1e-6, f_reltol=1e-8, successive_f_tol = 10, show_trace=true, extended_trace=false, store_trace=true)

X_opt, optim_energy, E_exact, info = GfPEPS.get_X_opt(lattice,Nf,Nv,params; doping_kwargs=dopingSet, optim_options=optopt);

Γ_fiducial = GfPEPS.Γ_fiducial(X_opt, Nv, Nf, lattice)
energy = GfPEPS.energy_CM(Γ_fiducial, Nf, params, lattice)

# loss_fn = GfPEPS.energy_loss(params, lattice.kvals, Nf)
# G_in = GfPEPS.G_in_Fourier(lattice.kvals, Nv)
# G = GfPEPS.GaussianMap(GfPEPS.get_Γ_blocks(Γ_fiducial, Nf)..., G_in)
# energy = loss_fn(G)

@show E_exact
@show optim_energy
@show energy

# doping = GfPEPS.doping_bcs(Γ_fiducial, Nf, lattice)
# doping2 = GfPEPS.doping_bcs2(Γ_fiducial, Nf, lattice)
# @show doping;
# @show doping2;