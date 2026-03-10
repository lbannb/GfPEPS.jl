#= Lattices =#
export RectLattice, InfiniteRectLattice

#= export BdG Hamiltonian parmeters =#
export RealSpaceBdGHamiltonian
export MomentumSpaceBdGHamiltonian
export DopingSettings
export default_BCS_hamiltonian, kitaev_BCS_hamiltonian
export get_isotropic_coupling_dict, get_anisotropic_coupling_dict

#= export constructor =#
export Gaussian_fPEPS

#= export per-site Gaussian map functions =#
export G_in_Fourier_persite, G_in_single_k_persite
export Γ_fiducial_blocks, Γ_fiducial_blocks_from_packed
export pack_Xs, X_matrix_form
export energy_loss_X_persite, doping_loss_X_persite

#= export utils functions =#
export init_ctmrg_env
export grow_env

#= export global variables =#
const root = normpath(joinpath(@__DIR__, ".."))
const config_path = joinpath(root, "conf")
const test_config_path = joinpath(root, "conf", "test_conf")
export root
export config_path
export test_config_path