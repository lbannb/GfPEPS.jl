#= Lattices =#
export RectLattice, InfiniteRectLattice
export InfiniteBrickWallLattice

#= export BdG Hamiltonian parmeters =#
export RealSpaceBdGHamiltonian
export MomentumSpaceBdGHamiltonian
export DopingSettings
export default_BCS_hamiltonian, kitaev_BCS_hamiltonian
export get_isotropic_coupling_dict, get_anisotropic_coupling_dict

#= export constructor =#
export Gaussian_fPEPS

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