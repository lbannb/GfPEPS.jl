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

#= export Gaussian map / loss functions =#
export GaussianMap, gaussian_map_inputs, contract_inner_modes, virtual_mode_partition
export CM_out_X, energy_loss_X, doping_loss_X, energy_CM, doping_CM

#= export global variables =#
const root = normpath(joinpath(@__DIR__, ".."))
export root