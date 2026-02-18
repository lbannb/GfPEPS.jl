#= Lattices =#
export RectLattice, InfiniteRectLattice

#= export BCS parmeter =#
export BCS
export DopingSettings

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