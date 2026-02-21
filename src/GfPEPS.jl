module GfPEPS

#= load external modules =#
using LinearAlgebra
using Statistics
using BlockDiagonals
using Optim
using Zygote
# using JSON: parsefile
# using Random
using TensorOperations
using SkewLinearAlgebra
using MatrixFactorizations
using Roots
# using LineSearches

# using SparseArrays: sparse, blockdiag, spdiagm
using TensorKit
using MPSKit
using PEPSKit
using MPSKitModels.TJOperators
import TensorKitTensors.HubbardOperators as hub
import TensorKitTensors.FermionOperators as FO
import TensorKitTensors.TJOperators as tJ

const V = FO.fermion_space()
const unit = TensorKit.id(V)

#= include local files =#
include("lib/lattices.jl")
include("lib/utils.jl")
include("lib/BdGHamiltonian.jl")
include("lib/GaussianMap.jl")
include("lib/loss.jl")
include("lib/states.jl")
include("lib/bogoliubov.jl")
include("lib/Xopt.jl")
include("lib/translate.jl")

include("models/bcs_spin.jl")
# include("models/kitaev.jl")
# include("models/tj_model.jl")

include("lib/constructor.jl")

include("exports.jl") # export functions

end
