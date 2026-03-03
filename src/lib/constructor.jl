
"""
    mutable struct Gaussian_fPEPS

A mutable structure representing a Gaussian fermionic Projected Entangled Pair State (fPEPS).
This structure holds all the relevant information about the fermionic Gaussian state including its PEPS representation.

# Fields
- `Nf::Int`: Number of Abrikosov fermions per site.
- `Λ::Int`: Number of virtual fermions per site (Bond dim. = 2^Λ).
- `lattice::Union{AbstractLattice,AbstractInfiniteLattice}`: The lattice geometry (finite or infinite).
- `H_bdg::AbstractBdGHamiltonian`: The BdG Hamiltonian object.
- `doping_kwargs::DopingSettings`: Configuration for doping calculations (e.g., enforcing density constraints).
- `optim_alg_options`: Optimization algorithm options (e.g., LBFGS or BFGS with manifold support).
- `optim_options`: General options for the Optim.jl optimizer (iterations, tolerances, etc.).
- `X_opt::Matrix{Float64}`: The optimized orthogonal matrix X defining the Gaussian map.
- `Γ_fiducial::Matrix{ComplexF64}`: The correlation matrix of the fiducial state in the Majorana representation.
- `peps::InfinitePEPS`: The resulting infinite PEPS tensor (compatible with PEPSKit.jl).
- `exact_energy::Float64`: The exact energy calculated from the BCS solution.
- `optim_energy::Float64`: The energy obtained after the optimization process.
- `optim_info_obj::NamedTuple`: Information about the optimization process (convergence, trace, etc.).

# Constructor
    Gaussian_fPEPS(
        Nf::Int, 
        Λ::Int, 
        lattice::Union{AbstractLattice,AbstractInfiniteLattice},
        H_bdg::AbstractBdGHamiltonian;
        doping_kwargs::DopingSettings=DopingSettings(),
        optim_alg_options::Union{Optim.LBFGS, Optim.BFGS} = Optim.LBFGS(; m=20, manifold = Optim.Stiefel(:CholQR)),
        optim_options::Optim.Options = Optim.Options(; iterations=1000, g_tol=1e-6, f_reltol=1e-8, successive_f_tol = 10, show_trace=false, extended_trace=false, store_trace=true)
    )

Constructs a `Gaussian_fPEPS` object by optimizing the fiducial state parameters `X` to minimize the energy
of the target BdG Hamiltonian or satisfy doping constraints.
"""
mutable struct Gaussian_fPEPS
    Nf::Int # number of physical fermions per site
    Λ::Int # number of virtual fermions per site (bond dimension = 2^Λ)

    lattice::Union{AbstractLattice,AbstractInfiniteLattice} # lattice structure
    H_bdg::AbstractBdGHamiltonian # BdG Hamiltonian object

    doping_kwargs::DopingSettings # settings for doping via Lagrange multiplier
    optim_alg_options::Union{Optim.LBFGS, Optim.BFGS} # options for the optimization algorithm (LBFGS or BFGS)
    optim_options::Optim.Options # options for the Optim.jl optimizer

    X_opt::Matrix{Float64} # optimal orthogonal matrix X
    Γ_fiducial::Matrix{ComplexF64} # fiducial state correlation matrix in Majorana representation
    peps::InfinitePEPS # iPEPS tensor (PEPSKit.jl format) TODO: add finite PEPS support here

    exact_energy::Float64 # exact energy from BCS solution
    optim_energy::Float64 # energy after optimization
    optim_info_obj::NamedTuple # custom info struct that contains: converged = Optim.converged(stage_res) and trace = optim_res.trace
    
    function Gaussian_fPEPS(
        Nf::Int, 
        Λ::Int, 
        lattice::Union{AbstractLattice,AbstractInfiniteLattice},
        H_bdg::AbstractBdGHamiltonian;
        doping_kwargs::DopingSettings=DopingSettings(),
        optim_alg_options::Union{Optim.LBFGS, Optim.BFGS} = Optim.LBFGS(; m=15, manifold = Optim.Stiefel(:CholQR)),
        # optim_alg_options::Union{Optim.LBFGS, Optim.BFGS} = Optim.BFGS(; manifold = Optim.Stiefel(:CholQR)),
        optim_options::Optim.Options = Optim.Options(; iterations=1000, g_tol=1e-6, f_reltol=1e-8, successive_f_tol = 10, show_trace=false, extended_trace=false, store_trace=true))

        X_opt, optim_energy, E_exact, info_obj = get_X_opt(lattice, Nf, Λ, H_bdg; doping_kwargs=doping_kwargs, optim_alg_options=optim_alg_options, optim_options=optim_options)
        
        # update chemical potential if we optimized for doping
        if doping_kwargs.solve_μ_from_δ && doping_kwargs.enforce_density
            H_bdg.μ = solve_for_mu(lattice.kvals, doping_kwargs.δ, H_bdg)
        end

        Γ = Γ_fiducial(X_opt, Nf, Λ, lattice)
        peps = translate(X_opt, Nf, Λ, lattice);

        new(Nf, Λ, lattice, H_bdg, doping_kwargs, optim_alg_options, optim_options, X_opt, Γ, peps, E_exact, optim_energy, info_obj)
    end
end