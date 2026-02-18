mutable struct Gaussian_fPEPS
    Nf::Int # number of physical fermions
    Nv::Int # number of virtual fermions

    # lattice size 
    Lx::Int # horizontal extent 
    Ly::Int # vertical extent
    bz::BrillouinZone2D # Brillouin zone

    # quadratic Hamiltonian parameters
    t::Float64 # hopping amplitude
    μ::Float64 # chemical potential
    Δ_options::Dict{String, Any} # pairing options

    # optimizer
    maxiter::Int # maximum iterations
    tol::Float64 # gradient tolerance
    
    X_opt::Matrix{Float64} # optimal orthogonal matrix X
    # peps::InfinitePEPS # iPEPS tensor (PEPSKit.jl format)

    exact_energy::Float64 # exact energy
    optim_energy::Float64 # energy after optimization

    function Gaussian_fPEPS(;conf::Dict=parsefile(joinpath(GfPEPS.config_path, "conf_default_BCS.json")))
        Random.seed!(conf["params"]["seed"])
        Nf = conf["params"]["N_physical_fermions_on_site"]
        Nv = conf["params"]["N_virtual_fermions_on_bond"]
        N = Nf + 4*Nv # total number of fermions per site

        Lx = conf["system_params"]["Lx"]
        Ly = conf["system_params"]["Ly"]
        x_bc = Symbol(conf["system_params"]["x_bc"])
        y_bc = Symbol(conf["system_params"]["y_bc"])

        bc = (x_bc, y_bc)
        # lattice_type = Symbol(conf["system_params"]["lattice_type"])

        # hamiltonian params
        t = conf["hamiltonian"]["t"]
        pairing_type = conf["hamiltonian"]["Δ_options"]["pairing_type"]
        Δ_options = conf["hamiltonian"]["Δ_options"]
        μ = conf["hamiltonian"]["μ"]
        δ = conf["hamiltonian"]["hole_density"]

        # build Δ_vec from dict
        Δ_vec = begin
            if pairing_type == "d_wave" || pairing_type == "p_wave" || pairing_type == "default"
                [Δ_options["Δ_x"], Δ_options["Δ_y"]]
            elseif pairing_type == "s_wave"
                [Δ_options["Δ_0"]]
            else
                error("Unknown pairing type: $pairing_type. Supported types are: d_wave, p_wave, s_wave.")
            end
        end

        # construct Brillouin zone
        bz = BrillouinZone2D(Lx,Ly,bc)
        has_dirac_points(bz,t,μ,pairing_type,Δ_vec...) # warn if dirac points are present
        Lx_init = isodd(Lx) ? 5 : 6
        Ly_init = isodd(Ly) ? 5 : 6
        bz_init = BrillouinZone2D(Lx_init,Ly_init,bc)
        has_dirac_points(bz_init,t,μ,pairing_type,Δ_vec...) # warn if dirac points are present

        # initial ortogonal matrix X to construct Γ_out with correct parity sector (even)
        _, X = rand_CM(Nf,Nv)
        @info "Created initial covariance matrix with even parity sector"

        if(conf["hamiltonian"]["μ_from_hole_density"])
            μ = solve_for_mu(bz,δ,t,Δ_x,Δ_y)
        end

        # build loss function
        loss_init = optimize_loss(t, μ, bz_init, Nf, Nv, pairing_type, Δ_vec...)
        loss = optimize_loss(t, μ, bz, Nf, Nv, pairing_type, Δ_vec...)

        # build gradients
        g_init(x) = first(Zygote.gradient(loss_init, x))
        g_init!(G,x) = copyto!(G, g_init(x)) # better for optim
        g(x) = first(Zygote.gradient(loss, x))
        g!(G,x) = copyto!(G, g(x)) # better for optim

        # First, find a better initial guess for X by solving for smaller system sizes (see: 10.1103/PhysRevLett.129.206401) 
        @info "Finding better initial guess for X by solving smaller system sizes..."
        # res_init = Optim.optimize(loss_init, g_init!, X, Optim.ConjugateGradient(manifold=Optim.Stiefel()), Optim.Options(
        res_init = Optim.optimize(loss_init, g_init!, X, Optim.LBFGS(;m=20,manifold=Optim.Stiefel()), Optim.Options(
            iterations = conf["params"]["maxiter"],
            g_tol = conf["params"]["grad_tol"],
            show_trace = conf["params"]["show_trace"],
            successive_f_tol = 10,
            f_reltol = conf["params"]["f_reltol"]
        ))

        @info "Finding optimal X for full system size..."
        # optimize X for the full system size
        # res = Optim.optimize(loss, g!, Optim.minimizer(res_init), Optim.ConjugateGradient(manifold=Optim.Stiefel()), Optim.Options(
        res = Optim.optimize(loss, g!, Optim.minimizer(res_init), Optim.LBFGS(;m=20,manifold=Optim.Stiefel()), Optim.Options(
        # res = Optim.optimize(loss, g!, X, Optim.LBFGS(;m=20,manifold=Optim.Stiefel()), Optim.Options(
            iterations = conf["params"]["maxiter"],
            g_tol = conf["params"]["grad_tol"],
            show_trace = conf["params"]["show_trace"],
            successive_f_tol = 10,
            f_reltol = conf["params"]["f_reltol"]
        ))

        @show Optim.minimum(res)
        println("Exact energy:", exact_energy_BCS_k(bz,t,μ,pairing_type,Δ_vec...))

        # @info "Building iPEPS from X..."
        # peps = translate(X, Nf, Nv)

        return new(
            Nf,
            Nv,
            Lx,
            Ly,
            bz,
            t,
            μ,
            Δ_options,
            conf["params"]["maxiter"],
            conf["params"]["grad_tol"],
            Optim.minimizer(res),
            # peps,
            exact_energy_BCS_k(bz,t,μ,pairing_type,Δ_vec...),
            Optim.minimum(res)
        )
    end
end