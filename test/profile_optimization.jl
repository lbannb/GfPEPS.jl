using Revise
using Profile
using Test
using GfPEPS
using JSON: parsefile
using BenchmarkTools
using Random
using Optim
using LinearAlgebra
using Zygote
using Logging

Random.seed!(1234) # for reproducibility

#= Test settings =#
Nf = 2
Λ = 2
lattice = InfiniteRectLattice(1,1;N_kx=6, N_ky=6, bc=(:PBC, :APBC))
doping_kwargs = DopingSettings(; δ=0.16, enforce_density=false)
t1 = 1.0
hopping = get_isotropic_coupling_dict(lattice, [t1]; interaction_type=["NN"])
μ = 1.0
# dwave pairing: Δ_y = -Δ_x 
Δ1_x = 1.0
Δ1_y = -Δ1_x
pairing = get_anisotropic_coupling_dict(lattice, [[Δ1_x,Δ1_x,Δ1_y,Δ1_y]]; interaction_type=["NN"])
H_bdg = default_BCS_hamiltonian(hopping, pairing, μ, lattice; interaction_type=["NN"])

#= Old code (do not change, this is for comparison) =#
function GaussianMap_old(A::AbstractMatrix, B::AbstractMatrix, D::AbstractMatrix, CM_in::AbstractArray)
    Bt = transpose(B)

    # Gaussian map for each (kx,ky)
    mats = map(s -> B * ((D .+ s) \ Bt) .+ A, eachslice(CM_in; dims=1)) # Hong hao paper

    # Stack into a 3D tensor [i, j, k]
    return stack(mats)
end

function energy_loss_old(Nf::Int, H_bdg_k, lattice)
    kvals = lattice.kvals

    E_shift_summed = sum(map(eachcol(kvals)) do k
        ξ_k = H_bdg_k.ξ_fct(k, H_bdg_k.hopping, H_bdg_k.μ)
        Δ_k = H_bdg_k.Δ_fct(k, H_bdg_k.pairing)
        return Nf * 0.5 * ξ_k + H_bdg_k.E_shift(k, ξ_k, Δ_k, H_bdg_k.μ)
    end)

    # divide by number of k-points
    Nk = size(kvals, 2)
    invN = 1.0 / (Nk * GfPEPS.get_number_of_sites(lattice)) # actually faster when precomputed, because multiplication is faster than division
    
    # Construct the Hamiltonian tensor (2Nf × 2Nf × Nk) (column-major order for all k values, to avoid allocations in the inner loop)
    # we need the adjoint here because dot(H, CM_out) = sum(H' .* CM_out))
    H_BdG_batched = stack(map(k -> GfPEPS.H_BdG_majorana_k(Nf, k, H_bdg_k, lattice)', eachcol(kvals)))

    function energy(CM_out::AbstractArray)
        # Fast Trace Formula: Tr(H * CM) = sum(H .* CM^T) = - sum(H .* CM) = - dot(H', CM)
        # Since input is already CM^T (see GaussianMap), this is just a dot product.

        # return real((E_shift_summed + 0.25 * sum(H_BdG_batched .* CM_out)) * invN)
        return real((E_shift_summed - 0.25 * dot(H_BdG_batched, CM_out)) * invN)
    end

    return energy
end

function energy_loss_X_old(lattice, Nf::Int, Λ::Int, H_bdg_k)
    G_in = GfPEPS.G_in_Fourier(Λ, lattice)
    energy = energy_loss_old(Nf, H_bdg_k, lattice)
    function loss(X)
        return real(energy(GaussianMap_old(GfPEPS.get_Γ_blocks(GfPEPS.Γ_fiducial(X, Nf, Λ, lattice), Nf, lattice)..., G_in)))
    end
    return loss
end

function doping_loss_old(Nf::Int, lattice)
    # divide by number of k-points
    Nk = size(lattice.kvals, 2)
    invN = 1.0 / (Nk * GfPEPS.get_number_of_sites(lattice)) # actually faster when precomputed, because multiplication is faster than division

    # Construct the symplectic form (2Nf × 2Nf × Nk) (column-major order for all k values, to avoid allocations in the inner loop)
    # occupation in the majorana basis
    J0 = [0 1; -1 0]
    J = kron(I(GfPEPS.get_Nf_in_uc(Nf,lattice)), J0)

    # repeat for all k-points
    J_batched = Array{eltype(J)}(undef, size(J,1), size(J,2), Nk)
    @inbounds for k in 1:Nk
        J_batched[:, :, k] = J
    end
    # J = stack(fill(J, Nk)) # repeat for all k-points

    function doping(CM_out::AbstractArray)
        # Fast Trace Formula: Tr(J * CM) = sum(J .* CM^T) = - sum(J .* CM) = - dot(J, CM)
        # Since input is already CM^T (see GaussianMap), this is just a dot product.
        return real((0.5 * Nf - 0.25*dot(J_batched, CM_out)) * invN)
        # return real((0.5 * Nf - 0.25*sum(J .* CM_out)) * invN)
    end
end

function doping_loss_X_old(lattice, Nf::Int, Λ::Int)
    G_in = GfPEPS.G_in_Fourier(Λ, lattice)
    doping = doping_loss_old(Nf, lattice)
    function loss(X)
        return real(doping(GaussianMap_old(GfPEPS.get_Γ_blocks(GfPEPS.Γ_fiducial(X, Nf, Λ, lattice), Nf, lattice)..., G_in)))
    end
    return loss
end

function get_X_opt_old(
    lattice, 
    Nf::Int, 
    Λ::Int,
    H_bdg;
    X_init::Union{AbstractMatrix, Nothing}=nothing,
    doping_kwargs::DopingSettings=GfPEPS.DopingSettings(),
    optim_alg_options::Union{Optim.LBFGS, Optim.BFGS} = Optim.LBFGS(; m=15, manifold = Optim.Stiefel()),
    optim_options::Optim.Options = Optim.Options(; iterations=1000, g_tol=1e-6, f_reltol=1e-8, successive_f_tol = 10, show_trace=false, extended_trace=false, store_trace=true))

    # TODO: implement / test odd parity optimization
    # initial ortogonal matrix X to construct Γ_Q with correct parity sector (even = 1, odd = -1)
    X_opt = begin
        if isnothing(X_init)
            GfPEPS.rand_CM(Nf, Λ, lattice; parity=1)[2]
        else
            @info "Using initial X matrix for optimization."
            X_init
        end
    end

    # warn if dirac points are present -> then optimization of Γ can be harder, so user can adjust different kvals set
    GfPEPS.has_dirac_points(lattice.kvals, H_bdg) 

    if doping_kwargs.enforce_density && doping_kwargs.solve_μ_from_δ
        H_bdg.μ = GfPEPS.solve_for_mu(lattice.kvals, doping_kwargs.δ, H_bdg)
    end

    # smaller set of momentum pairs for initial optimization for faster convergence
    N_kx_inits, N_ky_inits = isnothing(X_init) ? GfPEPS.get_kgrids(lattice) : ([lattice.N_kx], [lattice.N_ky])
    
    if doping_kwargs.enforce_density
        @info "Target hole density δ = $(doping_kwargs.δ) will be enforced with tolerance $(doping_kwargs.density_tol)."
    end

    if !isempty(N_kx_inits) && !isempty(N_ky_inits)
       @info "Finding better initial guess for X by solving smaller system sizes..."
    end

    size_stages = collect(zip(N_kx_inits, N_ky_inits))
    stage_res = nothing
    stage_doping = nothing
    for (stage_idx, (N_kx_init, N_ky_init)) in enumerate(size_stages)
        stage_label = !(N_kx_init==lattice.N_kx) ? "Warmup stage $(stage_idx) (N_kx=$(N_kx_init), N_ky=$(N_ky_init))" : "Final optimization stage (N_kx=$(N_kx_init), N_ky=$(N_ky_init))"
        @info "Optimize X for: N_kx = $(N_kx_init), N_ky = $(N_ky_init)"

        training_lattice = InfiniteRectLattice(lattice.Lx, lattice.Ly; N_kx=N_kx_init, N_ky=N_ky_init, bc=lattice.bc, shift_x=lattice.shift_x, shift_y=lattice.shift_y)
        GfPEPS.has_dirac_points(training_lattice.kvals, H_bdg)

        loss_fct = energy_loss_X_old(training_lattice, Nf, Λ, H_bdg)
        doping_fct = doping_loss_X_old(training_lattice, Nf, Λ)

        # optimize X for current stage and get energy and doping results
        X_opt, stage_res, stage_doping = optimize_X_old(X_opt, loss_fct, doping_fct; doping_kwargs=doping_kwargs, optim_alg_options=optim_alg_options, optim_options=optim_options)

        if Optim.converged(stage_res)
            if doping_kwargs.enforce_density
                @info "$(stage_label) converged after $(stage_res.iterations) iterations." energy=Optim.minimum(stage_res) doping=stage_doping
            else
                @info "$(stage_label) converged after $(stage_res.iterations) iterations." energy=Optim.minimum(stage_res)
            end
        else
            if doping_kwargs.enforce_density
                @warn "$(stage_label) did not converge after $(stage_res.iterations) iterations." gradient_norm=stage_res.g_residual energy=Optim.minimum(stage_res) doping=stage_doping
            else
                @warn "$(stage_label) did not converge after $(stage_res.iterations) iterations." gradient_norm=stage_res.g_residual energy=Optim.minimum(stage_res)
            end
        end
    end

    # final results summary and check if density constraint is satisfied
    constraint_final = doping_kwargs.enforce_density ? stage_doping - doping_kwargs.δ : nothing
    if doping_kwargs.enforce_density
        @info "Final doping summary" target=doping_kwargs.δ achieved=stage_doping deviation=constraint_final
    end
    if doping_kwargs.enforce_density && abs(constraint_final) > doping_kwargs.density_tol
        @warn "Final doping deviates from target by $(constraint_final). Consider increasing density_opt_iters or penalty_growth."
    end

    # compute final energy and compare to exact energy
    E_exact = GfPEPS.exact_energy(lattice, H_bdg, Nf)
    optim_energy = Optim.minimum(stage_res)
    deviation = abs(optim_energy - E_exact)
    
    @info "Final energy summary" target=E_exact achieved=optim_energy deviation=deviation
    println()

    # return final results and optimization info
    info_obj = (
        converged = Optim.converged(stage_res),
        trace = stage_res.trace
    )

    return X_opt, optim_energy, E_exact, info_obj
end

function optimize_X_old(X::AbstractMatrix, loss_fct::Function, doping_fct::Function;
    doping_kwargs::DopingSettings=DopingSettings(),
    optim_alg_options::Union{Optim.LBFGS, Optim.BFGS},
    optim_options::Optim.Options)

     # No density constraint: minimize the pure energy objective on the Stiefel manifold.
    if !doping_kwargs.enforce_density
        grad_energy(x) = first(Zygote.gradient(loss_fct, x))
        grad_energy!(G, x) = copyto!(G, grad_energy(x))

        res = Optim.optimize(loss_fct, grad_energy!, X, optim_alg_options, optim_options)
        X_opt = Optim.minimizer(res)
        return X_opt, res, nothing
    end

    # Density constraint: use a penalty term to enforce the density constraint while minimizing the energy.

    # Set up augmented-Lagrangian variables.
    η = 0.0
    λ = doping_kwargs.λ

    # run density optimization loop, where in each iteration we minimize the augmented Lagrangian with the current penalty and multiplier, then update those based on the constraint violation.
    X_current = X
    last_res = nothing
    last_doping = doping_fct(X_current)
    total_iters = 0
    total_trace = []
    for _ in 1:max(doping_kwargs.density_opt_iters, 1) # usually only a few iterations are needed
        η_local = η
        λ_local = λ

        # new loss funciton with penalty term for density constraint
        loss_augmented(x) = begin
            dens = doping_fct(x)
            constraint = dens - doping_kwargs.δ
            return loss_fct(x) + η_local * constraint + 0.5 * λ_local * constraint^2
        end
        grad_aug(x) = first(Zygote.gradient(loss_augmented, x))
        grad_aug!(G, x) = copyto!(G, grad_aug(x))

        res = Optim.optimize(loss_augmented, grad_aug!, X_current, optim_alg_options, optim_options)
        total_iters += res.iterations
        total_trace = vcat(total_trace, res.trace)

        # update augmented Lagrangian parameters based on constraint violation
        last_res = res
        X_current = Optim.minimizer(res)
        last_doping = doping_fct(X_current)
        constraint = last_doping - doping_kwargs.δ

        if abs(constraint) <= doping_kwargs.density_tol
            λ = λ_local
            break
        end

        η = η_local + λ_local * constraint
        λ = max(λ_local * doping_kwargs.penalty_growth, DEFAULT_PENALTY_FALLBACK)
    end
    last_res === nothing && error("Augmented Lagrangian did not run for stage $(stage_label).")

    # update iterations
    last_res.iterations = total_iters
    last_res.trace = total_trace

    return X_current, last_res, last_doping
end

# print("Old code:")
# time_old = @benchmark begin
#     X_opt, optim_energy, E_exact, info_obj = get_X_opt_old(lattice, Nf, Λ, H_bdg; doping_kwargs=doping_kwargs)
# end
# println()

#= New code =#
println("New code:")
time_new = with_logger(NullLogger()) do 
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            # begin
                @benchmark X_opt, optim_energy, E_exact, info_obj = GfPEPS.get_X_opt(lattice, Nf, Λ, H_bdg; doping_kwargs=doping_kwargs)
            # end
        end
    end
end

# print("Old code:")
# display(time_old)
println()
print("New code:")
display(time_new);