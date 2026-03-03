using Test
using GfPEPS
using Random
using LinearAlgebra
using Zygote
using BenchmarkTools
import ChainRulesCore: rrule, NoTangent, unthunk

Random.seed!(42)

@testset "GaussianMap optimizations" begin
    # Set up small test case: Nf=1, Λ=2, trivial unit cell
    Nf = 1
    Λ = 2
    lattice = InfiniteRectLattice(1, 1; N_kx=6, N_ky=6, bc=(:APBC, :PBC))

    # Build test inputs
    G_in = GfPEPS.G_in_Fourier(Λ, lattice)
    _, X = GfPEPS.rand_CM(Nf, Λ, lattice; parity=1)
    Γ = GfPEPS.Γ_fiducial(X, Nf, Λ, lattice)
    A, B, D = GfPEPS.get_Γ_blocks(Γ, Nf, lattice)

    @testset "Forward pass: new matches reference" begin
        # Reference: functional map+stack (the old implementation)
        println("Old:")
        CM_ref = @btime begin
            Bt = transpose($B)
            mats_ref = map(s -> $B * (($D .+ s) \ Bt) .+ $A, eachslice($G_in; dims=1))
            stack(mats_ref)
        end

        # New: pre-allocated threaded implementation (dispatches through rrule forward)
        println("New:")
        CM_new = @btime GfPEPS.GaussianMap($A, $B, $D, $G_in)

        @test CM_new ≈ CM_ref atol=1e-12
    end

    @testset "Custom rrule gradient vs finite differences" begin
        # Scalar loss function: sum of real parts
        function scalar_loss(A, B, D, G_in)
            out = GfPEPS.GaussianMap(A, B, D, G_in)
            return real(sum(out))
        end

        # Compute gradient via custom rrule (Zygote will use it)
        grad_A, grad_B, grad_D, grad_CM = Zygote.gradient(scalar_loss, A, B, D, G_in)

        # Finite difference gradient for A
        ε = 1e-6
        fd_A = similar(A)
        for i in eachindex(A)
            A_plus = copy(A); A_plus[i] += ε
            A_minus = copy(A); A_minus[i] -= ε
            fd_A[i] = (scalar_loss(A_plus, B, D, G_in) - scalar_loss(A_minus, B, D, G_in)) / (2ε)
        end
        @test grad_A ≈ fd_A atol=1e-5

        # Finite difference gradient for B
        fd_B = similar(B)
        for i in eachindex(B)
            B_plus = copy(B); B_plus[i] += ε
            B_minus = copy(B); B_minus[i] -= ε
            fd_B[i] = (scalar_loss(A, B_plus, D, G_in) - scalar_loss(A, B_minus, D, G_in)) / (2ε)
        end
        @test grad_B ≈ fd_B atol=1e-5

        # Finite difference gradient for D
        fd_D = similar(D)
        for i in eachindex(D)
            D_plus = copy(D); D_plus[i] += ε
            D_minus = copy(D); D_minus[i] -= ε
            fd_D[i] = (scalar_loss(A, B, D_plus, G_in) - scalar_loss(A, B, D_minus, G_in)) / (2ε)
        end
        @test grad_D ≈ fd_D atol=1e-5
    end

    @testset "Custom rrule gradient matches Zygote generic AD" begin
        # Build a loss using the full pipeline (as in energy_loss_X)
        Jx = 1.0; Jy = 1.0; Jz = 1.0
        H_BdG = kitaev_BCS_hamiltonian(Jx, Jy, Jz, lattice; interaction_type=["NN"])
        loss_fct = GfPEPS.energy_loss_X(lattice, Nf, Λ, H_BdG)

        # Gradient via custom rrule
        grad_rrule = first(Zygote.gradient(loss_fct, X))

        # Reference gradient via Zygote generic AD (using old map+stack code inline)
        G_in_ref = GfPEPS.G_in_Fourier(Λ, lattice)
        energy_fct = GfPEPS.energy_loss(Nf, H_BdG, lattice)

        function loss_reference(X_in)
            Γ_loc = GfPEPS.Γ_fiducial(X_in, Nf, Λ, lattice)
            A_loc, B_loc, D_loc = GfPEPS.get_Γ_blocks(Γ_loc, Nf, lattice)
            Bt_loc = transpose(B_loc)
            mats = map(s -> B_loc * ((D_loc .+ s) \ Bt_loc) .+ A_loc, eachslice(G_in_ref; dims=1))
            CM_out = stack(mats)
            return real(energy_fct(CM_out))
        end

        grad_ref = first(Zygote.gradient(loss_reference, X))

        @test grad_rrule ≈ grad_ref atol=1e-10
    end

    @testset "BCS d-wave energy optimization" begin
        # This is the standard BCS test that is known to work  
        Nf_bcs = 2
        Λ_bcs = 4
        lattice_bcs = InfiniteRectLattice(1, 1; N_kx=6, N_ky=6, bc=(:PBC, :APBC))

        t1 = 1.0
        hopping = get_isotropic_coupling_dict(lattice_bcs, [t1]; interaction_type=["NN"])
        μ = 1.0
        Δ1_x = 1.0
        Δ1_y = -Δ1_x
        pairing = get_anisotropic_coupling_dict(lattice_bcs, [[Δ1_x, Δ1_x, Δ1_y, Δ1_y]]; interaction_type=["NN"])
        H_BdG = default_BCS_hamiltonian(hopping, pairing, μ, lattice_bcs; interaction_type=["NN"])

        Ψ_trial = Gaussian_fPEPS(Nf_bcs, Λ_bcs, lattice_bcs, H_BdG)

        @test Ψ_trial.exact_energy ≈ Ψ_trial.optim_energy atol=1e-5
    end
end;
