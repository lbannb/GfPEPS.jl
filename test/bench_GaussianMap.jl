"""
Benchmark script: compare old (map+stack + Zygote generic AD) vs new
(pre-allocated + threaded + custom rrule) GaussianMap implementation.

Measures both the isolated forward pass and a full forward+backward (gradient)
pass at two problem sizes.
"""

using GfPEPS
using Random
using LinearAlgebra
using Zygote
using BenchmarkTools
import ChainRulesCore: rrule, NoTangent, unthunk

Random.seed!(42)

# ── Old implementation (for reference) ──────────────────────────────────────
function GaussianMap_old(A, B, D, CM_in)
    Bt = transpose(B)
    mats = map(s -> B * ((D .+ s) \ Bt) .+ A, eachslice(CM_in; dims=1))
    return stack(mats)
end

# ── Benchmark configs ───────────────────────────────────────────────────────
configs = [
    (label="Small (Nf=1,Λ=2,6×6)", Nf=1, Λ=2, Nk=6),
    (label="Medium (Nf=2,Λ=4,12×12)", Nf=2, Λ=4, Nk=12),
]

println("Julia threads: ", Threads.nthreads())
println("="^72)

for cfg in configs
    println("\n", cfg.label)
    println("-"^72)

    lattice = InfiniteRectLattice(1, 1; N_kx=cfg.Nk, N_ky=cfg.Nk, bc=(:PBC, :APBC))

    # Build common inputs
    G_in = GfPEPS.G_in_Fourier(cfg.Λ, lattice)
    _, X = GfPEPS.rand_CM(cfg.Nf, cfg.Λ, lattice; parity=1)
    Γ = GfPEPS.Γ_fiducial(X, cfg.Nf, cfg.Λ, lattice)
    A, B, D = GfPEPS.get_Γ_blocks(Γ, cfg.Nf, lattice)

    d = size(D, 1)
    n = size(A, 1)
    Nk_total = size(G_in, 1)
    println("  Matrix sizes: A=$(n)×$(n), B=$(n)×$(d), D=$(d)×$(d), Nk=$(Nk_total)")

    # ── Forward pass ──
    println("\n  Forward pass (GaussianMap only):")
    print("    OLD (map+stack):         ")
    @btime GaussianMap_old($A, $B, $D, $G_in)
    print("    NEW (prealloc+threads):  ")
    @btime GfPEPS.GaussianMap($A, $B, $D, $G_in)

    # ── Forward + backward (gradient via Zygote) ──
    # Precompute the energy function outside the gradient closure (matches real code)
    t1 = 1.0
    hopping = get_isotropic_coupling_dict(lattice, [t1]; interaction_type=["NN"])
    μ = 1.0; Δ = 1.0
    pairing = get_isotropic_coupling_dict(lattice, [Δ]; interaction_type=["NN"])
    H_BdG = default_BCS_hamiltonian(hopping, pairing, μ, lattice; interaction_type=["NN"])

    # Precompute energy function (not differentiated through)
    energy_fct = GfPEPS.energy_loss(cfg.Nf, H_BdG, lattice)

    # Old loss: uses GaussianMap_old (Zygote generic AD for backprop)
    function loss_old_X(X_in)
        Γ_loc = GfPEPS.Γ_fiducial(X_in, cfg.Nf, cfg.Λ, lattice)
        A_loc, B_loc, D_loc = GfPEPS.get_Γ_blocks(Γ_loc, cfg.Nf, lattice)
        Bt_loc = transpose(B_loc)
        mats = map(s -> B_loc * ((D_loc .+ s) \ Bt_loc) .+ A_loc, eachslice(G_in; dims=1))
        CM_out = stack(mats)
        return real(energy_fct(CM_out))
    end

    # New loss: uses GfPEPS.GaussianMap with custom rrule
    function loss_new_X(X_in)
        Γ_loc = GfPEPS.Γ_fiducial(X_in, cfg.Nf, cfg.Λ, lattice)
        A_loc, B_loc, D_loc = GfPEPS.get_Γ_blocks(Γ_loc, cfg.Nf, lattice)
        return real(energy_fct(GfPEPS.GaussianMap(A_loc, B_loc, D_loc, G_in)))
    end

    # warm-up
    Zygote.gradient(loss_old_X, X)
    Zygote.gradient(loss_new_X, X)

    println("\n  Full gradient (forward + backward through Zygote):")
    print("    OLD (Zygote generic AD): ")
    @btime Zygote.gradient($loss_old_X, $X)
    print("    NEW (custom rrule):      ")
    @btime Zygote.gradient($loss_new_X, $X)
end

println("\n", "="^72)
println("Done.")
