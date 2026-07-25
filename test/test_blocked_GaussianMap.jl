using Test
using GfPEPS
using LinearAlgebra
using Random
using Zygote

Random.seed!(20260707)

#=
    Consistency of the blocked Gaussian map (inner/boundary mode ordering + k-independent
    Schur complement over the intra-cell bonds) against the full Gaussian map on all
    virtual modes, for values and Zygote gradients.
=#

# full-map reference loss (contracts all 8Λ·Lx·Ly virtual modes per k-point)
function full_energy_loss_X(lattice, Nf, Λ, H_BdG)
    G_in = GfPEPS.G_in_Fourier(Λ, lattice)
    energy = GfPEPS.energy_loss(Nf, H_BdG, lattice)
    return X_vec -> real(energy(GaussianMap(GfPEPS.get_Γ_blocks(X_vec, Nf, Λ, lattice)..., G_in)))
end

@testset "Blocked Gaussian map == full Gaussian map" begin
    cases = (
        (2, 2, [1 2; 2 1], 2, 2), # 2x2 unit cell with two sublattices
        (2, 1, [1 2], 1, 2), # 1x2 with two sublattices
        (2, 2, [2 2; 1 1], 2, 1), # layout where labels differ from linear site indices
        (1, 1, fill(1, 1, 1), 2, 2), # trivial unit cell: blocked map must reduce to full map
    )

    # check that the blocked Gaussian map (inner/boundary mode ordering + k-independent Schur complement over the intra-cell bonds) gives the same energy loss and doping loss as the full Gaussian map on all virtual modes, for values and Zygote gradients
    for (Lx, Ly, layout, Nf, Λ) in cases
        lattice = InfiniteRectLattice(Lx, Ly; N_kx=6, N_ky=6, bc=(:PBC, :APBC), uc_layout=layout)
        labels = unique(vec(layout))

        # create a BCS Hamiltonian with nonzero hopping and pairing on all bonds, so that the energy loss is nontrivial
        hop = Dict(s => Dict((1, 0) => 1.0, (-1, 0) => 1.0, (0, 1) => 1.0, (0, -1) => 1.0) for s in labels)
        pair = Dict(s => Dict((1, 0) => 2.0, (-1, 0) => 2.0, (0, 1) => 2.0, (0, -1) => 2.0) for s in labels)
        H_BdG = default_BCS_hamiltonian(hop, pair, 3.0, lattice; Nf=Nf)

        X_vec = [GfPEPS.rand_CM(Nf, Λ; parity=1)[2] for _ in labels] # random CM states for each distinct site in the unit cell

        # compute energy losses using both the blocked and full Gaussian maps
        loss_blocked = energy_loss_X(lattice, Nf, Λ, H_BdG)
        loss_full = full_energy_loss_X(lattice, Nf, Λ, H_BdG)
        @test loss_blocked(X_vec) ≈ loss_full(X_vec) atol=1e-10

        # compute Zygote gradients of the energy losses using both the blocked and full Gaussian maps
        g_blocked = Zygote.gradient(loss_blocked, X_vec)[1]
        g_full = Zygote.gradient(loss_full, X_vec)[1]
        @test maximum(maximum(abs, g_blocked[i] - g_full[i]) for i in eachindex(g_blocked)) < 1e-9

        # doping path
        doping_blocked = doping_loss_X(lattice, Nf, Λ)(X_vec)
        doping_full = real(GfPEPS.doping_loss(Nf, lattice)(GaussianMap(GfPEPS.get_Γ_blocks(X_vec, Nf, Λ, lattice)..., GfPEPS.G_in_Fourier(Λ, lattice))))
        @test doping_blocked ≈ doping_full atol=1e-10
    end
end;

@testset "Mode partition properties" begin
    for (Lx, Ly, Λ) in ((2, 2, 2), (3, 2, 1), (1, 2, 3), (1, 1, 2))
        lattice = InfiniteRectLattice(Lx, Ly; N_kx=4, N_ky=4)
        inner, boundary = virtual_mode_partition(Λ, lattice)
        # boundary modes: 4Λ(Lx+Ly) Majorana modes minus double counting for Lx==1 / Ly==1,
        # where l/r (u/d) coincide on the same wrap bond
        n_boundary_x = (Lx == 1 ? 4Λ : 4Λ) * Ly       # per row: l of col 1 + r of col Lx (2Λ + 2Λ)
        n_boundary_y = (Ly == 1 ? 4Λ : 4Λ) * Lx
        @test length(boundary) == n_boundary_x + n_boundary_y
        @test sort(vcat(inner, boundary)) == collect(1:8Λ * Lx * Ly)

        # all k-dependence must live in the boundary block
        gm = gaussian_map_inputs(Λ, lattice)
        k = randn(2) * 2π # random k-point
        G_k = GfPEPS.G_in_single_k(k, Λ, lattice)
        G_rebuilt = complex(copy(gm.G_intra))
        G_rebuilt[boundary, boundary] .= G_k[boundary, boundary]
        @test maximum(abs, G_k - G_rebuilt) < 1e-14
    end
end;

@testset "Unit-cell layout indexing of get_Γ_blocks" begin
    # layout whose labels do NOT coincide with the linear site index of their first occurrence
    Nf, Λ = 2, 1
    layout = [2 2; 1 1]
    lattice = InfiniteRectLattice(2, 2; N_kx=4, N_ky=4, uc_layout=layout)
    X_vec = [GfPEPS.rand_CM(Nf, Λ; parity=1)[2] for _ in 1:2]
    A, B, D = GfPEPS.get_Γ_blocks(X_vec, Nf, Λ, lattice)

    # Check that the blocks of the full Γ_fiducial matrix are correctly indexed by the unit-cell layout, i.e. that the blocks of A, B, D correspond to the correct site in the unit cell.
    for x in 1:2, y in 1:2
        s = GfPEPS.get_site_index(x, y, lattice)
        Γ_site = GfPEPS.Γ_fiducial(X_vec[layout[y, x]], Nf, Λ)
        pa, pv = 2Nf * (s - 1), 8Λ * (s - 1)
        @test A[pa+1:pa+2Nf, pa+1:pa+2Nf] ≈ Γ_site[1:2Nf, 1:2Nf]
        @test B[pa+1:pa+2Nf, pv+1:pv+8Λ] ≈ Γ_site[1:2Nf, 2Nf+1:end]
        @test D[pv+1:pv+8Λ, pv+1:pv+8Λ] ≈ Γ_site[2Nf+1:end, 2Nf+1:end]
    end
end;
