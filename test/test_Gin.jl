using Test
using GfPEPS

@testset "G_in_single_k_persite correctness" begin
    σ_x = [0 1; 1 0]

    @testset "1×1 unit cell (Λ=1)" begin
        Λ = 1
        lattice = InfiniteRectLattice(1, 1; N_kx=4, N_ky=4)
        k = [0.3, 0.7]

        # For a 1×1 cell every bond is inter-cell.
        # One site, Λ=1 → 8 Majorana modes ordered as:
        #   l(1:2), r(3:4), u(5:6), d(7:8)
        #
        # x-direction: right(1,1) ↔ left(1,1) with phase e^{ikx}   (wraps)
        # y-direction: down(1,1)  ↔ up(1,1)   with phase e^{iky}   (wraps)

        G_ref = zeros(ComplexF64, 8, 8)

        # x-bond: right(3:4) ↔ left(1:2), inter-cell phase kx
        ph_x = cis(k[1])
        # G[left, right] = -e^{ikx} σ_x   → G[1:2, 3:4]
        # G[right, left] =  e^{-ikx} σ_x  → G[3:4, 1:2]
        G_ref[1:2, 3:4] = -ph_x * σ_x
        G_ref[3:4, 1:2] =  conj(ph_x) * σ_x

        # y-bond: down(7:8) ↔ up(5:6), inter-cell phase ky
        ph_y = cis(k[2])
        G_ref[5:6, 7:8] = -ph_y * σ_x
        G_ref[7:8, 5:6] =  conj(ph_y) * σ_x

        G_test = GfPEPS.G_in_single_k_persite(k, Λ, lattice)

        @test size(G_test) == (8, 8)
        @test G_test ≈ G_ref atol=1e-14
    end

    @testset "1×1 unit cell (Λ=2)" begin
        Λ = 2
        lattice = InfiniteRectLattice(1, 1; N_kx=4, N_ky=4)
        k = [0.3, 0.7]

        # One site, Λ=2 → 16 Majorana modes ordered as:
        #   l₁(1:2), r₁(3:4), l₂(5:6), r₂(7:8),
        #   u₁(9:10), d₁(11:12), u₂(13:14), d₂(15:16)

        G_ref = zeros(ComplexF64, 16, 16)

        ph_x = cis(k[1])
        ph_y = cis(k[2])

        # x-bonds for both Λ flavors (all inter-cell for 1×1)
        for α in 1:2
            r0 = 4(α - 1) + 3   # right modes start
            l0 = 4(α - 1) + 1   # left modes start
            G_ref[l0:l0+1, r0:r0+1] = -ph_x * σ_x
            G_ref[r0:r0+1, l0:l0+1] =  conj(ph_x) * σ_x
        end

        # y-bonds for both Λ flavors (all inter-cell for 1×1)
        for α in 1:2
            d0 = 4Λ + 4(α - 1) + 3   # down modes start
            u0 = 4Λ + 4(α - 1) + 1   # up modes start
            G_ref[u0:u0+1, d0:d0+1] = -ph_y * σ_x
            G_ref[d0:d0+1, u0:u0+1] =  conj(ph_y) * σ_x
        end

        G_test = GfPEPS.G_in_single_k_persite(k, Λ, lattice)

        @test size(G_test) == (16, 16)
        @test G_test ≈ G_ref atol=1e-14
    end

    @testset "1×2 unit cell (Λ=1)" begin
        Λ = 1
        lattice = InfiniteRectLattice(1, 2; N_kx=4, N_ky=4)
        k = [0.5, -0.4]

        # Two sites: A = site 1 (ix=1,iy=1), B = site 2 (ix=1,iy=2)
        # Lx=1, Ly=2.  Linear index: s = ix + (iy-1)*Lx → A=1, B=2
        #
        # Per site 8 modes: l(1:2), r(3:4), u(5:6), d(7:8)
        # Site A modes: 1..8,  Site B modes: 9..16
        #
        # x-direction (Lx=1): right(ix=1) → left(ix=1, wraps), always inter-cell
        #   For site A: right_A(3:4) ↔ left_A(1:2), phase e^{ikx}
        #   For site B: right_B(11:12) ↔ left_B(9:10), phase e^{ikx}
        #
        # y-direction (Ly=2):
        #   Site A (iy=1) → Site B (iy=2): down_A(7:8) ↔ up_B(13:14), intra-cell (phase=0)
        #   Site B (iy=2) → Site A (iy=1): down_B(15:16) ↔ up_A(5:6), inter-cell (phase=ky)

        n = 16  # 8Λ * Lx * Ly = 8 * 1 * 2
        G_ref = zeros(ComplexF64, n, n)

        ph_x = cis(k[1])
        ph_y = cis(k[2])

        # x-bonds (both sites wrap in x since Lx=1)
        # Site A: right_A(3:4) ↔ left_A(1:2)
        G_ref[1:2, 3:4] = -ph_x * σ_x
        G_ref[3:4, 1:2] =  conj(ph_x) * σ_x

        # Site B: right_B(11:12) ↔ left_B(9:10)
        G_ref[9:10, 11:12] = -ph_x * σ_x
        G_ref[11:12, 9:10] =  conj(ph_x) * σ_x

        # y-bonds
        # A→B intra-cell: down_A(7:8) ↔ up_B(13:14), phase=1
        ph_intra = cis(0.0)  # = 1
        G_ref[13:14, 7:8] = -ph_intra * σ_x
        G_ref[7:8, 13:14] =  conj(ph_intra) * σ_x

        # B→A inter-cell: down_B(15:16) ↔ up_A(5:6), phase=e^{iky}
        G_ref[5:6, 15:16] = -ph_y * σ_x
        G_ref[15:16, 5:6] =  conj(ph_y) * σ_x

        G_test = GfPEPS.G_in_single_k_persite(k, Λ, lattice)

        @test size(G_test) == (n, n)
        @test G_test ≈ G_ref atol=1e-14
        display(G_ref)
    end

    @testset "2×2 unit cell (Λ=1)" begin
        Λ = 1
        lattice = InfiniteRectLattice(2, 2; N_kx=4, N_ky=4)
        k = [1.2, -0.8]

        # Four sites, column-major: s = ix + (ix-1)*Lx
        #   s=1: (1,1), s=2: (2,1), s=3: (1,2), s=4: (2,2)
        # Per site 8 modes → 32 total
        # Site s modes: (s-1)*8+1 .. s*8
        #   within each site: l(1:2), r(3:4), u(5:6), d(7:8) offset by (s-1)*8

        n = 32
        G_ref = zeros(ComplexF64, n, n)

        ph_x = cis(k[1])
        ph_y = cis(k[2])

        Lx, Ly = 2, 2
        n_per_site = 8

        # Build all bonds explicitly
        for iy in 1:Ly, ix in 1:Lx
            s = ix + (iy - 1) * Lx
            base_s = n_per_site * (s - 1)

            # x-bond: right(ix,iy) → left(ix+1,iy)
            ix_next = mod1(ix + 1, Lx)
            s_next_x = ix_next + (iy - 1) * Lx
            base_next_x = n_per_site * (s_next_x - 1)
            phase_x = (ix == Lx) ? k[1] : 0.0

            r0 = base_s + 3      # right modes of site s
            l0 = base_next_x + 1 # left modes of next site in x

            ph = cis(phase_x)
            G_ref[l0:l0+1, r0:r0+1] = -ph * σ_x
            G_ref[r0:r0+1, l0:l0+1] =  conj(ph) * σ_x

            # y-bond: down(ix,iy) → up(ix,iy+1)
            iy_next = mod1(iy + 1, Ly)
            s_next_y = ix + (iy_next - 1) * Lx
            base_next_y = n_per_site * (s_next_y - 1)
            phase_y = (iy == Ly) ? k[2] : 0.0

            d0 = base_s + 4Λ + 3     # down modes of site s (offset by LR block)
            u0 = base_next_y + 4Λ + 1 # up modes of next site in y

            ph2 = cis(phase_y)
            G_ref[u0:u0+1, d0:d0+1] = -ph2 * σ_x
            G_ref[d0:d0+1, u0:u0+1] =  conj(ph2) * σ_x
        end

        G_test = GfPEPS.G_in_single_k_persite(k, Λ, lattice)

        @test size(G_test) == (n, n)
        @test G_test ≈ G_ref atol=1e-14
    end

    @testset "1×1 persite matches original G_in_single_k" begin
        # For a 1×1 unit cell, G_in_single_k_persite should match
        # the original G_in_single_k (up to identical mode ordering)
        for Λ in [1, 2, 3]
            lattice = InfiniteRectLattice(1, 1; N_kx=4, N_ky=4)
            k = [0.3, 0.7]

            G_old = GfPEPS.G_in_single_k(k, Λ, lattice)
            G_new = GfPEPS.G_in_single_k_persite(k, Λ, lattice)

            # The old function orders modes as:
            #   [⊕_{α=1}^{Λ} helper(kx,1)] ⊕ [⊕_{α=1}^{Λ} helper(ky,1)]
            # where helper(k,1) gives (l,r) block of size 4×4
            # So old ordering: l₁r₁ l₂r₂ ... lΛrΛ u₁d₁ u₂d₂ ... uΛdΛ
            #
            # The new function orders modes per site, but for 1×1 this is:
            #   l₁r₁ l₂r₂ ... lΛrΛ u₁d₁ u₂d₂ ... uΛdΛ
            # which is identical.

            @test size(G_old) == size(G_new)
            @test G_old ≈ G_new atol=1e-14
        end
    end
end