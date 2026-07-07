using GfPEPS, LinearAlgebra, Random, Zygote, Printf
Random.seed!(11)

function full_energy_loss_X(lattice, Nf, Λ, H_BdG)
    G_in = GfPEPS.G_in_Fourier(Λ, lattice)
    energy = GfPEPS.energy_loss(Nf, H_BdG, lattice)
    return X_vec -> real(energy(GaussianMap(GfPEPS.get_Γ_blocks(X_vec, Nf, Λ, lattice)..., G_in)))
end

function bench(f, X; n=5)
    f(X); Zygote.gradient(f, X) # warm up
    tv = minimum((@elapsed f(X)) for _ in 1:n)
    tg = minimum((@elapsed Zygote.gradient(f, X)) for _ in 1:n)
    return tv, tg
end

println("case | d_full -> d_bdry | loss: full / blocked (speedup) | grad: full / blocked (speedup)")
for (Lx, Ly, Λ, Nf, Nk) in ((1,1,2,2,24), (2,2,2,2,12), (2,2,3,2,12), (4,4,2,2,6), (4,4,3,1,6))
    layout = reshape(collect(1:Lx*Ly), Ly, Lx)
    lattice = InfiniteRectLattice(Lx, Ly; N_kx=Nk, N_ky=Nk, bc=(:PBC,:APBC), uc_layout=layout)
    labels = unique(vec(layout))
    hop = Dict(s => Dict((1,0)=>1.0, (-1,0)=>1.0, (0,1)=>1.0, (0,-1)=>1.0) for s in labels)
    pair = Dict(s => Dict((1,0)=>2.0, (-1,0)=>2.0, (0,1)=>2.0, (0,-1)=>2.0) for s in labels)
    H = default_BCS_hamiltonian(hop, pair, 3.0, lattice; Nf=Nf)
    X = [GfPEPS.rand_CM(Nf, Λ; parity=1)[2] for _ in labels]

    lb = energy_loss_X(lattice, Nf, Λ, H)
    lf = full_energy_loss_X(lattice, Nf, Λ, H)
    tvf, tgf = bench(lf, X)
    tvb, tgb = bench(lb, X)
    inner, bdry = virtual_mode_partition(Λ, lattice)
    @printf("%dx%d Λ=%d Nf=%d Nk=%d² | %4d -> %3d | %8.4fs / %8.4fs (%4.1fx) | %8.4fs / %8.4fs (%4.1fx)\n",
        Lx, Ly, Λ, Nf, Nk, 8Λ*Lx*Ly, length(bdry), tvf, tvb, tvf/tvb, tgf, tgb, tgf/tgb)
end
