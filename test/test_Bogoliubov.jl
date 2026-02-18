using Revise
using Test
using GfPEPS
using LinearAlgebra

# Create a parent Hamiltonian for a random CM state and test the properties of the Bogoliubov transformation that diagonalizes it. 
# Also test the Bloch-Messiah decomposition of the Bogoliubov transformation.
Nf = 2
Λ = 2
N = (Nf + 4*Λ)
Γ,_ = GfPEPS.rand_CM(Nf, Λ)
H = GfPEPS.get_parent_hamiltonian(Γ, Nf, Λ)
_, M = GfPEPS.bogoliubov(H)

@testset "Bogoliubov transformation" begin
    U,V = GfPEPS.get_bogoliubov_blocks(M)
    V_conj = M[1:N, N+1:end]
    U_conj = M[N+1:end, N+1:end]

    # test properties of Bogoliubov transformation
    @test U_conj ≈ conj(U)
    @test V_conj ≈ conj(V)
    @test U'U + V'V ≈ I
    @test transpose(U) * V ≈ - transpose(V) * U 

    # test Bloch-Messiah decomposition
    Dmat, UVmat, Cmat = GfPEPS.bloch_messiah_decomposition(M)
    @test M ≈ Dmat * UVmat * Cmat
end;