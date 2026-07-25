struct BCS
    t::Real
    μ::Real
    pairing_type::String
    Δ_0::Real
    Δ_02::Real
end
BCS(t, μ, pairing_type, Δ_0) = BCS(t, μ, pairing_type, Δ_0, 0.0)

struct Kitaev
    Jx::Real
    Jy::Real
    Jz::Real
end