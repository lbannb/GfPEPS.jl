mutable struct BCS
    t::Real # hopping amplitude
    μ::Real # chemical potential -> This can be changed when solve_μ_from_δ = true in DopingSettings
    pairing_type::String # "s_wave", "d_wave", "p_wave"
    Δ_0::Real # pairing amplitude
    Δ_02::Real 
end
BCS(t, μ, pairing_type, Δ_0) = BCS(t, μ, pairing_type, Δ_0, 0.0)

struct Kitaev
    Jx::Real
    Jy::Real
    Jz::Real
end