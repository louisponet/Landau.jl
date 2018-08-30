
function Fl(P::Vec{3, T}, α::Vec{3}) where T
    P2 = Vec{3, T}((P[1]^2, P[2]^2, P[3]^2))
    return (α[1] * sum(P2) +
           α[2] * (P[1]^4 + P[2]^4 + P[3]^4)) +
           α[3] * ((P2[1] * P2[2]  + P2[2]*P2[3]) + P2[1]*P2[3])
end
function Fl(P::Vec{3, T}, α::Vec{6}) where T
    P2 = Vec{3, T}((P[1]^2, P[2]^2, P[3]^2))
    P4 = Vec{3, T}((P[1]^4, P[2]^4, P[3]^4))
    return (((((α[1] * sum(P2) +
           α[2] * sum(P4)) +
           α[3] * ((P2[1] * P2[2]  + P2[2]*P2[3]) + P2[1]*P2[3])) +
           α[4] * ((P[1]^6 + P[2]^6) + P[3]^6)) +
           α[5] * (P4[1]*(P2[2] + P2[3]) + P4[2]*(P2[1] + P2[3]) + P4[3]*(P2[1] + P2[2]))) +
           α[6] * prod(P2))
end
@inline Fc(ε, C) = 0.5(ε ⊡ C) ⊡ ɛ
@inline Fq(P, ε, q) = -(((q ⊡ ε) ⋅ P) ⋅ P)
@inline Fg(∇P, G) = 0.5(∇P ⊡ G) ⊡ ∇P
@inline Ff(∇P, ε, F, C) = 0.5((∇P ⊡ F) ⊡ C) ⊡ ε

F(P, ε, ∇P, params) = (((Fc(ε, params.C) + Fg(∇P, params.G)) + Fl(P, params.α)) + Ff(∇P, ε, params.F ,params.C)) + Fq(P, ε, params.Q)


calc_cell(x, cache::CellCache) = cache.potential(x, cache.cellvalues, cache.parameters, cache.force)
