
const α4thTuple{T} = NamedTuple{(:α1, :α11, :α12), NTuple{3, T}}
function Flandau(P::Vec{3, T}, α::α4thTuple{T}) where T
    P2 = Vec{3, T}((P[1]^2, P[2]^2, P[3]^2))
    return (α.α1 * sum(P2) +
           α.α11 * (P[1]^4 + P[2]^4 + P[3]^4)) +
           α.α12 * ((P2[1] * P2[2]  + P2[2]*P2[3]) + P2[1]*P2[3])
end

const α6thTuple{T} = NamedTuple{(:α1, :α11, :α111, :α12, :α112, :α123), NTuple{6, T}}

function Flandau(P::Vec{3, T}, α) where T
    P2 = Vec{3, T}((P[1]^2, P[2]^2, P[3]^2))
    P4 = Vec{3, T}((P[1]^4, P[2]^4, P[3]^4))
    return (((((α.α1 * sum(P2) +
           α.α11 * sum(P4)) +
           α.α12 * ((P2[1] * P2[2]  + P2[2]*P2[3]) + P2[1]*P2[3])) +
           α.α111 * ((P[1]^6 + P[2]^6) + P[3]^6)) +
           α.α112 * (P4[1]*(P2[2] + P2[3]) + P4[2]*(P2[1] + P2[3]) + P4[3]*(P2[1] + P2[2]))) +
           α.α123 * prod(P2))
end
@inline Felastic(ε, C) = 0.5(ε ⊡ C) ⊡ ɛ
@inline Felectrostriction(P, ε, q) = -(((q ⊡ ε) ⋅ P) ⋅ P)
@inline Fginzburg(∇P, G) = 0.5(∇P ⊡ G) ⊡ ∇P
@inline Fflexoelectric(∇P, ε, F, C) = 0.5((∇P ⊡ F) ⊡ C) ⊡ ε

@inline F(P, ε, ∇P, α, G, C, Q, F) =
	(((Felastic(ε, C) + Fginzburg(∇P, G)) + Flandau(P, α)) + Fflexoelectric(∇P, ε, F, C)) + Felectrostriction(P, ε, Q)
