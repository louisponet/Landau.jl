@export function GLDparameters(α, G11, G12, G44, C11, C12, C44, Q11, Q12, Q44, F11, F12, F44, εᵣ=1.0)
    # q11 = C11*Q11 + 2C12*Q12
    # q12 = C11*Q12 + C12*(Q11 + Q12)
    # q44 = C44*Q44
    q11 = Q11
    q12 = Q12
    q44 = Q44
    V2T(p11, p12, p44) = Tensor{4, 3}((i,j,k,l) -> p11 * δ(i,j)*δ(k,l)*δ(i,k) + p12*δ(i,j)*δ(k,l)*(1 - δ(i,k)) + p44*δ(i,k)*δ(j,l)*(1 - δ(i,j)))
    C = V2T(C11, C12, C44)
    G = V2T(G11, G12, G44)
    Q = V2T(q11, q12, q44)
    F = V2T(F11, F12, F44)
    (α=α, C=C, G=G, Q=Q, F=F, εᵣ=εᵣ)
end
