using Zygote, Landau, JuAFEM

function Flandau(P::Vec{3, T}, α::Vec{6}) where T
    P2 = Vec{3, T}((P[1]^2, P[2]^2, P[3]^2))
    P4 = Vec{3, T}((P[1]^4, P[2]^4, P[3]^4))
    return (((((α[1] * sum(P2) +
           α[2] * sum(P4)) +
           α[3] * ((P2[1] * P2[2]  + P2[2]*P2[3]) + P2[1]*P2[3])) +
           α[4] * ((P[1]^6 + P[2]^6) + P[3]^6)) +
           α[5] * (P4[1]*(P2[2] + P2[3]) + P4[2]*(P2[1] + P2[3]) + P4[3]*(P2[1] + P2[2]))) +
           α[6] * prod(P2))
end
function Landau.Flandau(P, α)
    P2 = [P[1]^2, P[2]^2, P[3]^2]
    P4 = [P[1]^4, P[2]^4, P[3]^4]
    return (((((α[1] * sum(P2) +
           α[2] * sum(P4)) +
           α[3] * ((P2[1] * P2[2]  + P2[2]*P2[3]) + P2[1]*P2[3])) +
           α[4] * ((P[1]^6 + P[2]^6) + P[3]^6)) +
           α[5] * (P4[1]*(P2[2] + P2[3]) + P4[2]*(P2[1] + P2[3]) + P4[3]*(P2[1] + P2[2]))) +
           α[6] * prod(P2))
end
αT(temp::T) where {T} = Vec{6, T}((-0.27054e8, -6.3817e8, 3.230e8, 7.8936e9, 4.470e9, 4.910e9))

BTOParams(T=300.0) = GLDparameters(αT(T), 5.1e-10,    -0.20e-10,  0.20e-10,
                                              2.75e11,     1.79e11,   0.543e11,
                                              0.1104,     -0.0452,    0.0289,
                                              0.3094e-11, -0.279e-11,-0.1335e-11)

testP = Vec{3, Float64}((1.0,0.0,0.0))
a = (BTOParams().α...,)
tlandau = x->Flandau(x..., BTOParams().α)
f(x, y) = 5x + 6y

Zygote.gradient(f, 2, 3)

@time Zygote.gradient(Landau.Flandau, (testP...,), a)


function element_potential(u::AbstractVector{T}, cellvalues, extradata, params) where T
    energy = zero(T)
    for qp=1:getnquadpoints(cellvalues.u)
        #δu = function_value(cellvalues.u, qp, u, extradata.ranges.u)
        P  = function_value(cellvalues.P, qp, u, 1:12)
        ∇P = function_gradient(cellvalues.P, qp, u, 1:12)
        ε  = function_symmetric_gradient(cellvalues.u, qp, u, 1:12)
        #δforce = function_value(cellvalues.u, qp, extradata.force)
        energy += (F(P, ε, ∇P, params)) * getdetJdV(cellvalues.u, qp)
        #energy += (F(P, ε, ∇P, params) -  ε[3,3] * δforce[3]) * getdetJdV(cellvalues.u, qp)

    end
    return energy
end
BTOModel(T::Float64, args...; kwargs...) = LandauModel(BTOParams(T), args...; kwargs...)

left = Vec{3}((-xsize, -ysize, -zsize))/2
right = Vec{3}((xsize, ysize, zsize))/2
#model = BTOModel(300.0, [(:u, 3),(:P, 3, (x)-> (0.0, 0.0, -0.265tanh((x[1])/20e-10)))], (nxels, nyels, nzels), left, right, element_potential; lagrangeorder=1, boundaryconds=[(:u, "bottom",(x,t)->[0.0],[3]),(:P, "bottom", (x,t)->[0.0], [3])]);
#model = BTOModel(300.0, [(:u, 3),(:P, 3, (x)-> (0.0, 0.0, -0.265tanh((x[1])/20e-10)))], (nxels, nyels, nzels), left, right, element_potential; lagrangeorder=1);
model = BTOModel(300.0, [(:u, 3),(:P, 3, (x)-> (0.0, 0.0, -0.265tanh((x[1]-0.5e-9)/0.5e-20)))], (nxels, nyels, nzels), left, right, element_potential; lagrangeorder=1, boundaryconds=[(:u, "bottom",(x,t)->[0.0],[3])]);

Zygote.gradient(model.threadcaches[1].efunc, zeros(12))
