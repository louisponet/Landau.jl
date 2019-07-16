using Landau
using JuAFEM
import JuAFEM: find_field, getorder, getdim, dof_range
αT(temp::T) where {T} = Vec{6, T}((-0.27054e8, -6.3817e8, 3.230e8, 7.8936e9, 4.470e9, 4.910e9))
BTOParams(T=300.0) = GLDparameters(αT(T), 5.1e-10,    -0.20e-10,  0.20e-10,
                                              2.75e11,     1.79e11,   0.543e11,
                                              0.1104,     -0.0452,    0.0289,
                                              0.3094e-11, -0.279e-11,-0.1335e-11)

const ε₀ = 8.8541878176e-12
function element_potential(u::AbstractVector{T}, cellvalues, extradata, params, ranges) where T
    energy = zero(T)
    for i in ranges.u
        u[i] *= 3.3e-11
    end
    for qp=1:getnquadpoints(cellvalues.u)
        # δu = function_value(cellvalues.u, qp, u, 1:12)
        function_value(cellvalues.P, qp, u, ranges.P)
        P  = function_value(cellvalues.P, qp, u, ranges.P)
        ∇P = function_gradient(cellvalues.P, qp, u, ranges.P)
        ε  = function_symmetric_gradient(cellvalues.u, qp, u, ranges.u)
        Ed = function_value(cellvalues.P, qp, extradata.Edepol)
        
        energy += F(P, ε, ∇P, params)* getdetJdV(cellvalues.u, qp)
        # energy += (F(P, ε, ∇P, params) - P ⋅ extradata.force - 1e-5*getdetJdV(cellvalues.u, qp)/(4π*ε₀*35)*P ⋅ Ed) * getdetJdV(cellvalues.u, qp)
        # energy += (F(P, ε, ∇P, params) - P ⋅ extradata.force ) * getdetJdV(cellvalues.u, qp)

    end
    return energy
end
function element_potential(u::AbstractVector{T}, cellvalues, extradata, params) where T
    energy = zero(T)
    for i in extradata.ranges.u
        u[i] *= 3.3e-11
    end
    for qp=1:getnquadpoints(cellvalues.u)
        # δu = function_value(cellvalues.u, qp, u, 1:12)
        function_value(cellvalues.P, qp, u, extradata.ranges.P)
        P  = function_value(cellvalues.P, qp, u, extradata.ranges.P)
        ∇P = function_gradient(cellvalues.P, qp, u, extradata.ranges.P)
        ε  = function_symmetric_gradient(cellvalues.u, qp, u, extradata.ranges.u)
        Ed = function_value(cellvalues.P, qp, extradata.Edepol)
        energy += F(P, ε, ∇P, params)* getdetJdV(cellvalues.u, qp)
        # energy += (F(P, ε, ∇P, params) - P ⋅ extradata.force - 1e-5*getdetJdV(cellvalues.u, qp)/(4π*ε₀*35)*P ⋅ Ed) * getdetJdV(cellvalues.u, qp)
        # energy += (F(P, ε, ∇P, params) - P ⋅ extradata.force ) * getdetJdV(cellvalues.u, qp)

    end
    return energy
end
left = Vec{3}((-75.e-9,-25.e-9,-20.e-9))
right = Vec{3}((75.e-9,25.e-9,20.e-9))
modelBFO = LandauModel(BTOParams(), [(:u, 3),(:P, 3, (x) -> (0.48, 0.48, 0.48tanh((-x[1]-x[2])/20e-10)))], (100, 10, 50), left, right, element_potential;lagrangeorder=1);

const testgrad = zeros(length(modelBFO.dofs))
@time ∇F!(testgrad, modelBFO.dofs, modelBFO)
sum(testgrad)
sum(testgrad)
