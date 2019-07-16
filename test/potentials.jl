

function element_potential_bubble(u::AbstractVector{T}, cvu, cvP, params, force) where T
    energy = zero(T)
    for qp=1:getnquadpoints(cvu)
        δu = function_value(cvu, qp, u, 1:12)
        P  = function_value(cvP, qp, u, 13:24)
        ∇P = function_gradient(cvP, qp, u, 13:24)
        ε  = function_symmetric_gradient(cvu, qp, u, 1:12)
        energy += (F(P, ε, ∇P, params) - 1.0e6*P[2]) * getdetJdV(cvu, qp)

    end
    return energy
end

function element_potential_pforce(u::AbstractVector{T}, cvu, cvP, params, force) where T
    energy = zero(T)
    for qp=1:getnquadpoints(cvu)
        P  = function_value(cvP, qp, u, 13:24)
        ∇P = function_gradient(cvP, qp, u, 13:24)
        ε  = function_symmetric_gradient(cvu, qp, u, 1:12)
        energy += (F(P, ε, ∇P, params) - P ⋅ force) * getdetJdV(cvu, qp)
    end
    return energy
end

function element_potential_tip(up::AbstractVector{T}, cvu, cvP, params, force) where T
    energy = zero(T)
    for qp=1:getnquadpoints(cvu)
        P  = function_value(cvP, qp, up, 13:24)
        ∇P = function_gradient(cvP, qp, up, 13:24)
        ε  = function_symmetric_gradient(cvu, qp, up, 1:12)
        energy += (F(P, ε, ∇P, params) - ε[2,2] * force[2]) * getdetJdV(cvu, qp)
    end
    return energy
end

# function element_potential(u::AbstractVector{T}, cvu, cvP, params, force) where T
#     energy = zero(T)
#     for qp=1:getnquadpoints(cvu)
#         δu = function_value(cvu, qp, u, 1:12)
#         P  = function_value(cvP, qp, u, 13:24)
#         ∇P = function_gradient(cvP, qp, u, 13:24)
#         ε  = function_symmetric_gradient(cvu, qp, u, 1:12)
#         energy += (F(P, ε, ∇P, params) - P ⋅ force) * getdetJdV(cvu, qp)
#
#     end
#     return energy
# end

function element_potential(u::AbstractVector{T}, cellvalues, extradata, params) where T
    energy = zero(T)
    for i in extradata.urange
        u[i] *= 3.3e-11
    end
    for qp=1:getnquadpoints(cellvalues.u)
        # δu = function_value(cellvalues.u, qp, u, 1:12)
        P  = function_value(cellvalues.p, qp, u, extradata.Prange)
        ∇P = function_gradient(cellvalues.p, qp, u, extradata.Prange)
        ε  = function_symmetric_gradient(cellvalues.u, qp, u, extradata.urange)
        energy += (F(P, ε, ∇P, params) - P ⋅ extradata.force) * getdetJdV(cellvalues.u, qp)

    end
    return energy
end
