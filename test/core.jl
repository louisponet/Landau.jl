
using Landau
# import Landau: CellCache, LandauModel, ModelParams

function element_potential(u::AbstractVector{T}, cellvalues, params, extradata) where T
    energy = zero(T)
    # u[1:12] .*= 1e-9
    for qp=1:getnquadpoints(cellvalues.u)
        # δu = function_value(cellvalues.u, qp, u, 1:12)
        P  = function_value(cellvalues.p, qp, u, 13:24)
        ∇P = function_gradient(cellvalues.p, qp, u, 13:24)
        ε  = function_symmetric_gradient(cellvalues.u, qp, u, 1:12)
        energy += (F(P, ε, ∇P, params) - P ⋅ extradata.force) * getdetJdV(cellvalues.u, qp)

    end
    return energy
end

function startingconditionsBFO!(up, dh)
    mid = div(ndofs_per_cell(dh), 2)
    for cell in CellIterator(dh)
        globaldofs = celldofs(cell)
        for i=1:3:mid
            it = div(i, 3) + 1

            up[globaldofs[mid+i]] = -0.53
            up[globaldofs[mid+i+1]] = 0.53
            up[globaldofs[mid+i+2]] =  0.53tanh((-cell.coords[it][1]-cell.coords[it][2])/20e-10)
        end
    end
end

α = Vec{3}((8.78e5 * (300 - 830), 4.71e8, 5.74e8))

params = Landau.ModelParams(α, 5.88e-11,   0.0,         5.88e-11,
                         3.02e11,     1.62e11,    0.68e11,
                         0.035,      -0.0175,      0.02015*4,
                            0., -0., -0.)

left = Vec{3}((-75.e-9,-25.e-9,-2.e-9))
right = Vec{3}((75.e-9,25.e-9,2.e-9))
modelBFO = LandauModel(params, [(:u, 3),(:P, 3)], (10, 10, 5), left, right, element_potential; startingconditions=startingconditionsBFO!);
const test = zeros(length(modelBFO.dofs))
∇F!(test, modelBFO.dofs, modelBFO, 0.0,0.0)
@time ∇F!(test, modelBFO.dofs, modelBFO, 0.0,0.0)
@test norm(test) == 2.0886596543709983e-6
const testhess = create_sparsity_pattern(modelBFO.dofhandler)
∇²F!(testhess, modelBFO.dofs, modelBFO, 0.0,0.0)
@time ∇²F!(testhess, modelBFO.dofs, modelBFO, 0.0,0.0)
@test norm(testhess) == 1.4519342133263154e6
