using Landau
import Landau: CellCache, LandauModel, ModelParams
using JuAFEM
using Optim, LineSearches

gaussian(x::Vec{DIM}, x0::Vec{DIM}, σ²) where DIM = 1/(2π * σ²)^(DIM/2) * ℯ^(-norm(x-x0)^2 / (2σ²))

αT(temp::T, model) where {T} = (model == :BTO ? Vec{6, T}((-0.27054e8, -6.3817e8, 3.230e8, 7.8936e9, 4.470e9, 4.910e9)) : Vec{3, T}((8.78e5 * (temp - 830), 4.71e8, 5.74e8)))

BTOParams(T=300.0) = ModelParams(αT(T, :BTO), 5.1e-10,    -0.20e-10,  0.20e-10,
                                              2.75e11,     1.79e11,   0.543e11,
                                              0.1104,     -0.0452,    0.0289,
                                              0.3094e-11, -0.279e-11,-0.1335e-11)


BFOParams(T=300.0) = ModelParams(αT(T, :BFO), 5.88e-11,   0.0,         5.88e-11,
                                              3.02e11,     1.62e11,    0.68e11,
                                              0.035,      -0.0175,      0.02015*4,
                                              0., -0., -0.)



function force!(clo, coords::Vec{3, T}, center::Vec{3, T}, prefac::T) where T
    if coords[2] == center[2]
        crel  = coords-center
        a²    = 10.0e-10^2
        r²    = crel[1]^2  + crel[3]^2
        clo.extradata = (force = r² <= a² ? -3prefac/(2pi*a²) * sqrt(1 - r²/a²) * Vec{3,T}((0.0, 1.0, 0.0)) : Vec{3, T}((0.0, 0.0, 0.0)),)
    else
        clo.extradata = (force = Vec{3, T}((0.0, 0.0, 0.0)),)
    end
end

function force!(clo, coords::Vec{3, T}, center::T, prefac::T) where T
#     clo.force = Vec{3, T}((0.0, 1e35, 0.0))
    clo.extradata = (force=prefac*Vec{3, T}((0.0, Landau.gaussian(coords[1]*1.0e9,center,0.5)*1e18ℯ^(-25.0+coords[2]*1e10), 0.0)),)
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


(clo::CellCache)(el) = element_potential(el, clo.cellvalues, clo.parameters, clo.force)

function startingconditionsBFO!(up, dh)
    mid = div(ndofs_per_cell(dh), 2)
    for cell in CellIterator(dh)
        globaldofs = celldofs(cell)
        for i=1:3:mid
            it = div(i, 3) + 1
#             up[globaldofs[i]] = cell.coords[it][1]
#             up[globaldofs[i+1]] = cell.coords[it][2]
#             up[globaldofs[i+2]] = cell.coords[it][3]

            up[globaldofs[mid+i]] = -0.53
            up[globaldofs[mid+i+1]] = 0.53
#             up[globaldofs[mid+i+2]] = 0.53tanh(cell.coords[it][1]/20e-10)
            up[globaldofs[mid+i+2]] =  0.53tanh((-cell.coords[it][1]-cell.coords[it][2])/20e-10)
        end
    end
end

function startingconditionsBTO!(up, dh)
    mid = div(ndofs_per_cell(dh), 2)
    for cell in CellIterator(dh)
        globaldofs = celldofs(cell)
        for i=1:3:mid
            it = div(i, 3) + 1
            up[globaldofs[mid+i]] = 0.
            up[globaldofs[mid+i+1]] = 0.29tanh((15e-9 - sqrt(cell.coords[it][1]^2-cell.coords[it][3]^2))/20e-10)
            up[globaldofs[mid+i+2]] =0.0
        end
    end
end

LandauModel(T::Float64, model, args...; kwargs...) = (model == :BTO ? LandauModel(BTOParams(T), args...; kwargs...) : LandauModel(BFOParams(T), args...; kwargs...))

left = Vec{3}((-75.e-9,-25.e-9,-2.e-9))
right = Vec{3}((75.e-9,25.e-9,2.e-9))
modelBFO = LandauModel(300.0, :BFO, [(:u, 3),(:P, 3)], (10, 10, 5), left, right, element_potential; startingconditions=startingconditionsBFO!);

function minimize!(model; kwargs...)
    dh = model.dofhandler
    dofdict = Landau.periodicmap_dim3(dh, "bottom", "top", left[3], right[3])


    dofs = model.dofs
    ∇f = fill(0.0, length(dofs))
    ∇²f = create_sparsity_pattern(dh)
    function preg!(storage, x)
        ∇F!(storage, x, model, 0.0, 0.0)
        for cell in CellIterator(dh)
            for i in celldofs(cell)[dof_range(dh, :P)]
                storage[i] = 0.0
            end
        end
        apply_zero!(storage, model.boundaryconds)
    end
    function g!(storage, x)
        # for (d, dn) in dofdict
        #     x[d] = x[dn]
        # end
        ∇F!(storage, x, model, 0.0, 0.0)
        apply_zero!(storage, model.boundaryconds)
    end
    function h!(storage, x)
        # for (d, dn) in dofdict
        #     x[d] = x[dn]
        # end
        ∇²F!(storage, x, model, 0.0,0.0)
        apply!(storage, model.boundaryconds)
    end
    function f(x)
        # for (d, dn) in dofdict
        #     x[d] = x[dn]
        # end
        F(x, model, 0.0,0.0) * 1e15
    end

    # preminimizer = OnceDifferentiable(f, preg!, model.dofs, 0.0, ∇f)
    od = TwiceDifferentiable(f, g!, h!, model.dofs, 0.0, ∇f, ∇²f)
    function cb(x)

        vtk_save("/home/ponet/tempsave$(x.iteration)", model, od.x_f)
        return false
    end

    # this way of minimizing is only beneficial when the initial guess is completely off,
    # then a quick couple of ConjuageGradient steps brings us easily closer to the minimum.
    # res = optimize(od, model.dofs, ConjugateGradient(linesearch=BackTracking()), Optim.Options(show_trace=true, show_every=1, g_tol=1e-20,iterations=1000))
    # model.dofs .= res.minimizer
    #to get the final convergence, Newton's method is more ideal since the energy landscape should be almost parabolic
    res = optimize(od, model.dofs, Newton(linesearch=BackTracking()), Optim.Options(show_trace=true, show_every=1, g_tol=1e-20, callback=cb))
    model.dofs .= res.minimizer
    return res
end

minimize!(modelBFO)
dofdict = Landau.periodicmap_dim3(modelBFO.dofhandler, "bottom", "top", left[3], right[3])
