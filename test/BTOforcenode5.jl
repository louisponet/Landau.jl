# using Pkg
# cd(homedir() * "/.julia/environments/Landau/")
# Pkg.activate(".")
# Pkg.instantiate()
using Landau
import Landau: LandauModel, ModelParams, @assemble!, ∇²F!, ∇F!, F, ThreadCache
using JuAFEM
import JuAFEM: find_field, getorder, getdim
using Base.Threads
using SparseArrays
using Optim, LineSearches
using JLD2
import ForwardDiff: gradient!, hessian!

using DiffResults

gaussian(x::Vec{DIM}, x0::Vec{DIM}, σ²) where DIM = 1/(2π * σ²)^(DIM/2) * ℯ^(-norm(x-x0)^2 / (2σ²))
gaussian(x::T, x0::T, σ²) where T<:AbstractFloat = 1/sqrt(2π * σ²) * ℯ^(-abs(x - x0)^2 / (2σ²))

αT(temp::T) where {T} = Vec{6, T}((-0.27054e8, -6.3817e8, 3.230e8, 7.8936e9, 4.470e9, 4.910e9))

BTOParams(T=300.0) = ModelParams(αT(T), 5.1e-10,    -0.20e-10,  0.20e-10,
                                              2.75e11,     1.79e11,   0.543e11,
                                              0.1104,     -0.0452,    0.0289,
                                              0.3094e-11, -0.279e-11,-0.1335e-11)
function calculateforce!(force::Vector{T}, nodes::Vector{Node{DIM, T}}, forceconstant, center) where {DIM, T}
    fill!(force, zero(T))
    for i = 1:length(nodes)
        force[(i-1)*DIM + 1] = zero(T)
        force[(i-1)*DIM + 2] = zero(T)
        force[(i-1)*DIM + 3] = forceconstant * gaussian(nodes[i].x[1]*1e9, center*1e9, 0.5) * ℯ^(-25.0 + nodes[i].x[3] * 1e10)
    end
end
function calculateforce(nodes::Vector{Node{DIM, T}} , forceconstant, center) where {DIM, T}
    out = zeros(T, length(nodes) * DIM)
    calculateforce!(out, nodes, forceconstant, center)
    return out
end


function force!(cache::ThreadCache{T, DIM} where T, nodeids, force) where DIM
    f = cache.extradata.force
    for (i, id) in enumerate(nodeids)
        f[(i-1)*DIM + 3] = force[(id-1)*DIM + 3]
    end
end

const ε₀ = 8.8541878176e-12
function element_potential(u::AbstractVector{T}, cellvalues, extradata, params) where T
    energy = zero(T)
    for i in extradata.urange
        u[i] *= 3.3e-11
    end
    for qp=1:getnquadpoints(cellvalues.u)
        δu = function_value(cellvalues.u, qp, u, extradata.urange)
        P  = function_value(cellvalues.p, qp, u, extradata.Prange)
        ∇P = function_gradient(cellvalues.p, qp, u, extradata.Prange)
        ε  = function_symmetric_gradient(cellvalues.u, qp, u, extradata.urange)
        δforce = function_value(cellvalues.u, qp, extradata.force)
        # Ed = function_value(cellvalues.p, qp, extradata.Edepol)
        # energy += F(P, ε, ∇P, params)* getdetJdV(cellvalues.u, qp)
        # energy += (F(P, ε, ∇P, params) - P ⋅ extradata.force - 1e-5*getdetJdV(cellvalues.u, qp)/(4π*ε₀*35)*P ⋅ Ed) * getdetJdV(cellvalues.u, qp)
        energy += (F(P, ε, ∇P, params) + δu ⋅ δforce) * getdetJdV(cellvalues.u, qp)

    end
    return energy
end

function startingconditionsBTO!(up, dh)
    mid = div(ndofs_per_cell(dh), 2)
    urange = dof_range(dh, :u)
    Prange = dof_range(dh, :P)
    Pinterp = dh.field_interpolations[find_field(dh, :P)]
    Poffset = JuAFEM.field_offset(dh, :P)
    Porder = getorder(Pinterp)
    Pdim = getdim(Pinterp)
    for cell in CellIterator(dh)
        globaldofs = celldofs(cell)
        for idx in 1:min(getnbasefunctions(Pinterp), length(cell.nodes))
            coord = cell.coords[idx]
            noderange = (Poffset + (idx - 1) * Pdim + 1):(Poffset + idx * Pdim)
            for i in noderange
                orderrenorm = mod1(i, Pdim)
                it = 1
                if orderrenorm == 1
                    up[globaldofs[i]] =0.
                elseif orderrenorm == 2
                    up[globaldofs[i]] = 0.
                elseif orderrenorm == 3
                    up[globaldofs[i]] = 0.265tanh((-coord[1])/20e-10)
                    it += 1
                end
            end
        end
    end
end

BTOModel(T::Float64, args...; kwargs...) = LandauModel(BTOParams(T), args...; kwargs...)

function F(dofvector::Vector{T}, model, forcevec) where T
    outs  = [zero(T) for t=1:nthreads()]
    @assemble! begin
        force!(cache, nodeids, forcevec)
        outs[threadid()] += cache.efunc(eldofs)
    end
    sum(outs)
end

function ∇F!(∇f::Vector{T}, dofvector::Vector{T}, model::LandauModel{T}, forcevec) where {T}
    fill!(∇f, zero(T))
    @assemble! begin
        force!(cache, nodeids, forcevec)
        gradient!(cache.gradient, cache.efunc, eldofs, cache.gradconf)
        @inbounds assemble!(∇f, cache.indices, cache.gradient)
    end
end
function ∇²F!(∇²f::SparseMatrixCSC, dofvector::Vector{T}, model::LandauModel{T}, forcevec) where T
    assemblers = [start_assemble(∇²f) for t=1:nthreads()]
    @assemble! begin
        force!(cache, nodeids, forcevec)
        hessian!(cache.hessresult, cache.efunc, eldofs, cache.hessconf)
        @inbounds assemble!(assemblers[threadid()], cache.indices, DiffResults.hessian(cache.hessresult))
    end
end

function minimize!(model, forceconstant, forcecenter; kwargs...)
    dh = model.dofhandler

    ∇²f = create_sparsity_pattern(dh)
    dofs = model.dofs[1:size(∇²f)[1]]
    ∇f = fill(0.0, length(dofs))
    force = calculateforce(dh.grid.nodes, forceconstant, forcecenter)
    function g!(storage, x)
        ∇F!(storage, x, model, force)
        apply_zero!(storage, model.boundaryconds)
        storage .*= 1e10
    end
    function h!(storage, x)
        ∇²F!(storage, x, model, force)
    end
    function f(x)
        F(x, model, force) * 1e15
    end

    od = TwiceDifferentiable(f, g!, h!, model.dofs, 0.0, ∇f, ∇²f)
    function cb(x)
        vtk_save("/home/ponet/BTO/node6/tempsave$(x.iteration)", model, od.x_f)
        return false
    end
    res = optimize(od, dofs, ConjugateGradient(), Optim.Options(g_tol=1e-15, iterations=10000, allow_f_increases=true, show_trace=true, show_every=20, callback=cb))
    # res = optimize(od, dofs, LBFGS(m=10,alphaguess = InitialQuadratic(),linesearch=BackTracking()), Optim.Options(show_trace=true, show_every=20, g_tol=1e-24, callback=cb, iterations=500, allow_f_increases=true))
    model.dofs[1:length(dofs)] .= res.minimizer
    # vtk_save("/home/ponet/BTO/node7/$(simname)", model, res.minimizer)
    return res
end
left = Vec{3}((-50.e-10, -2.e-10, -25.e-10 ))
right = Vec{3}((50.e-10, 2.e-10, 25.e-10))
model = BTOModel(300.0, [(:u, 3),(:P, 3)], (100, 4, 50), left, right, element_potential; startingconditions=startingconditionsBTO!, lagrangeorder=1);
const gradvec = zeros(length(model.dofs))
@time ∇F!(gradvec, model.dofs, model, 0.0, 0.0)
@time F(model.dofs, model, 0.0, 0.0)
@time ∇F!(gradvec, model.dofs, model, 0.0, 0.0)
gradvec .*= 1e15
sum(gradvec)
sum(gradvec)
# model = BTOModel(300.0, [(:u, 3),(:P, 3)], (100, 4, 50), left, right, element_potential; startingconditions=startingconditionsBTO!, lagrangeorder=1, boundaryconds=[(:u,"bottom", (x,t)-> [0.0] ,[3]),(:P, "bottom", (x,t) -> [0.0, 0.0, 0.0], [1,2,3])]);
@load "/home/ponet/BTO/node6/finaldofs.jld" finishedofs
model.dofs .= finishedofs
for i=2:2:9
	minimize!(model, 1e19, i*1.0e-10)
	@save "/home/ponet/BTO/fixedbottomwall/finaldofs$i.jld" model.dofs
	model.dofs .= finishedofs
end
