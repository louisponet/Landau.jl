# using Pkg
# cd(homedir() * "/.julia/environments/Landau/")
# Pkg.activate(".")
using Landau
import Landau: @assemble!, ∇²F!, ∇F!, F, ThreadCache
using Base.Threads
using JuAFEM
import JuAFEM: getfaceset
using Optim, LineSearches
using JLD2
import ForwardDiff: gradient!, hessian!
using SparseArrays

# using DiffResults

gaussian(x::Vec{DIM}, x0::Vec{DIM}, σ²) where DIM = 1/(2π * σ²)^(DIM/2) * ℯ^(-norm(x-x0)^2 / (2σ²))
gaussian(x::T, x0::T, σ²) where T<:AbstractFloat = 1/sqrt(2π * σ²) * ℯ^(-abs(x - x0)^2 / (2σ²))

αT(temp::T) where {T} = Vec{6, T}((-0.27054e8, -6.3817e8, 3.230e8, 7.8936e9, 4.470e9, 4.910e9))

BTOParams(T=300.0) = GLDparameters(αT(T), 5.1e-10,    -0.20e-10,  0.20e-10,
                                              2.75e11,     1.79e11,   0.543e11,
                                              0.1104,     -0.0452,    0.0289,
                                              0.3094e-11, -0.279e-11,-0.1335e-11)
const ε₀ = 8.8541878176e-12

const tipradius   = 10.0e-9
const loadforce   = 1.0e-6
# const xsize       = 100.0
# const ysize       = 4.0
# const zsize       = 50.0
const xsize       = 20.0
const ysize       = 20.0
const zsize       = 20.0
const elementsize = 1.0
const nxels       = round(Int, xsize/elementsize)
const nyels       = round(Int, ysize/elementsize)
const nzels       = round(Int, zsize/elementsize)

#minimization related constants
const uprefac       = 3.3e-11
const ∇Fprefac      = 1e10
const Fprefac       = 1e15
const gtol          = 1e-13
const minmethod     = LBFGS
const lsearchmethod = HagerZhang


function calculateforce!(force, nodes::Vector{Node{DIM, T}}, prefac, center::Vec{DIM, T}) where {DIM,T}
    fill!(force, zero(T))
    for i = 1:length(nodes)
        #if nodes[i].x[3] == center[3]
            crel  = nodes[i].x-center
            a²    = (3/4 * prefac * tipradius/2.75e11)^(2/3)
            r²    = crel[1]^2
            force[(i-1)*DIM + 3] = r² <= a² ? -3prefac/(2pi*a²) * sqrt(1 - r²/a²)/nzels * 1/(1+crel[3]^2/a²) : zero(T)
        #end
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

function element_potential(u::AbstractVector{T}, cellvalues, extradata, params) where T
    energy = zero(T)
    for i in extradata.ranges.u
        u[i] *= uprefac
    end
    for qp=1:getnquadpoints(cellvalues.u)
        #δu = function_value(cellvalues.u, qp, u, extradata.ranges.u)
        P  = function_value(cellvalues.P, qp, u, extradata.ranges.P)
        ∇P = function_gradient(cellvalues.P, qp, u, extradata.ranges.P)
        ε  = function_symmetric_gradient(cellvalues.u, qp, u, extradata.ranges.u)
        δforce = function_value(cellvalues.u, qp, extradata.force)
        # energy += (F(P, ε, ∇P, params)) * getdetJdV(cellvalues.u, qp)
        energy += (F(P, ε, ∇P, params) -  ε[3,3] * δforce[3]) * getdetJdV(cellvalues.u, qp)

    end
    return energy
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

function periodicmap_dim1u(dofhandler, faceset1, faceset2, edgecoord1, edgecoord2)
    mid = div(ndofs_per_cell(dofhandler), 2)
    dofdict_ = Dict{Int, Int}()
    ci1 = CellIterator(dofhandler)
    ci2 = deepcopy(ci1)
    dofs1 = zeros(Int, ndofs_per_cell(dofhandler))
    dofs2 = zeros(Int, ndofs_per_cell(dofhandler))
    for (cellidx1, faceidx1) in getfaceset(dofhandler.grid, faceset1), (cellidx2, faceidx2) in getfaceset(dofhandler.grid, faceset2)
        reinit!(ci1, cellidx1)
        reinit!(ci2, cellidx2)
        for (ic1, coord1) in enumerate(ci1.coords), (ic2, coord2) in enumerate(ci2.coords)
            if coord1[3] == coord2[3] && coord1[2] == coord2[2] && coord1[1] == edgecoord1 && coord2[1] == edgecoord2
                celldofs!(dofs1, ci1)
                celldofs!(dofs2, ci2)
                for (r1, r2) in zip((ic1-1)*3+1:ic1*3, (ic2-1)*3+1:ic2*3)
                    dofdict_[dofs2[r2]] = dofs1[r1]
                end
                # for (r1, r2) in zip(mid+(ic1-1)*3+1:mid+ic1*3, mid+(ic2-1)*3+1:mid+ic2*3)
                #     dofdict_[dofs2[r2]] = dofs1[r1]
                # end
            end
        end
    end
    dofdict_
end
function periodicmap_dim2uP(dofhandler, faceset1, faceset2, edgecoord1, edgecoord2)
    mid = div(ndofs_per_cell(dofhandler), 2)
    dofdict_ = Dict{Int, Int}()
    ci1 = CellIterator(dofhandler)
    ci2 = deepcopy(ci1)
    dofs1 = zeros(Int, ndofs_per_cell(dofhandler))
    dofs2 = zeros(Int, ndofs_per_cell(dofhandler))
    for (cellidx1, faceidx1) in getfaceset(dofhandler.grid, faceset1), (cellidx2, faceidx2) in getfaceset(dofhandler.grid, faceset2)
        reinit!(ci1, cellidx1)
        reinit!(ci2, cellidx2)
        for (ic1, coord1) in enumerate(ci1.coords), (ic2, coord2) in enumerate(ci2.coords)
            if coord1[1] == coord2[1] && coord1[3] == coord2[3] && coord1[2] == edgecoord1 && coord2[2] == edgecoord2
                celldofs!(dofs1, ci1)
                celldofs!(dofs2, ci2)
                for (r1, r2) in zip((ic1-1)*3+1:ic1*3, (ic2-1)*3+1:ic2*3)
                    dofdict_[dofs2[r2]] = dofs1[r1]
                end
                for (r1, r2) in zip(mid+(ic1-1)*3+1:mid+ic1*3, mid+(ic2-1)*3+1:mid+ic2*3)
                    dofdict_[dofs2[r2]] = dofs1[r1]
                end
            end
        end
    end
    dofdict_
end


function push_tip!(dofvec::Vector{T}, dofnodes, center, radius=tipradius) where T
    radius² = radius^2
    constindices = Int[]
    for d in dofnodes
        realpos = d.coord + uprefac * Vec{3, T}((dofvec[d.dofs.u]...,))
        r² = (realpos[1]-center[1])^2 + (realpos[3]-center[3])^2
        if r² < radius²
            # println(-sqrt(radius² - (realpos[1]- center[1])^2) + center[3]- d.coord[3])
            dofvec[d.dofs.u[3]] = (-sqrt(radius² - (realpos[1]- center[1])^2) + center[3]- d.coord[3])/uprefac
            push!(constindices, d.dofs.u[3])
        end
    end
    return constindices
end


function minimize!(model, stencils, center; kwargs...)
    dh = model.dofhandler

    dirname = String(@__DIR__) *"/job$(Int(center[1]/elementsize))/"
    mkdir(dirname)
    ∇²f = create_sparsity_pattern(dh)
    dofs = model.dofs[1:size(∇²f)[1]]
    ∇f = fill(0.0, length(dofs))
    # force = calculateforce(dh.grid.nodes, forceconstant, Vec{3, Float64}((i * elementsize, 0.0, zsize/2)))
    dnodes   = dofnodes(dh)
    topnodes = filter(x -> x.coord[3] == zsize/2, dnodes)
    constindices = push_tip!(dofs, topnodes, center)
    function gpre!(storage, x)
        # ∇F!(storage, x, model, force)
        ∇F!(storage, x, model)
        apply_zero!(storage, model.boundaryconds)
        for d in dnodes
            pdof = d.dofs.P
            storage[pdof] .= 0.0
        end
        storage .*= ∇Fprefac
    end
    function g!(storage, x)
        # ∇F!(storage, x, model, force)
        ∇F!(storage, x, model)
        apply_zero!(storage, model.boundaryconds)
        storage[constindices] .*= 0.0
        #for d in dnodes
        #    pdof = d.dofs.P
        #    storage[pdof] .= 0.0
        #end
        storage .*= ∇Fprefac
    end
    function h!(storage, x)
        ∇²F!(storage, x, model, force)
    end
    function f(x)
        # F(x, model, force) * Fprefac
        F(x, model) * Fprefac
    end

    # od = TwiceDifferentiable(f, gpre!, h!, dofs, 0.0, ∇f, ∇²f)
    function cb(x)
        vtk_save(dirname*"tempsave$(x.iteration)", model, od.x_f)
        return false
    end
    # res = optimize(od, dofs, minmethod(linesearch=lsearchmethod()), Optim.Options(g_tol=gtol, iterations=10000, allow_f_increases=true,show_trace=true,show_every=50,callback=cb))
    # model.dofs[1:length(dofs)] .= res.minimizer
    od = TwiceDifferentiable(f, g!, h!, dofs, 0.0, ∇f, ∇²f)
    res = optimize(od, dofs, minmethod(linesearch=lsearchmethod()), Optim.Options(g_tol=gtol, iterations=10000, allow_f_increases=true,show_trace=true,show_every=50,callback=cb))
    model.dofs[1:length(dofs)] .= res.minimizer
    # finishedofs = copy(model.dofs)
    # savepath = dirname*"finaldofs.jld"
    # @save savepath finishedofs
    return res
end

left = Vec{3}((-xsize, -ysize, -zsize))/2
right = Vec{3}((xsize, ysize, zsize))/2
# model = BTOModel(300.0, [(:u, 3),(:P, 3, (x)-> (0.0, 0.0, -0.265tanh((x[1])/20e-10)))], (nxels, nyels, nzels), left, right, element_potential; lagrangeorder=1, boundaryconds=[(:u, "bottom",(x,t)->[0.0],[3]),(:P, "bottom", (x,t)->[0.0], [3])]);
model = BTOModel(300.0, [(:u, 3),(:P, 3, (x)-> (0.0, 0.0, -0.265tanh((x[1])/20e-10)))], (20, 20, 20), left, right, element_potential; lagrangeorder=1, boundaryconds=[(:u, "bottom",(x,t)->[0.0],[3]),(:P, "bottom", (x,t)->[0.0], [3])]);
stencils = construct_stencils(model.dofhandler, (21, 21, 21), (-18:4:18, -18:4:18, -18:4:18), 20)

size(model.dofhandler.grid.nodes)
using Plots
stencils[20]
plotStencil(stencils[end])
stencils[end]
plot
toplotx = Float64[]
toploty = Float64[]
for i=1:length(stencils[20].connections)
    Rrel = stencils[4].Rhat[i] * stencils[4].R⁻³[i]^(-1/3)
    push!(toplotx, Rrel[1])
    push!(toploty, Rrel[2])
end
stencils[4]
toplotx
scatter(toplotx, toploty)
plotStencil(stencils[4])
#model = BTOModel(300.0, [(:u, 3),(:P, 3, (x)-> (0.0, 0.0, -0.265tanh((x[1])/20e-10)))], (nxels, nyels, nzels), left, right, element_potential; lagrangeorder=1);
#model = BTOModel(300.0, [(:u, 3),(:P, 3, (x)-> (0.0, 0.0, -0.265tanh((x[1])/20e-10)))], (nxels, nyels, nzels), left, right, element_potential; lagrangeorder=1, boundaryconds=[(:u, "bottom",(x,t)->[0.0],[3])]);

@load String(@__DIR__)*"/finaldofs.jld" finishedofs
# @load homedir()*"/Documents/PhD/BTO/nm0.1force/finaldofs.jld" finishedofs
model.dofs .= finishedofs
const dd =periodicmap_dim1u(model.dofhandler, "left", "right", left[1], right[1])
merge!(dd, periodicmap_dim2uP(model.dofhandler, "front", "back", left[2], right[2]))

for i=1:length(model.dofhandler.cell_dofs)
    d = model.dofhandler.cell_dofs[i]
    dn = get(dd, d, d)
    model.dofhandler.cell_dofs[i] = dn
end
for j=-5:5
    dnodes   = dofnodes(model.dofhandler)
    topnodes = filter(x -> x.coord[3] == zsize/2, dnodes)
    umax = uprefac * maximum(model.dofs[map(x->x.dofs.u[3], topnodes)])
    umin = uprefac * minimum(model.dofs[map(x->x.dofs.u[3], topnodes)])
    ustep = (umax-umin) /2
    Fprev = F(model.dofs, model)
    toplotF = Float64[Fprev]
    Fnew  = Fprev
    i = 1
    while abs(Fnew - Fprev)/ustep < loadforce
        i += 1
        res = minimize!(model, Vec{3, Float64}((j*elementsize, 0.0, zsize/2 + tipradius + i*ustep)))
        Fnew = res.minimum / Fprefac
        push!(toplotF, Fnew)
    end
    @save String(@__DIR__)*"/job$(j)/toplotF.jld" toplotF
    finishedofs = copy(model.dofs)
    @save String(@__DIR__)*"/job$(j)/finishedofs.jld" finishedofs
end
dnodes   = dofnodes(model.dofhandler)
topnodes = filter(x -> x.coord[3] == zsize/2, dnodes)
tdofs = copy(model.dofs)
push_tip!(tdofs, topnodes, Vec{3, Float64}((0.0, 0.0, 25e-9 + 3.0e-9+ 7.0e-9)))
finishedofs = copy(model.dofs)
@save homedir()*"/Documents/PhD/BTO/nm0.1force/finaldofs.jld" finishedofs
for i=-2:0
     minimize!(model, loadforce, i)
     model.dofs .= finishedofs
end
