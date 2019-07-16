
using ForwardDiff
import ForwardDiff: GradientConfig, Chunk
using JuAFEM
using Optim, LineSearches
using Tensors

function Fl(P::Vec{3, T}, α::Vec{3}) where T
    P2 = Vec{3, T}((P[1]^2, P[2]^2, P[3]^2))
    return (α[1] * sum(P2) +
           α[2] * (P[1]^4 + P[2]^4 + P[3]^4)) +
           α[3] * ((P2[1] * P2[2]  + P2[2]*P2[3]) + P2[1]*P2[3])
end
@inline Fg(∇P, G) = 0.5(∇P ⊡ G) ⊡ ∇P

F(P, ∇P, params)  = Fl(P, params.α) + Fg(∇P, params.G)

function element_potential(eldofs::AbstractVector{T}, cvP, params) where T
    energy = zero(T)
    for qp=1:getnquadpoints(cvP)
        P  = function_value(cvP, qp, eldofs)
        ∇P = function_gradient(cvP, qp, eldofs)
        energy += F(P, ∇P, params) * getdetJdV(cvP, qp)
    end
    return energy
end

function startingconditions!(dofvector, dofhandler)
    for cell in CellIterator(dofhandler)
        globaldofs = celldofs(cell)
        it = 1
        for i=1:3:length(globaldofs)
            dofvector[globaldofs[i]]   = 0.0
            dofvector[globaldofs[i+1]] = 0.0
            dofvector[globaldofs[i+2]] = 0.83tanh((-cell.coords[it][1]-0.30cell.coords[it][2])/20e-10)
            it += 1
        end
    end
end

struct ModelParams{V, T}
    α::V
    G::T
end

mutable struct CellCache{CV, MP, F <: Function}
    cvP::CV
    params::MP
    elpotential::F
    function CellCache(cvP::CV, params::MP, elpotential::Function) where {CV, MP}
        potfunc = x -> elpotential(x, cvP, params)
        return new{CV, MP, typeof(potfunc)}(cvP, params, potfunc)
    end
end

struct ThreadCache{T, DIM, CC <: CellCache, GC <: GradientConfig}
    dofindices       ::Vector{Int}
    element_dofs     ::Vector{T}
    element_gradient ::Vector{T}
    cellcache        ::CC
    gradconf         ::GC
    element_coords   ::Vector{Vec{DIM, T}}
end
function ThreadCache(dpc::Int, nodespercell, args...)
    dofindices       = zeros(Int, dpc)
    element_dofs     = zeros(dpc)
    element_gradient = zeros(dpc)
    cellcache        = CellCache(args...)
    gradconf         = GradientConfig(nothing, zeros(12), Chunk{12}())
    coords           = zeros(Vec{3}, nodespercell)
    return ThreadCache(dofindices, element_dofs, element_gradient, cellcache, gradconf, coords)
end

mutable struct LandauModel{T, DH <: DofHandler, CH <: ConstraintHandler, TC <: ThreadCache}
    dofs          ::Vector{T}
    dofhandler    ::DH
    boundaryconds ::CH
    threadindices ::Vector{Vector{Int}}
    threadcaches  ::Vector{TC}
end

function LandauModel(α, G, gridsize, left::Vec{DIM, T}, right::Vec{DIM, T}, elpotential) where {DIM, T}
    grid = generate_grid(Tetrahedron, gridsize, left, right)
    questionmark, threadindices = JuAFEM.create_coloring(grid)

    qr  = QuadratureRule{DIM, RefTetrahedron}(2)
    cvP = CellVectorValues(qr, Lagrange{DIM, RefTetrahedron, 1}())

    dofhandler = DofHandler(grid)
    push!(dofhandler, :P, 3)
    close!(dofhandler)

    dofvector = zeros(ndofs(dofhandler))
    startingconditions!(dofvector, dofhandler)

    boundaryconds = ConstraintHandler(dofhandler)
    add!(boundaryconds, Dirichlet(:P, getfaceset(grid, "left"), (x, t) -> [0.53], [3]))
    add!(boundaryconds, Dirichlet(:P, getfaceset(grid, "right"), (x, t) -> [-0.53], [3]))
    close!(boundaryconds)
    update!(boundaryconds, 0.0)

    apply!(dofvector, boundaryconds)

    dpc = ndofs_per_cell(dofhandler)
    cpc = length(grid.cells[1].nodes)
    caches = [ThreadCache(dpc, cpc, copy(cvP), ModelParams(α, G), elpotential) for t=1:Threads.nthreads()]
    return LandauModel(dofvector, dofhandler, boundaryconds, threadindices, caches)
end

function JuAFEM.vtk_save(path, model, dofs=model.dofs)
    vtkfile = vtk_grid(path, model.dofhandler)
    vtk_point_data(vtkfile, model.dofhandler, dofs)
    vtk_save(vtkfile)
end


macro assemble!(innerbody)
    esc(quote
        dofhandler = model.dofhandler
        for indices in model.threadindices
            Threads.@threads for i in indices
                cache  = model.threadcaches[Threads.threadid()]
                cellcache = cache.cellcache
                eldofs = cache.element_dofs
                nodeids = dofhandler.grid.cells[i].nodes
                for j=1:length(cache.element_coords)
                    cache.element_coords[j] = dofhandler.grid.nodes[nodeids[j]].x
                end
                reinit!(cellcache.cvP, cache.element_coords)

                celldofs!(cache.dofindices, dofhandler, i)
                for j=1:length(cache.element_dofs)
                    eldofs[j] = dofvector[cache.dofindices[j]]
                end
                $innerbody
            end
        end
    end)
end

# function ∇F!(∇f::Vector{T}, dofvector::Vector{T}, model) where {T}
#     fill!(∇f, zero(T))
#     dh = model.dofhandler
#     for color in model.threadindices
#         Threads.@threads for t in color
#             tid = Threads.threadid()
#             cache = model.threadcaches[tid]
#             el = cache.element_dofs
#             fe = cache.element_gradient
#             config = cache.gradconf
#             eldofs = cache.dofindices
#             coords = cache.element_coords
#             clo    = cache.cellcache
#             cvP = clo.cvP
#             nodeids = dh.grid.cells[t].nodes
#             for j=1:length(coords)
#                 coords[j] = dh.grid.nodes[nodeids[j]].x
#             end
#             reinit!(cvP, coords)
#             celldofs!(eldofs, dh, t)
#             for i=1:12
#                 el[i] = dofvector[eldofs[i]]
#             end
#             ForwardDiff.gradient!(fe, clo.elpotential, el, config)
#             @inbounds assemble!(∇f, eldofs, fe)
#         end
#     end
# end
function ∇F!(∇f::Vector{T}, dofvector::Vector{T}, model::LandauModel{T}) where T
    fill!(∇f, zero(T))
    @assemble! begin
        ForwardDiff.gradient!(cache.element_gradient, cellcache.elpotential, eldofs, cache.gradconf)

        @inbounds assemble!(∇f, cache.dofindices, cache.element_gradient)
    end
end

function F(dofvector::Vector{T}, model) where T
    outs = fill(zero(T), Threads.nthreads())
    @assemble! begin
        outs[Threads.threadid()] += cache.cellcache.elpotential(eldofs)
    end
    return sum(outs)
end

function minimize!(model; kwargs...)
    dh = model.dofhandler
    uP = model.dofs
    ∇f = zeros(uP)
    function g!(storage, x)
        ∇F!(storage, x, model)
        storage .*= 1e18
        apply_zero!(storage, model.boundaryconds)
    end
    f(x) = F(x, model)*1e18
    od = OnceDifferentiable(f, g!, model.dofs, 0.0, ∇f)

    res = optimize(od, model.dofs, ConjugateGradient(linesearch=BackTracking()), Optim.Options(kwargs...))
    model.dofs .= res.minimizer
    # save("/home/ponet/Documents/PhD/BFODW/up.jld", "up", model.uP)
    return res
    # save("/Users/ponet/Documents/Fysica/PhD/BFODW/up.jld", "up", model.uP)
end


δ(i, j) = i == j ? one(i) : zero(i)
V2T(p11, p12, p44) = Tensor{4, 3}((i,j,k,l) -> p11 * δ(i,j)*δ(k,l)*δ(i,k) + p12*δ(i,j)*δ(k,l)*(1 - δ(i,k)) + p44*δ(i,k)*δ(j,l)*(1 - δ(i,j)))

const G = V2T(5.88e-11, 0.0, 5.88e-11)

left = Vec{3}((-75.e-9,-25.e-9,-2.e-9))
right = Vec{3}((75.e-9,25.e-9,2.e-9))

model = LandauModel(Vec{3}((-4.6534e8, 4.71e8, 5.74e8)), G, (150, 20, 5), left, right, element_potential)

vtk_save(homedir()*"/orig", model)
minimize!(model)
vtk_save(homedir()*"/test", model)
sum(model.dofs)

const test = zeros(model.dofs)

@time ∇F!(test, model.dofs, model)
sum(test)

@time F(model.dofs, model)

for cache in model.threadcaches
    println(pointer_from_objref(cache.cellcache.cvP.))
end
