using Landau
import Landau: LandauModel, ModelParams, @assemble!, ∇²F!, ∇F!, F
using JuAFEM
import JuAFEM: find_field, getorder, getdim
using Optim, LineSearches
using Base.Threads
using Base.Cartesian
using SparseArrays
using LinearAlgebra
using Plots

gaussian(x::Vec{DIM}, x0::Vec{DIM}, σ²) where DIM = 1/(2π * σ²)^(DIM/2) * ℯ^(-norm(x-x0)^2 / (2σ²))

αT(temp::T, model) where {T} = (model == :BTO ? Vec{6, T}((-0.27054e8, -6.3817e8, 3.230e8, 7.8936e9, 4.470e9, 4.910e9)) : Vec{3, T}((8.78e5 * (temp - 830), 4.71e8, 5.74e8)))

BTOParams(T=300.0) = ModelParams(αT(T, :BTO), 5.1e-10,    -0.20e-10,  0.20e-10,
                                              2.75e11,     1.79e11,   0.543e11,
                                              0.1104,     -0.0452,    0.0289,
                                              0.3094e-11, -0.279e-11,-0.1335e-11)


BFOParams(T=300.0) = ModelParams(αT(T, :BFO), 5.88e-11,   0.0,         5.88e-11,
                                              3.02e11,     1.62e11,    0.68e11,
                                              0.035,      -0.0175,      0.02015,
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
  clo.extradata = (force=prefac*Vec{3, T}((0.0, Landau.gaussian(Vec{1, T}((coords[1],))*1.0e9,Vec{1, T}((center,)),0.5)*1e18ℯ^(-25.0+coords[2]*1e10), 0.0)), clo.extradata...)
end
const ε₀ = 8.8541878176e-12
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
        Ed = function_value(cellvalues.p, qp, extradata.Edepol)
        energy += F(P, ε, ∇P, params)* getdetJdV(cellvalues.u, qp)
        # energy += (F(P, ε, ∇P, params) - P ⋅ extradata.force - 1e-5*getdetJdV(cellvalues.u, qp)/(4π*ε₀*35)*P ⋅ Ed) * getdetJdV(cellvalues.u, qp)
        # energy += (F(P, ε, ∇P, params) - P ⋅ extradata.force ) * getdetJdV(cellvalues.u, qp)

    end
    return energy
end

function dof2node(dh::DofHandler)
    cdofs = dofnodes(dh)
    dof2nodevec = Vector{Int}(undef, ndofs(dh))
    for (i, d) in enumerate(cdofs)
        for fdofs in d.dofs
            dof2nodevec[fdofs] .= i
        end
    end
    return dof2nodevec
end



# function periodicmap(dofhandler::DofHandler{DIM}, dimension, fields...) where DIM
#     @assert dimension <= DIM
#     if dimension == 1
#         facesets = ["left", "right"]
#     elseif dimension == 2
#         facesets = ["front", "back"]
#     elseif dimension == 3
#         facesets = ["bottom", "top"]
#     end
#     bhandler = ConstraintHandler(dofhandler)
#     for faceset in facesets
#         for field in fields
#             add!(bhandler, Dirichlet(field, getfaceset(dofhandler.grid, faceset), (x,t) -> 0, collect(1:DIM)))
#         end
#     end
#     close!(bhandler)

function periodicmap_dim3(dofhandler, faceset1, faceset2, edgecoord1, edgecoord2)
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
            if coord1[1] == coord2[1] && coord1[2] == coord2[2] && coord1[3] == edgecoord1 && coord2[3] == edgecoord2
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




function minimize!(model; kwargs...)
    dh = model.dofhandler
    # dofdict =periodicmap_dim3(dh, "bottom", "top", left[3], right[3])

    ∇²f = create_sparsity_pattern(dh)
    dofs = model.dofs[1:size(∇²f)[1]]
    ∇f = fill(0.0, length(dofs))
    println(length(dofs))
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
        ∇F!(storage, x, model, 0.0, 0.0, )
        storage .*= 1e15
        # apply_zero!(storage, model.boundaryconds)
    end
    function h!(storage, x)
        # for (d, dn) in dofdict
        #     x[d] = x[dn]
        # end
        ∇²F!(storage, x, model, 0.0,0.0, )
        storage .*= 1e-4
        # apply!(storage, model.boundaryconds)
    end
    function f(x)
        # for (d, dn) in dofdict
        #     x[d] = x[dn]
        # end
        F(x, model, 0.0,0.0, ) * 1e15
    end

    # preminimizer = OnceDifferentiable(f, preg!, model.dofs, 0.0, ∇f)
    od = TwiceDifferentiable(f, g!, h!, model.dofs, 0.0, ∇f, ∇²f)
    function cb(x)

        # vtk_save("/home/ponet/tempsave$(x.iteration)", model, od.x_f)
        return false
    end
    res = optimize(od, dofs, LBFGS(linesearch=BackTracking()), Optim.Options(show_trace=true, show_every=100, g_tol=1e-24, iterations=500, allow_f_increases=true))
    model.dofs[1:length(dofs)] .= res.minimizer
    return res
end
function minimize!(model, stencils; kwargs...)
    dh = model.dofhandler
    stencils = construct_stencils(dh, 3, 3, 4e-9)
    # dofdict =periodicmap_dim3(dh, "bottom", "top", left[3], right[3])

    ∇²f = create_sparsity_pattern(dh)
    dofs = model.dofs[1:size(∇²f)[1]]
    ∇f = fill(0.0, length(dofs))
    println(length(dofs))
    function preg!(storage, x)
        ∇F!(storage, x, model, 0.0, 0.0, stencils)
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
        ∇F!(storage, x, model, 0.0, 0.0, stencils)
        storage .*= 1e15
        # apply_zero!(storage, model.boundaryconds)
    end
    function h!(storage, x)
        # for (d, dn) in dofdict
        #     x[d] = x[dn]
        # end
        ∇²F!(storage, x, model, 0.0,0.0, stencils)
        storage .*= 1e-4
        # apply!(storage, model.boundaryconds)
    end
    function f(x)
        # for (d, dn) in dofdict
        #     x[d] = x[dn]
        # end
        F(x, model, 0.0,0.0, stencils) * 1e15
    end

    # preminimizer = OnceDifferentiable(f, preg!, model.dofs, 0.0, ∇f)
    od = TwiceDifferentiable(f, g!, h!, model.dofs, 0.0, ∇f, ∇²f)
    function cb(x)

        # vtk_save("/home/ponet/tempsave$(x.iteration)", model, od.x_f)
        return false
    end
    res = optimize(od, dofs, ConjugateGradient(linesearch=BackTracking()), Optim.Options(show_trace=true, show_every=100, g_tol=1e-24, callback=cb, iterations=1000, allow_f_increases=true))
    model.dofs[1:length(dofs)] .= res.minimizer
    return res
end

minimize!(modelBFO)
function calculateEdep!(efield, stencil::Stencil{DIM, T}, alldofs, localdofs) where {DIM, T}
    fill!(efield, zero(T))
    for i = 1:length(stencil.connections)
        factor = zero(T)
        for j=1:DIM
            factor += stencil.Rhat[i][j] * alldofs[stencil.dofs[i].P[j]]
        end
        factor *= 3
        for j=1:DIM
            efield[j] += stencil.R⁻³[i] * (factor * stencil.Rhat[i][j] - alldofs[stencil.dofs[i].P[j]])
        end
    end
    efield .*= stencil.volume
end
function calculateEdep!(Edep, stencils::Vector{<:Stencil{DIM, T}}, alldofs) where {DIM, T}
    localdofs = [zeros(T, DIM) for i=1:nthreads()]
    @threads for i=1:length(stencils)
        calculateEdep!(Edep[i], stencils[i], alldofs, localdofs[threadid()])
    end
end
function calculateEdep(stencils::Vector{<:Stencil{DIM, T}}, alldofs) where {DIM, T}
    outfield = [zeros(T, DIM) for i=1:length(stencils)]
    calculateEdep!(outfield, stencils, alldofs)
    return outfield
end

function calculateEdep(dofnodes, stencil::Stencil{DIM ,T}, alldofs) where {DIM, T}
    totnodes = length(dofnodes)
    outfield = [zeros(T, DIM) for i=1:totnodes]
    # localdofs = [zeros(T, DIM) for i=1:nthreads()]
    for i=1:length(outfield)
        for (conn, Rhat, R⁻³) in zip(stencil.connections, stencil.Rhat, stencil.R⁻³)
            if i + conn > totnodes || i + conn <= 0
                continue
            end
            n2 = dofnodes[i + conn]
            factor = zero(T)
            for d=1:DIM
                factor += Rhat[d] * alldofs[n2.dofs.P[d]]
            end
            factor *= 3
            for d=1:DIM
                # outfield[i][d] += R⁻³[d] * (factor * Rhat[d] - alldofs[n2.dofs.P[d]])
                outfield[i][d] += R⁻³*(factor * Rhat[d] - alldofs[n2.dofs.P[d]])
            end
        end
    end
    outfield .*= stencil.volume
end

newedep = calculateEdep(cnodes, stencil, modelBFO.dofs)

oldedep = calculateEdep(stencils, modelBFO.dofs)

edepfield = [zeros(Float64, 3) for i=1:length(stencils)]
@time calculateEdep!(edepfield, stencils, modelBFO.dofs)
sum(edepfield)
function plotEdep(nodes, Edep::Vector{Vector{T}}, zcoord) where T
    toplotx = T[]
    toploty = T[]
    toheatmapx = T[]
    toheatmapy = T[]
    toheatmapz = T[]
    for (node, Ed) in zip(nodes, Edep)
        if node.x[3] == zcoord
            push!(toplotx, node.x[1])
            push!(toploty, node.x[2])
            push!(toheatmapx, Ed[1])
            push!(toheatmapy, Ed[2])
            push!(toheatmapz, Ed[3])
        end
    end
    plot(heatmap(reshape(toheatmapx, length(unique(toplotx)), length(unique(toploty)))', title="x"),
        heatmap(reshape(toheatmapy, length(unique(toplotx)), length(unique(toploty)))', title="y"),
        heatmap(reshape(toheatmapz, length(unique(toplotx)), length(unique(toploty)))', title="z"))
end

# modelBFO.dofhandler.grid.nodes[1]
# stencils = construct_stencils(modelBFO.dofhandler, 30e-9)

# @time depfield = calculateEdep(stencils, modelBFO.dofs)

# @time F(modelBFO.dofs, modelBFO, depfield)
# F(modelBFO.dofs, modelBFO, depfield)

# plotEdep( modelBFO.dofhandler.grid.nodes, oldedep, 0.0e-9)
# plotEdep( modelBFO.dofhandler.grid.nodes, newedep, 0.0e-9)

function fibonacci_sphere(samples)
    points = []
    offset = 2/samples
    increment = π * (3. - sqrt(5.))

    for i=0:samples-1
        y = ((i * offset)-1.0) + (offset / 2)
        r = sqrt(1 - y^2)

        ϕ = ((i + 1) % samples) * increment

        x = cos(ϕ) * r
        z = sin(ϕ) * r

        push!(points, 5*Vec{3, Float64}((x,y,z)))
    end
    return points
end

left = Vec{3}((-75.e-9,-25.e-9,-20.e-9))
right = Vec{3}((75.e-9,25.e-9,20.e-9))
modelBFO = LandauModel(300.0, :BFO, [(:u, 3),(:P, 3)], (4, 4, 4), left, right, element_potential; startingconditions=startingconditionsBFO!, lagrangeorder=1);

const dd =periodicmap_dim3(modelBFO.dofhandler, "bottom", "top", left[3], right[3])
for i=1:length(modelBFO.dofhandler.cell_dofs)
    d = modelBFO.dofhandler.cell_dofs[i]
    dn = get(dd, d, d)
    modelBFO.dofhandler.cell_dofs[i] = dn
end
fullstencils = construct_stencils(modelBFO.dofhandler, 20e-9)
function refinecells(cells, dnodes, grid, thresholds, alldofs)
    for ic = 1:length(cells)
        found = false
        for edge in JuAFEM.edges(cells[ic])
            # println(edge)
            if edge[1] > length(dnodes) || edge[2] > length(dnodes)
                continue
            end
            dofs1 = dnodes[edge[1]].dofs
            dofs2 = dnodes[edge[2]].dofs
            for (n, v) in pairs(thresholds)
                if norm(getindex.((alldofs,), dofs1[n]) .- getindex.((alldofs,), dofs2[n])) > v
                    newid = length(grid.nodes) + 1
                    # println(edge)
                    eid1 = findfirst(x -> x == edge[1], cells[ic].nodes)
                    eid2 = findfirst(x -> x == edge[2], cells[ic].nodes)
                    cells[ic] = Tetrahedron(Base.setindex(cells[ic].nodes, newid, eid2))
                    push!(grid.cells, Tetrahedron(Base.setindex(cells[ic].nodes, newid, eid1)))
                    found = true
                    push!(grid.nodes, Node((dnodes[edge[1]].coord + dnodes[edge[2]].coord)/2))
                    refinecells([cells[ic], grid.cells[end]], dnodes, grid, thresholds, alldofs)
                    break
                    # push!(edgestochange, edge)
                end
            end
            if found
                break
            end
        end
    end
end
function refinegrid(model, thresholds::NamedTuple, alldofs)
    dnodes = dofnodes(model.dofhandler)
    grid = model.dofhandler.grid
    refinecells(grid.cells, dnodes, grid, thresholds, alldofs)

end
function refinegrid(model, thresholds::NamedTuple, alldofs)
    dnodes = dofnodes(model.dofhandler)
    grid = model.dofhandler.grid
    cells = grid.cells
    foundedges = 3000
    newnodes = Node{3, Float64}[]
    while foundedges > 0
        foundedges = 0
        for ic = 1:length(cells)
            found = false
            for edge in JuAFEM.edges(cells[ic])
                if edge[1] > length(dnodes) || edge[2] > length(dnodes)
                    continue
                end
                dofs1 = dnodes[edge[1]].dofs
                dofs2 = dnodes[edge[2]].dofs
                for (n, v) in pairs(thresholds)
                    if norm(getindex.((alldofs,), dofs1[n]) .- getindex.((alldofs,), dofs2[n])) > v
                        foundedges += 1
                        found = true
                        newnode = Node((dnodes[edge[1]].coord + dnodes[edge[2]].coord)/2)
                        # newid = length(grid.nodes) + 1
                        newid = findfirst(x->x==newnode, grid.nodes)
                        if newid == nothing
                            newid = length(grid.nodes) + 1
                            push!(grid.nodes, newnode)
                        end
                        eid1 = findfirst(x -> x == edge[1], cells[ic].nodes)
                        eid2 = findfirst(x -> x == edge[2], cells[ic].nodes)
                        cells[ic] = Tetrahedron(Base.setindex(cells[ic].nodes, newid, eid2))
                        push!(grid.cells, Tetrahedron(Base.setindex(cells[ic].nodes, newid, eid1)))
                        # refinecells([cells[ic], grid.cells[end]], dnodes, grid, thresholds, alldofs)
                        break
                        # push!(edgestochange, edge)
                    end
                end
                if found
                    break
                end
            end
        end
    end
    grid.nodes = unique(grid.nodes)
end

function findedges(model, thresholds::NamedTuple, alldofs)
    dnodes = dofnodes(model.dofhandler)
    grid = model.dofhandler.grid
    cells = grid.cells
    edgestochange = Tuple{Int, Int}[]
    edgetocell = Vector{Int}[]
    newnodeids = Int[]
    newnodes = Node[]
    for ic = 1:length(cells)
        for edge in JuAFEM.edges(cells[ic])
            dofs1 = dnodes[edge[1]].dofs
            dofs2 = dnodes[edge[2]].dofs
            for (n, v) in pairs(thresholds)
                if any(abs.(getindex.((alldofs,), dofs1[n]) .- getindex.((alldofs,), dofs2[n])) .> v)
                    edgeid = findfirst(x -> x == edge, edgestochange)
                    edgeidrev = findfirst(x-> x == reverse(edge), edgestochange)
                    if edgeid != nothing
                        push!(edgetocell[edgeid], ic)
                        break
                    end
                    if edgeidrev != nothing
                        push!(edgetocell[edgeidrev], ic)
                        break
                    end
                    push!(edgestochange, edge)
                    push!(edgetocell, [ic])
                    break
                end
            end
        end
    end
    return edgestochange, edgetocell
end

function refinecell(cell, newcells, edgestochange, dnodes, totnodes)
    if !any(JuAFEM.edges(cell) .∈ (edgestochange,))
        push!(newcells, cell)
    end
    for edge in JuAFEM.edges(cell)
        if edge ∈ edgestochange
            newnode = Node((dnodes[edge[1]].coord + dnodes[edge[2]].coord)/2)
            # newid = length(grid.nodes) + 1
            newid = findfirst(x->x == newnode, totnodes)
            if newid == nothing
                newid = length(totnodes) + 1
                push!(totnodes, newnode)
            end
            eid1 = findfirst(x -> x == edge[1], cell.nodes)
            eid2 = findfirst(x -> x == edge[2], cell.nodes)
            newcell1 = Tetrahedron(Base.setindex(cell.nodes, newid, eid2))
            newcell2 = Tetrahedron(Base.setindex(cell.nodes, newid, eid1))
            refinecell(newcell1, newcells, edgestochange, dnodes, totnodes)
            refinecell(newcell2, newcells, edgestochange, dnodes, totnodes)
            break
        end
    end
end

function refinegrid(model, thresholds::NamedTuple, alldofs)
    dnodes = dofnodes(model.dofhandler)
    grid = model.dofhandler.grid
    newcells = Tetrahedron[]
    totnodes = grid.nodes
    edgestochange, edgetocell = findedges(model, thresholds, alldofs)
    for cell in grid.cells
        refinecell(cell, newcells, edgestochange, dnodes, totnodes)
    end
    grid.cells = newcells
end

function refinecell(cell, newcells, edgestochange, dnodes, totnodes)
    if !any(JuAFEM.edges(cell) .∈ (edgestochange,))
        push!(newcells, cell)
    end
    for edge in JuAFEM.edges(cell)
        if edge ∈ edgestochange
            newnode = Node((dnodes[edge[1]].coord + dnodes[edge[2]].coord)/2)
            # newid = length(grid.nodes) + 1
            newid = findfirst(x->x == newnode, totnodes)
            if newid == nothing
                newid = length(totnodes) + 1
                push!(totnodes, newnode)
            end
            eid1 = findfirst(x -> x == edge[1], cell.nodes)
            eid2 = findfirst(x -> x == edge[2], cell.nodes)
            newcell1 = Tetrahedron(Base.setindex(cell.nodes, newid, eid2))
            newcell2 = Tetrahedron(Base.setindex(cell.nodes, newid, eid1))
            refinecell(newcell1, newcells, edgestochange, dnodes, totnodes)
            refinecell(newcell2, newcells, edgestochange, dnodes, totnodes)
            break
        end
    end
end

function refinegrid(model, thresholds::NamedTuple, alldofs)
    dnodes = dofnodes(model.dofhandler)
    grid = model.dofhandler.grid
    newcells = copy(grid.cells)
    totnodes = grid.nodes
    edgestochange, edgetocell = findedges(model, thresholds, alldofs)
    while !isempty(edgestochange)
        cellstobeadded = Tetrahedron[]
        edge = pop!(edgestochange)
        newnode = Node((dnodes[edge[1]].coord + dnodes[edge[2]].coord)/2)
        newid = findfirst(x->x == newnode, totnodes)
        if newid == nothing
            newid = length(totnodes) + 1
            push!(totnodes, newnode)
        end
        for (ic, cell) in enumerate(newcells)
            if edge ∈ JuAFEM.edges(cell) || reverse(edge) ∈ JuAFEM.edges(cell)
                eid1 = findfirst(x -> x == edge[1], cell.nodes)
                eid2 = findfirst(x -> x == edge[2], cell.nodes)
                push!(cellstobeadded, Tetrahedron(Base.setindex(cell.nodes, newid, eid2)))
                newcells[ic] = Tetrahedron(Base.setindex(cell.nodes, newid, eid1))
            end
        end
        append!(newcells, cellstobeadded)
    end
    grid.cells = newcells
end

for cell in CellIterator(modelBFO.dofhandler)
    up = modelBFO.dofs[celldofs(cell)]
    println(up[13:24])
end

modelBFO = LandauModel(300.0, :BFO, [(:u, 3),(:P, 3)], (10, 10, 2), left, right, element_potential; startingconditions=startingconditionsBFO!, lagrangeorder=1);
startingconditionsBFO!(modelBFO.dofs, modelBFO.dofhandler)
@time refinegrid(modelBFO, (u=3, P=1e-1), modelBFO.dofs)
newdh = DofHandler(modelBFO.dofhandler.grid)
push!(newdh, :u, 3)
push!(newdh, :P, 3)
close!(newdh)

modelBFO.dofhandler = newdh
modelBFO.dofs = zeros(ndofs(newdh))
vtk_save("/home/ponet/testafter", modelBFO)
minimize!(modelBFO)
const dd =periodicmap_dim3(modelBFO.dofhandler, "bottom", "top", left[3], right[3])
for i=1:length(modelBFO.dofhandler.cell_dofs)
    d = modelBFO.dofhandler.cell_dofs[i]
    dn = get(dd, d, d)
    modelBFO.dofhandler.cell_dofs[i] = dn
end
modelBFO.dofhandler.cell_dofs_offset

Base.length(::Type{Tensor{1, 3, Float64, 3}}) = 3
using NearestNeighbors
tree = KDTree([x.x for x in modelBFO.dofhandler.grid.nodes])

@time inrange(tree, Vec{3}((1.2e-9,0.0,0.0)), 3e-8, true)

modelBFO.dofhandler.grid.nodes[101169].x
