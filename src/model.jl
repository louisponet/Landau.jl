import JuAFEM: vtk_save
import ForwardDiff: GradientConfig, HessianConfig, JacobianConfig, Chunk
import ForwardDiff.DiffResults: HessianResult, DiffResult

mutable struct ThreadCache{T, DIM, CV <: NamedTuple, EX <: NamedTuple, GC <: GradientConfig, HC <: HessianConfig, EF <: Function, HR <: DiffResult}
    indices   ::Vector{Int}
    dofs      ::Vector{T}
    gradient  ::Vector{T}
    hessian   ::Matrix{T}
    coords    ::Vector{Vec{DIM, T}}
    cellvalues::CV
    extradata ::EX
    gradconf  ::GC
    hessconf  ::HC
    efunc     ::EF
    hessresult::HR
end

function ThreadCache(dpc::Int, nodespercell::Int, cellvalues, extradata, element_energy)
    indices  = zeros(Int, dpc)
    dofs     = zeros(dpc)
    gradient = zeros(dpc)
    hessian  = zeros(dpc, dpc)
    coords   = zeros(Vec{3}, nodespercell)
    efunc = x -> element_energy(x, cellvalues, extradata)
    gradconf = GradientConfig(efunc, zeros(dpc), Chunk{12}())

    hessresult = HessianResult(zeros(dpc))
    hessconf = HessianConfig(efunc, hessresult, zeros(dpc), Chunk{6}())
    return ThreadCache(indices, dofs, gradient, hessian, coords, cellvalues, extradata, gradconf, hessconf, efunc, hessresult)
end

abstract type AbstractModel end

mutable struct LandauModel{T, DH <: DofHandler, CH <: ConstraintHandler, TC <: ThreadCache, DN <: DofNode} <: AbstractModel
    dofs          ::Vector{T}
    dofhandler    ::DH
    dofnodes      ::Vector{DN}
    boundaryconds ::CH
    threadindices ::Vector{Vector{Int}}
    threadcaches  ::Vector{TC}
end

"""
    LandauModel(fields::AbstractVector{Tuple{Symbol, Int, Function}},
                gridsize::NTuple{3, Int},
                left::Vec{DIM, T},
                right::Vec{DIM, T},
                element_function::Function;
                boundaryconds = [],
                elgeom = nothing,
                gridgeom = nothing,
                lagrangeorder = 1,
                quadratureorder= 2) where {DIM, T}

Creates a LandauModel, holding all the information necessary for energy minimization and further post-processing. 
"""
function LandauModel(fields::AbstractVector{Tuple{Symbol, Int, Function}},
                     gridsize::NTuple{3, Int},
                     left::Vec{DIM, T},
                     right::Vec{DIM, T},
                     element_function::Function;
                     boundaryconds = [],
                     elgeom = nothing,
                     gridgeom = nothing,
                     lagrangeorder = 1,
                     quadratureorder= 2) where {DIM, T}
    if elgeom == nothing; elgeom = RefTetrahedron end
    if gridgeom == nothing
        if DIM==3
            if lagrangeorder == 2
                gridgeom = QuadraticTetrahedron
            else
                gridgeom = Tetrahedron
            end
        else
            gridgeom = Triangle
        end
    end

    grid = generate_grid(gridgeom, gridsize, left, right)
    bleirgh, colors = JuAFEM.create_coloring(grid)

    qr  = QuadratureRule{DIM, elgeom}(quadratureorder)
    interpolation = Lagrange{DIM, elgeom, lagrangeorder}()
    geominterp = Lagrange{DIM, elgeom, lagrangeorder}()
    dh = DofHandler(grid)
    for f in fields
        push!(dh, f[1], f[2], interpolation)
    end

    close!(dh)
    dofvec = zeros(ndofs(dh))

    uranges = UnitRange[]
    cvs = CellValues[] 
    for field in fields
        push!(uranges, dof_range(dh, field[1]))
        if field[2] > 1
            push!(cvs, CellVectorValues(qr, Lagrange{field[2], elgeom, lagrangeorder}(), geominterp))
        else
            push!(cvs, CellScalarValues(qr, interpolation, geominterp))
        end
            
        if length(field) == 3
            startingconditions!(dofvec, dh, field[1], field[3])
        end
    end
    ranges = NamedTuple{(dh.field_names...,)}(uranges)
    cellvalues = NamedTuple{(dh.field_names...,)}(cvs)

    bdcs_ = ConstraintHandler(dh)
    for bdc in boundaryconds
        add!(bdcs_, Dirichlet(bdc[1], getfaceset(grid, bdc[2]), bdc[3], bdc[4]))
    end
    close!(bdcs_)
    JuAFEM.update!(bdcs_, 0.0)

    apply!(dofvec, bdcs_)
    dpc = ndofs_per_cell(dh)
    cpc = length(dh.grid.cells[1].nodes)

    extradata = (force=zeros(T, DIM*cpc), Edepol=zeros(T, DIM*cpc), ranges=ranges) 
    #TODO generalize
    caches = [ThreadCache(dpc, cpc, deepcopy(cellvalues), extradata, element_function) for t=1:Threads.nthreads()]
	dnodes = dofnodes(dh)
    LandauModel(dofvec, dh, dnodes, bdcs_, colors, caches)
end

"""
    LandauModel(model::LandauModel, element_potential::Function)
    
To create a new model using everything from the old one except for the new `element_potential`.
"""
LandauModel(model::LandauModel, element_potential::Function) =
	LandauModel(model.dofs,
                model.dofhandler,
                model.dofnodes,
                model.boundaryconds,
                model.threadindices,
                [ThreadCache(length(t.dofs), length(t.coords), t.cellvalues, t.extradata, element_potential) for t in model.threadcaches])

"""
    vtk_save(path::AbstractString, model::LandauModel, dofs::AbstractVector=model.dofs)

Saves the `dofs` in a format readable by Paraview.
"""
function vtk_save(path::AbstractString, model::LandauModel, dofs::AbstractVector=model.dofs)
    vtkfile = vtk_grid(path, model.dofhandler)
    vtk_point_data(vtkfile, model.dofhandler, dofs)
    vtk_save(vtkfile)
end

landau_model(model::LandauModel) =
	model

dofs(model::AbstractModel) =
	landau_model(model).dofs

nodes(model::AbstractModel) =
	landau_model(model).dofhandler.grid.nodes

dofnodes(model::AbstractModel) =
	landau_model(model).dofnodes

dofhandler(model::AbstractModel) =
	landau_model(model).dofhandler

boundaryconds(model::AbstractModel) =
	landau_model(model).boundaryconds

#TODO do we need drange?
function startingconditions!(dofvector, dh, fieldsym, fieldfunction)
    drange = JuAFEM.dof_range(dh, fieldsym)
    offset = JuAFEM.field_offset(dh, fieldsym)
    n = ndofs_per_cell(dh)
    t_dofs = [zeros(Int, n) for i=1:Threads.nthreads()]
    t_coords = [zeros(Vec{3, Float64}, JuAFEM.nnodes(typeof(dh).parameters[2])) for i=1:Threads.nthreads()]
    interp = dh.field_interpolations[JuAFEM.find_field(dh, fieldsym)]
    dim    = JuAFEM.ndim(dh, fieldsym)
    Threads.@threads for i = 1:length(dh.grid.cells)
        cell = dh.grid.cells[i]
        globaldofs = t_dofs[Threads.threadid()]
        coords = t_coords[Threads.threadid()]
        JuAFEM.celldofs!(globaldofs, dh, i)
        JuAFEM.cellcoords!(coords, dh, i)
        for (idx, coord) in enumerate(coords)
            noderange = (offset + (idx - 1) * dim + 1):(offset + idx * dim)
            dofvector[globaldofs[noderange]] .= fieldfunction(coord)
        end
    end
end

NearestNeighbors.BallTree(grid::JuAFEM.Grid) =
    BallTree(map(x -> mean(map(y-> grid.nodes[y].x, x.nodes)), grid.cells))

NearestNeighbors.BallTree(dofhandler::JuAFEM.DofHandler) = BallTree(dofhandler.grid) 
NearestNeighbors.BallTree(m::LandauModel) = BallTree(m.dofhandler)

function JuAFEM.function_value(dofhandler::DofHandler{dim, C, T}, alldofs, point::Vec, tree) where {dim, C, T}
    cellids = knn(tree, point, 4)[1]
    cell_coords = zeros(Vec{dim, T}, JuAFEM.nnodes(C))
    best = typemax(T)
    bestid = 0
    for id in cellids
        JuAFEM.cellcoords!(cell_coords, dofhandler, id)
        s = mapreduce(x -> norm(point - x), +, cell_coords)
        if s < best
            best = s
            bestid = id
        end
    end
    JuAFEM.cellcoords!(cell_coords, dofhandler, bestid)
    cdofs = alldofs[celldofs(dofhandler, bestid)]
    r = point - cell_coords[1]
    cell_param = [cell_coords[2]-cell_coords[1] cell_coords[3]-cell_coords[1] cell_coords[4]-cell_coords[1]]
    quad_point = Vec{3}(inv(cell_param) * r)

    vals = [zero(Vec{d, typeof(dofhandler).parameters[end]}) for d in dofhandler.field_dims]
    for (i, (d, interp, drange)) in enumerate(zip(dofhandler.field_dims, dofhandler.field_interpolations, dof_range.((dofhandler,), dofhandler.field_names) ))
        for base_func in 1:getnbasefunctions(interp)
            vals[i] += Vec{d}(JuAFEM.value(interp, base_func, quad_point) * cdofs[drange[(base_func-1) * d + 1:base_func * d]])
        end
    end
    return NamedTuple{(dofhandler.field_names...,)}((vals...,))
end

## DATA EXTRACTION
extract_data(m::LandauModel, points::Vector{<:Vec}, dofs = m.dofs, tree = BallTree(m)) = function_value.((m.dofhandler,), (dofs,), points, (tree,))

extract_data_line(m::LandauModel, start::Vec, stop::Vec, length::Int, args...) =
    extract_data(m, range(start, stop, length=length) , args...)
