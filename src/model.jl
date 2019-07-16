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
    # cellcache        ::CC
    gradconf  ::GC
    hessconf  ::HC
    # jacconf   ::JC
    efunc     ::EF
    hessresult::HR
    # gfunc     ::GF
end
function ThreadCache(dpc::Int, nodespercell::Int, cellvalues, extradata, element_energy)
    indices  = zeros(Int, dpc)
    dofs     = zeros(dpc)
    gradient = zeros(dpc)
    hessian  = zeros(dpc, dpc)
    coords   = zeros(Vec{3}, nodespercell)
    # cellcache        = CellCache(args...)
    efunc = x -> element_energy(x, cellvalues, extradata)
    gradconf = GradientConfig(efunc, zeros(dpc), Chunk{12}())

    hessresult = HessianResult(zeros(dpc))
    hessconf = HessianConfig(efunc, hessresult, zeros(dpc), Chunk{6}())
    # hessconf = HessianConfig(efunc, zeros(dpc), Chunk{12}())
    return ThreadCache(indices, dofs, gradient, hessian, coords, cellvalues, extradata, gradconf, hessconf, efunc, hessresult)
    # return ThreadCache(indices, dofs, gradient, hessian, coords, cellvalues, extradata, gradconf, jacconf, efunc, gfunc)
end


mutable struct LandauModel{T, DH <: DofHandler, CH <: ConstraintHandler, TC <: ThreadCache}
    dofs          ::Vector{T}
    dofhandler    ::DH
    boundaryconds ::CH
    threadindices ::Vector{Vector{Int}}
    threadcaches  ::Vector{TC}
end

function LandauModel(fields, gridsize, left::Vec{DIM, T}, right::Vec{DIM, T}, element_function;
                     boundaryconds = [], elgeom = nothing, gridgeom = nothing, lagrangeorder = 1, quadratureorder= 2) where {DIM, T}
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
    # grid = generate_grid(Tetrahedron, gridsize, left, right)
    bleirgh, colors = JuAFEM.create_coloring(grid)

    qr  = QuadratureRule{DIM, elgeom}(quadratureorder)
    interpolation = Lagrange{DIM, elgeom, lagrangeorder}()
    geominterp = Lagrange{DIM, elgeom, lagrangeorder}()
    # cvu = CellVectorValues(qr, interpolation)
    # cvP = CellVectorValues(qr, interpolation)
    dh = DofHandler(grid)
    for f in fields
        # push!(dh, f[1], f[2])
        push!(dh, f[1], f[2], interpolation)
    end

    close!(dh)
    dofvec = zeros(ndofs(dh))

    uranges = UnitRange[]
    cvs = CellVectorValues[]
    for field in fields
        push!(uranges, dof_range(dh, field[1]))
        push!(cvs, CellVectorValues(qr, interpolation, geominterp))
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
    #TODO generalize


    caches = [ThreadCache(dpc, cpc, deepcopy(cellvalues), (force=zeros(T, DIM*cpc), Edepol=zeros(T, DIM*cpc), ranges=ranges), element_function) for t=1:Threads.nthreads()]
    LandauModel(dofvec, dh, bdcs_, colors, caches)
end

#To create a new model using everything from the old one except for the element_potential with the params already set
@export LandauModel(model::LandauModel, element_potential::Function) =
	LandauModel(model.dofs,
                model.dofhandler,
                model.boundaryconds,
                model.threadindices,
                [ThreadCache(length(t.dofs), length(t.coords), t.cellvalues, t.extradata, element_potential) for t in model.threadcaches])

function vtk_save(path, model::LandauModel, up=model.dofs)
    vtkfile = vtk_grid(path, model.dofhandler)
    vtk_point_data(vtkfile, model.dofhandler, up)
    vtk_save(vtkfile)
end
