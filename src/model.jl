import JuAFEM: vtk_save
import ForwardDiff: GradientConfig, HessianConfig, Chunk

struct ModelParams{V, T}
    α::V
    C::T
    G::T
    Q::T
    F::T
end

function ModelParams(α, G11, G12, G44, C11, C12, C44, Q11, Q12, Q44, F11, F12, F44)
    q11 = C11*Q11 + 2C12*Q12
    q12 = C11*Q12 + C12*(Q11 + Q12)
    q44 = C44*Q44
    V2T(p11, p12, p44) = Tensor{4, 3}((i,j,k,l) -> p11 * δ(i,j)*δ(k,l)*δ(i,k) + p12*δ(i,j)*δ(k,l)*(1 - δ(i,k)) + p44*δ(i,k)*δ(j,l)*(1 - δ(i,j)))
    C = V2T(C11, C12, C44)
    G = V2T(G11, G12, G44)
    Q = V2T(q11, q12, q44)
    F = V2T(F11, F12, F44)
    ModelParams(α, C, G, Q, F)
end

mutable struct CellCache{CV <: NamedTuple, MP <: ModelParams, F <: Function, EX <: NamedTuple}
    cellvalues::CV
    parameters::MP
    potential ::F
    extradata ::EX
    function CellCache(cellvalues::CV, parameters::MP, elfunction::Function, extra::EX) where {CV, MP, EX}
        potfunc = x -> elfunction(x, cellvalues, parameters, extra)
        return new{CV, MP, typeof(potfunc), EX}(cellvalues, parameters, potfunc, extra)
    end
end

struct ThreadCache{T, DIM, CC <: CellCache, GC <: GradientConfig, HC <: HessianConfig}
    element_indices  ::Vector{Int}
    element_dofs     ::Vector{T}
    element_gradient ::Vector{T}
    element_hessian  ::Matrix{T}
    cellcache        ::CC
    gradconf         ::GC
    hessconf         ::HC
    element_coords   ::Vector{Vec{DIM, T}}
end
function ThreadCache(dpc::Int, nodespercell,  args...)
    element_indices  = zeros(Int, dpc)
    element_dofs     = zeros(dpc)
    element_gradient = zeros(dpc)
    element_hessian  = zeros(dpc, dpc)
    cellcache        = CellCache(args...)
    gradconf         = GradientConfig(nothing, zeros(dpc), Chunk{12}())
    hessconf         = HessianConfig(nothing, zeros(dpc), Chunk{12}())
    coords           = zeros(Vec{3}, nodespercell)
    return ThreadCache(element_indices, element_dofs, element_gradient, element_hessian, cellcache, gradconf, hessconf, coords )
end


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
                    dofdict_[dofs1[r1]] = dofs2[r2]
                end
                for (r1, r2) in zip(mid+(ic1-1)*3+1:mid+ic1*3, mid+(ic2-1)*3+1:mid+ic2*3)
                    dofdict_[dofs1[r1]] = dofs2[r2]
                end
            end
        end
    end
    dofdict_
end

mutable struct LandauModel{T, DH <: DofHandler, CH <: ConstraintHandler, TC <: ThreadCache}
    dofs          ::Vector{T}
    dofhandler    ::DH
    boundaryconds ::CH
    threadindices ::Vector{Vector{Int}}
    threadcaches  ::Vector{TC}
end

function LandauModel(parameters::ModelParams, fields, gridsize, left::Vec{DIM, T}, right::Vec{DIM, T}, element_function;
                     boundaryconds = [], startingconditions = nothing, elgeom = nothing, gridgeom = nothing, lagrangeorder = 1, quadratureorder= 2) where {DIM, T}
    if elgeom == nothing; elgeom = RefTetrahedron end
    if gridgeom == nothing; gridgeom = DIM==3 ? Tetrahedron : Triangle end

    grid = generate_grid(gridgeom, gridsize, left, right)
    bleirgh, colors = JuAFEM.create_coloring(grid)

    qr  = QuadratureRule{DIM, elgeom}(quadratureorder)
    cvu = CellVectorValues(qr, Lagrange{DIM, elgeom, lagrangeorder}())
    cvP = CellVectorValues(qr, Lagrange{DIM, elgeom, lagrangeorder}())
    dh = DofHandler(grid)
    for f in fields
        push!(dh, f[1], f[2])
    end
    close!(dh)

    up = zeros(ndofs(dh))
    if startingconditions != nothing
        startingconditions(up, dh)
    end


    bdcs_ = ConstraintHandler(dh)
    for bdc in boundaryconds
        add!(bdcs_, Dirichlet(bdc[1], getfaceset(grid, bdc[2]), bdc[3], bdc[4]))
    end
    close!(bdcs_)
    JuAFEM.update!(bdcs_, 0.0)

    apply!(up, bdcs_)
    dpc = ndofs_per_cell(dh)
    cpc = length(dh.grid.cells[1].nodes)
    #TODO generalize
    caches = [ThreadCache(dpc, cpc, (u=copy(cvu), p=copy(cvP)), parameters, element_function, (force=zero(Vec{3, T}),)) for t=1:Threads.nthreads()]
    LandauModel(up, dh, bdcs_, colors, caches)
end

function vtk_save(path, model::LandauModel, up=model.dofs)
    vtkfile = vtk_grid(path, model.dofhandler)
    vtk_point_data(vtkfile, model.dofhandler, up)
    vtk_save(vtkfile)
end
