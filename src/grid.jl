using LinearAlgebra
#Based on JuAFEM's generate_grid(Tetrahedron, ...) function
function JuAFEM.generate_grid(::Type{QuadraticTetrahedron}, cells_per_dim::NTuple{3,Int}, left::Vec{3,T}=Vec{3}((-1.0,-1.0,-1.0)), right::Vec{3,T}=Vec{3}((1.0,1.0,1.0))) where {T}
    nodes_per_dim = (2 .* cells_per_dim) .+ 1
    cells_per_cube = 6
    total_nodes = prod(nodes_per_dim)
    total_elements = cells_per_cube * prod(cells_per_dim)
    n_nodes_x, n_nodes_y, n_nodes_z = nodes_per_dim
    n_cells_x, n_cells_y, n_cells_z = cells_per_dim
    # Generate nodes
    coords_x = range(left[1], stop=right[1], length=n_nodes_x)
    coords_y = range(left[2], stop=right[2], length=n_nodes_y)
    coords_z = range(left[3], stop=right[3], length=n_nodes_z)
    numbering = reshape(1:total_nodes, nodes_per_dim)
    # Pre-allocate the nodes & cells
    nodes = Vector{Node{3,T}}(undef,total_nodes)
    cells = Vector{QuadraticTetrahedron}(undef,total_elements)
    # Generate nodes
    node_idx = 1
    @inbounds for k in 1:n_nodes_z, j in 1:n_nodes_y, i in 1:n_nodes_x
        nodes[node_idx] = Node((coords_x[i], coords_y[j], coords_z[k]))
        node_idx += 1
    end
    # Generate cells, case 1 from: http://www.baumanneduard.ch/Splitting%20a%20cube%20in%20tetrahedras2.htm
    # cube = (1, 2, 3, 4, 5, 6, 7, 8)
    # left = (1, 4, 5, 8), right = (2, 3, 6, 7)
    # front = (1, 2, 5, 6), back = (3, 4, 7, 8)
    # bottom = (1, 2, 3, 4), top = (5, 6, 7, 8)
    localnodes = [  ((1,1,1),(3,1,1),(1,3,1),(1,3,3)),
                    ((1,1,1),(1,1,3),(3,1,1),(1,3,3)),
                    ((3,1,1),(3,3,1),(1,3,1),(1,3,3)),
                    ((3,1,1),(3,3,3),(3,3,1),(1,3,3)),
                    ((3,1,1),(1,1,3),(3,1,3),(1,3,3)),
                    ((3,1,1),(3,1,3),(3,3,3),(1,3,3))
                    ]
    avg(x,y) = (x == 1 && y == 3) || (x == 3 && y == 1) ? 2 : x
    indexavg(x,y) = CartesianIndex(avg.(Tuple(x),Tuple(y)))

    cell_idx = 0
    @inbounds for k in 1:n_cells_z, j in 1:n_cells_y, i in 1:n_cells_x
        cube = numbering[(2*(i-1) + 1):(2*i + 1), (2*(j-1)+1): 2*j + 1, (2*(k-1) +1): (2*k +1)]
        for (idx, p1vertices) in enumerate(localnodes)
            v1,v2,v3,v4 = map(CartesianIndex,p1vertices)
            cells[cell_idx + idx] = QuadraticTetrahedron((cube[v1],cube[v2],cube[v3],cube[v4],
                        cube[indexavg(v1,v2)],cube[indexavg(v2,v3)],cube[indexavg(v1,v3)],cube[indexavg(v1,v4)],
                        cube[indexavg(v2,v4)],cube[indexavg(v3,v4)]))
        end
        cell_idx += cells_per_cube
    end
    # Order the cells as c_nxyz[n, x, y, z] such that we can look up boundary cells
    c_nxyz = reshape(1:total_elements, (cells_per_cube, cells_per_dim...))
    @views le = [map(x -> (x,4), c_nxyz[1, 1, :, :][:])   ; map(x -> (x,2), c_nxyz[2, 1, :, :][:])]
    @views ri = [map(x -> (x,1), c_nxyz[4, end, :, :][:]) ; map(x -> (x,1), c_nxyz[6, end, :, :][:])]
    @views fr = [map(x -> (x,1), c_nxyz[2, :, 1, :][:])   ; map(x -> (x,1), c_nxyz[5, :, 1, :][:])]
    @views ba = [map(x -> (x,3), c_nxyz[3, :, end, :][:]) ; map(x -> (x,3), c_nxyz[4, :, end, :][:])]
    @views bo = [map(x -> (x,1), c_nxyz[1, :, :, 1][:])   ; map(x -> (x,1), c_nxyz[3, :, :, 1][:])]
    @views to = [map(x -> (x,3), c_nxyz[5, :, :, end][:]) ; map(x -> (x,3), c_nxyz[6, :, :, end][:])]
    boundary_matrix = JuAFEM.boundaries_to_sparse([le; ri; bo; to; fr; ba])
    facesets = Dict(
        "left" => Set(le),
        "right" => Set(ri),
        "front" => Set(fr),
        "back" => Set(ba),
        "bottom" => Set(bo),
        "top" => Set(to),
    )
    return JuAFEM.Grid(cells, nodes, facesets=facesets, boundary_matrix=boundary_matrix)
end

function connectivity(dh)
    out = [Int[] for i=1:getnnodes(dh.grid)]
    for cell in CellIterator(dh)
        for n in cell.nodes, (i2, n2) in enumerate(cell.nodes)
            if !in(n2, out[n])
                push!(out[n], n2)
            end
        end
    end
end

#Stencils for stuff like Edep, can be used for calculations that involve r-r'
mutable struct Stencil{DIM, T, NT <: NamedTuple}
    volume::T
    connections::Vector{Int}
    Rhat::Vector{Vec{DIM, T}}
    R⁻³ ::Vector{T}
    dofs::Vector{NT} #vector{namedtuple}
end
Base.zero(::Type{Stencil{DIM, T, NT}}) where {DIM, T, NT} = Stencil(zero(T), Int[], Vec{DIM, T}[], T[], NT[])

function construct_stencils(dh::DofHandler{DIM, N, T} where N, gridsize, ranges, Rmax) where {DIM, T}
    grid = dh.grid
    stencils = [zero(Stencil{DIM, T, NamedTuple{(dh.field_names...,), Tuple{Vector{Int}, Vector{Int}}}}) for i=1:getnnodes(grid)]
    cnodes = dofnodes(dh)
    totnodes = getnnodes(grid)
    linids = LinearIndices(gridsize)
    cartids = CartesianIndices(gridsize)
    steps  = getfield.(ranges, :step)
    volume = abs(prod(grid.nodes[1].x-grid.nodes[linids[(1 .+ steps)...]].x))
    for (i, n) in enumerate(grid.nodes)
        radius = zero(T)
        cid = cartids[i]
        for r in [(r1, r2, r3) for r1=ranges[1], r2=ranges[2], r3 =ranges[3]]
            if norm(r) == 0
                continue
            end
            temp_cid = Tuple(cid) .+ r
            if prod(temp_cid) > totnodes || any(temp_cid .> gridsize) || any(x -> x <= 0, temp_cid)
                continue
            end
            temp_lid = linids[temp_cid...]
            n2 = cnodes[temp_lid]
            if norm(n2.coord - n.x) > Rmax
                continue
            end

            Rrel = n.x - n2.coord
            d = norm(Rrel)
            push!(stencils[i].connections, temp_lid)
            push!(stencils[i].R⁻³, d^-3)
            push!(stencils[i].Rhat, Vec{3}((normalize(Rrel)...,)))
            push!(stencils[i].dofs, cnodes[temp_lid].dofs)
        end
        stencils[i].volume = volume
    end
    return stencils
end

function construct_stencil(dh::DofHandler{DIM, N, T} where N, gridsize, ranges, Rmax) where {DIM, T}
    grid = dh.grid
    stencil = zero(Stencil{DIM, T, NamedTuple{(dh.field_names...,), Tuple{Vector{Int}, Vector{Int}}}})
    cnodes = dofnodes(dh)
    totnodes = getnnodes(grid)
    linids = LinearIndices(gridsize)
    cartids = CartesianIndices(gridsize)
    midcartid = Int.(ceil.(gridsize ./ 2))
    midlinid = linids[midcartid...]
    node = cnodes[midlinid]
    steps  = getfield.(ranges, :step)
    volume = abs(prod(grid.nodes[1].x-grid.nodes[linids[(1 .+ steps)...]].x))
    for r in [(r1, r2, r3) for r1=ranges[1], r2=ranges[2], r3 =ranges[3]]
        if norm(r) == 0
            continue
        end
        cartid = midcartid .+ r
        if prod(cartid) > totnodes || any(cartid .> gridsize) || any(x -> x <= 0, cartid)
            continue
        end
        linid = linids[cartid...]
        n2 = cnodes[linid]
        if norm(n2.coord - node.coord) > Rmax
            continue
        end

        Rrel = node.coord - n2.coord
        d    = norm(Rrel)
        push!(stencil.connections, linid - midlinid)
        push!(stencil.R⁻³, d^-3)
        push!(stencil.Rhat, Vec{3}((normalize(Rrel)...,)))
        push!(stencil.dofs, cnodes[linid].dofs)
    end
    stencil.volume = volume
    return stencil
end

function plotStencil(stencil)
    toplotx = Float64[]
    toploty = Float64[]
    for i=1:length(stencil.connections)
        Rrel = stencil.Rhat[i] * stencil.R⁻³[i]^(-1/3)
        push!(toplotx, Rrel[1])
        push!(toploty, Rrel[2])
    end
    scatter(toplotx, toploty)
end
