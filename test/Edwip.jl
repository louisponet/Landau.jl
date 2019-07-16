using JuAFEM
left = Vec{3}((-75.e-9,-25.e-9,-2.e-9))
right = Vec{3}((75.e-9,25.e-9,2.e-9))
grid = generate_grid(QuadraticTetrahedron, (2, 2, 2), left, right)
dh = DofHandler(grid)
push!(dh, :P, 3, Lagrange{3, RefTetrahedron, 2}())
push!(dh, :u, 3, Lagrange{3, RefTetrahedron, 2}())
close!(dh)
test  = construct_stencils(dh)
grid.nodes[end]
pointer_from_objref(test[1].coords)
pointer_from_objref(test[2].coords)
cellit = CellIterator(dh)
filter(x->x.x == Vec{3}((0.0,0.0,0.0)), grid.nodes)
reinit!(cellit, 44)
