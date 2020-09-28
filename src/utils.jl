δ(i, j) = i == j ? one(i) : zero(i)

voigt_to_tensor(p11, p12, p44) =
	Tensor{4, 3}((i, j, k, l) -> p11 * δ(i, j)*δ(k, l)*δ(i, k) + p12*δ(i, j)*δ(k, l)*(1 - δ(i, k)) + p44*δ(i, k)*δ(j, l)*(1 - δ(i, j)))

mutable struct DofNode{DIM, T, NT <: NamedTuple}
    coord       ::Vec{DIM, T}
    dofs        ::NT
end

"Construct nodes that also hold the dofs associated with them from a DofHandler."
function dofnodes(dh::DofHandler{DIM, N, T} where N) where {DIM, T}
    grid = dh.grid
    out = [DofNode(node.x, NamedTuple{(dh.field_names...,)}([zeros(Int, DIM) for i=1:length(dh.field_names)])) for node in grid.nodes]
    celldofs_ = zeros(Int, ndofs_per_cell(dh))
    for cell in CellIterator(dh)
        celldofs!(celldofs_, cell)
        for (i1, n) in enumerate(cell.nodes)
            if sum(out[n].dofs[1]) == 0
                nodedofs = Vector{Int}[]
                for (name, dim) in zip(dh.field_names, dh.field_dims)
                    nodeoffset = (i1 - 1) * dim
                    r = dof_range(dh, name)[nodeoffset+1:nodeoffset+dim]
                    nodedofs = push!(nodedofs, celldofs_[r])
                end
                out[n].dofs = NamedTuple{(dh.field_names...,)}(nodedofs)
            end
        end
    end
    return out
end

alldofs(d::DofNode) = vcat(values(d.dofs)...)

line_intersect(p1, p2, p3) = 0.99999 <= dot(normalize(p3 - p1), normalize(p2 - p1)) <= 1.000001

function extract_data(model, var_sym, p1::Vec{3, Float64}, p2::Vec{3, Float64}, dofs=model.dofs)
	dnodes = dofnodes(model.dofhandler)
	out = []
	xyz = []
	for d in dnodes
		if line_intersect(p1, p2, d.coord)
			push!(out, dofs[d.dofs[var_sym]])
			push!(xyz, d.coord)
		end
	end
	return out, xyz
end

plane_intersect(origin, up, right, p) = abs(dot(normalize(p - origin), cross(normalize(up - origin), normalize(right - origin)))) < 1e-5

function extract_data(model, var_sym, origin, up::Vec{3, Float64}, right::Vec{3, Float64}, dofs=model.dofs)
	dnodes = dofnodes(model.dofhandler)
	out = []
	xyz = []
	for d in dnodes
		if plane_intersect(origin, up, right, d.coord)
			push!(out, dofs[d.dofs[var_sym]])
			push!(xyz, d.coord)
		end
	end
	return out, xyz
end

