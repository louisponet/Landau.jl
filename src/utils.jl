δ(i, j) = i == j ? one(i) : zero(i)

voigt_to_tensor(p11, p12, p44) =
	Tensor{4, 3}((i,j,k,l) -> p11 * δ(i,j)*δ(k,l)*δ(i,k) + p12*δ(i,j)*δ(k,l)*(1 - δ(i,k)) + p44*δ(i,k)*δ(j,l)*(1 - δ(i,j)))

gaussian(x::Vec{DIM}, x0::Vec{DIM}, σ²) where DIM =
	1/(2π * σ²)^(DIM/2) * ℯ^(-norm(x-x0)^2 / (2σ²))

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
    clo.extradata = (force=prefac*Vec{3, T}((0.0, gaussian(Vec{1, T}((coords[1],))*1.0e9,Vec{1, T}((center,)),0.5)*1e18ℯ^(-25.0+coords[2]*1e10), 0.0)), clo.extradata...)
end


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
