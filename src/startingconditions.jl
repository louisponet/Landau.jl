import JuAFEM: field_offset, getdim, find_field, ndim

function startingconditions!(dofvector, dh, fieldsym, fieldfunction)
    drange = dof_range(dh, fieldsym)
    offset = field_offset(dh, fieldsym)
    interp = dh.field_interpolations[find_field(dh, fieldsym)]
    dim    = ndim(dh, fieldsym)
    for cell in CellIterator(dh)
        globaldofs = celldofs(cell)
        for idx in 1:min(getnbasefunctions(interp), length(cell.nodes))
            coord = cell.coords[idx]
            noderange = (offset + (idx - 1) * dim + 1):(offset + idx * dim)
            if fieldfunction(coord) === nothing
                @show coord
                @show fieldsym
            end
            dofvector[globaldofs[noderange]] .= fieldfunction(coord)
        end
    end
end
