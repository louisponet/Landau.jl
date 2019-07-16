import JuAFEM: field_offset, getdim, find_field

function startingconditions!(dofvector, dh, fieldsym, fieldfunction)
    drange = dof_range(dh, fieldsym)
    offset = field_offset(dh, fieldsym)
    interp = dh.field_interpolations[find_field(dh, fieldsym)]
    dim    = getdim(interp)
    for cell in CellIterator(dh)
        globaldofs = celldofs(cell)
        for idx in 1:min(getnbasefunctions(interp), length(cell.nodes))
            coord = cell.coords[idx]
            noderange = (offset + (idx - 1) * dim + 1):(offset + idx * dim)
            dofvector[globaldofs[noderange]] .= fieldfunction(coord)
        end
    end
end
