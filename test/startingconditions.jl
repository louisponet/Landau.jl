function startingconditionsBFO!(up, dh)
    mid = div(ndofs_per_cell(dh), 2)
    urange = dof_range(dh, :u)
    Prange = dof_range(dh, :P)
    Pinterp = dh.field_interpolations[find_field(dh, :P)]
    Poffset = JuAFEM.field_offset(dh, :P)
    Porder = getorder(Pinterp)
    Pdim = getdim(Pinterp)
    for cell in CellIterator(dh)
        globaldofs = celldofs(cell)
        for idx in 1:min(getnbasefunctions(Pinterp), length(cell.nodes))
            coord = cell.coords[idx]
            noderange = (Poffset + (idx - 1) * Pdim + 1):(Poffset + idx * Pdim)
            for i in noderange
                orderrenorm = mod1(i, Pdim)
                it = 1
                if orderrenorm == 1
                    up[globaldofs[i]] = -0.48
                elseif orderrenorm == 2
                    up[globaldofs[i]] = 0.48
                elseif orderrenorm == 3
                    up[globaldofs[i]] = 0.48tanh((-coord[1]-coord[2])/20e-10)
                    it += 1
                end
            end
        end
    end
end
# function startingconditionsBFO!(up, dh)
#     mid = div(ndofs_per_cell(dh), 2)
#     for cell in CellIterator(dh)
#         globaldofs = celldofs(cell)
#         for i=1:3:mid
#             it = div(i, 3) + 1
# #             up[globaldofs[i]] = cell.coords[it][1]
# #             up[globaldofs[i+1]] = cell.coords[it][2]
# #             up[globaldofs[i+2]] = cell.coords[it][3]
#
#             up[globaldofs[mid+i]] = 0.48tanh((-cell.coords[it][1]+cell.coords[it][2])/20e-10)
#             up[globaldofs[mid+i+1]] = 0.48tanh((-cell.coords[it][1]+cell.coords[it][2])/20e-10)
#             up[globaldofs[mid+i+2]] = -0.48tanh((-cell.coords[it][1]+cell.coords[it][2])/20e-10)
# #             up[globaldofs[mid+i+2]] =  0.0053*rand()-0.0053
#         end
#     end
# end
function startingconditionsBTO!(up, dh)
    mid = div(ndofs_per_cell(dh), 2)
    for cell in CellIterator(dh)
        globaldofs = celldofs(cell)
        for i=1:3:mid
            it = div(i, 3) + 1
            up[globaldofs[mid+i]] = 0.
            up[globaldofs[mid+i+1]] = 0.29tanh((15e-9 - sqrt(cell.coords[it][1]^2-cell.coords[it][3]^2))/20e-10)
            up[globaldofs[mid+i+2]] =0.0
        end
    end
end
