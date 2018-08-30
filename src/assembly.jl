
import ForwardDiff: gradient!, hessian!
import Statistics: mean
import SparseArrays: SparseMatrixCSC

function precond!(f, r, val)
    for i in r
        f[i] *= val
    end
end


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
    clo.extradata = (force=prefac*Vec{3, T}((0.0, Landau.gaussian(Vec{1, T}((coords[1],))*1.0e9,Vec{1, T}((center,)),0.5)*1e18ℯ^(-25.0+coords[2]*1e10), 0.0)),)
end

macro assemble!(innerbody)
    esc(quote
        dofhandler = model.dofhandler
        for indices in model.threadindices
            @threads for i in indices
                cache     = model.threadcaches[threadid()]
                cellcache = cache.cellcache
                eldofs    = cache.element_dofs
                nodeids   = dofhandler.grid.cells[i].nodes
                for j=1:length(cache.element_coords)
                    cache.element_coords[j] = dofhandler.grid.nodes[nodeids[j]].x
                end
                for cellvalue in cellcache.cellvalues
                    reinit!(cellvalue, cache.element_coords)
                end

                celldofs!(cache.element_indices, dofhandler, i)
                for j=1:length(cache.element_dofs)
                    eldofs[j] = dofvector[cache.element_indices[j]]
                end
                $innerbody
            end
        end
    end)
end

function F(dofvector::Vector{T}, model, offset, forceconstant::T) where T
    outs  = [zero(T) for t=1:nthreads()]
    @assemble! begin
        force!(cellcache, mean(cache.element_coords), offset, forceconstant)
        outs[threadid()] += cellcache.potential(eldofs)
    end
    sum(outs)
end

function ∇F!(∇f::Vector{T}, dofvector::Vector{T}, model::LandauModel{T}, offset, forceconstant::T) where {T}
    fill!(∇f, zero(T))
    @assemble! begin
        force!(cellcache, mean(cache.element_coords), offset, forceconstant)
        gradient!(cache.element_gradient, cellcache.potential, eldofs, cache.gradconf)
        @inbounds assemble!(∇f, cache.element_indices, cache.element_gradient)
    end
end

function ∇²F!(∇²f::SparseMatrixCSC, dofvector::Vector{T}, model::LandauModel{T}, offset, forceconstant::T) where T
    assemblers = [start_assemble(∇²f) for t=1:nthreads()]
    @assemble! begin
        force!(cellcache, mean(cache.element_coords), offset, forceconstant)
        ForwardDiff.hessian!(cache.element_hessian, cellcache.potential, eldofs, cache.hessconf)
        @inbounds assemble!(assemblers[threadid()], cache.element_indices, cache.element_hessian)
    end
end



# function ∇F!(∇f::Vector{T}, uP::Vector{T}, model::LandauModel{T}, r,offset, forceconstant::T, uprefac= 1.0e6, Pprefac= 1.0e17) where {T}
#     fill!(∇f, zero(T))
#     for color in model.threadindices
#         Threads.@threads for t in color
#             tid = Threads.threadid()
#             cache = model.threadcaches[tid]
#             el = cache.element_dofs
#             fe = cache.element_gradient
#             config = cache.gradconf
#             eldofs = cache.dofindices
#             coords = cache.element_coords
#             clo    = cache.cellcache
#             cvu,  cvP = clo.cellvalues
#             nodeids = dh.grid.cells[t].nodes
#             for j=1:length(coords)
#                 coords[j] = dh.grid.nodes[nodeids[j]].x
#             end
#             reinit!(cvP, coords)
#             reinit!(cvu, coords)
#             force!(clo, mean(coords), offset, forceconstant)
#             celldofs!(eldofs, dh, t)
#             for i=1:24
#                 el[i] = uP[eldofs[i]]
#             end
#             ForwardDiff.gradient!(fe, clo.potential, el, config)
#             precond!(fe, 1:12, uprefac)
#             precond!(fe, 13:24, Pprefac)
#             precond!(fe, r, 0.0)
#             @inbounds assemble!(∇f, eldofs, fe)
#         end
#     end
# end

# function F(uP::Vector{T}, model, offset, forceconstant::T) where T
#     outs  = [zero(T) for t=1:nthreads()]
#     dh = model.dofhandler
#     for c=1:length(model.threadindices)
#         color = model.threadindices[c]
#         @threads for t in color
#             tid = threadid()
#             cache = model.threadcaches[tid]
#             el = cache.element_dofs
#             eldofs = cache.element_dofs
#             coords = cache.element_coords
#             clo    = cache.cellcache
#             cvu,  cvP = clo.cellvalues
#             nodeids = dh.grid.cells[t].nodes
#             for j=1:length(coords)
#                 coords[j] = dh.grid.nodes[nodeids[j]].x
#             end
#             reinit!(cvP, coords)
#             reinit!(cvu, coords)
#             celldofs!(eldofs, model.dh, t)
#             for i=1:24
#                 el[i] = uP[eldofs[i]]
#             end
#             force!(clo, mean(coords), offset, forceconstant)
#             outs[tid] += clo.potential(el)
#         end
#     end
#     return sum(outs)
# end
