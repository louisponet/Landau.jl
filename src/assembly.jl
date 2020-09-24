import ForwardDiff: gradient!, hessian!
import Statistics: mean
import SparseArrays: SparseMatrixCSC

function precond!(f, r, val)
    for i in r
        f[i] *= val
    end
end

macro assemble!(innerbody)
    esc(quote
        dofhandler = model.dofhandler
        for indices in model.threadindices
            Threads.@threads for i in indices
                cache     = model.threadcaches[threadid()]
                eldofs    = cache.dofs
                nodeids   = dofhandler.grid.cells[i].nodes
                for j=1:length(cache.coords)
                    cache.coords[j] = dofhandler.grid.nodes[nodeids[j]].x
                end
                map(x -> reinit!(x, cache.coords), cache.cellvalues)
                celldofs!(cache.indices, dofhandler, i)
                for j=1:length(cache.dofs)
                    eldofs[j] = dofvector[cache.indices[j]]
                end
                $innerbody
            end
        end
    end)
end

function F(dofvector::Vector{T}, model) where T
    outs  = [zero(T) for t=1:nthreads()]
    @assemble! begin
        outs[threadid()] += cache.efunc(eldofs)
    end
    sum(outs)
end

function ∇F!(∇f::Vector{T}, dofvector::Vector{T}, model::LandauModel{T}) where {T}
    fill!(∇f, zero(T))
    @assemble! begin
        gradient!(cache.gradient, cache.efunc, eldofs, cache.gradconf)
        @inbounds assemble!(∇f, cache.indices, cache.gradient)
    end
end

function ∇²F!(∇²f::SparseMatrixCSC, dofvector::Vector{T}, model::LandauModel{T}) where T
    assemblers = [start_assemble(∇²f) for t=1:nthreads()]
    @assemble! begin
        ForwardDiff.hessian!(cache.hessresult, cache.efunc, eldofs, cache.hessconf)
        @inbounds assemble!(assemblers[threadid()], cache.indices, DiffResults.hessian(cache.hessresult))
    end
end

function force!(cache::ThreadCache{<:Any, DIM}, nodeids, force) where DIM
    f = cache.extradata.force
    for (i, id) in enumerate(nodeids)
        for i_f = 1:DIM
            f[(i-1)*DIM + i_f] = force[(id-1)*DIM + i_f]
        end
    end
end

function Landau.F(dofvector::Vector{T}, model, forcevec) where T
    outs  = [zero(T) for t=1:nthreads()]
    @assemble! begin
        force!(cache, nodeids, forcevec)
        outs[threadid()] += cache.efunc(eldofs)
    end
    sum(outs)
end

function Landau.∇F!(∇f::Vector{T}, dofvector::Vector{T}, model::Landau.LandauModel{T}, forcevec) where {T}
    fill!(∇f, zero(T))
    @assemble! begin
        force!(cache, nodeids, forcevec)
        gradient!(cache.gradient, cache.efunc, eldofs, cache.gradconf)
        @inbounds assemble!(∇f, cache.indices, cache.gradient)
    end
end

function ∇²F!(∇²f::SparseMatrixCSC, dofvector::Vector{T}, model::LandauModel{T}, forcevec) where T
    assemblers = [start_assemble(∇²f) for t=1:nthreads()]
    @assemble! begin
        force!(cache, nodeids, forcevec)
        ForwardDiff.hessian!(cache.hessresult, cache.efunc, eldofs, cache.hessconf)
        @inbounds assemble!(assemblers[threadid()], cache.indices, DiffResults.hessian(cache.hessresult))
    end
end

function reinitEdep!(cache::ThreadCache{T, DIM} where T, nodeids, Edep) where DIM
    edepol = cache.extradata.Edepol
    for (i, id) in enumerate(nodeids)
        for j=1:DIM
            edepol[(i-1)*DIM + j] = Edep[id][j]
        end
    end
end

# function F(dofvector::Vector{T}, model, Edep) where T
#     outs  = [zero(T) for t=1:nthreads()]
#     @assemble! begin
#         reinitEdep!(cache, nodeids, Edep)
#         # force!(cache, mean(cache.coords), offset, forceconstant)
#         outs[threadid()] += cache.efunc(eldofs)
#     end
#     sum(outs)
# end

# function ∇F!(∇f::Vector{T}, dofvector::Vector{T}, model::LandauModel{T}, Edep) where {T}
#     fill!(∇f, zero(T))
#     @assemble! begin
#         reinitEdep!(cache, nodeids, Edep)
#         # force!(cache, mean(cache.coords), offset, forceconstant)
#         gradient!(cache.gradient, cache.efunc, eldofs, cache.gradconf)
#         # cache.gfunc(eldofs)
#         @inbounds assemble!(∇f, cache.indices, cache.gradient)
#     end
# end
# function ∇²F!(∇²f::SparseMatrixCSC, dofvector::Vector{T}, model::LandauModel{T}, Edep) where T
#     assemblers = [start_assemble(∇²f) for t=1:nthreads()]
#     @assemble! begin
#         reinitEdep!(cache, nodeids, Edep)
#         # force!(cache, mean(cache.coords), offset, forceconstant)
#         ForwardDiff.hessian!(cache.hessresult, cache.efunc, eldofs, cache.hessconf)
#         # @time ForwardDiff.hessian!(cache.hessian, cache.efunc, eldofs, cache.hessconf)
#         # ForwardDiff.jacobian!(cache.hessian, x->gradient!(cache.gradient, cache.efunc, eldofs, cache.gradconf), eldofs, cache.jacconf, Val{false}())
#         # ForwardDiff.jacobian!(cache.hessian, cache.gfunc, eldofs, cache.jacconf, Val{false}())
#         @inbounds assemble!(assemblers[threadid()], cache.indices, DiffResults.hessian(cache.hessresult))
#     end
# end
