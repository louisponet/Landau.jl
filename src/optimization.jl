"""
    optimize(model::LandauModel;
             force::Union{Vector{<:AbstractFloat}, Nothing} = nothing,
             fixed_fields::Vector{Symbol}                   = Symbol[],
             savedir::Union{AbstractString, Nothing}        = nothing,
             kwargs...)

Minimize the energy for the Landau `model`.
`force` is an optional vector containing a possible force field that can be used in the energy.
Any fields in `fixed_fields` will be kept constant during the minimization.
In the optional `savedir` the temporary dofs will be during the optimization iterations.
Optional `kwargs` are passed to the optimization options. 
"""
function Optim.optimize(model::LandauModel;
                        force::Union{Vector{<:AbstractFloat}, Nothing} = nothing,
                        fixed_fields::Vector{Symbol}                   = Symbol[],
                        savedir::Union{AbstractString, Nothing}        = nothing,
                        kwargs...)
    dofs       = model.dofs
    ∇f_storage = fill(0.0, length(dofs))
    # Setting up possible save directory
    if savedir !== nothing
        if ispath(savedir)
            rm.(filter(x -> occursin("iteration", x) && occursin("vtu", x), readdir(savedir)))
            rm.(filter(x -> occursin("dofs", x) && occursin("dat", x), readdir(savedir)))
        else
            savedir = mkpath(savedir)
            savedir = mkdir(savedir)
        end
        vtk_save(joinpath(savedir, "start.vtu"), model, model.dofs)
    end

    fixed_dof_ids = map(s -> dof_range(model.dofhandler, s), fixed_fields)    
    function g!(storage, x)
       force !== nothing ?  ∇F!(storage, x, model, force) : ∇F!(storage, x, model)
        if !isempty(fixed_fields)
            for cell in JuAFEM.CellIterator(model.dofhandler)
    	        globaldofs = JuAFEM.celldofs(cell)
    	        for idx in 1:length(cell.nodes)
        	        for r in fixed_dof_ids
            	        storage[globaldofs[r]] .= 0.0
        	        end
    	        end
            end
        end
        JuAFEM.apply_zero!(storage, model.boundaryconds)
    end
    function f(x)
        return force !== nothing ? F(x, model, force) : F(x, model)
    end

    od = OnceDifferentiable(f, g!, dofs, 0.0, ∇f_storage)
    function cb(x)
        savedir !== nothing &&
            serialize(joinpath(savedir, "dofs$(x.iteration).dat"), od.x_f)
        return false
    end
    optimize_options = merge((g_tol=1e-6, iterations=3000, allow_f_increases=true, show_trace=true, show_every=1, callback=cb), kwargs) 
    res = optimize(od, dofs, ConjugateGradient(), Optim.Options(;optimize_options...))
    return res
end

iteration_range(savedir) = parse.(Int, map(x -> splitext(x)[1][5:end], filter(x -> occursin("dat", x), readdir(savedir))))

"""
    write_vtu_files(model::LandauModel, savedir::AbstractString, is = iteration_range(savedir))

Writes the vtu files for the iteration dof data saved in `savedir`.
"""
function write_vtu_files(model::LandauModel, savedir::AbstractString, is = iteration_range(savedir))
    @info "Writing a total of $(length(is)) vtu files."
    p = Progress(length(is), 1, "Writing files...")
    Threads.@threads for i in is
        dfs = deserialize(joinpath(savedir, "dofs$(i).dat"))
        vtk_save(joinpath(savedir, "iteration$(i).vtu"), model, dfs)
        next!(p)
    end
end
