module Landau
    using Reexport
    @reexport using Tensors
    using JuAFEM
    using ForwardDiff
    using Optim, LineSearches
    using Base.Threads
    using InlineExports

    include("utils.jl")
    include("model.jl")
    include("assembly.jl")
    include("energy.jl")
    include("parameters.jl")
    include("startingconditions.jl")
    include("grid.jl")

    export LandauModel
    export Flandau, Fginzburg, Felastic, Felectrostriction, Fflexoelectric
    export F, ∇F!,∇²F!
    export GLDparameters
    export startingconditions!
    export DofNode, dofnodes
    export construct_stencils


end # module
