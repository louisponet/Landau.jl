module Landau
    using Reexport
    @reexport using Tensors
    using JuAFEM
    using ForwardDiff
    using Optim, LineSearches
    using Base.Threads

    include("utils.jl")
    include("model.jl")
    include("assembly.jl")
    include("energy.jl")

    export LandauModel
    export F, ∇F!,∇²F!


end # module
