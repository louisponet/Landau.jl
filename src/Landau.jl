module Landau
    using Reexport
    @reexport using JuAFEM
    @reexport using Tensors
    using Base.Threads
    using LinearAlgebra
    using SparseArrays
    using ForwardDiff
    
    include("utils.jl")
    include("model.jl")
    include("assembly.jl")
    include("grid.jl")

    export LandauModel
    export F, ∇F!,∇²F!
    export startingconditions!
    export DofNode, dofnodes
    export construct_stencils
end # module
