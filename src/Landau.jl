module Landau
    using Reexport
    @reexport using JuAFEM
    @reexport using Tensors
    using Base.Threads
    using LinearAlgebra
    using SparseArrays
    using ForwardDiff
    using NearestNeighbors
    @reexport using Optim
    using ProgressMeter
    using Serialization
    
    include("utils.jl")
    include("model.jl")
    include("assembly.jl")
    include("grid.jl")
    include("optimization.jl")

    export LandauModel
    export F, ∇F!,∇²F!
    export startingconditions!
    export DofNode, dofnodes
    export construct_stencils
    export extract_data, extract_data_line
    export voigt_to_tensor, write_vtu_files
end # module
