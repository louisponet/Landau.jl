# using Pkg
# cd(homedir() * "/.julia/environments/Landau2/")
# Pkg.activate(".")

using Landau
using JLD2
using Plots

dirname = String(@__DIR__)
cd(@__DIR__)

αT(temp::T) where {T} = Vec{6, T}((-0.27054e8, -6.3817e8, 3.230e8, 7.8936e9, 4.470e9, 4.910e9))

BTOParams(T=300.0) = GLDparameters(αT(T), 5.1e-10,    -0.20e-10,  0.20e-10,
                                              2.75e11,     1.79e11,   0.543e11,
                                              0.1104,     -0.0452,    0.0289,
                                              0.3094e-11, -0.279e-11,-0.1335e-11)


searchdir(path::String, key) = filter(x -> occursin(key, x), readdir(path))

BTOModel(T::Float64, args...; kwargs...) = LandauModel(BTOParams(T), args...; kwargs...)

left = Vec{3}((-50.e-10, -2.e-10, -25.e-10 ))
right = Vec{3}((50.e-10, 2.e-10, 25.e-10))
model = BTOModel(300.0, [(:u, 3),(:P, 3)], (100, 4, 50), left, right, (x)->nothing; lagrangeorder=1);
dnodes = dofnodes(model.dofhandler)

cd("/home/ponet/Documents/PhD/BTO/smallhertz/")
@load "finaldofs.jld" finishedofs
startdofs = copy(finishedofs)

dirnames = filter(x->isdir(dirname * "/$x"), searchdir(dirname, "job"))
for n in dirnames
    @load "$n/finaldofs.jld" finishedofs
    reldofs = finishedofs - startdofs

    xcoords = Float64[]
    ycoord = minimum(x -> abs(x.coord[2]), dnodes)
    zcoord = 25.e-10
    toplot = Float64[]
    for node in dnodes
        if node.coord[1] ∉ xcoords && node.coord[2] == ycoord && node.coord[3] == zcoord
            push!(xcoords, node.coord[1])
            push!(toplot, reldofs[node.dofs.u[3]])
        end
    end
    plt = plot(xcoords .* 1e10, toplot*3.3e-1, xlabel="x (angstrom)", ylabel = "u - u0 (angstrom)")
    savefig("$n/deltausurface.png", plt)
end
