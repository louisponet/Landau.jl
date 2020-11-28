using Revise
using Landau
using LinearAlgebra
# Following: [Marton, P., & Hlinka, J. (2010). https://doi.org/10.1103/PhysRevB.81.144125]
function f_Landau(P, α1, α11, α12, α111, α112, α123)
    P2 = (P[1]^2, P[2]^2, P[3]^2)
    P4 = (P2[1]^2, P2[2]^2, P2[3]^2)
    return (α1 * sum(P2) +
           α11 * sum(P4) +
           α12 * (P2[1] * (P2[2] + P2[3]) + P2[2]*P2[3]) +
           α111 * dot(P2,P4) +
           α112 * (P4[1]*(P2[2] + P2[3]) + P4[2]*(P2[1] + P2[3]) + P4[3]*(P2[1] + P2[2])) +
           α123 * prod(P2))
end

f_Ginzburg(∇P, G) = 0.5 * ((G ⊡ ∇P) ⊡ ∇P) # ⊡ signifies double contraction  
f_elastic(ε, C)   = 0.5 * ((C ⊡ ε) ⊡ ɛ)
f_electrostriction(P, ε, q) = -((q ⊡ ε) ⋅ P) ⋅ P # ⋅ is single contraction
f_flexoelectric(∇P, ε, F, C) = 0.5 * ((∇P ⊡ F) ⊡ (C ⊡ ε))

f_tot(P, ε, ∇P, mp) = f_Landau(P, mp.α1, mp.α11, mp.α12, mp.α111, mp.α112, mp.α123) +
                      f_Ginzburg(∇P, mp.G) +
                      f_elastic(ε, mp.C) +
                      f_electrostriction(P, ε, mp.Q) 
                      # f_flexoelectric(∇P, ε, mp.F, mp.C)

# This function will be evaluated to calculate the energy contribution of each element.
# up is the vector of degrees of freedom corresponding to the edges of the current element,
# cellvalues are the preevaluated geometric values to interpolate the function values
# at the central points inside the cell.
# extradata holds more info that can be used inside this function such as ranges of up
# that correspond to the different fields (they are stored contiguous).
# mp are the model parameters
function element_energy(up::Vector{T}, cellvalues, extradata, mp) where {T}
    energy = zero(T)    
    # add contribution to the energy for each quadrature point inside the element
    # with weight getdetJdV(cellvalues.u, qp).
    # function_value, function_gradient and function_symmetric_gradient calculate the interpolated values
    # of the fields and their spatial gradients.
    for qp=1:getnquadpoints(cellvalues.u)
        P  = function_value(cellvalues.P, qp, up, extradata.ranges.P)
        ∇P = function_gradient(cellvalues.P, qp, up, extradata.ranges.P)
        ε  = function_symmetric_gradient(cellvalues.u, qp, up, extradata.ranges.u)
        energy += (f_tot(P, ε, ∇P, mp)) * getdetJdV(cellvalues.u, qp)
    end
    return energy
end
                                            
# Definition of fields (name, dimension, initial value function f(r))
u_field = (:u, 3, x -> Vec(0.0,0.0,0.0))
P_field = (:P, 3, x -> x[1] > 0.0 ? Vec(0.0, 0.0, -0.2) : Vec(0.0, 0.0, 0.2))

# The 1e20 factor is to make the energy a reasonable order of magnitude.
# const is to tell Julia that the type of model_params is not going to change.
const model_params = (
	α1   = -2.7054e7 * 1e-10,
	α11  = -6.3817e8 * 1e-10,
	α111 = 7.8936e9  * 1e-10,
	α12  = 3.23e8    * 1e-10,
	α112 = 4.47e9    * 1e-10,
	α123 = 4.91e9    * 1e-10,
	G = voigt_to_tensor(51e-11, -2e-11, 2e-11) * 1e-10,
	C = voigt_to_tensor(27.5e10, 17.9e10, 5.43e10) * 1e-10,
	Q = voigt_to_tensor(14.2e9, -0.74e9, 1.57e9) * 1e-10,
	F = voigt_to_tensor(0.3094e9, -0.279e9, -0.1335e9) * 1e-10,
) #1e-10 so energy scales as unity and we can use angstrom lengthscales

gridsize = (20, 20, 20)
left = Vec(-20.0, -20.0, -20.0) # nm
right = Vec(20.0, 20.0, 20.0)

# We set the boundary condition of the 3rd dimension of the u field at
# the bottom boundary to not vary at all x and t.
boundary_conditions = [(:u, "bottom",(x, t) -> [0.0], [3])]

# This function will be used in Landau.jl, it should always take 3 parameters,
# and output 1 value for the energy/element.
# So what we do here is basically "fill in" the model parameters and then pass it to the algorithm
element_function =
    (up, cellvalues, extra) -> element_energy(up, cellvalues, extra, model_params)
    
model = LandauModel([u_field, P_field],
                    gridsize,
                    left,
                    right,
                    element_function,
                    boundaryconds = boundary_conditions);

# Minimize the energy
res = optimize(model)

# we save the solution to view with paraview
vtk_save("solution_noforce.vtu", model, res.minimizer)

# store the new dofvector
model.dofs = res.minimizer

# Define the energy density when a force is applied to u.
function element_energy_force(up::Vector{T}, cellvalues, extradata, mp) where {T}
    energy = zero(T)    
    for qp=1:getnquadpoints(cellvalues.u)
        P  = function_value(cellvalues.P, qp, up, extradata.ranges.P)
        ∇P = function_gradient(cellvalues.P, qp, up, extradata.ranges.P)
        ε  = function_symmetric_gradient(cellvalues.u, qp, up, extradata.ranges.u)
        δforce = function_value(cellvalues.u, qp, extradata.force)
        δu = function_value(cellvalues.u, qp, up, extradata.ranges.u)
        energy += (f_tot(P, ε, ∇P, mp) -  δu[3] * δforce[3]) * getdetJdV(cellvalues.u, qp)
    end
    return energy
end

element_function =
    (up, cellvalues, extra) -> element_energy_force(up, cellvalues, extra, model_params)

model = LandauModel(model, element_function);

# Make a gaussian force field applied to the surface
center = Vec(4.0, 0.0, 20.0)
nodes = model.dofhandler.grid.nodes
nnodes = length(nodes)
const force = fill(0.0, 3 * nnodes) # const to make calculations faster
for i = 1:nnodes
    crel  = nodes[i].x - center 
    r     = sqrt(crel[1]^2 + crel[2]^2)

    force[(i-1)*3 + 3] = -exp(-r^2/3) * exp(-4 * abs(crel[3]))
end

# now optimize energy with the applied force
res = optimize(model, force=force)

vtk_save("solution.vtu", model, res.minimizer)
