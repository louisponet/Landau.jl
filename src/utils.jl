δ(i, j) = i == j ? one(i) : zero(i)
gaussian(x::Vec{DIM}, x0::Vec{DIM}, σ²) where DIM = 1/(2π * σ²)^(DIM/2) * ℯ^(-norm(x-x0)^2 / (2σ²))
