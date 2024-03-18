module BrenierTwoFluid

using Distances
using Base.Threads
using Plots
using LaTeXStrings
using LinearAlgebra
using KrylovKit

include("costs.jl")
export LazyCost, CostCollection, c_periodic, ∇c_periodic, scale, set_scale!

include("sinkhornvariable.jl")
export SinkhornVariable, initialize_potentials_nolog!, initialize_potentials_log!

include("sinkhornparameters.jl")
export SinkhornParameters

include("sinkhorndivergence.jl")
export SinkhornDivergence, softmin, sinkhorn_step!, value, compute!, x_gradient!, x_gradient, y_gradient, y_gradient!
export maxit, tol, acceleration, marginal_error, scale, set_scale!
export initialize_potentials!

include("barycenter.jl")
export SinkhornBarycenter, compute!

include("transportplans.jl")
export TransportPlan, transportmatrix

include("plotting.jl")
export plot

include("rbf.jl")
export bspline, kde

include("resampling.jl")
export resample!

end
