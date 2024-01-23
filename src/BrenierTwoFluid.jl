module BrenierTwoFluid

using Distances
using Base.Threads
using Plots
using LaTeXStrings
using LinearAlgebra

include("costs.jl")
export LazyCost, CostCollection #,LazySlice

include("sinkhornvariable.jl")
export SinkhornVariable, initialize_potentials!, positions, density, logdensity, potential, debiasing_potential

include("sinkhorndivergence.jl")
export SinkhornDivergence, SinkhornParameters, softmin, sinkhorn_step!, value, compute!, x_gradient!, x_gradient, y_gradient, y_gradient!

include("barycenter.jl")
export barycenter_sinkhorn!

include("transportplans.jl")
export TransportPlan, transportmatrix

include("plotting.jl")
export plot

end
