module BrenierTwoFluid

using Distances
using Base.Threads
using Plots
using LaTeXStrings

include("costs.jl")
export LazyCost, CostCollection #,LazySlice

include("sinkhornvariable.jl")
export SinkhornVariable, initialize_potentials!, positions, density, logdensity, potential, debiasing_potential

include("sinkhorndivergence.jl")
export SinkhornDivergence, softmin, sinkhorn_step!, value, compute!, x_gradient!, x_gradient

include("transportplans.jl")
export TransportPlan, transportmatrix

include("plotting.jl")
export plot

end
