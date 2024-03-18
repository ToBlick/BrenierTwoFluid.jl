using BrenierTwoFluid
using Test
using Distances
using Random
using LinearAlgebra

@testset "BrenierTwoFluid.jl" begin
    include("costtests.jl")
    include("distancetests.jl")
    #include("gradienttests.jl")
    include("barycentertests.jl")
end
