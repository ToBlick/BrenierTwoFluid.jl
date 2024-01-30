using BrenierTwoFluid
using Test
using Distances
using Random
using LinearAlgebra

@testset "BrenierTwoFluid.jl" begin
    include("distancetests.jl")
    include("gradienttests.jl")
end
