using Distances
using Plots
using LatinHypercubeSampling
using Sobol
using BrenierTwoFluid
using LinearAlgebra
using Random
using Statistics

using ApproxFun

d = 2

_x = -0.5:0.01:0.5

c = (x,y) -> 0.5 * sqeuclidean(x,y)
∇c = (x,y) -> x-y

sp = (-1..1)^d
x,y = Fun(sp)

A = [Dirichlet(sp,1); Laplacian()]

N = 1000

X = rand(N,2) .* 2 .- 1 .- 0.01;

A