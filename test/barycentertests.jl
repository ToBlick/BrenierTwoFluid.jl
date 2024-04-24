# compute W_2 barycenter between two distributions shifted by +/- an offset - barycenter should be the original distribution
using BrenierTwoFluid
using Test
using Distances
using Random
using LinearAlgebra
using Plots

const dotime = false
# const dotime = true

c = (x,y) -> 0.5 * sqeuclidean(x,y)
∇c = (x,y) -> x - y
d = 2

N = 32^2
M = N
α = ones(N) / N
β = ones(M) / M
μ = ones(N) / N

X = rand(N,d) .- 0.5
Y = rand(M,d) .+ 0.5
Z = rand(N,d) .+ 0.1
Z_true = rand(N,d)

Vα = SinkhornVariable(X,α)
Vβ = SinkhornVariable(Y,β)
Vμ = SinkhornVariable(Z,μ)

V_vec = [Vα, Vβ];

ε = 0.1
tol = 1e-5

params = SinkhornParameters(ε=ε,tol=tol,sym=false,acc=true);

ω = [0.5, 0.5];
B = SinkhornBarycenter(ω, [ SinkhornDivergence(Vμ, V_vec[i], c, params, islog=true) for i in eachindex(V_vec) ], ∇c, 10, 1e-3, zero(Z));

dotime ? @time(compute!(B)) : compute!(B)

Vμ = SinkhornVariable(Z, μ);
Vμ_true = SinkhornVariable(Z_true, μ);
S = SinkhornDivergence(Vμ_true, Vμ, c, params, islog=true);
@test compute!(S) < 1e-3
