using BrenierTwoFluid
using Test
using Distances
using Random
using LinearAlgebra

c = (x,y) -> 0.5 * sqeuclidean(x,y)
∇c = (x,y) -> x - y
d = 3
N = 30^2
M = 30^2
ε = 0.1

Random.seed!(123)
α = ones(N) / N
β = ones(M) / M
X = rand(N,d) .- 0.5
Y = rand(M,d)

params = SinkhornParameters(ε = ε);

S = SinkhornDivergence(SinkhornVariable(X, α),
                        SinkhornVariable(Y, β),
                        c,
                        params,
                        true);

C_naive = 0.5 * [ norm(X[i,:] - Y[j,:])^2 for i in 1:N, j in 1:M ];
norm_C = norm(C_naive)

@test norm(S.CC.C_xy - C_naive) < 1e-12
@test norm(S.CC.C_xy - S.CC.C_yx') < 1e-12
@test norm( exp.(-S.CC.C_xy ./ scale(S)) - S.CC.K_xy) < 1e-12

set_scale!(S, 0.5)
@test norm( exp.(-S.CC.C_xy ./ 0.5) - S.CC.K_xy) < 1e-12