using Distances
using Plots
using LatinHypercubeSampling
using Sobol
using BrenierTwoFluid
using LinearAlgebra
using Random
using Statistics

d = 2

function c_periodic(x::VT,y::VT,D) where {T,VT <: AbstractVector{T}}
    d = 0
    for i in eachindex(x)
        if x[i] - y[i] > D[i]/2
            d += (x[i] - y[i] - D[i])^2
        elseif x[i] - y[i] < -D[i]/2
            d += (x[i] - y[i] + D[i])^2
        else
            d += (x[i] - y[i])^2
        end
    end
    0.5 * d
end

function ∇c_periodic(x,y,D)
    ∇c = zero(x)
    for i in eachindex(x)
        if x[i] - y[i] > D[i]/2
            ∇c[i] = x[i] - y[i] - D[i]
        elseif x[i] - y[i] < -D[i]/2
            ∇c[i] = (x[i] - y[i] + D[i])
        else
            ∇c[i] = x[i] - y[i]
        end
    end
    ∇c
end

c = (x,y) -> 0.5 * sqeuclidean(x,y)
∇c = (x,y) -> x-y
#D = 1*ones(2);
#c = (x,y) -> c_periodic(x,y,D)
#∇c = (x,y) -> ∇c_periodic(x,y,D)

S_vec = []
∇S_vec = []
N_vec = []

for i in 1:5

    d′ = 2*floor(d/2)
    N = 64*2^i
    M = N
    ε = N^(-4/3/d) # N^(-1/(d′+4))  #  #
    q = 0.7
    Δ = 1.0

    α = ones(N) / N
    β = ones(M) / M

    Random.seed!(123)

    X = rand(N,d) .- 0.5
    Y = rand(M,d) .- 0.5

    CC = CostCollection(X, Y, c)
    V1 = SinkhornVariable(X,α)
    V2 = SinkhornVariable(Y,β)
    S = SinkhornDivergence(V1, V2, CC; ε=ε, q=q, Δ=Δ, tol=1e-3)
    ∇S = zero(X)

    initialize_potentials!(V1,V2,CC)
    @time valS = compute!(S);
    x_gradient!(∇S, S, ∇c);

    norm∇S = [ norm(∇S[i,:]) for i in axes(X,1) ];

    push!(N_vec, N)
    push!(S_vec, value(S))
    push!(∇S_vec, mean(norm∇S))
end

plot(N_vec, abs.(S_vec), minorgrid=true, legend = :bottomleft,
    xlabel = "N", linewidth = 2, xaxis = :log, yaxis=:log, label = "S" )
plot!(N_vec, ∇S_vec, linewidth = 2, label = "|∇S|" )
plot!(N_vec, 0.01 * N_vec .^(-2/(d′+4)), label = "0.01 N^(-2/(d′+4))" )
plot!(N_vec, 0.01 * N_vec .^(-4/3/d), label = "0.01 N^(-4/3/d)" )
plot!(N_vec, 0.01 * N_vec .^(-1), label = "0.01 N^(-1)" )