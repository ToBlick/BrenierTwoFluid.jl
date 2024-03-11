# compute W_2 distance between two uniform distributions
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
α = ones(N) / N
β = ones(M) / M

d′ = 2*Int(floor(d/2))
d′′ = 2*Int(ceil(d/2))
ε = 0.1                 # entropic regularization. √ε is a length.
q = 1.0                 # annealing parameter
Δ = 1.0                 # characteristic domain size
s = ε                   # current scale: no annealing -> equals ε
tol = 1e-8              # marginal condition tolerance
crit_it = 20            # acceleration inferrence
p_η = 2

offset = 0.5

Random.seed!(123)
X = rand(N,d) .- 0.5
Y = rand(M,d) .- 0.5

for i = 1:2
        if i == 1 # Scaling test
                truevalue = 1/2 * d * 1/12 * offset^2 #1/2 * d * offset^2
                Y .= (rand(M,d) .- 0.5) .* (1 + offset)
        else # Shifting test
                truevalue = 1/2 * d * offset^2
                Y .= (rand(M,d) .- 0.5) .+ offset
        end

        CC = CostCollection(X, Y, c)
        V = SinkhornVariable(X,α)
        W = SinkhornVariable(Y,β)

        # acc, no sym
        params = SinkhornParameters(ε=ε,q=1.0,Δ=1.0,s=s,tol=tol,crit_it=crit_it,p_η=p_η,sym=false,acc=true)
        S = SinkhornDivergence(V,W,c,params,true)
        initialize_potentials!(S)
        @time valueS = compute!(S)
        ∇S_x = x_gradient!(S, ∇c)
        ∇S_y = y_gradient!(S, ∇c)

        if i == 1
                @test dot(X - ∇S_x - (1 + offset) * X, α .* (X - ∇S_x - (1 + offset) * X)) < (sqrt(N*M))^(-4/(2*d′′ + 8)) * log(sqrt(N*M))
                println(dot(X - ∇S_x - (1 + offset) * X, α .* (X - ∇S_x - (1 + offset) * X)))
                @test dot(Y - ∇S_y - 1/(1 + offset) * Y, β .* (Y - ∇S_y - 1/(1 + offset) * Y)) < (sqrt(N*M))^(-4/(2*d′′ + 8)) * log(sqrt(N*M))
                println(dot(Y - ∇S_y - 1/(1 + offset) * Y, β .* (Y - ∇S_y - 1/(1 + offset) * Y)))
        else
                @test dot(∇S_x + offset * X ./ X, α .* (∇S_x + offset * X ./ X)) < (sqrt(N*M))^(-4/(2*d′′ + 8)) * log(sqrt(N*M))
                println(dot(∇S_x + offset * X ./ X, α .* (∇S_x + offset * X ./ X)))
                @test dot(∇S_y - offset * Y ./ Y, β .* (∇S_y - offset * Y ./ Y)) < (sqrt(N*M))^(-4/(2*d′′ + 8)) * log(sqrt(N*M))
                println(dot(∇S_y - offset * Y ./ Y, β .* (∇S_y - offset * Y ./ Y)))
        end
end
