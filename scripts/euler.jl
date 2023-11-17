using Distances
using Plots
using LatinHypercubeSampling
using BrenierTwoFluid

const N = 16^2
const M = 16^2
const d = 2
gens = 1000

c = (x,y) -> 0.5 * sqeuclidean(x,y)
∇c = (x,y) -> x-y

α = ones(N) / N
β = ones(M) / M

@time plan, _ = LHCoptim(N+M,2,gens);
scaled_plan = scaleLHC(plan,[(-0.5,0.5),(-0.5,0.5)])
X = scaled_plan[1:N,:]
Y = scaled_plan[N+1:end,:];

#uniform grid
for k in 1:Int(sqrt(M))
    for l in 1:Int(sqrt(M))
        Y[(k-1)*Int(sqrt(M)) + l,:] .= [ k/(Int(sqrt(M))+1), l/(Int(sqrt(M))+1) ] .- 1/2
    end
end

scatter(X[:,1], X[:,2], label = false, color = :blue)
scatter!(Y[:,1], Y[:,2], label = false, color = :red)

X .= X[sortperm(X[:,1]), :]
Y .= Y[sortperm(Y[:,1]), :];

u0(x) =(-cos(π*x[1])*sin(π*x[2]), sin(π*x[1])*cos(π*x[2]))
V = zero(X)
for i in axes(X)[1]
    V[i,:] .= u0(X[i,:])
end

solX = []
solV = []
push!(solX, copy(X))
push!(solV, copy(V));

j = 1
plt = scatter(solX[j][1:div(N,3),1], solX[j][1:div(N,3),2], label = false, color = :blue)
scatter!(solX[j][div(N,3)+1:div(2N,3),1], solX[j][div(N,3)+1:div(2N,3),2], label = false, color = :green)
scatter!(solX[j][div(2N,3)+1:end,1], solX[j][div(2N,3)+1:end,2], label = false, color = :red)
scatter!(Y[:,1],Y[:,2], label = false, color = :black, markersize=1)
plt

ε = 1/N^(2/d)
λ = N^(d/2)
t = 0
Δt = 0.5 / λ
q = 0.9

CC = CostCollection(X, Y, c)
V1 = SinkhornVariable(X,α)
V2 = SinkhornVariable(Y,β)
initialize_potentials!(V1,V2,CC)
S = SinkhornDivergence(V1,V2,CC;ε=ε,q=q);
∇S = zero(X);

@time while t < 0.5

    X .+= Δt * V

    S.s[1] = ε / q^3
    compute!(S)
    x_gradient!(∇S, S, ∇c)
    V .-= Δt * λ .* ∇S ./ α

    push!(solX, copy(X))
    push!(solV, copy(V))

    t += Δt
end

T = length(solX)
j = T
plt = scatter(solX[j][1:div(N,3),1], solX[j][1:div(N,3),2], label = false, color = :blue)
scatter!(solX[j][div(N,3)+1:div(2N,3),1], solX[j][div(N,3)+1:div(2N,3),2], label = false, color = :green)
scatter!(solX[j][div(2N,3)+1:end,1], solX[j][div(2N,3)+1:end,2], label = false, color = :red)






#=
X = zeros(N,2)
for i in 1:N
    t = -π + (i-1)/N * 2π
    X[i,:] .= [0.1 * cos(t) - 0.5, 0.1 * sin(t) - 0.25]
end
Y = zeros(M,2)
for i in 1:M
    t = -π + (i-1)/N * 2π
    Y[i,:] .= [cos(t) - 1/(1+4t^2), sin(t)]
end
=#

#=
plt = scatter(X[:,1], X[:,2], label = false, color = :blue)
scatter!(Y[:,1], Y[:,2], label = false, color = :red)
for i in eachindex(α), j in eachindex(β)
    plot!([X[i,1], Y[j,1]], [X[i,2], Y[j,2]], alpha=sqrt(N*M)*sol.plan[i,j],legend=:false, color = :black, grid = false, size = (800,800))
end
plt
=#

"""
Potential is equal to S_ε(α, unif) = S_ε( ∑_i α_i δx_i,  ∑_j β_j δy_j ), hence 
∇_X_i(t) S_ε(α, unif) is known
"""

Y = rand(M,2)
    for k in 1:Int(sqrt(M))
        for l in 1:Int(sqrt(M))
            Y[(k-1)*Int(sqrt(M)) + l,:] .= [ k/(Int(sqrt(M))+1), l/(Int(sqrt(M))+1) ] .- 1/2
        end
    end

X = copy(Y) # rand(N,2) .- 1/2
X .= X[sortperm(X[:,1]), :]

u0(x) =(-cos(π*x[1])*sin(π*x[2]), sin(π*x[1])*cos(π*x[2]))

σ = 1/N^(1/d)
q = 0.9
Δ = 1
λ = N^(d/2)

V = zero(X)
for i in axes(X)[1]
    V[i,:] .= u0(X[i,:])
end

solX = []
solV = []
push!(solX, copy(X))
push!(solV, copy(V))

j = 1
plt = scatter(solX[j][1:div(N,3),1], solX[j][1:div(N,3),2], label = false, color = :blue)
scatter!(solX[j][div(N,3)+1:div(2N,3),1], solX[j][div(N,3)+1:div(2N,3),2], label = false, color = :green)
scatter!(solX[j][div(2N,3)+1:end,1], solX[j][div(2N,3)+1:end,2], label = false, color = :red)
plt

t = 0
Δt = 1 / λ

f = zero(α)
g = zero(β)
h_x = zero(α)
h_y = zero(β)

@time while t < 0.5

    X .+= Δt * V
    for x in X
        #if x > 1/2
        #    x = x - 2*(x-1/2)
        #elseif x < -1/2
        #    x = x + 2*(x+1/2)
        #end
    end
    dist = sinkhorndivergence(X, α, Y, β, c, f, g, h_x, h_y, σ/q^3 , σ, q);
    V .-= Δt * λ .* dist.grad_x

    push!(solX, copy(X))
    push!(solV, copy(V))

    f .= dist.f
    g .= dist.g
    h_x .= dist.h_x
    h_y .= dist.h_y

    t += Δt
end

T = length(solX)
j = T
plt = scatter(solX[j][1:div(N,3),1], solX[j][1:div(N,3),2], label = false, color = :blue)
scatter!(solX[j][div(N,3)+1:div(2N,3),1], solX[j][div(N,3)+1:div(2N,3),2], label = false, color = :green)
scatter!(solX[j][div(2N,3)+1:end,1], solX[j][div(2N,3)+1:end,2], label = false, color = :red)
#scatter!(solX[1][:,1], solX[1][:,2], label = false, color = :black, markersize = 1)

gradp(x) = [π * cos(π*x[1]) * sin(π*x[1]), π * cos(π*x[2]) * sin(π*x[2])]

dist = sinkhorndivergence(X, α, Y, β, c, f, g, h_x, h_y, σ/q^3 , σ, q);

plt = scatter(X[:,1], X[:,2], label = false, size = (1000,800), color = :black)
scatter!(Y[:,1], Y[:,2], label = false, size = (1000,800), color = :gray, markersize = 1)
for i in eachindex(α)
    plot!([X[i,1], X[i,1] - 2 * dist.grad_x[i,1]], 
          [X[i,2], X[i,2] - 2 * dist.grad_x[i,2]], 
          legend=:false, color = :red)
    plot!([X[i,1], X[i,1] - 0.02 * gradp(X[i,:])[1]], 
          [X[i,2], X[i,2] - 0.02 * gradp(X[i,:])[2]],
          legend=:false, color = :blue)
end
plt


λ

2/0.02

#=
K = div(M+N,2)
μ = ones(K) / K;
Z = (copy(X) + copy(Y))/2

#plt = scatter(Z[:,1], Z[:,2], label = false, color = :purple, xlim = (-0.1,1.1), ylim = (-0.1,1.1))

plt = scatter(X[:,1], X[:,2], label = false)
for t in [0.25, 0.5, 0.75]
    for i in 1:3
        sol1 = sinkhorndivergence(Z, μ, X, α, c, 1, 3e-2, 0.9);
        sol2 = sinkhorndivergence(Z, μ, Y, β, c, 1, 3e-2, 0.9);

        Z .-= t * sol1.grad_x
        Z .-= (1-t) * sol2.grad_x
    end
    scatter!(Z[:,1], Z[:,2], label = false)
end
scatter!(Y[:,1], Y[:,2], label = false, xlim = (-1.5,0.7), ylim = (-1.1,1.1), size = (600,600))


for i in eachindex(μ), j in eachindex(α)
    plot!([Z[i,1], X[j,1]], [Z[i,2], X[j,2]], 
        alpha=sqrt(length(α)*length(μ))*sol1.plan[i,j],
        legend=:false, color = :black)
end
plt

scatter!(Z[:,1], Z[:,2], label = false, color = :red)

plt = scatter(X[:,1], X[:,2], label = false, color = :blue, xlim = (-0.1,1.1), ylim = (-0.1,1.1))
scatter!(Y[:,1], Y[:,2], label = false, color = :red)
scatter!(Z[:,1], Z[:,2], label = false, color = :purple)
#for i in eachindex(α), j in eachindex(β)
#    plot!([X[i,1], Y[j,1]], [X[i,2], Y[j,2]], alpha=sqrt(N*M)*π_αβ[i,j],legend=:false, color = :black)
#end
plt

sol1.grad_x
    for i in eachindex(μ)
        plot!([Z[i,1], Z[i,1] - sol1.grad_x[i,1]], 
              [Z[i,2], Z[i,2] - sol1.grad_x[i,2]], 
              legend=:false, color = :red)
        plot!([Z[i,1], Z[i,1] - sol2.grad_x[i,1]], 
              [Z[i,2], Z[i,2] - sol2.grad_x[i,2]], 
              legend=:false, color = :blue)
    end
    scatter!(X[:,1], X[:,2], color = :red)
    scatter!(Y[:,1], Y[:,2], color = :blue)
    plt

=#