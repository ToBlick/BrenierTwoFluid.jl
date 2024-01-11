using Distances
using Plots
using LatinHypercubeSampling
using Sobol
using BrenierTwoFluid
using LinearAlgebra
using Random
using LaTeXStrings

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

δ = 5e-2

N = 30^2 # Int(ceil(δ^(-3/4*d)))
M = N
d′ = 2*floor(d/2)
ε = δ^2 #N^(-1/(d′+4))
sqrt(ε)

K₀ = 0.25

Δt = 1/50
λ = 2*K₀/δ^2 # Δt^(-2) # N^(d/2) * N

ε^-1 * N^(-1/2) * log(N)

t = 0

q = 0.8
Δ = 1.0

α = ones(N) / N
β = ones(M) / M

#=
r = sqrt.(rand(N))
θ = rand(N) * 2π
X = hcat( r.*cos.(θ), r.*sin.(θ) ) .* 0.5

r = sqrt.(rand(M))
θ = rand(M) * 2π
Y = hcat( r.*cos.(θ), r.*sin.(θ) ) .* 0.5
=#
Random.seed!(123)

X = rand(N,d) .- 0.5;
Y = rand(M,d) .- 0.5;

#uniform grid
for k in 1:Int(sqrt(M))
    for l in 1:Int(sqrt(M))
        Y[(k-1)*Int(sqrt(M)) + l,:] .= [ k/(Int(sqrt(M))) - 1/(2*Int(sqrt(M))), l/(Int(sqrt(M))) - 1/(2*Int(sqrt(M)))] .- 1/2
    end
end
X .= Y .+ rand(N,d) * sqrt(ε) .- sqrt(ε)/2

scatter(X[:,1], X[:,2], label = false, color = :blue)
scatter!(Y[:,1], Y[:,2], label = false, color = :red)

X .= X[sortperm(X[:,1]), :]
Y .= Y[sortperm(Y[:,1]), :];

u0(x) = [-cos(π*x[1])*sin(π*x[2]), sin(π*x[1])*cos(π*x[2])]
V = zero(X)
for i in axes(X)[1]
    V[i,:] .= u0(X[i,:])
end

norm(V)^2/2/N

p0(x) = 0.5 * (sin(π*x[1])^2 + sin(π*x[2])^2)
∇p(x) = π * [sin(π*x[1])*cos(π*x[1]), sin(π*x[2])*cos(π*x[2])]

solX = []
solV = []
solD = []
sol∇S = []

CC = CostCollection(X, Y, c)
V1 = SinkhornVariable(X,α)
V2 = SinkhornVariable(Y,β)
S = SinkhornDivergence(V1,V2,CC; ε=ε, q=q, Δ=Δ, tol=1e-3)
∇S = zero(X)
initialize_potentials!(S.V1,S.V2,S.CC)
@time compute!(S)
push!(solX, copy(X))
push!(solV, copy(V))
push!(solD, value(S))
push!(sol∇S, copy(x_gradient!(∇S, S, ∇c)))

j = 1
plt = scatter(solX[j][1:div(N,3),1], solX[j][1:div(N,3),2], label = false, color = :blue)
scatter!(solX[j][div(N,3)+1:div(2N,3),1], solX[j][div(N,3)+1:div(2N,3),2], label = false, color = :green)
scatter!(solX[j][div(2N,3)+1:end,1], solX[j][div(2N,3)+1:end,2], label = false, color = :red)
scatter!(Y[:,1],Y[:,2], label = false, color = :black, markersize=1)
plt

@time while t < 1.0

    X .+= 0.5 * Δt * V
    
    #=for i in axes(X,1)
        for j in axes(X,2)
            if X[i,j] > 0.5
                X[i,j] -= D[j]
            elseif X[i,j] < -0.5
                X[i,j] += D[j]
            end
        end
    end=#

    S.params.s = Δ
    initialize_potentials!(V1,V2,CC)
    compute!(S)
    x_gradient!(∇S, S, ∇c)
    push!(solD, value(S))
    push!(sol∇S, copy(∇S))
    V .-= Δt * λ .* ∇S ./ α
    
    #for i in axes(X,1)
    #    V[i,:] .-= Δt * ∇p(X[i,:])
    #end

    X .+= 0.5 * Δt * V
    
    #=for i in axes(X,1)
        for j in axes(X,2)
            if X[i,j] > 0.5
                X[i,j] -= D[j]
            elseif X[i,j] < -0.5
                X[i,j] += D[j]
            end
        end
    end=#

    push!(solX, copy(X))
    push!(solV, copy(V))
    #push!(solD, value(S))

    t += Δt
    #Y .= rand(M,d) .- 0.5;
end

solK = [norm(V)^2 for V in solV]/2/N;

solK

plot(solK .- solK[1], linewidth=2, label=L"\frac{1}{2} \sum_i V_i^2(t) - \frac{1}{2} \sum_i V_i^2(0)", legend = :bottomright)
plot!(λ/2/N * (solD .- solD[1]), linewidth=2, label=L"\frac{\lambda}{2} S_\varepsilon(t) - \frac{\lambda}{2} S_\varepsilon(0)")

T = length(solX)
j = T
plt = scatter(solX[j][1:div(N,3),1], solX[j][1:div(N,3),2], label = false, color = :blue)
scatter!(solX[j][div(N,3)+1:div(2N,3),1], solX[j][div(N,3)+1:div(2N,3),2], label = false, color = :green)
scatter!(solX[j][div(2N,3)+1:end,1], solX[j][div(2N,3)+1:end,2], label = false, color = :red)

Π = TransportPlan(S);
sum(Matrix(Π))

f(x,y) = cos(π*x) * cos(π*y);

F = [ f(solX[1][i,1],solX[1][i,2]) for i in 1:N ];

anim = @animate for i in 1:T
    scatter(solX[i][:,1], solX[i][:,2]; label = false, zcolor = F, color=:seismic,
    ylim = (-0.6,0.6), xlim = (-0.6,0.6))
end
gif(anim, "anim_fps10.gif", fps = 10)

scatter(solX[end][:,1], solX[end][:,2]; label = false, zcolor = F, color=:seismic,
    ylim = (-0.6,0.6), xlim = (-0.6,0.6), size = (1000,1000), colorbar = false)

quiver!(X[:,1],X[:,2],quiver= -Δt * λ .* (sol∇S[end][:,1] ,sol∇S[end][:,2] ),
        alpha = 0.5, color = :red, size = (1000,1000))
quiver!(X[:,1],X[:,2],quiver= -Δt .* ([∇p(X[i,:])[1] for i in axes(X,1)], [∇p(X[i,:])[2] for i in axes(X,1)]),
        alpha = 0.5, color = :blue)
#scatter!(X[:,1],X[:,2],color=:black,legend = false, markersize = 1)
#scatter!(Y[:,1],Y[:,2],color=:blue)

savefig("pressure.pdf")

Statistics.median(λ .* [norm(sol∇S[end][i,:]) for i in axes(X,1)])

Statistics.median([norm(∇p(X[i,:])) for i in axes(X,1)])


1


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
solD = []
push!(solX, copy(X))
push!(solV, copy(V))
push!(solD, 0.0)

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
    push!(solD, value(dist))
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
=#