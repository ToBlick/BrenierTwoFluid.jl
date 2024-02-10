using Distances
using Plots
using LatinHypercubeSampling
using Sobol
using BrenierTwoFluid
using LinearAlgebra
using Random
using Statistics

d = 2
c = (x,y) -> 0.5 * sqeuclidean(x,y)
∇c = (x,y) -> x-y
#D = 1*ones(2);
#c = (x,y) -> c_periodic(x,y,D)
#∇c = (x,y) -> ∇c_periodic(x,y,D)

d′ = 2*floor(d/2)
δ = 5e-2
Int(ceil(δ^(-3/2*d)))
N = 20^2
M = N
ε = N^(-2/3/d) #N^(-1/(d′+4)) #1e-3 # 0.1 * δ^(1/2) #1/N^(2/d)
λ = 1000 # N^(d/2) * N
t = 0
Δt = 1/10 #1 / λ * N
q = 0.7
Δ = 1.0

α = ones(N) / N
β = ones(M) / M

#@time plan, _ = LHCoptim(N+M,2,gens);
#scaled_plan = scaleLHC(plan,[(-0.5,0.5),(-0.5,0.5)])
#X = scaled_plan[1:N,:]
#Y = scaled_plan[N+1:end,:];

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
X .= Y .+ rand(N,d) * 0.01 .- 0.005

scatter(X[:,1], X[:,2], label = false, color = :blue)
scatter!(Y[:,1], Y[:,2], label = false, color = :red)

X .= X[sortperm(X[:,1]), :]
Y .= Y[sortperm(Y[:,1]), :];

u0(x) = [-cos(π*x[1])*sin(π*x[2]), sin(π*x[1])*cos(π*x[2])]
V = zero(X)
for i in axes(X)[1]
    V[i,:] .= u0(X[i,:])
end
p0(x) = 0.5 * (sin(π*x[1])^2 + sin(π*x[2])^2)
∇p(x) = π * [sin(π*x[1])*cos(π*x[1]), sin(π*x[2])*cos(π*x[2])]

CC = CostCollection(X, Y, c)
V1 = SinkhornVariable(X,α)
V2 = SinkhornVariable(Y,β)
S = SinkhornDivergence(V1, V2, CC; ε=ε, q=q, Δ=Δ, tol=1e-3)
∇S = zero(X);
∇P = zero(X);

Δt = 0.2
X .+= 0.5*Δt * V;

initialize_potentials!(V1,V2,CC)
@time valS = compute!(S);
@time x_gradient!(∇S, S, ∇c);

for i in axes(X,1)
    ∇P[i,:] .= ∇p(X[i,:])
end
#=
plt = scatter(X[:,1], X[:,2]; label = false, color = :black, markersize = 1,
    ylim = (-0.6,0.6), xlim = (-0.51,0.51), size = (1000,1000))
for i in eachindex(α)
    plot!(  [X[i,1], X[i,1] - 0.02 * ∇S[i,1]], 
            [X[i,2], X[i,2] - 0.02 * ∇S[i,2]], 
            legend=:false, color = :red, alpha = 0.66)
    plot!(  [X[i,1], X[i,1] - 0.02 * ∇P[i,1]], 
            [X[i,2], X[i,2] - 0.02 * ∇P[i,2]], 
            legend=:false, color = :blue, alpha = 0.33)
    end
scatter!(Y[:,1], Y[:,2], label = false, color = :black, markersize = 1, alpha = 0.5)
plt
=#


valS
(value(S))

norm∇P = [ norm(∇P[i,:]) for i in axes(X,1) ];
norm∇S = [ norm(∇S[i,:]) for i in axes(X,1) ];

mean(norm∇P)
std(norm∇P)
median(norm∇P)

mean(norm∇S)
std(norm∇S)
median(norm∇S)

N
valS
ε^2

N^(-1/(d′+4))
plot([500,1000,2000,4000,8000,16000], [2.8e-5, 1.1e-5, 3.6e-6, 3.3e-6, 9.8e-7, 4.1e-7], minorgrid=true,
    xlabel = "N", ylabel = "avg. |∇S(Xᵢ)|",linewidth = 2, xaxis = :log, yaxis=:log, label = "random, ε = N^(-1/(d+4))" )
plot!([484,961,1936,3969,7921], [5.4e-6, 3.0e-6, 1.6e-6, 8.7e-7, 4.7e-7], linewidth = 2, label = "grid, ε = N^(-1/(d+4))" )
plot!([484,961,1936,3969,7921], [1.1e-5, 6.1e-6, 3.2e-6, 1.6e-6, 8.4e-7], linewidth = 2, label = "grid, ε = N^(-2/(3d))" )
plot!([500,1000,2000,4000,8000,16000], x -> 1/x * 1e-2, label = "0.01/N" )