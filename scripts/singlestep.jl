# compute W_2 distance between two uniform distributions
using BrenierTwoFluid
using Test
using Distances
using Random
using LinearAlgebra
using Plots
using Printf
using ProgressBars

p0(x) = 0.5 * (sin(π*x[1])^2 + sin(π*x[2])^2)
∇p(x) = π * [sin(π*x[1])*cos(π*x[1]), sin(π*x[2])*cos(π*x[2])]
u0(x) = [-cos(π*x[1])*sin(π*x[2]), sin(π*x[1])*cos(π*x[2])]

results = []
ε_vec = []
N_vec = []

Δt = 1/25

d = 2
N = 50^2
M = N
α = ones(N) / N
β = ones(M) / M

global Y = zeros(N,d)
for k in 1:Int(sqrt(M))
    for l in 1:Int(sqrt(M))
        Y[(k-1)*Int(sqrt(M)) + l,:] .= [ k/(Int(sqrt(M))) - 1/(2*Int(sqrt(M))), l/(Int(sqrt(M))) - 1/(2*Int(sqrt(M)))] .- 1/2
    end
end
global X = copy(Y)
Y .= rand(N,d) .- 0.5
X .= rand(N,d) .- 0.5

D = [1.0,1.0];

c = (x,y) -> 0.5 * sqeuclidean(x,y)
∇c = (x,y) -> x - y
#c = (x,y) -> c_periodic(x,y,D)
#∇c = (x,y) -> ∇c_periodic(x,y,D)

V = zero(X)
for i in axes(X)[1]
    V[i,:] .= u0(X[i,:])
end

v̄ = sum([norm(V[i,:]) for i in 1:N ])/N

d′ = 2*Int(floor(d/2))
q = 1.0
Δ = 1.0

tol = 1e-6

p_ω = 2

ε = 1e-3 # κ*(v̄ * 1/2 * Δt)^2
crit_it = maximum((Int(ceil(0.1 * Δ / ε)), 20))
s = ε
X .= Y .+ 1/2 * Δt .* V;
params = SinkhornParameters(ε=ε,
                            q=q,
                            Δ=Δ,
                            s=s,
                            tol=tol,
                            crit_it=crit_it,
                            max_it=10000,
                            p_ω=p_ω,
                            sym=false,
                            acc=true);
S = SinkhornDivergence(SinkhornVariable(X,α),
                    SinkhornVariable(Y,β),
                    c,params,true);
initialize_potentials!(S);
@time compute!(S)
marginal_error(S)

Π = Matrix(TransportPlan(S));
svdΠ = svd(Π);
plot(svdΠ.S[:]./svdΠ.S[1], yaxis = :log)

K = exp.(-S.CC.C_xy/ε);
svdK = svd(K);
plot(svdK.S[:]./svdK.S[1], yaxis = :log)

global ϕ = S.V1.f - S.V1.h;
global p0X = [p0(S.V1.X[i,:]) for i in 1:N];
λ_vec = [10*i for i in 1:1000]
err = [ norm(p0X .- sum(p0X)/N - λ * (ϕ .- sum(ϕ)/N), 2)/N 
        for λ in λ_vec ];
push!(results, err)
push!(N_vec, N)
push!(ε_vec, ε)



λ_vec = [10*i for i in 1:1000];
plt = plot();
for i in eachindex(results)
    str = @sprintf "ε = %.1E, N = %.1E" ε_vec[i] N_vec[i]
    plot!(λ_vec, results[i], minorgrid = true, legend = :bottomright,
        label = str, color=palette(:viridis,length(results))[i],
        yaxis = :log, ylabel = "|λϕ - p|₂", xlabel = "λ where Δt = $Δt")
end
plt

λ = 800 # 10*argmin(results[end])

scatter(X[:,1], X[:,2], p0X .- sum(p0X)/N, markersize = 1)
scatter!(X[:,1], X[:,2], λ * (ϕ .- sum(ϕ)/N), markersize = 1)