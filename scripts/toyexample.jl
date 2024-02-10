# compute W_2 distance between two uniform distributions
using BrenierTwoFluid
using Test
using Distances
using Random
using LinearAlgebra
using Plots
using Printf

p0(x) = 0.5 * (sin(π*x[1])^2 + sin(π*x[2])^2)
∇p(x) = π * [sin(π*x[1])*cos(π*x[1]), sin(π*x[2])*cos(π*x[2])]
u0(x) = [-cos(π*x[1])*sin(π*x[2]), sin(π*x[1])*cos(π*x[2])]

d = 2
N = 50^2
M = 50^2
α = ones(N) / N
β = ones(M) / M

Y = zeros(N,d)
for k in 1:Int(sqrt(M))
    for l in 1:Int(sqrt(M))
        Y[(k-1)*Int(sqrt(M)) + l,:] .= [ k/(Int(sqrt(M))) - 1/(2*Int(sqrt(M))), l/(Int(sqrt(M))) - 1/(2*Int(sqrt(M)))] .- 1/2
    end
end
X = copy(Y)

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
Δt = 1/25

results = []
for κ in [2^i for i in 0:10]
    ε = κ*(v̄ * 1/2 * Δt)^2
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

    #Π = Matrix(TransportPlan(S));
    #svdΠ = svd(Π);
    #plot(svdΠ.S[1:1000], yaxis = :log)

    ϕ = S.V1.f - S.V1.h;

    p0X = [p0(S.V1.X[i,:]) for i in 1:N];
    λ_vec = [10*i for i in 1:1000]
    err = [ norm(p0X .- sum(p0X)/N - λ * (ϕ .- sum(ϕ)/N), 2)/N 
            for λ in λ_vec ];

    push!(results, err)
end

λ_vec = [10*i for i in 1:1000]
plt = plot()
for i in eachindex(results)
    str = @sprintf "ε = %.2E" (2^(i-1)*(v̄ * 1/2 * Δt)^2)
    plot!(λ_vec, results[i], minorgrid = true,
        label = str, color=palette(:viridis,length(results))[i],
        yaxis = :log, ylabel = "|λϕ - p|₂", xlabel = "λ")
end
plt

scatter(S.V1.X[:,1], S.V1.X[:,2], λ_vec[k] * (ϕ .- sum(ϕ)/N), markersize=1)
#scatter(S.V2.X[:,1], S.V2.X[:,2], λ_vec[k] .* S.V2.f, markersize=1)
scatter!(S.V1.X[:,1], S.V1.X[:,2], p0X .- sum(p0X)/N, markersize=1)

plot(λ_vec, results[1])

minimum.(results)