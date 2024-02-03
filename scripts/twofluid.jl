using Distances
using Plots
using LatinHypercubeSampling
using Sobol
using BrenierTwoFluid
using LinearAlgebra
using Random
using LaTeXStrings

Random.seed!(123)

d = 2

c = (x,y) -> 0.5 * sqeuclidean(x,y)
∇c = (x,y) -> x-y
#D = 1*ones(2);
#c = (x,y) -> c_periodic(x,y,D)
#∇c = (x,y) -> ∇c_periodic(x,y,D)

δ = 5e-2
N = 25^2 # Int(ceil(δ^(-3/4*d)))
M = N
d′ = 2*floor(d/2)
ε = δ^2 #N^(-1/(d′+4))
sqrt(ε)

mα = 1
mβ = 1

Zβ = 1

X = rand(N,d) .- 0.5;
Y = rand(M,d) .- 0.5;
#uniform grid
for k in 1:Int(sqrt(M))
    for l in 1:Int(sqrt(M))
        Y[(k-1)*Int(sqrt(M)) + l,:] .= [ k/(Int(sqrt(M))) - 1/(2*Int(sqrt(M))), l/(Int(sqrt(M))) - 1/(2*Int(sqrt(M)))] .- 1/2
    end
end
X .= Y .+ rand(N,d) * sqrt(ε) .- sqrt(ε)/2
Y .+= rand(N,d) * sqrt(ε) .- sqrt(ε)/2

α = ones(N) / N
β = ones(M) / M

X .= X[sortperm(X[:,1]), :]
Y .= Y[sortperm(Y[:,1]), :];

U = zero(X)
for i in axes(X)[1]
    U[i,:] .= 0.1
end

V = zero(Y)
for i in axes(Y)[1]
    V[i,:] .= 0
end

K₀ = mα * norm(U)^2/2/N + mβ * norm(V)^2/2/N
Δt = 1/20
λ = 2*K₀/δ^2 # Δt^(-2) # N^(d/2) * N
t = 0
q = 0.8
Δ = 1.0

solX = []
solY = []
solU = []
solV = []
solD = []

Z = rand(N,d)
Z .= Y .+ rand(N,d) * sqrt(ε) .- sqrt(ε)/2
μ = ones(N) / N

Vα = SinkhornVariable(X,α)
Vβ = SinkhornVariable(Y,β)
Vμ = SinkhornVariable(Z,μ);

CCαβ = CostCollection(X, Y, c)
CCμα = CostCollection(Z, X, c)
CCμβ = CostCollection(Z, Y, c)
CCαμ = CostCollection(X, Z, c)
CCβμ = CostCollection(Y, Z, c)

Sαβ = SinkhornDivergence(Vα, Vβ, CCαβ; ε=ε, q=q , Δ=Δ, tol = 1e-2);
Vμ, val = barycenter_sinkhorn!(ω, Vμ, [Vα, Vβ], [CCμα, CCμβ], ∇c, (Sαβ.params), 5, δ);
val

Sαμ = SinkhornDivergence(Vα, Vμ, CCαμ; ε=ε, q=q , Δ=Δ, tol = 1e-3)
Sβμ = SinkhornDivergence(Vβ, Vμ, CCβμ; ε=ε, q=q , Δ=Δ, tol = 1e-3)

push!(solX, copy(X))
push!(solY, copy(Y))
push!(solU, copy(U))
push!(solV, copy(V))
push!(solD, val)

scatter(X[:,1],X[:,2],color=:red, label = "α")
scatter!(Y[:,1],Y[:,2],color=:green, label = "β")
scatter!(Z[:,1],Z[:,2],color=:blue, label = "μ")

@time while t < 1.0

    X .+= 0.5 * Δt * U
    Y .+= 0.5 * Δt * V

    Z .= 0.5 .* X .+ 0.5 .* Y

    # compute barycenter
    Vμ, val, ∇S_vec = barycenter_sinkhorn!(ω, Vμ, [Vα, Vβ], [CCμα, CCμβ], ∇c, (Sαβ.params), 5, δ);
    # compute forces
    # on α
    Sαμ.params.s = Δ
    initialize_potentials!(Sαμ.V1,Sαμ.V2,CCαμ)
    compute!(Sαμ)
    ∇_α_Sαμ = x_gradient(Sαμ, ∇c)
    # on β
    Sβμ.params.s = Δ
    initialize_potentials!(Sβμ.V1,Sβμ.V2,CCβμ)
    compute!(Sβμ)
    ∇_β_Sβμ = x_gradient(Sβμ, ∇c)

    U .-= Δt * λ .* ∇_α_Sαμ ./ α ./ mα
    V .-= Δt * λ .* ∇_β_Sβμ ./ β ./ mβ ./ Zβ

    X .+= 0.5 * Δt * U
    Y .+= 0.5 * Δt * V

    push!(solX, copy(X))
    push!(solY, copy(Y))
    push!(solU, copy(U))
    push!(solV, copy(V))
    push!(solD, val)

    t += Δt
end

solK = [mα * norm(solU[t])^2/2/N + mβ * norm(solV[t])^2/2/N for t in eachindex(solU)]

solK

plot(solK .- solK[1], linewidth=2, label=L"\frac{1}{2} \sum_i V_i^2(t) - \frac{1}{2} \sum_i V_i^2(0)", legend = :bottomright)
plot!(λ/2 * (solD .- solD[1]), linewidth=2, label=L"\frac{\lambda}{2} S_\varepsilon(t) - \frac{\lambda}{2} S_\varepsilon(0)")

T = length(solX)
j = T

scatter(solX[j][:,1],solX[j][:,2],color=:red, label = "α")
scatter!(solY[j][:,1],solY[j][:,2],color=:green, label = "β")

anim = @animate for j in 1:T
    scatter(solX[j][:,1],solX[j][:,2],color=:red, label = "α", ylim = (-0.6,0.6), xlim = (-0.6,0.6))
    scatter!(solY[j][:,1],solY[j][:,2],color=:green, label = "β", ylim = (-0.6,0.6), xlim = (-0.6,0.6))
end
gif(anim, "anim_fps10.gif", fps = 10)

sqrt.(abs.(solD))