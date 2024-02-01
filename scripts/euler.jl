using BrenierTwoFluid
using Distances
using Plots
using LinearAlgebra
using Random
using LaTeXStrings
using ProgressBars
using HDF5
using Dates

### Set output file
results = "runs/$(now()).hdf5"
###

Random.seed!(123)

d = 2
c = (x,y) -> 0.5 * sqeuclidean(x,y)
∇c = (x,y) -> x-y

d′ = 2*floor(d/2)
δ = 0.05    # spatial tolerance 
ε = δ^2     # entropic regularization parameter

N = Int((ceil(1e-1/ε))^(d))  
#N = Int((ceil(1e-2/ε))^(d′+4))                  # particle number
M = N #Int((ceil(N^(1/d))^d))

q = 1.0         # ε-scaling rate
Δ = 1.0         # characteristic domain size
s = ε           # initial scale (ε)
tol = 1e-1 * δ      # tolerance on marginals (absolute)
crit_it = Int(ceil(0.05 * Δ / ε))    # when to compute acceleration
p_ω = 2         # acceleration heuristic

# initial conditions - identical
α = ones(N) / N
β = ones(M) / M
X = rand(N,d) .- 0.5;
Y = rand(M,d) .- 0.5;
#  uniform grid for background density
for k in 1:Int(sqrt(M))
    for l in 1:Int(sqrt(M))
        Y[(k-1)*Int(sqrt(M)) + l,:] .= [ k/(Int(sqrt(M))) - 1/(2*Int(sqrt(M))), l/(Int(sqrt(M))) - 1/(2*Int(sqrt(M)))] .- 1/2
    end
end
#X .= Y .+ rand(N,d) * δ .- δ/2   # wiggle by δ
X .= X[sortperm(X[:,1]), :]
Y .= Y[sortperm(Y[:,1]), :];

# initial velocity
u0(x) = [-cos(π*x[1])*sin(π*x[2]), sin(π*x[1])*cos(π*x[2])]
V = zero(X)
for i in axes(X)[1]
    V[i,:] .= u0(X[i,:])
end

# calculate initial distance
# Setup Sinkhorn
CC = CostCollection(X, Y, c)
V1 = SinkhornVariable(X, α)
V2 = SinkhornVariable(Y, β)
# no scaling, no symmetrization, with acceleration
params = SinkhornParameters(CC;ε=ε,q=1.0,Δ=1.0,s=s,tol=tol,crit_it=crit_it,p_ω=p_ω,sym=false,acc=true);
S = SinkhornDivergence(V1,V2,CC,params)
∇S = zero(X)
initialize_potentials!(S.V1,S.V2,S.CC)
@time valS = compute!(S)
δ^2
@assert δ^2 > valS

K₀ = 0.5 * dot(V,diagm(α) * V) #0.25    # initial kinetic energy
λ² = 2*K₀/(δ^2 - valS)                  # relaxation to enforce dist < δ

Δt = 1/25                               # time-step               
t = 0

Δt * √(λ²) * √2

T = 1.0     # final time
nt = Int(ceil((T-t)/Δt))

p0(x) = 0.5 * (sin(π*x[1])^2 + sin(π*x[2])^2)
∇p(x) = π * [sin(π*x[1])*cos(π*x[1]), sin(π*x[2])*cos(π*x[2])]

solX = [ zero(X) for i in 1:(nt + 1) ]
solV = [ zero(V) for i in 1:(nt + 1) ]
solD = [ 0.0 for i in 1:(nt + 1) ]
sol∇S = [ zero(X) for i in 1:(nt + 1) ]

solX[1] = copy(X)
solV[1] = copy(V)
solD[1] = value(S)
sol∇S[1] = copy(x_gradient!(∇S, S, ∇c));

# plot initial condition and background
j = 1
plt = scatter(solX[j][1:div(N,3),1], solX[j][1:div(N,3),2], label = false, color = :blue)
scatter!(solX[j][div(N,3)+1:div(2N,3),1], solX[j][div(N,3)+1:div(2N,3),2], label = false, color = :green)
scatter!(solX[j][div(2N,3)+1:end,1], solX[j][div(2N,3)+1:end,2], label = false, color = :red)
scatter!(Y[:,1],Y[:,2], label = false, color = :black, markersize=1)
plt

# integrate
_, time = @timed for it in ProgressBar(1:nt)

    X .+= 0.5 * Δt * V

    S.params.s = s  # if scaling is used it should be reset here
    initialize_potentials!(V1,V2,CC)
    compute!(S)
    x_gradient!(∇S, S, ∇c)

    V .-= Δt .* λ² .* ∇S ./ α
    # exact dynamics
    #for i in axes(V,1)
    #    V[i,:] .-= Δt * ∇p(X[i,:])
    #end

    X .+= 0.5 .* Δt .* V

    # diagnostics
    #initialize_potentials!(V1,V2,CC)
    #compute!(S)
    solX[1+it] = copy(X)
    solV[1+it] = copy(V)
    solD[1+it] = value(S)
    sol∇S[1+it] = copy(x_gradient!(∇S, S, ∇c))
end

fid = h5open(results, "w")
fid["X"] = [ solX[i][j,k] for i in eachindex(solX), j in axes(X,1), k in axes(X,2) ];
fid["V"] = [ solV[i][j,k] for i in eachindex(solV), j in axes(V,1), k in axes(V,2) ];
fid["D"] = solD
fid["alpha"] = α
fid["beta"] = α
fid["grad"] = [ sol∇S[i][j,k] for i in eachindex(sol∇S), j in axes(X,1), k in axes(X,2) ];
fid["delta"] = δ
fid["lambda"] = sqrt(λ²)
fid["epsilon"] = ε
fid["tol"] = tol
fid["crit_it"] = crit_it
fid["p"] = p_ω
fid["deltat"] = Δt
close(fid)