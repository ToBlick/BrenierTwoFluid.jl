using BrenierTwoFluid
using Distances
using Plots
using LinearAlgebra
using Random
using LaTeXStrings
using ProgressBars
using HDF5
using Dates

function run_euler(path, d, c, ∇c, seed, Δt, T, λ², ε, q, Δ, s, tol, crit_it, p_ω, max_it, sym, acc)
    Random.seed!(seed)

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
    X .= Y #.+ rand(N,d) * δ .- δ/2   # wiggle by δ
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
    # no scaling, no symmetrization, with acceleration
    params = SinkhornParameters(ε=ε,q=q,Δ=Δ,s=s,tol=tol,crit_it=crit_it,p_ω=p_ω,max_it=max_it,sym=sym,acc=acc);
    S = SinkhornDivergence(SinkhornVariable(X, α),
                           SinkhornVariable(Y, β),
                           c,params,true)
    initialize_potentials!(S)
    valS = compute!(S)

    #K₀ = 0.5 * dot(V,diagm(α) * V) #0.25    # initial kinetic energy
    #λ² = (Δt)^(-2) # 2*K₀/(δ^2) # 2*K₀/(δ^2 - valS)                  # relaxation to enforce dist < δ
          
    t = 0
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
    sol∇S[1] = copy(x_gradient!(S, ∇c));

    # integrate
    for it in ProgressBar(1:nt)

        X .+= 0.5 * Δt * V

        S.params.s = s  # if scaling is used it should be reset here
        #initialize_potentials!(S)
        compute!(S)
        ∇S = x_gradient!(S, ∇c)

        # note: the gradient alrady comes divided by the weights
        V .-= Δt .* λ² .* ∇S
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
        sol∇S[1+it] = copy(x_gradient!(S, ∇c))
    end

    fid = h5open(path, "w")
    fid["X"] = [ solX[i][j,k] for i in eachindex(solX), j in axes(X,1), k in axes(X,2) ];
    fid["V"] = [ solV[i][j,k] for i in eachindex(solV), j in axes(V,1), k in axes(V,2) ];
    fid["D"] = solD
    fid["alpha"] = α
    fid["beta"] = α
    fid["grad"] = [ sol∇S[i][j,k] for i in eachindex(sol∇S), j in axes(X,1), k in axes(X,2) ];
    fid["lambda"] = sqrt(λ²)
    fid["epsilon"] = ε
    fid["tol"] = tol
    fid["crit_it"] = crit_it
    fid["p"] = p_ω
    fid["deltat"] = Δt
    close(fid)

    return S
end

### Set output file
path = "runs/$(now()).hdf5"
###

### parameters
d = 2
c = (x,y) -> 0.5 * sqeuclidean(x,y)
∇c = (x,y) -> x-y

d′ = 2*floor(d/2)
ε = 0.01    # entropic regularization parameter
λ² = 5000

N = 50^2 #Int((ceil(1e-1/ε))^(d))  
#N = Int((ceil(1e-2/ε))^(d′+4))                  # particle number
M = N #Int((ceil(N^(1/d))^d))

q = 1.0         # ε-scaling rate
Δ = 1.0         # characteristic domain size
s = ε           # initial scale (ε)
tol = 1e-4      # tolerance on marginals (absolute)
crit_it = 20 # Int(ceil(0.1 * Δ / ε))    # when to compute acceleration
p_ω = 2         # acceleration heuristic
T = 1.0

sym = false
acc = true

seed = 123

Δt = 1/50

max_it = 10000

λ² * Δt^2

S = run_euler(path, d, c, ∇c, seed, Δt, T, λ², ε, q, Δ, s, tol, crit_it, p_ω, max_it, sym, acc)


#=
params_coarse = SinkhornParameters(ε=ε,q=q,Δ=Δ,s=ε,tol=tol,crit_it=crit_it,p_ω=p_ω,max_it=max_it,sym=sym,acc=acc);
S = SinkhornDivergence(S.V1,S.V2,S.CC,params_coarse,true);
scale(S)
initialize_potentials!(S);
compute!(S);

Π = Matrix(TransportPlan(S));
suppΠ = zero(Π);
for i in axes(Π,1), j in axes(Π,2)
    suppΠ[i,j] = (S.V1.f[i] + S.V2.f[j] - S.CC.C_xy[i,j] > -3ε ? 1 : 0)
end

sum(suppΠ)/(N^2)

Παα = (S.V1.α*S.V1.α') .* exp.( (-S.CC.C_xx + S.V1.h * ones(N)' + ones(N)*S.V1.h') / ε );
Πββ = (S.V2.α*S.V2.α') .* exp.( (-S.CC.C_yy + S.V2.h * ones(N)' + ones(N)*S.V2.h') / ε );

(dot(S.CC.C_xy, Π) + ε * dot(log.(Π) .- log.(S.V1.α*S.V2.α'), Π) 
- 0.5 * (dot(S.CC.C_xx, Παα) + ε * dot(log.(Παα) .- log.(S.V1.α*S.V1.α'), Παα))
- 0.5 * (dot(S.CC.C_yy, Πββ) + ε * dot(log.(Πββ) .- log.(S.V2.α*S.V2.α'), Πββ)) )

# dot(S.CC.C_xy, S.V1.α*S.V2.α') - 0.5*dot(S.CC.C_xx, S.V1.α*S.V1.α') - 0.5*dot(S.CC.C_yy, S.V2.α*S.V2.α')

dot(S.V1.f - S.V1.h, S.V1.α) + dot(S.V2.f - S.V2.h, S.V2.α)

Π*ones(N)*N
Π'*ones(N)*N

svdΠ = svd(Π);

plot(svdΠ.S[1:1000], yaxis = :log)

scatter(S.V1.X[:,1], S.V1.X[:,2], S.V1.f, markersize=1)
scatter(S.V2.X[:,1], S.V2.X[:,2], λ²/2 .* S.V2.f, markersize=1)

p0(x) = 0.5 * (sin(π*x[1])^2 + sin(π*x[2])^2)

scatter(S.V2.X[:,1], S.V2.X[:,2], [p0(S.V2.X[i,:]) for i in 1:N], markersize=1)
=#