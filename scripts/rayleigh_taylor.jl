using BrenierTwoFluid
using Distances
using Plots
using LinearAlgebra
using Random
using LaTeXStrings
using ProgressBars
using HDF5
using Dates

###
# WIP!
###

### Set output file
path = "runs/$(now()).hdf5"
###

### parameters
d = 2
c = (x,y) -> 0.5 * sqeuclidean(x,y)
∇c = (x,y) -> x-y

d′ = 2*floor(d/2)
ε = 0.001    # entropic regularization parameter

N = 40^2 #Int((ceil(1e-1/ε))^(d))  
#N = Int((ceil(1e-2/ε))^(d′+4))                  # particle number
M = N #Int((ceil(N^(1/d))^d))

q = 1.0         # ε-scaling rate
Δ = 1.0       # characteristic domain size
s = ε           # initial scale (ε)
tol = 1e-3      # tolerance on marginals (absolute)
max_it = 100
crit_it = 100    # when to compute acceleration
p_ω = 2         # acceleration heuristic

sym = false
acc = true

seed = 123

Δt = 1/200

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
    X .= Y
    X .= X[sortperm(X[:,1]), :]
    Y .= Y[sortperm(Y[:,1]), :];

    Mass = ones(N)
    for i in axes(X,1)
        if X[i,2] > 0.3 * cos(2π*X[i,1])
            Mass[i] = 3
        end
    end
    G = zero(X)
    G[:,2] .= -10.0;

    # initial velocity
    u0(x) = [0,0] #[-cos(π*x[1])*sin(π*x[2]), sin(π*x[1])*cos(π*x[2])]
    V = zero(X)
    for i in axes(X)[1]
        V[i,:] .= u0(X[i,:])
    end

    # calculate initial distance
    # Setup Sinkhorn
    params = SinkhornParameters(ε=ε,q=q,Δ=Δ,s=s,tol=tol,ω=1.5,
                                crit_it=crit_it,p_ω=p_ω,max_it=max_it,sym=sym,acc=acc,tol_it=2);
    S = SinkhornDivergence(SinkhornVariable(X, α),
                           SinkhornVariable(Y, β),
                           c,params,true)
    initialize_potentials!(S)
    ∇S = zero(X)
    initialize_potentials!(S)
    valS = compute!(S)
    
    K₀ = 0.5 * dot(V,diagm(α) * V) #0.25    # initial kinetic energy
    λ² = 2 * Δt^-2 # 2*K₀/(δ^2) # 2*K₀/(δ^2 - valS)                  # relaxation to enforce dist < δ
          
    t = 0

    T = 0.5     # final time
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

    plt = scatter()
    for i in axes(X,1)
        if Mass[i] == 1
            scatter!([X[i,1]], [X[i,2]], label = false, markerstrokewidth=0, markersize = 2.5, color = :black)
        else
            scatter!([X[i,1]], [X[i,2]], label = false, markerstrokewidth=0, markersize = 2.5, color = :red)
        end
    end
    plt

    # integrate
    for it in ProgressBar(1:nt)

        X .+= 0.5 * Δt * V

        S.params.s = s  # if scaling is used it should be reset here
        initialize_potentials!(S)
        compute!(S)
        ∇S = x_gradient!(S, ∇c)

        V .-= Δt .* λ² .* ∇S ./ Mass
        V .+= Δt .* G

        X .+= 0.5 .* Δt .* V

        # diagnostics
        #initialize_potentials!(V1,V2,CC)
        #compute!(S)
        solX[1+it] = copy(X)
        solV[1+it] = copy(V)
        solD[1+it] = value(S)
        sol∇S[1+it] = copy(x_gradient!(S, ∇c))
    end

nt

j = 100

    plt = scatter()
    for i in axes(solX[j],1)
        if Mass[i] == 1
            scatter!([solX[j][i,1]], [solX[j][i,2]], label = false, markerstrokewidth=0, markersize = 2.5, color = :black)
        else
            scatter!([solX[j][i,1]], [solX[j][i,2]], label = false, markerstrokewidth=0, markersize = 2.5, color = :red)
        end
    end
    plt


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


