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
const PATH_OUT = "runs/$(now()).hdf5"
###

### parameters
const d = 2
const c = (x,y) -> 0.5 * sqeuclidean(x,y)
const ∇c = (x,y) -> x-y

const N = 30^2
const M = 30^2

const DOMAIN = (1.0, 1.0)   # only cubes for now

const T = 1.0               # final time

const ENTROPIC_REG = 0.1 * N^(-1/3)    # entropic regularization parameter ε

const SCALING_RATE = 1.0    # ε-scaling rate
const R = 0.5 * sum([l^2 for l in DOMAIN])   # L infinity norm of the cost function
const INITIAL_SCALE = ENTROPIC_REG     # initial scale (ε for no scaling, C_INFTY otherwise)
const SINKHORN_TOLERANCE = 1e-3   # tolerance to end the Sinkhorn iterations. Relative change of the dual potentials. L_inf norm of the marginals violations is also an option.
const SINKHORN_MAX_IT = 100 # maximum number of Sinkhorn iterations
const ACCELERATION_IT = 20  # when to compute acceleration. Set this to SINKHORN_MAX_IT + 1 when using a fixed, given value for the acceleration parameter.
const INITIAL_ETA = 1.5     # initial value for the acceleration parameter
const ACCELERATION_P = 2    # acceleration is computed using the difference of the value at ACCELERATION_IT - ACCELERATION_P and at ACCELERATION_IT.
const TOLERANCE_FREQUENCY = 2 # how often to check the tolerance

const PRECISE_DIAGNOSTICS = false # whether to re-compute the Sinkhorn divergence at the end of each time step

const SYM = false           # symmetrization
const ACC = true            # acceleration
const LOG = false            # log-domain
const DEB = true            # debiasing
const SAFE = true           # safe stopping criterion versus fixed heuristic number of iterations

const SEED = 123            # for reproducibility

const DELTA_T = 1/50                    # time step
const LAMBDA_SQUARE = 2 / (DELTA_T^2)   # relaxation parameter

const X_ON_GRID = true
const Y_ON_GRID = true

const u0(x) = [-cos(π*x[1])*sin(π*x[2]), sin(π*x[1])*cos(π*x[2])]
const p0(x) = 0.5 * (sin(π*x[1])^2 + sin(π*x[2])^2)
const ∇p(x) = π * [sin(π*x[1])*cos(π*x[1]), sin(π*x[2])*cos(π*x[2])]

function run_euler()
    Random.seed!(SEED)

    α = ones(N) / N
    β = ones(M) / M
    Y = stack(vec([ [x,y] for x in range(-0.5,0.5,length=Int(sqrt(M))), y in range(-0.5,0.5,length=Int(sqrt(M))) ]), dims = 1)
    X = stack(vec([ [x,y] for x in range(-0.5,0.5,length=Int(sqrt(N))), y in range(-0.5,0.5,length=Int(sqrt(N))) ]), dims = 1)
    X_ON_GRID ? nothing : X = rand(N,d) .- 0.5
    Y_ON_GRID ? nothing : Y = rand(M,d) .- 0.5

    # initial velocity
    V = zero(X)
    for i in axes(X)[1] V[i,:] .= u0(X[i,:]) end

    # Setup Sinkhorn
    params = SinkhornParameters(ε=ENTROPIC_REG,
                                q=SCALING_RATE,
                                Δ=R,
                                s=INITIAL_SCALE,
                                tol=SINKHORN_TOLERANCE,
                                η=INITIAL_ETA,
                                crit_it=ACCELERATION_IT,
                                p_η=ACCELERATION_P,
                                max_it=SINKHORN_MAX_IT,
                                tol_it=TOLERANCE_FREQUENCY,
                                sym=SYM,
                                acc=ACC,
                                deb=DEB,
                                safe=SAFE,
                                );
    S = SinkhornDivergence(SinkhornVariable(X, α),
                           SinkhornVariable(Y, β),
                           c,
                           params,
                           LOG)
    initialize_potentials!(S)
    compute!(S)
          
    t = 0
    Δt = DELTA_T
    nt = Int(ceil((T-t)/Δt))

    solX = [ zero(X) for i in 1:(nt + 1) ]
    solV = [ zero(V) for i in 1:(nt + 1) ]
    solD = [ 0.0 for i in 1:(nt + 1) ]
    sol∇S = [ zero(X) for i in 1:(nt + 1) ]

    solX[1] = copy(X)
    solV[1] = copy(V)
    solD[1] = value(S)
    sol∇S[1] = copy(x_gradient!(S, ∇c))
    # integrate
    for it in ProgressBar(1:nt)

        X .+= 0.5 * Δt * V

        SCALING_RATE == 1.0 ? nothing : set_scale!(S, INITIAL_SCALE)

        initialize_potentials!(S)
        compute!(S)
        ∇S = x_gradient!(S, ∇c)

        V .-= Δt .* λ² .* ∇S
        
        X .+= 0.5 .* Δt .* V

        if PRECISE_DIAGNOSTICS
            initialize_potentials!(S)
            compute!(S)
            ∇S = x_gradient!(S, ∇c)
        end

        solX[1+it] = copy(X)
        solV[1+it] = copy(V)
        solD[1+it] = value(S)
        sol∇S[1+it] = copy(x_gradient!(S, ∇c))
        sol_species[1+it] = copy(species)
        sol_alpha[1+it] = copy(α)
    end

    fid = h5open(path, "w")
    fid["X"] = [ solX[i][j,k] for i in eachindex(solX), j in axes(X,1), k in axes(X,2) ];
    fid["V"] = [ solV[i][j,k] for i in eachindex(solV), j in axes(V,1), k in axes(V,2) ];
    fid["D"] = solD
    fid["alpha"] = [ sol_alpha[i][j] for i in eachindex(sol_alpha), j in axes(α,1) ];
    fid["species"] = [ sol_species[i][j] for i in eachindex(sol_species), j in axes(species,1) ]
    fid["beta"] = α
    fid["grad"] = [ sol∇S[i][j,k] for i in eachindex(sol∇S), j in axes(X,1), k in axes(X,2) ];
    fid["lambda"] = sqrt(λ²)
    fid["epsilon"] = ε
    fid["tol"] = tol
    fid["crit_it"] = crit_it
    fid["p"] = p_η
    fid["deltat"] = Δt
    close(fid)

    return S
end


S = run_euler()

#=
Π = Matrix(TransportPlan(S));

suppΠ = zero(Π);
for i in axes(Π,1), j in axes(Π,2)
    suppΠ[i,j] = (S.V1.f[i] + S.V2.f[j] - S.CC.C_xy[i,j] > -5ε ? 1 : 0)
end
sum(suppΠ)/(N^2)

svdΠ = svd(Π);
plot(svdΠ.S ./ svdΠ.S[1], yaxis = :log)
=#

#=
params_coarse = SinkhornParameters(ε=ε,q=q,Δ=Δ,s=ε,tol=tol,crit_it=crit_it,p_η=p_η,max_it=max_it,sym=sym,acc=acc);
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