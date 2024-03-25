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
const DOMAIN = [1,1]
const D = DOMAIN
const c = (x,y) -> c_periodic_x(x,y,DOMAIN)
const ∇c = (x,y) -> ∇c_periodic_x(x,y,DOMAIN)

const N = 50^2
const M = 50^2

const T = 0.2             # final time

const ENTROPIC_REG = N^(-1/3)    # entropic regularization parameter ε

const SCALING_RATE = 1.0    # ε-scaling rate
const R = 0.5 * sum([l^2 for l in DOMAIN])   # L infinity norm of the cost function
const INITIAL_SCALE = ENTROPIC_REG     # initial scale (ε for no scaling, C_INFTY otherwise)
const SINKHORN_TOLERANCE = 1e-3   # tolerance to end the Sinkhorn iterations. Relative change of the dual potentials. L_inf norm of the marginals violations is also an option.
const SINKHORN_MAX_IT = 100 # maximum number of Sinkhorn iterations
const ACCELERATION_IT = 20  # when to compute acceleration. Set this to SINKHORN_MAX_IT + 1 when using a fixed, given value for the acceleration parameter.
const INITIAL_ETA = 1.0     # initial value for the acceleration parameter
const ACCELERATION_P = 2    # acceleration is computed using the difference of the value at ACCELERATION_IT - ACCELERATION_P and at ACCELERATION_IT.
const TOLERANCE_FREQUENCY = 2 # how often to check the tolerance

const PRECISE_DIAGNOSTICS = false # whether to re-compute the Sinkhorn divergence at the end of each time step

const SYM = false           # symmetrization
const ACC = true            # acceleration
const LOG = true            # log-domain
const DEB = true            # debiasing
const SAFE = true           # safe stopping criterion versus fixed heuristic number of iterations

const SEED = 123            # for reproducibility

const DELTA_T = 1/200                    # time step
const LAMBDA_SQUARE = 2 / (DELTA_T^2)   # relaxation parameter

const X_ON_GRID = true
const Y_ON_GRID = true

# initial velocity
const u0(x) = x[2] < 0.05 * cos(x[1]*2π*2) ? [1.0,0] : [0.5,0]

function run_kelvin_helmholtz()

    function enforce_periodicity_x!(X, D)
        for i in axes(X,1)
            if X[i,1] > D[1]/2
                X[i,1] -= D[1]
            elseif X[i,1] < -D[1]/2
                X[i,1] += D[1]
            end
        end
    end

    Random.seed!(SEED)

    α = ones(N) / N
    β = ones(M) / M
    #initial spacing
    h = 1.0 / sqrt(N)
    Y = stack(vec([ [x,y] for x in range(-D[1]/2,D[1]/2 - h,length=Int(sqrt(M))), y in range(-D[2]/2,D[2]/2,length=Int(sqrt(M))) ]), dims = 1)
    X = stack(vec([ [x,y] for x in range(-D[1]/2,D[1]/2 - h,length=Int(sqrt(M))), y in range(-D[2]/2,D[2]/2,length=Int(sqrt(N))) ]), dims = 1)
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

    λ² = LAMBDA_SQUARE
    ε  = ENTROPIC_REG

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
        enforce_periodicity_x!(X, D)

        SCALING_RATE == 1.0 ? nothing : set_scale!(S, INITIAL_SCALE)

        initialize_potentials!(S)
        compute!(S)
        ∇S = x_gradient!(S, ∇c)

        V .-= Δt .* λ² .* ∇S
        
        X .+= 0.5 .* Δt .* V
        enforce_periodicity_x!(X, D)

        if PRECISE_DIAGNOSTICS
            initialize_potentials!(S)
            compute!(S)
            ∇S = x_gradient!(S, ∇c)
        end

        solX[1+it] = copy(X)
        solV[1+it] = copy(V)
        solD[1+it] = value(S)
        sol∇S[1+it] = copy(x_gradient!(S, ∇c))
    end

    fid = h5open(PATH_OUT, "w")
    fid["X"] = [ solX[i][j,k] for i in eachindex(solX), j in axes(X,1), k in axes(X,2) ]
    fid["V"] = [ solV[i][j,k] for i in eachindex(solV), j in axes(V,1), k in axes(V,2) ]
    fid["D"] = solD
    fid["alpha"] = α
    fid["beta"] = α
    fid["grad"] = [ sol∇S[i][j,k] for i in eachindex(sol∇S), j in axes(X,1), k in axes(X,2) ]
    fid["lambda"] = sqrt(λ²)
    fid["epsilon"] = ε
    fid["tol"] = SINKHORN_TOLERANCE
    fid["crit_it"] = ACCELERATION_IT
    fid["p"] = ACCELERATION_P
    fid["deltat"] = Δt

    close(fid)

    return S
end

S = run_kelvin_helmholtz();

# read results and parameters
fid = h5open(PATH_OUT, "r")
solX =    read(fid["X"])
solV =    read(fid["V"])
solD =    read(fid["D"])
sol∇S =   read(fid["grad"])
λ² =      read(fid["lambda"])^2
ε =       read(fid["epsilon"])
tol =     read(fid["tol"])
crit_it = read(fid["crit_it"])
p_ω =     read(fid["p"])
Δt =      read(fid["deltat"])
sol_α =   read(fid["alpha"])
β =       read(fid["beta"])
close(fid)
nt = size(solX,1) - 1;

massvec = [ solV[begin,i,1] == 1.0  for i in 1:N];
nomassvec = [!x for x in massvec ];

anim = @animate for j in axes(solX,1)
    scatter([solX[j,nomassvec,1]], [solX[j,nomassvec,2]], label = false, markerstrokewidth=0, markersize = 2.5, color = :black, xlims = (-0.55,0.55), ylims = (-0.55,0.55))
    scatter!([solX[j,massvec,1]], [solX[j,massvec,2]], label = false, markerstrokewidth=0, markersize = 2.5, color = :red, xlims = (-0.55,0.55), ylims = (-0.55,0.55))
end
gif(anim, "figs/kelvin_helmholtz.gif", fps = 8)


