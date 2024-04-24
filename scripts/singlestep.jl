using BrenierTwoFluid
using Distances
using Plots
using LinearAlgebra
using Random
using LaTeXStrings
using ProgressBars
using HDF5
using Dates

### parameters
const d = 2
const c = (x,y) -> 0.5 * sqeuclidean(x,y)
const ∇c = (x,y) -> x-y

const N = 30^2
const M = 30^2

const DOMAIN = (1.0, 1.0)   # only cubes for now

const T = 1.0               # final time

const ENTROPIC_REG = 0.5 * N^(-1/3)    # entropic regularization parameter ε

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


α = ones(N) / N
β = ones(M) / M
Y = stack(vec([ [x,y] for x in range(-0.5,0.5,length=Int(sqrt(M))), y in range(-0.5,0.5,length=Int(sqrt(M))) ]), dims = 1)
X = stack(vec([ [x,y] for x in range(-0.5,0.5,length=Int(sqrt(N))), y in range(-0.5,0.5,length=Int(sqrt(N))) ]), dims = 1)

X .*= 1.1

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
@time compute!(S)

using KrylovKit

const r = 50;
K_lr = []
@time for C in (S.CC.C_xy, S.CC.C_yx, S.CC.C_xx, S.CC.C_yy)
    vals, lvecs, rvecs, info = svdsolve(S.CC.K_xy, r; krylovdim=r, tol = SINKHORN_TOLERANCE);
    push!(K_lr, LowRankMatrix(lvecs, vals, rvecs, zero(vals)))
end

[ minimum(K_lr[i].σ) for i in 1:4 ]

CC = CostCollection(S.CC.C_xy,
                    S.CC.C_yx,
                    S.CC.C_xx,
                    S.CC.C_yy,
                    K_lr[1],
                    K_lr[2],
                    K_lr[3],
                    K_lr[4]);

S_lr = SinkhornDivergence(SinkhornVariable(X, α),
                          SinkhornVariable(Y, β),
                          CC,
                          params,
                          LOG);
initialize_potentials!(S_lr);
@time compute!(S_lr);


#function recursive_rls_nyström(X::AT, Y::AT, k::Base.Callable, λ, δ) where {AT}
ε = ENTROPIC_REG
ϵ = ε
ϵ′ = minimum((1, ε*ϵ/(50 * (4*R^2*ε + log(N/ε/ϵ)))))

δ = SINKHORN_TOLERANCE

    X = rand(N,2)
    m = size(X,1)
    if m < 192 * log(1/δ)
        return diagm(ones(m))
    end
    mask = bitrand(m)
    S_bar = zeros(m,m)
    
    X̄ = X[mask,:]
    Ȳ = Y[mask,:]
    S_tilde = recursive_rls_nyström(X̄, Ȳ, k, λ, δ/3)
    S_tilde = S_bar .* S_tilde

    K = [ k(X[i,:], Y[i,:]) for i in 1:m, j in 1:m ]

    l_λ = 3/(2λ) .* [ (K - K * Ŝ * inv(Ŝ'*K*Ŝ) * Ŝ' * K)[i,i] for i in 1:m ]

    f = 16 * log( sum(l_λ) / δ)
    p = [ min(1, l_λ[i] * f ) for i in 1:m ]

    S = []