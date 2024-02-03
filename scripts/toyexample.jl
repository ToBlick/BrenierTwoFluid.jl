# compute W_2 distance between two uniform distributions

using Plots
using LinearAlgebra

d = 2
N = 500
M = 500
α = ones(N) / N
β = ones(M) / M

offset = 0.0
truevalue = 0.0

X = rand(N,d) .- 0.5
Y = (rand(M,d) .- 0.5) .* (1 + offset)
c = (x,y) -> 0.5 * sqeuclidean(x,y)
∇c = (x,y) -> x - y

CC = CostCollection(X, Y, c)
V = SinkhornVariable(X,α)
W = SinkhornVariable(Y,β)

d′ = 2*Int(floor(d/2))
δ = N^(-1/2)
ε = 0.01 * N^(-1/(d′+4)) # δ^2 #N^(-3/(d′+4))
#0.1 * N^(-1/(d′+4))
q = 1.0
Δ = 1.0
s = ε
tol = 1e-4
crit_it = 10
p_ω = 2

### Safe version

params = SinkhornParameters(CC;
                            ε=ε,
                            q=1.0,
                            Δ=1.0,
                            s=s,
                            tol=tol,
                            crit_it=crit_it,
                            p_ω=p_ω,
                            sym=false,
                            acc=true);
S = SinkhornDivergence(V,W,CC,params);
initialize_potentials!(V,W,CC);
@time valS = compute!(S);
value(S)
abs(value(S) - truevalue) * sqrt(sqrt(N*M))
Π = TransportPlan(S);
sum(Matrix(Π))
S.params.ω

# dual potential violation: 
BrenierTwoFluid.marginal_errors(S)
norm(exp.(V.f/ε), Inf)
norm(exp.(W.f/ε), Inf)

norm(exp.(V.f/ε), 1)
norm(exp.(W.f/ε), 1)

sum(V.f .* α)
sum(W.f .* β)

# f ⊕ g - c ≤ 0
# const = (∫fμ - ∫gν)/2
# f ← f + const
# g ← g - const

π1_err = 0
π2_err = 0
@inbounds for i in eachindex(S.V1.f)
    sm_i = softmin(i, S.CC.C_yx, S.V2.f, S.V2.log_α, BrenierTwoFluid.scale(S))
    π2_err += abs( exp( (S.V1.f[i] - sm_i) / BrenierTwoFluid.scale(S) ) - 1 ) * S.V1.α[i]
end
@inbounds for j in eachindex(S.V2.f)
    sm_j = softmin(j, S.CC.C_xy, S.V1.f, S.V1.log_α, BrenierTwoFluid.scale(S))
    π1_err += abs( exp( (S.V2.f[j] - sm_j) / BrenierTwoFluid.scale(S) ) - 1 ) * S.V2.α[j]
end
π1_err, π2_err

π1_err = 0
π2_err = 0
@inbounds for i in eachindex(S.V1.f)
    sm_i = softmin(i, S.CC.C_yx, S.V2.f, S.V2.log_α, BrenierTwoFluid.scale(S))
    π2_err += abs( exp( (- sm_i) / BrenierTwoFluid.scale(S) ) - exp( (-S.V1.f[i]) / BrenierTwoFluid.scale(S) ) ) * S.V1.α[i]
end
@inbounds for j in eachindex(S.V2.f)
    sm_j = softmin(j, S.CC.C_xy, S.V1.f, S.V1.log_α, BrenierTwoFluid.scale(S))
    π1_err += abs( exp( (- sm_j) / BrenierTwoFluid.scale(S) ) - exp( (-S.V2.f[j]) / BrenierTwoFluid.scale(S) ) ) * S.V2.α[j]
end
π1_err, π2_err


π1_err = 0
π2_err = 0
@inbounds for i in eachindex(S.V1.h)
    sm_i = softmin(i, S.CC.C_xx, S.V1.h, S.V1.log_α, BrenierTwoFluid.scale(S))
    π2_err += abs( exp( (S.V1.h[i] - sm_i) / BrenierTwoFluid.scale(S) ) - 1 ) * S.V1.α[i]
end
@inbounds for j in eachindex(S.V2.h)
    sm_j = softmin(j, S.CC.C_yy, S.V2.h, S.V2.log_α, BrenierTwoFluid.scale(S))
    π1_err += abs( exp( (S.V2.h[j] - sm_j) / BrenierTwoFluid.scale(S) ) - 1 ) * S.V2.α[j]
end
π1_err, π2_err

norm(exp.(V.h/ε), Inf)
norm(exp.(W.h/ε), Inf)

norm(exp.(V.h/ε), 1)
norm(exp.(W.h/ε), 1)

sum(V.h .* α)

sum(W.h .* β)

maximum(S.V1.f * ones(N)' + ones(N) * S.V2.f' - S.CC.C_yx)
ε
minimum(S.V1.f * ones(N)' + ones(N) * S.V2.f' - S.CC.C_yx)
norm(S.V1.f * ones(N)' + ones(N) * S.V2.f' - S.CC.C_yx, 1) / N^2