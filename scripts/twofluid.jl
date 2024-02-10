using Distances
using Plots
using BrenierTwoFluid
using LinearAlgebra
using Random
using LaTeXStrings
using ProgressBars

Random.seed!(123)

d = 2
c = (x,y) -> 0.5 * sqeuclidean(x,y)
∇c = (x,y) -> x-y

#
# Attempting to re-create a test like the Beltrami incompressible Euler one for two fluids
# with equal mass and charge
#


### parameters
d = 2
c = (x,y) -> 0.5 * sqeuclidean(x,y)
∇c = (x,y) -> x-y
d′ = 2*floor(d/2)
ε = 0.01     # entropic regularization parameter
λ² = 5000
N = 50^2    # particle number
M = N
q = 1.0         # ε-scaling rate
Δ = 1.0         # characteristic domain size
s = ε           # initial scale (ε)
tol = 1e-5      # tolerance on marginals (absolute)
crit_it = 20    # Int(ceil(0.1 * Δ / ε))    # when to compute acceleration
p_ω = 2         # acceleration heuristic
sym = false
acc = true
seed = 123
Δt = 1/50
max_it = 1000

# draw samples
X = rand(N,d) .- 0.5;
Y = rand(M,d) .- 0.5;
#uniform grid
#for k in 1:Int(sqrt(M))
#    for l in 1:Int(sqrt(M))
#        Y[(k-1)*Int(sqrt(M)) + l,:] .= [ k/(Int(sqrt(M))) - 1/(2*Int(sqrt(M))), l/(Int(sqrt(M))) - 1/(2*Int(sqrt(M)))] .- 1/2
#    end
#end
#X .= Y
α = ones(N) / N
β = ones(M) / M
X .= X[sortperm(X[:,1]), :]
Y .= Y[sortperm(Y[:,1]), :];

# variables for the barycenter
Z = rand(N,d) .- 0.5;
μ = copy(α)

# initial velocity for species one
u0(x) = [-cos(π*x[1])*sin(π*x[2]), sin(π*x[1])*cos(π*x[2])]
V = zero(X)
W = zero(Y)
for i in axes(X)[1]
    V[i,:] .= u0(X[i,:])
end

params = SinkhornParameters(ε=ε,q=q,Δ=Δ,s=s,tol=tol,crit_it=crit_it,p_ω=p_ω,max_it=max_it,sym=sym,acc=acc);
Vα  = SinkhornVariable(X, α)
Vβ = SinkhornVariable(Y, β)
S = SinkhornDivergence(Vα, Vβ, c, params, true)
compute!(S)

t = 0
T = 0.1     # final time
nt = Int(ceil((T-t)/Δt))

solX = [ zero(X) for i in 1:(nt) ]
solV = [ zero(V) for i in 1:(nt) ]
solY = [ zero(Y) for i in 1:(nt) ]
solW = [ zero(V) for i in 1:(nt) ]
solD = [ 0.0 for i in 1:(nt + 1) ]

ω = [0.5, 0.5];
B = SinkhornBarycenter(ω, Z, μ, [Vα, Vβ], c, ∇c, params, 10, 1e-3, true);

for it in ProgressBar(1:nt)

    X .+= 0.5 * Δt * V
    Y .+= 0.5 * Δt * W

    valB = compute!(B)

    V .-= Δt * λ² .* y_gradient!(B.Ss[1], B.∇c)
    W .-= Δt * λ² .* y_gradient!(B.Ss[2], B.∇c)

    X .+= 0.5 * Δt * V
    Y .+= 0.5 * Δt * W

    solX[it] = copy(X)
    solV[it] = copy(V)
    solY[it] = copy(Y)
    solW[it] = copy(W)
    solD[it] = valB
end

anim = @animate for j in eachindex(solX)
    scatter(solX[j][:,1],solX[j][:,2], legend = :topright, color=:red, label = "α", xlims = (-0.55,0.55), ylims = (-0.55,0.55))
    scatter!(solY[j][:,1],solY[j][:,2],color=:green, label = "β")
end
gif(anim, "figs/twofluid_vortex.gif", fps = 1)

plot(λ²/2 * solD, linewidth = 2, label = "λ² dist²")
plot!( 1/2 * [dot(solV[i], diagm(α) * solV[i]) for i in axes(solV,1)], linewidth = 2, label = "K(α)" )
plot!( 1/2 * [dot(solW[i], diagm(β) * solW[i]) for i in axes(solV,1)], linewidth = 2, label = "K(β)" )
savefig("figs/twofluid_vortex_energy.png")