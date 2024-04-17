# compute W_2 distance between two uniform distributions
using BrenierTwoFluid
using Test
using Distances
using Random
using LinearAlgebra

const dotime = false
# const dotime = true

c = (x,y) -> 0.5 * sqeuclidean(x,y)
∇c = (x,y) -> x - y
d = 3
N = 20^2
M = 20^2
α = ones(N) / N
β = ones(M) / M

d′ = 2*Int(floor(d/2))
d′′ = 2*Int(ceil(d/2))
ε = 0.1                 # entropic regularization. √ε is a length.
q = 1.0                 # annealing parameter
Δ = 1.0                 # characteristic domain size
s = ε                   # current scale: no annealing -> equals ε
tol = 1e-6             # marginal condition tolerance
crit_it = 20            # acceleration inferrence
p_η = 2

offset = 0.5

Random.seed!(123)
X = rand(N,d) .- 0.5
Y = rand(M,d) .- 0.5

const TOL = (sqrt(N*M))^(-2/(d′+4))

for i in 1:2
        if i == 1 # Scaling test
                truevalue = 1/2 * d * 1/12 * offset^2 #1/2 * d * offset^2
                Y .= (rand(M,d) .- 0.5) .* (1 + offset)
        else # Shifting test
                truevalue = 1/2 * d * offset^2
                Y .= (rand(M,d) .- 0.5) .+ offset
        end

        V = SinkhornVariable(X,α)
        W = SinkhornVariable(Y,β)

        for islog in (true, false), sym in (true, false), acc in (true, false), deb in (true, false), safe in (true, false)
                if sym && !islog
                        continue # symmetrization is not implemented in the non-log domain
                end
                if safe
                        params = SinkhornParameters(ε=ε,q=1.0,Δ=1.0,s=s,tol=tol,crit_it=crit_it,p_η=p_η,sym=sym,acc=acc,deb=deb,safe=safe)
                else
                        params = SinkhornParameters(ε=ε,q=0.9,Δ=1.0,s=Δ,tol=tol,crit_it=crit_it,p_η=p_η,sym=sym,acc=acc,deb=deb,safe=safe)
                end

                S = SinkhornDivergence(V,W,c,params,islog=islog)
                initialize_potentials!(S)
                dotime ? valueS = @time(compute!(S)) : valueS = compute!(S)

                if deb
                        @test abs(valueS - truevalue) < (sqrt(N*M))^(-2/(d′+4))
                else
                        @test abs(valueS - truevalue) < (sqrt(N*M))^(-1/(d′+4))
                end

                ∇S_x = x_gradient!(S, ∇c)
                ∇S_y = y_gradient!(S, ∇c)

                if i == 1
                        @test dot(X - ∇S_x - (1 + offset) * X, α .* (X - ∇S_x - (1 + offset) * X)) < (sqrt(N*M))^(-4/(2*d′′ + 8)) * log(sqrt(N*M))
                        @test dot(Y - ∇S_y - 1/(1 + offset) * Y, β .* (Y - ∇S_y - 1/(1 + offset) * Y)) < (sqrt(N*M))^(-4/(2*d′′ + 8)) * log(sqrt(N*M))
                else
                        @test dot(∇S_x + offset * X ./ X, α .* (∇S_x + offset * X ./ X)) < (sqrt(N*M))^(-4/(2*d′′ + 8)) * log(sqrt(N*M))
                        @test dot(∇S_y - offset * Y ./ Y, β .* (∇S_y - offset * Y ./ Y)) < (sqrt(N*M))^(-4/(2*d′′ + 8)) * log(sqrt(N*M))
                end
        end
end
