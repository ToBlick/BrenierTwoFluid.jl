# compute W_2 distance between two uniform distributions

d = 3
N = 200
M = 200
α = ones(N) / N
β = ones(M) / M

X = rand(N,d)
Y = rand(M,d) .+ 1
c = (x,y) -> 0.5 * sqeuclidean(x,y)
∇c = (x,y) -> x - y

CC = CostCollection(X, Y, c)
V = SinkhornVariable(X,α)
W = SinkhornVariable(Y,β)

δ = N^(-1/2)
ε = δ^2 #N^(-3/(d′+4))
q = 0.7
Δ = 1.0

### Unsafe version
S = SinkhornDivergence(V,W,CC;ε=ε,q=0.9,Δ=1.0);
initialize_potentials!(V,W,CC);
@time compute!(S);
∇S = zero(X);
@time x_gradient!(∇S, S, ∇c);

@test abs(value(S) - 0.5 * 1^2 * d) * sqrt(sqrt(N*M)) < 1

Π = TransportPlan(S);
sum(Matrix(Π))

### Safe version
S = SinkhornDivergence(V,W,CC;ε=ε,q=0.9,Δ=1.0,tol=1e-3);
initialize_potentials!(V,W,CC);
@time compute!(S);
∇S_x = zero(X);
∇S_y = zero(X);
@time x_gradient!(∇S_x, S, ∇c)
@time y_gradient!(∇S_y, S, ∇c)

∇S

value(S)
@test abs(value(S) - 0.5 * 1^2 * d) * sqrt(sqrt(N*M)) < 1

#=
using Plots
quiver(X[:,1],X[:,2],quiver= -1 .* (∇S_x[:,1] ./ α ,∇S_x[:,2] ./ α ),
        alpha = 0.5, color = :black)
scatter!(X[:,1],X[:,2],color=:red,legend = false)
scatter!(Y[:,1],Y[:,2],color=:blue)

quiver(Y[:,1],Y[:,2],quiver= -1 .* (∇S_y[:,1] ./ β ,∇S_y[:,2] ./ β ),
        alpha = 0.5, color = :black)
scatter!(X[:,1],X[:,2],color=:red,legend = false)
scatter!(Y[:,1],Y[:,2],color=:blue)
=#

