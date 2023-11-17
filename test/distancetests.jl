# compute W_2 distance between two uniform distributions

d = 2
N = 50
M = 50
α = ones(N) / N
β = ones(M) / M

X = rand(N,d)
Y = rand(M,d) .+ 2
c = (x,y) -> 0.5 * sqeuclidean(x,y)
∇c = (x,y) -> x - y

CC = CostCollection(X, Y, c)
V = SinkhornVariable(X,α)
W = SinkhornVariable(Y,β)

S = SinkhornDivergence(V,W,CC;ε=0.01,q=0.5);
initialize_potentials!(V,W,CC);
@report_opt compute!(S)
S.s

@test abs(value(S) - 0.5 * d * 2^2) * sqrt(sqrt(N*M)) < 1
#=
Π = TransportPlan(S);
sum(Matrix(Π))

plot(Π)

∇S = x_gradient(S, ∇c)

quiver(X[:,1],X[:,2],quiver= -1 .* (∇S[:,1] ./ α ,∇S[:,2] ./ α ),
        alpha = 0.5, color = :black)
scatter!(X[:,1],X[:,2],color=:red,legend = false)
scatter!(Y[:,1],Y[:,2],color=:blue)
=#