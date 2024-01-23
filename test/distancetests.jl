# compute W_2 distance between two uniform distributions

using Plots
using LinearAlgebra

d = 2
N = 1000
M = 1000
α = ones(N) / N
β = ones(M) / M

offset = 0.1
truevalue = 1/2 * d * 1/12 * offset^2 #1/2 * d * offset^2

X = rand(N,d) .- 0.5
Y = (rand(M,d) .- 0.5) .* (1 + offset)
c = (x,y) -> 0.5 * sqeuclidean(x,y)
∇c = (x,y) -> x - y

CC = CostCollection(X, Y, c)
V = SinkhornVariable(X,α)
W = SinkhornVariable(Y,β)

d′ = 2*Int(floor(d/2))
δ = N^(-1/2)
ε = 0.1 * N^(-1/(d′+4)) # δ^2 #N^(-3/(d′+4))
q = 0.7
Δ = 1.0
s = ε
tol = 1e-3
crit_it = 5
p_ω = 2

N^(-2/(d+4))

### Safe version
params = SinkhornParameters(CC; ε=ε,q=1.0,Δ=1.0,s=s,tol=tol,crit_it=crit_it,p_ω=p_ω,sym=true,acc=false);
S = SinkhornDivergence(V,W,CC,params);
initialize_potentials!(V,W,CC);
@time _, trace_sym_noacc = compute!(S);
value(S)
@test abs(value(S) - truevalue) * sqrt(sqrt(N*M)) < 10
Π = TransportPlan(S);
sum(Matrix(Π))

params = SinkhornParameters(CC;ε=ε,q=1.0,Δ=1.0,s=s,tol=tol,crit_it=crit_it,p_ω=p_ω,sym=false,acc=false);
S = SinkhornDivergence(V,W,CC,params);
initialize_potentials!(V,W,CC);
@time _, trace_nosym_noacc = compute!(S);
value(S)
@test abs(value(S) - truevalue) * sqrt(sqrt(N*M)) < 10
Π = TransportPlan(S);
sum(Matrix(Π))

π_opt = Matrix(Π);
norm(π_opt * ones(N) - α,1)
norm(π_opt' * ones(N) - β,1)

M_opt = diagm(1 ./ α) * π_opt * diagm(1 ./ β) * π_opt'
evd_M = eigen(M_opt, sortby = x -> - abs(x))
evd_M.values[2]
ω_opt = Real(2/(1 + sqrt(1 - evd_M.values[2])))

params = SinkhornParameters(CC; ε=ε,q=1.0,Δ=1.0,s=s,tol=tol,crit_it=crit_it,p_ω=p_ω,sym=true,acc=true);
S = SinkhornDivergence(V,W,CC,params);
initialize_potentials!(V,W,CC);
@time _, trace_sym_acc = compute!(S);
value(S)
@test abs(value(S) - truevalue) * sqrt(sqrt(N*M)) < 10
Π = TransportPlan(S);
sum(Matrix(Π))
S.params.ω

params = SinkhornParameters(CC;ε=ε,q=1.0,Δ=1.0,s=s,tol=tol,crit_it=crit_it,p_ω=p_ω,sym=false,acc=true);
S = SinkhornDivergence(V,W,CC,params);
initialize_potentials!(V,W,CC);
@time compute!(S);
value(S)
@test abs(value(S) - truevalue) * sqrt(sqrt(N*M)) < 10
Π = TransportPlan(S);
sum(Matrix(Π))
S.params.ω

using Plots

plot([trace_sym_noacc[i][2] for i in eachindex(trace_sym_noacc)], yaxis = :log, color = :blue)
plot!([trace_nosym_noacc[i][2] for i in eachindex(trace_nosym_noacc)], color = :red)
plot!([trace_nosym_acc[i][2] for i in eachindex(trace_nosym_acc)], color = :green)
plot!([trace_sym_acc[i][2] for i in eachindex(trace_sym_acc)], color = :purple)

#plot(ones(S.params.maxit) * truevalue, xaxis = :log, yaxis =:log, color = :black)
scatter([abs(trace_sym_noacc[i][3] - truevalue) for i in eachindex(trace_sym_noacc)], xaxis = :log, yaxis =:log, color = :blue, markersize = 2)
scatter!([abs(trace_nosym_noacc[i][3] - truevalue) for i in eachindex(trace_nosym_noacc)], color = :red, markersize = 2)
scatter!([abs(trace_nosym_acc[i][3] - truevalue) for i in eachindex(trace_nosym_acc)], color = :green, markersize = 2)
scatter!([abs(trace_sym_acc[i][3]- truevalue) for i in eachindex(trace_sym_acc)], color = :purple, markersize = 2)

∇S_x = x_gradient(S, ∇c);
∇S_y = y_gradient(S, ∇c);

quiver(X[:,1],X[:,2],quiver= -1 .* (∇S_x[:,1] ./ α ,∇S_x[:,2] ./ α ), alpha = 0.5, color = :black)
scatter!(X[:,1],X[:,2],color=:red,legend = false)
scatter!(Y[:,1],Y[:,2],color=:blue)

quiver(Y[:,1],Y[:,2],quiver= -1 .* (∇S_y[:,1] ./ β ,∇S_y[:,2] ./ β ),
        alpha = 0.5, color = :black)
scatter!(X[:,1],X[:,2],color=:red,legend = false)
scatter!(Y[:,1],Y[:,2],color=:blue)

θ_est = []
m = 2
for i in eachindex(trace_nosym_noacc)
        if i <= m
                push!(θ_est, 0)
        else
                push!(θ_est, (trace_nosym_noacc[i][2]/trace_nosym_noacc[i-m][2])^(1/m))
        end
end

ω_opt_vec = Real.(2 ./ (1 .+ sqrt.(1 .- θ_est)))

plot(ω_opt_vec)
plot!(ones(length(ω_opt_vec)) * ω_opt)

ω_opt_vec[20]