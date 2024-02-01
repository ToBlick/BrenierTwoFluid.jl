using Distances
using Plots
using BrenierTwoFluid
using LinearAlgebra
using Random
using Statistics
using LaTeXStrings

d = 2
d′ = 2*Int(floor(d/2))

c = (x,y) -> 0.5 * sqeuclidean(x,y)
∇c = (x,y) -> x-y

"""
Illustrate the effect of acceleration for a small distance
"""

d = 2
d′ = 2*Int(floor(d/2))
N = 16^2
M = 16^2
α = ones(N) / N
β = ones(M) / M

offset = 0.05
truevalue = 1/2 * d * 1/12 * offset^2 #1/2 * d * offset^2

# common parameters
tol = 1e-15
q = 1.0
Δ = 1.0
ε = 0.1 * N^(-1/(d′+4))
s = ε
crit_it = 20
p_ω = 2


# for maxit = 10, 20, ... calculate the distance between two uniform distributions.
# maximum iterations are 10*i_max
i_max = 10 #Int(ceil(Δ/ε/10))
it_vec = [ i*10 for i in 1:i_max ]

epochs = 10
S_vec = zeros(2, length(it_vec), epochs)
marginal_error_vec = zero(S_vec)
times = zero(S_vec);

Random.seed!(123)
@time for i in 1:2
    if i == 1
        acc = false
    else
        acc = true
    end
    for j in eachindex(it_vec)
        for k in 1:epochs
            X = rand(N,d) .- 0.5
            Y = (rand(M,d) .- 0.5) .* (1 + offset)

            CC = CostCollection(X, Y, c)
            V = SinkhornVariable(X,α)
            W = SinkhornVariable(Y,β)

            params = SinkhornParameters(CC;ε=ε,q=q,Δ=Δ,s=s,crit_it=crit_it,p_ω=p_ω,tol=tol,maxit=it_vec[j],sym=false,acc=acc)
            
            S = SinkhornDivergence(V,W,CC,params)
            initialize_potentials!(V,W,CC)

            S_vec[i,j,k], times[i,j,k] = @timed compute!(S)
            marginal_error_vec[i,j,k] = marginal_error(S)
        end
    end
end

Random.seed!(123)
X = rand(N,d) .- 0.5
Y = (rand(M,d) .- 0.5) .* (1 + offset)
CC = CostCollection(X, Y, c)
V = SinkhornVariable(X,α)
W = SinkhornVariable(Y,β)
params = SinkhornParameters(CC;ε=ε,q=q,Δ=Δ,s=s,crit_it=crit_it,p_ω=p_ω,tol=tol,maxit=it_vec[end],sym=false,acc=true)
S = SinkhornDivergence(V,W,CC,params)
initialize_potentials!(V,W,CC)
compute!(S)
π_opt = Matrix(TransportPlan(S))
M_opt = diagm(1 ./ α) * π_opt * diagm(1 ./ β) * π_opt'
evd_M = eigen(M_opt, sortby = x -> - abs(x))
ω_true = 2/(1 + sqrt( 1 - Real(evd_M.values[2])))
println("true η: $ω_true")
ω_inferred = acceleration(S)
println("inferrred η: $ω_inferred")



p = plot(it_vec, mean(abs.(marginal_error_vec[1,:,:]), dims = 2), 
        minorgrid = true, xlabel = L"$\mathrm{iteration}$", ylabel = L"$\Vert 1 - \int \exp ( (\phi \oplus \psi - c) / \varepsilon ) d n^\beta \Vert_{L^1(n^\alpha)}$", yaxis=:log,
        legend = :topright, legendfontsize=14, tickfontsize=10, xguidefontsize=14, yguidefontsize=14,
        linewidth = 2, label = L"$\eta = 0$", color = palette(:default)[1])
    plot!(it_vec, mean(abs.(marginal_error_vec[2,:,:]), dims = 2),
        linewidth = 2, fillalpha=0.33, label = L"$\eta = {%$(round(ω_inferred * 1e4)*1e-4)}$", color = palette(:default)[2])
for i in 1:epochs
    plot!(it_vec, abs.(marginal_error_vec[1,:,i]), 
        linewidth = 1, alpha=0.3, label = false, color = palette(:default)[1])
    plot!(it_vec, abs.(marginal_error_vec[2,:,i]), 
        linewidth = 1, alpha=0.3, label = false, color = palette(:default)[2])
end
p
savefig("figs/acceleration.pdf")

#=
plot(it_vec, mean(abs.(marginal_error_vec[1,:,:]), dims = 2), minorgrid = true, xlabel = L"$\mathrm{iteration}$", ylabel = L"$\Vert 1 - \int \exp ( (\phi \oplus \psi - c) / \varepsilon ) d n^\beta \Vert_{L^1(n^\alpha)}$", yaxis=:log,
    legendfontsize=14, tickfontsize=10, xguidefontsize=14, yguidefontsize=14,
    legend = :topright,
    ribbon = (minimum(abs.(marginal_error_vec[1,:,:]), dims = 2),maximum(marginal_error_vec[1,:,:], dims = 2)),
    linewidth = 2, fillalpha=0.33, label = L"$\eta = 0$" )
plot!(it_vec, mean(abs.(marginal_error_vec[2,:,:]), dims = 2),
    ribbon = (minimum(abs.(marginal_error_vec[2,:,:]), dims = 2),maximum(marginal_error_vec[2,:,:], dims = 2)),
    linewidth = 2, fillalpha=0.33, label = L"$\eta = {%$(round(ω_inferred * 1e4)*1e-4)}$")
vline!([crit_it], color="grey", style=:dash, label=false)
savefig("../figs/acceleration.pdf")
=#

plot(it_vec, mean(abs.(S_vec[1,:,:] .- truevalue), dims = 2), minorgrid = true, xlabel = "iterations", ylabel = "S² error", yaxis=:log,
    title = "tol = $tol, N = $N", legend = :topright,
    ribbon = (minimum(abs.(S_vec[1,:,:] .- truevalue), dims = 2),maximum(abs.(S_vec[1,:,:] .- truevalue), dims = 2)),
    linewidth = 2, fillalpha=0.33, label = "no acceleration" )
plot!(it_vec, mean(abs.(S_vec[2,:,:]), dims = 2),
    ribbon = (minimum(abs.(S_vec[2,:,:] .- truevalue), dims = 2),maximum(abs.(S_vec[2,:,:] .- truevalue), dims = 2)),
    linewidth = 2, fillalpha=0.33, label = "accelerated" )

# check the computed omega

Random.seed!(123)
X = rand(N,d) .- 0.5
Y = (rand(M,d) .- 0.5) .* (1 + offset)
CC = CostCollection(X, Y, c)
V = SinkhornVariable(X,α)
W = SinkhornVariable(Y,β)
params = SinkhornParameters(CC;ε=ε,q=q,Δ=Δ,s=s,crit_it=crit_it,p_ω=p_ω,tol=tol,maxit=it_vec[end],sym=false,acc=true)
S = SinkhornDivergence(V,W,CC,params)
initialize_potentials!(V,W,CC)
compute!(S)
π_opt = Matrix(TransportPlan(S))
M_opt = diagm(1 ./ α) * π_opt * diagm(1 ./ β) * π_opt'
evd_M = eigen(M_opt, sortby = x -> - abs(x))
ω_true = 2/(1 + sqrt( 1 - Real(evd_M.values[2])))
println("true η: $ω_true")
ω_inferred = acceleration(S)
println("inferrred η: $ω_inferred")


