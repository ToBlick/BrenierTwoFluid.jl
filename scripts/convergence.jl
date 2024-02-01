using Distances
using Plots
using BrenierTwoFluid
using LinearAlgebra
using Random
using Statistics
using ProgressBars

d = 2
d′ = 2*Int(floor(d/2))

c = (x,y) -> 0.5 * sqeuclidean(x,y)
∇c = (x,y) -> x-y

# for N = 10^2, 20^2, 30^2, ..., calculate the distance between two uniform distributions. This quantity is zero for all ε >= 0, hence this should give us the convergence in N.

# we use three different ε: 
#   theory-based ε ∼ N^(-1/(d′+4))
#   optimistic ε ∼ N^(-1/d)

# maximum N is 64 * 2^(i_max-1)
i_max = 7
64 * 2^(i_max-1)

# acceleration is inferred from convergence rate after crit_it iterations. No scaling.
tol = 1e-6
crit_it = 20
p_ω = 2
q = 1.0
Δ = 1.0
acc = true
offset = 0.05
truevalue = 1/2 * d * 1/12 * offset^2 #1/2 * d * offset^2

epochs = 10

N_vec = [ 64 * 2^(i-1) for i in 1:i_max ]
S_vec = zeros(2, length(N_vec), epochs)
times = zero(S_vec);

Random.seed!(123)
for j in ProgressBar(eachindex(N_vec))

    N = N_vec[j]
    M = N

    α = ones(N) / N
    β = ones(M) / M

    for k in 1:epochs
        X = rand(N,d) .- 0.5
        Y = (rand(M,d) .- 0.5) .* (1 + offset)

        CC = CostCollection(X, Y, c)
        V = SinkhornVariable(X,α)
        W = SinkhornVariable(Y,β)

        ε = 1.0
        for i in 1:2
            if i == 1
                ε = 0.1*N^(-1/(d′+4))
            elseif i == 2
                ε = 0.1*N^(-1/d)
            #else
            #    ε = N^(-2/d)
            end
            
            s = ε

            params = SinkhornParameters(CC;ε=ε,q=q,Δ=Δ,s=s,crit_it=crit_it,p_ω=p_ω,tol=tol,maxit=10*Int(ceil(Δ/ε)),sym=false,acc=acc)
            
            S = SinkhornDivergence(V,W,CC,params)
            initialize_potentials!(V,W,CC)

            S_vec[i,j,k], times[i,j,k] = @timed compute!(S)
        end
    end
end

p = plot(N_vec, mean(abs.(S_vec[1,:,:] .- truevalue), dims = 2), 
        minorgrid = true, xlabel = L"N", ylabel = L"| S_\varepsilon(n^\alpha_N, n^\beta_N)^2 - W_2(n^\alpha, n^\beta)^2 |", yaxis=:log, xaxis = :log,
legend = :bottomleft, legendfontsize=14, tickfontsize=10, xguidefontsize=14, yguidefontsize=14,
        linewidth = 2, label = L"\varepsilon = 0.1 N^{-1/(d′+4)}", color = palette(:default)[1])
    plot!(N_vec, mean(abs.(S_vec[2,:,:] .- truevalue), dims = 2),
        linewidth = 2, fillalpha=0.33, label = L"\varepsilon = 0.1 N^{-1/d}", color = palette(:default)[2])
for i in 1:epochs
    plot!(N_vec, abs.(S_vec[1,:,i] .- truevalue), 
        linewidth = 1, alpha=0.3, label = false, color = palette(:default)[1])
    plot!(N_vec, abs.(S_vec[2,:,i] .- truevalue), 
        linewidth = 1, alpha=0.3, label = false, color = palette(:default)[2])
end
plot!(N_vec, x -> 1e-1 * x^(-2/(d′+4)),
    linewidth = 2, label = L"0.1 N^{-2/(d'+4)}", linestyle = :dash, color = palette(:default)[3])
plot!(N_vec, x -> 1e-0 * x^(-2/d′),
    linewidth = 2, label = L"N^{-2/d'}", linestyle = :dash, color = palette(:default)[4])
p
savefig("figs/error_convergence.pdf")



p = plot(N_vec, mean(abs.(times[1,:,:]), dims = 2),
        minorgrid = true, xlabel = L"N", ylabel = L"\mathrm{comput. time}", yaxis=:log, xaxis = :log,
legend = :topleft, legendfontsize=14, tickfontsize=10, xguidefontsize=14, yguidefontsize=14,
        linewidth = 2, label = L"\varepsilon = 0.1 N^{-1/(d′+4)}", color = palette(:default)[1])
    plot!(N_vec, mean(abs.(times[2,:,:]), dims = 2),
        linewidth = 2, fillalpha=0.33, label = L"\varepsilon = 0.1 N^{-1/d}", color = palette(:default)[2])
for i in 1:epochs
    plot!(N_vec, abs.(times[1,:,i]), 
        linewidth = 1, alpha=0.3, label = false, color = palette(:default)[1])
    plot!(N_vec, abs.(times[2,:,i]), 
        linewidth = 1, alpha=0.3, label = false, color = palette(:default)[2])
end
plot!(N_vec, x -> 1e-5 * x^(2),
    linewidth = 2, label = L"10^{-5} N^2", linestyle = :dash, color = palette(:default)[3])

p
savefig("figs/computing_time.pdf")

#=    
plot!(N_vec, mean(abs.(S_vec[1,:,:]), dims = 2), minorgrid = true, xlabel = L"N", ylabel = L"| S_\varepsilon(n^\alpha_N, n^\beta_N)^2 - W_2(n^\alpha, n^\beta)^2 |", yaxis=:log, xaxis = :log,
    legend = :bottomleft, legendfontsize=14, tickfontsize=10, xguidefontsize=14, yguidefontsize=14,
    ribbon = (mean(abs.(S_vec[1,:,:]), dims = 2) - std(abs.(S_vec[1,:,:]), dims = 2),
              mean(abs.(S_vec[1,:,:]), dims = 2) + std(abs.(S_vec[1,:,:]), dims = 2)),
    linewidth = 2, fillalpha=0.33, label = L"\varepsilon = 0.1 \times N^{-1/(d′+4)}" )
plot!(N_vec, mean(abs.(S_vec[2,:,:] .- truevalue), dims = 2),
    ribbon = (minimum(abs.(S_vec[2,:,:] .- truevalue), dims = 2),maximum(abs.(S_vec[2,:,:] .- truevalue), dims = 2)),
    linewidth = 2, fillalpha=0.33, label = L"\varepsilon = 0.1 \times N^{-1/d}" )
plot!(N_vec, x -> 1e-1 * x^(-2/(d′+4)),
    linewidth = 2, label = L"\sim N^{-2/(d′+4)}", linestyle = :dash )
#plot!(N_vec, x -> mean(abs.(S_vec[1,1,:])) * N_vec[1,1]^(1/2) * x^(-1/2),
#    linewidth = 2, label = "∼ N^(-1/2)" )
plot!(N_vec, x -> 1e-0 * x^(-1),
    linewidth = 2, label = L"\sim N^{-1}", linestyle = :dash )
savefig("../figs/error_convergence.pdf")

plot(N_vec, mean(abs.(times[1,:,:]), dims = 2), minorgrid = true, xlabel = "N", ylabel = "comput. time", xaxis = :log, yaxis=:log,
    title = "tol = $tol, acc = $acc", legend = :topleft,
    ribbon = (minimum(abs.(times[1,:,:]), dims = 2),maximum(times[1,:,:], dims = 2)),
    linewidth = 2, fillalpha=0.33, label = "ε = N^(-1/(d′+4))" )
plot!(N_vec, mean(abs.(times[2,:,:]), dims = 2),
    ribbon = (minimum(abs.(times[2,:,:]), dims = 2),maximum(times[2,:,:], dims = 2)),
    linewidth = 2, fillalpha=0.33, label = "ε = N^(-1/d)" )
plot!(N_vec, x ->  mean(times[1,1,:]) / N_vec[1,1]^2 * x^2,
    linewidth = 2, label = "∼ N^2" )
savefig("computing_time.pdf")
=#