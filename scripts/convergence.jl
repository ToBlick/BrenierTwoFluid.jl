using Distances
using Plots
using BrenierTwoFluid
using LinearAlgebra
using Random
using Statistics
using ProgressBars
using LaTeXStrings

d = 2
d′ = 2*Int(floor(d/2))

c = (x,y) -> 0.5 * sqeuclidean(x,y)
∇c = (x,y) -> x-y

# for N = 10^2, 20^2, 30^2, ..., calculate the distance between two uniform distributions. This quantity is zero for all ε >= 0, hence this should give us the convergence in N.

# we use three different ε: 
#   theory-based ε ∼ N^(-1/(d′+4))
#   optimistic ε ∼ N^(-1/d)

# maximum N is 64 * 2^(i_max-1)
i_max = 4
64 * 2^(i_max-1)
(10 * i_max)^2
j_max = 5
4.0^(-j_max)

# acceleration is inferred from convergence rate after crit_it iterations. No scaling.
tol = 1e-6
crit_it = 20
max_it = 100
p_ω = 2
q = 1.0
Δ = 1.0
acc = true
offset = 0.0
truevalue = 1/2 * d * 1/12 * offset^2 #1/2 * d * offset^2

epochs = 5

N_vec = [ (10 * i)^2 for i in 1:i_max ]
ε_vec = [ 4.0^(-j) for j in 1:j_max ]
S_vec = zeros(length(ε_vec), length(N_vec), epochs)
norm_∇S_vec = zero(S_vec)
times = zero(S_vec);

@time for j in ProgressBar(eachindex(N_vec))
    N = N_vec[j]
    M = N

    α = ones(N) / N
    β = ones(M) / M

    for k in 1:epochs
        Random.seed!(k)
        X = rand(N,d) .- 0.5
        Y = (rand(M,d) .- 0.5) .* (1 + offset)
        for k in 1:Int(sqrt(M))
            for l in 1:Int(sqrt(M))
                Y[(k-1)*Int(sqrt(M)) + l,:] .= [ k/(Int(sqrt(M))) - 1/(2*Int(sqrt(M))), l/(Int(sqrt(M))) - 1/(2*Int(sqrt(M)))] .- 1/2
            end
        end
        X .= Y .+ randn(N,d) * 1e-2

        for i in eachindex(ε_vec)
            
            ε = ε_vec[i]
            s = ε

            params = SinkhornParameters(ε=ε,q=q,Δ=Δ,s=s,crit_it=crit_it,p_ω=p_ω,tol=tol,max_it=10*Int(ceil(Δ/ε)),sym=false,acc=acc)
            
            S = SinkhornDivergence(SinkhornVariable(X, α),
                                    SinkhornVariable(Y, β),
                                    c,params,true)
            initialize_potentials!(S)

            S_vec[i,j,k], times[i,j,k] = @timed compute!(S)

            ∇S = x_gradient!(S, ∇c)
            norm_∇S = [ norm(∇S[i,:]) for i in axes(∇S,1)]
            norm_∇S_vec[i,j,k] = sum(norm_∇S)/N
        end
    end
end

S_avg = [ mean((abs.(S_vec[i,:,:] .- truevalue)), dims = 2) for i in axes(S_vec,1) ];
S_err = [ std((abs.(S_vec[i,:,:] .- truevalue)), dims = 2) for i in axes(S_vec,1) ];

∇S_avg = [ mean(norm_∇S_vec[i,:,:], dims = 2) for i in axes(norm_∇S_vec,1) ];
∇S_err = [ std(norm_∇S_vec[i,:,:], dims = 2) for i in axes(norm_∇S_vec,1) ];


p = plot()
for i in 2:length(ε_vec)
    plot!(N_vec, abs.(S_avg[i]) .+ 1e-8, ribbon = S_err[i],
        minorgrid = true, xlabel = L"N", ylabel = L"| S_\varepsilon(n^\alpha_N, n^\beta_N)^2 - W_2(n^\alpha, n^\beta)^2 |", xaxis = :log, yaxis = :log,
        linewidth = 2, label = "ε = $((round(ε_vec[i]*1e5))/100000)", fillalpha = 0.2, color=palette(:berlin,length(ε_vec))[i])
end
p

p = plot()
for i in eachindex(ε_vec)
    plot!(N_vec, ∇S_avg[i], ribbon = ∇S_err[i],
        minorgrid = true, xlabel = L"N", ylabel = L" \frac{1}{N} \sum_{i=1}^N | \nabla_{X_i} S^2_\varepsilon(n^\alpha_N, n^\beta_N) |", xaxis = :log, yaxis = :log,
        linewidth = 2, label = "ε = $((round(ε_vec[i]*1e5))/100000)", fillalpha = 0.2, color=palette(:berlin,length(ε_vec))[i])
end
p

#=
p = plot(N_vec, S_avg_1, ribbon = S_err_1, ylim = (1e-5, 2e-2),
        minorgrid = true, xlabel = L"N", ylabel = L"| S_\varepsilon(n^\alpha_N, n^\beta_N)^2 - W_2(n^\alpha, n^\beta)^2 |", yaxis=:log, xaxis = :log,
legend = :topright, legendfontsize=14, tickfontsize=10, xguidefontsize=14, yguidefontsize=14,
        linewidth = 2, label = L"\varepsilon = 0.5", color = palette(:default)[1])
plot!(N_vec, S_avg_2, ribbon = S_err_2,
        linewidth = 2, fillalpha=0.33, label = L"\varepsilon = 0.1", color = palette(:default)[2])
#=for i in 1:epochs
    plot!(N_vec, abs.(S_vec[1,:,i] .- truevalue), 
        linewidth = 1, alpha=0.3, label = false, color = palette(:default)[1])
    plot!(N_vec, abs.(S_vec[2,:,i] .- truevalue), 
        linewidth = 1, alpha=0.3, label = false, color = palette(:default)[2])
end
=#
#plot!(N_vec, x -> 5e-2 * x^(-1/3),
#    linewidth = 2, label = L"\sim \, N^{-1/3}", linestyle = :dash, color = palette(:default)[3])
plot!(N_vec, x -> 0.1 * x^(-1),
    linewidth = 2, label = L"N^{-1/2}", linestyle = :dash, color = palette(:default)[4])

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