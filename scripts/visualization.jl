#=
results = "runs/results.hdf5"
fid = h5open(results, "r")
solX = fid["X"] = [ solX[i][j,k] for i in eachindex(solX), j in axes(X,1), k in axes(X,2) ];
fid["V"] = [ solV[i][j,k] for i in eachindex(solV), j in axes(V,1), k in axes(V,2) ];
fid["D"] = solD
fid["grad"] = [ sol∇S[i][j,k] for i in eachindex(sol∇S), j in axes(X,1), k in axes(X,2) ];
fid["delta"] = δ
fid["lambda"] = sqrt(λ²)
fid["epsilon"] = ε
fid["tol"] = tol
fid["crit_it"] = crit_it
fid["p"] = p_ω
fid["deltat"] = Δt
close(fid)
=#

# calculate diagnostics
K = 1/2 * [dot(V, diagm(α) * V) for V in solV]; #kinetic energy

# energy Plot
plot((0:nt)/Δt, K .- K[1], minorgrid = true, xlabel = L"t",
    legendfontsize=14, tickfontsize=10, xguidefontsize=14, yguidefontsize=14,
    linewidth = 2, label=L"\frac{1}{2} \sum_i w_i (V_i^2(t) - V_i^2(0))")
plot!((0:nt)/Δt, λ²/2 * solD .- λ²/2 * solD[1],
    linewidth = 2, label=L"\frac{\lambda^2}{2} (S_\varepsilon(t) - S_\varepsilon(0))")
#plot!((0:nt)/Δt, K .- K[1] .+ λ²/2 * solD .- λ²/2 * solD[1],
#    linewidth = 2, label=L"H(t) - H(0)")
savefig("figs/energy.pdf")

# Particle plots
j = 1 #div(1 *length(solX), 4)
plt = scatter(solX[j][1:div(N,3),1], solX[j][1:div(N,3),2], label = false, color = palette(:default)[1], markerstrokewidth=0, markersize = 2,
            legendfontsize=14, tickfontsize=10, xguidefontsize=14, yguidefontsize=14, xlabel = L"x_1", ylabel = L"x_2")
scatter!(solX[j][div(N,3)+1:div(2N,3),1], solX[j][div(N,3)+1:div(2N,3),2], label = false, 
        color = palette(:default)[2], markerstrokewidth=0, markersize = 2)
scatter!(solX[j][div(2N,3)+1:end,1], solX[j][div(2N,3)+1:end,2], label = false, 
color = palette(:default)[3], markerstrokewidth=0, markersize = 2)
savefig("figs/particles_1.pdf")