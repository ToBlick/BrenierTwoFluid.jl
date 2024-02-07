using HDF5
using Plots
using LaTeXStrings
using LinearAlgebra

### Input the .hdf5 here
results = "runs/2024-02-07T14:54:12.987.hdf5"
###

# read results and parameters
fid = h5open(results, "r")
solX =    read(fid["X"])
solV =    read(fid["V"])
solD =    read(fid["D"])
sol∇S =   read(fid["grad"])
δ =       read(fid["delta"])
λ² =      read(fid["lambda"])^2
ε =       read(fid["epsilon"])
tol =     read(fid["tol"])
crit_it = read(fid["crit_it"])
p_ω =     read(fid["p"])
Δt =      read(fid["deltat"])
α =       read(fid["alpha"])
β =       read(fid["beta"])
close(fid)
nt = size(solX,1) - 1
N = size(solX,2)

λ²
N
ε
tol
Δt
crit_it
p_ω


# calculate diagnostics
K = 1/2 * [dot(solV[i,:,:], diagm(α) * solV[i,:,:]) for i in axes(solV,1)]; #kinetic energy

# energy Plot
plot(0:Δt:1, K .- K[1], minorgrid = true, xlabel = L"t",
    legendfontsize=12, tickfontsize=10, xguidefontsize=14, yguidefontsize=14,
    linewidth = 2, label=L"\frac{1}{2} \sum_i w_i (V_i^2(t) - V_i^2(0))")
plot!(0:Δt:1, λ²/2 * solD .- λ²/2 * solD[1],
    linewidth = 2, label=L"\frac{\lambda^2}{2} (S^2_\varepsilon(t) - S^2_\varepsilon(0))")
# total energy if needed
#plot!((0:nt)/Δt, K .- K[1] .+ λ²/2 * solD .- λ²/2 * solD[1], linewidth = 2, label=L"H(t) - H(0)")
savefig("figs/energy.pdf")

# Particle plots
for j in axes(solX,1)
    plt = scatter(solX[j,1:div(N,3),1], solX[j,1:div(N,3),2], label = false, color = palette(:default)[1], markerstrokewidth=0, markersize = 2.5,
                legendfontsize=14, tickfontsize=10, xguidefontsize=14, yguidefontsize=14, xlabel = L"x_1", ylabel = L"x_2")
    scatter!(solX[j,div(N,3)+1:div(2N,3),1], solX[j,div(N,3)+1:div(2N,3),2], label = false, 
            color = palette(:default)[2], markerstrokewidth=0, markersize = 2.5)
    scatter!(solX[j,div(2N,3)+1:end,1], solX[j,div(2N,3)+1:end,2], label = false, 
    color = palette(:default)[3], markerstrokewidth=0, markersize = 2.5)
    savefig("figs/particles_$(j).pdf")
end

u0(x) = [-cos(π*x[1])*sin(π*x[2]), sin(π*x[1])*cos(π*x[2])]
ΔV = zero(solD);
for i in eachindex(ΔV)
    for j in axes(solV,2)
        ΔV[i] += norm(solV[i,j,:] - u0(solX[i,j,:]))
    end
end
ΔV ./= N;
plot(0:Δt:1, ΔV)
