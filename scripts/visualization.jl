using HDF5
using Plots
using LaTeXStrings
using LinearAlgebra
using BrenierTwoFluid

### Input the .hdf5 here
results = "runs/2024-03-15T12:29:44.698.hdf5"
###

# read results and parameters
fid = h5open(results, "r")
solX =    read(fid["X"])
solV =    read(fid["V"])
solD =    read(fid["D"])
sol∇S =   read(fid["grad"])
#δ =       read(fid["delta"])
λ² =      read(fid["lambda"])^2
ε =       read(fid["epsilon"])
tol =     read(fid["tol"])
crit_it = read(fid["crit_it"])
p_ω =     read(fid["p"])
Δt =      read(fid["deltat"])
sol_α =       read(fid["alpha"])
β =       read(fid["beta"])
sol_species = read(fid["species"])
close(fid)
nt = size(solX,1) - 1
N = size(solX,2)

    # colors
    species = zero(α)
    for i in 1:N
        if X[i,1] < -1/6
            species[i] = 1
        elseif X[i,1] > 1/6
            species[i] = 3
        else
            species[i] = 2
        end
    end


# calculate diagnostics
K = 1/2 * [dot(solV[i,:,:], diagm(sol_α[i,:]) * solV[i,:,:]) for i in axes(solV,1)]; #kinetic energy

# energy Plot
plot(K .- K[1], minorgrid = true, xlabel = L"t",
    legendfontsize=12, tickfontsize=10, xguidefontsize=14, yguidefontsize=14,
    linewidth = 2, label=L"\frac{1}{2} \sum_i w_i (V_i^2(t) - V_i^2(0))")
plot!(λ²/2 * solD .- λ²/2 * solD[1],
    linewidth = 2, label=L"\frac{\lambda^2}{2} (S^2_\varepsilon(t) - S^2_\varepsilon(0))")
# total energy if needed
#plot!((0:nt)/Δt, K .- K[1] .+ λ²/2 * solD .- λ²/2 * solD[1], linewidth = 2, label=L"H(t) - H(0)")
savefig("figs/energy.pdf")

# Particle plots
for j in axes(solX,1)
    scatter(solX[j,:,1], solX[j,:,2], label = false, color = palette(:default)[Int.(sol_species[j,:])], markerstrokewidth=0, markersize = 2.5, legendfontsize=14, tickfontsize=10, xguidefontsize=14, yguidefontsize=14, xlabel = L"x_1", ylabel = L"x_2", xlim=(-0.55,0.55), ylim=(-0.55,0.55))
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
plot(ΔV)
savefig("figs/V_error.pdf")

anim = @animate for j in axes(solX,1)
    scatter(solX[j,:,1], solX[j,:,2], label = false, color = palette(:default)[Int.(sol_species[j,:])], markerstrokewidth=0, markersize = 2.5, legendfontsize=14, tickfontsize=10, xguidefontsize=14, yguidefontsize=14, xlabel = L"x_1", ylabel = L"x_2", xlim=(-0.55,0.55), ylim=(-0.55,0.55))
end
gif(anim, "figs/euler.gif", fps = 8)

#=
massvec = []
nomassvec = []
for i in axes(solX,2)
    if color[i] == 1
        push!(nomassvec, i)
    else
        push!(massvec, i)
    end
end
anim = @animate for j in axes(solX,1)
    scatter([solX[j,nomassvec,1]], [solX[j,nomassvec,2]], label = false, markerstrokewidth=0, markersize = 2.5, color = :black, xlims = (-0.55,0.55), ylims = (-0.55,0.55))
    scatter!([solX[j,massvec,1]], [solX[j,massvec,2]], label = false, markerstrokewidth=0, markersize = 2.5, color = :red, xlims = (-0.55,0.55), ylims = (-0.55,0.55))
end
gif(anim, "figs/raleigh_taylor.gif", fps = 8)
=#

# Rank Stuff
Y = stack(vec([ [x,y] for x in range(-0.5,0.5,length=Int(sqrt(M))), y in range(-0.5,0.5,length=Int(sqrt(M))) ]), dims = 1);
Y .= Y[sortperm(Y[:,1]), :];

plt = plot(xlabel = L"i", ylabel = L"\sigma_i")
for j in axes(solX,1)
    if j % 10 == 0 || j==1
        CC = CostCollection(solX[j,:,:], Y, (x,y) -> 0.5 * sqeuclidean(x,y));
        K = exp.(-CC.C_xy/ε)
        svdK = svd(K);
        plot!(svdK.S, yscale=:log10, label = "$j") 
    end
end
plt

#=
p = 3
h = sqrt(ε)
s_vec = [1,2,3];
ρ_vec = [ vec([ kde([x, y], solX[end,:,:], α, h, p, s, species) for x in range(-0.5,0.5,length=Int(sqrt(N))), y in range(-0.5,0.5,length=Int(sqrt(N)))]) for s in s_vec ]
for ρ in ρ_vec
    ρ ./= sum(ρ)
    ρ ./= length(s_vec) # number of species
end
=#

