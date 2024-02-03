# compute W_2 barycenter between two distributions shifted by +/- an offset - barycenter should be the original distribution

N = 200
M = N
α = ones(N) / N
β = ones(M) / M
μ = ones(N) / N

X = rand(N,d) .- 0.2
Y = rand(M,d) .+ 0.2
Z = rand(N,d)
Z_true = rand(N,d) 

Vα = SinkhornVariable(X,α)
Vβ = SinkhornVariable(Y,β)
Vμ = SinkhornVariable(Z,μ);
Vμ_true = SinkhornVariable(Z_true,μ);

CCαβ = CostCollection(X, Y, c)
CCμα = CostCollection(Z, X, c)
CCμβ = CostCollection(Z, Y, c)
CCαμ = CostCollection(X, Z, c)
CCβμ = CostCollection(Y, Z, c)

# weights
ω = [0.5, 0.5];





scatter(X[:,1],X[:,2],color=:red, label = "α")
scatter!(Y[:,1],Y[:,2],color=:green, label = "β")
scatter!(Z[:,1],Z[:,2],color=:blue, label = "μ")

Sαβ = SinkhornDivergence(Vα, Vβ, CCαβ; ε=ε, q=q , Δ=Δ, tol = 1e-3);
Vμ, val = barycenter_sinkhorn!(ω, Vμ, [Vα, Vβ], [CCμα, CCμβ], ∇c, (Sαβ.params), 5, δ);
val
Vα = SinkhornVariable(X,α)
Vβ = SinkhornVariable(Y,β)

scatter(X[:,1],X[:,2],color=:red, label = "α")
scatter!(Y[:,1],Y[:,2],color=:green, label = "β")
scatter!(Z[:,1],Z[:,2],color=:blue, label = "μ")

Sαμ = SinkhornDivergence(Vα, Vμ, CCαμ; ε=ε, q=q , Δ=Δ, tol = 1e-3)
Sαμ.params.s = Δ
initialize_potentials!(Sαμ.V1,Sαμ.V2,CCαμ)
compute!(Sαμ)
Π = TransportPlan(Sαμ);
sum(Matrix(Π))
maximum([Sαμ.V1.f[i] + Sαμ.V2.f[j] - CCαμ.C_xy[i,j] for i in 1:N, j in 1:N])
∇_μ_Sαμ = y_gradient(Sαμ, ∇c)
∇_α_Sαμ = x_gradient(Sαμ, ∇c)

Sβμ = SinkhornDivergence(Vβ, Vμ, CCβμ; ε=ε, q=q , Δ=Δ, tol = 1e-3)
Sβμ.params.s = Δ
initialize_potentials!(Sβμ.V1,Sβμ.V2,CCβμ)
compute!(Sβμ)
Π = TransportPlan(Sβμ);
sum(Matrix(Π))
maximum([Sβμ.V1.f[i] + Sβμ.V2.f[j] - CCβμ.C_xy[i,j] for i in 1:N, j in 1:N])
∇_μ_Sβμ = y_gradient(Sβμ, ∇c)
∇_β_Sβμ = x_gradient(Sβμ, ∇c)

scatter(X[:,1],X[:,2],color=:red, label = "α")
scatter!(Y[:,1],Y[:,2],color=:green, label = "β")
scatter!(Z[:,1],Z[:,2],color=:blue, label = "μ")


quiver(X[:,1],X[:,2],quiver= -1/2 .* (∇_α_Sαμ[:,1] ./ α, ∇_α_Sαμ[:,2] ./ α),
        alpha = 0.5, color = :red)
quiver!(Y[:,1],Y[:,2],quiver= -1/2 .* (∇_β_Sβμ[:,1] ./ β, ∇_β_Sβμ[:,2] ./ β),
        alpha = 0.5, color = :green)
scatter!(Z[:,1],Z[:,2],color=:blue)
scatter!(X[:,1],X[:,2],color=:red,legend = false)
scatter!(Y[:,1],Y[:,2],color=:green)

quiver(Z[:,1],Z[:,2],quiver= -1/2 .* (∇_μ_Sαμ[:,1] ./ μ ,∇_μ_Sαμ[:,2] ./ μ),
        alpha = 0.5, color = :red)
quiver!(Z[:,1],Z[:,2],quiver= -1/2 .* (∇_μ_Sβμ[:,1] ./ μ ,∇_μ_Sβμ[:,2] ./ μ),
        alpha = 0.5, color = :green)
scatter!(Z[:,1],Z[:,2],color=:blue)
scatter!(X[:,1],X[:,2],color=:red,legend = false)
scatter!(Y[:,1],Y[:,2],color=:green)

δX = (ω' * [∇_μ_Sαμ, ∇_μ_Sβμ]) ./ μ;

quiver(Z[:,1],Z[:,2],quiver= -1/2 .* (δX[:,1] ,δX[:,2] ),
        alpha = 0.5, color = :red)
scatter!(Z[:,1],Z[:,2],color=:blue)