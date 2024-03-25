function resample!(X, V, α, h, p, s_vec, species)
    N = length(α)

    X_new = stack(vec([ [x,y] for x in range(-0.5,0.5,length=Int(sqrt(N))), y in range(-0.5,0.5,length=Int(sqrt(N))) ]), dims = 1)
    V_new = zero(V)
    α_new = zero(α)

    ρ_vec = [ vec([ kde(X_new[i,:], X, α, h, p, s, species) for i in axes(X_new,1) ]) for s in s_vec ]

    for ρ in ρ_vec
        ρ ./= sum(ρ)
        ρ ./= length(s_vec) # number of species
    end

    for i in axes(X,1)
        α_new[i] = sum([ ρ_vec[j][i] for j in axes(ρ_vec,1)] )
        species[i] = argmax([ ρ_vec[j][i] for j in axes(ρ_vec,1)] )
    end

    for i in axes(X,1)
        for j in axes(X,1)
            V_new[i,:] += V[j,:] .* α[j] * bspline(norm(X_new[j,:]-X[i,:])/h, -p/2-0.5, p) / α_new[i]
        end
    end

    X .= X_new
    V .= V_new
    α .= α_new
end