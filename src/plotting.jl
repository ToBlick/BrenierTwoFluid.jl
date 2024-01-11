function Plots.plot(Π::TransportPlan{T,2}) where {T}
    X = Π.V1.X
    Y = Π.V2.X
    α = Π.V1.α
    β = Π.V2.α
    N = length(α)
    M = length(β)
    plt = plot()
    scatter!(X[:,1],X[:,2],label = L"α",color = :red)
    scatter!(Y[:,1],Y[:,2],label = L"β",color = :blue)
    C = sum(Matrix(Π))
    for i in eachindex(α), j in eachindex(β)
        v = sqrt(N*M)*Matrix(Π)[i,j]/C
        if v > 1e-2
            plot!([X[i,1], Y[j,1]], [X[i,2], Y[j,2]], 
                alpha=v,
                label = false, color = :black, grid = false, 
                size = (600,600))
        end
    end
    plt
end