# compute W_2 distance between two uniform distributions
d′ = 2*Int(floor(d/2))
d′′ = 2*Int(ceil(d/2))

Random.seed!(123)
X .= rand(N,d) .- 0.5
Y .= rand(M,d) .- 0.5

∇S_x = zero(X)
∇S_y = zero(Y)

for i in 1:2
        if i == 1 # Scaling test
                truevalue = 1/2 * d * 1/12 * offset^2 #1/2 * d * offset^2
                Y .= (rand(M,d) .- 0.5) .* (1 + offset)
        else # Shifting test
                truevalue = 1/2 * d * offset^2
                Y .= (rand(M,d) .- 0.5) .+ offset
        end

        CC = CostCollection(X, Y, c)
        V = SinkhornVariable(X,α)
        W = SinkhornVariable(Y,β)

        # acc, no sym
        params = SinkhornParameters(CC;ε=ε,q=1.0,Δ=1.0,s=s,tol=tol,crit_it=crit_it,p_ω=p_ω,sym=false,acc=true)
        S = SinkhornDivergence(V,W,CC,params)
        initialize_potentials!(V,W,CC)
        valueS = compute!(S)
        ∇S_x = x_gradient!(S, ∇c)
        ∇S_y = y_gradient!(S, ∇c)

        if i == 1
                @test dot(X - ∇S_x - (1 + offset) * X, α .* (X - ∇S_x - (1 + offset) * X)) < (sqrt(N*M))^(-4/(2*d′′ + 8)) * log(sqrt(N*M))
                #println(dot(X - ∇S_x - (1 + offset) * X, α .* (X - ∇S_x - (1 + offset) * X)))
                @test dot(Y - ∇S_y - 1/(1 + offset) * Y, β .* (Y - ∇S_y - 1/(1 + offset) * Y)) < (sqrt(N*M))^(-4/(2*d′′ + 8)) * log(sqrt(N*M))
                #println(dot(Y - ∇S_y - 1/(1 + offset) * Y, β .* (Y - ∇S_y - 1/(1 + offset) * Y)))
        else
                @test dot(∇S_x ./ α + offset * X ./ X, α .* (∇S_x ./ α + offset * X ./ X)) < (sqrt(N*M))^(-4/(2*d′′ + 8)) * log(sqrt(N*M))
                #println(dot(∇S_x ./ α + offset * X ./ X, α .* (∇S_x ./ α + offset * X ./ X)))
                @test dot(∇S_y ./ β - offset * Y ./ Y, β .* (∇S_y ./ β - offset * Y ./ Y)) < (sqrt(N*M))^(-4/(2*d′′ + 8)) * log(sqrt(N*M))
                #println(dot(∇S_y ./ β - offset * Y ./ Y, β .* (∇S_y ./ β - offset * Y ./ Y)))
        end
end
