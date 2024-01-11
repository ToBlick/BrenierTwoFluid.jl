function barycenter_sinkhorn!(ω, Vμ ,V_vec, CC_vec, ∇c, params, maxit, tol) where T

    S_vec = [SinkhornDivergence(Vμ, V_vec[s], CC_vec[s];
                                ε=params.ε, q=params.q, Δ=params.Δ, tol=params.tol) 
                for s in eachindex(V_vec)]
    ∇S_μ_vec = [zero(Vμ.X) for _ in V_vec]
    ∇S_α_vec = [zero(V.X) for V in V_vec]
    N = size(Vμ.X, 1)

    for i in 1:maxit
        for s in eachindex(S_vec)
            S = S_vec[s]
            S.params.s = params.Δ
            initialize_potentials!(S.V1,S.V2,CC_vec[s])
            compute!(S)
            x_gradient!(∇S_μ_vec[s], S, ∇c)
        end

        δX = (ω' * ∇S_μ_vec) ./ Vμ.α
        #println(norm(δX)^2/N)

        if norm(δX)^2/N < tol^2
            println("terminated at i = $i")
            for s in eachindex(S_vec)
                S = S_vec[s]
                y_gradient!(∇S_α_vec[s], S, ∇c)
            end
            return Vμ, ω' * [value(S) for S in S_vec], ∇S_α_vec
        else
            Vμ.X .-= δX
        end
    end

    for s in eachindex(S_vec)
        S = S_vec[s]
        S.params.s = params.Δ
        initialize_potentials!(S.V1,S.V2,CC_vec[s])
        compute!(S)
        x_gradient!(∇S_μ_vec[s], S, ∇c)
        y_gradient!(∇S_α_vec[s], S, ∇c)
    end

    return Vμ, ω' * [value(S) for S in S_vec], ∇S_α_vec
end