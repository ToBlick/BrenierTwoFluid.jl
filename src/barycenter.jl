"""
    SinkhornBarycenter

A type representing a Sinkhorn barycenter problem.
"""
struct SinkhornBarycenter{LOG, SAFE, SYM, ACC, T, d, AT, VT, CT}
    ω::Vector{T}
    Ss::Vector{SinkhornDivergence{LOG, SAFE, SYM, ACC, T, d, AT, VT, CT}}
    CCs::Vector{CostCollection{T,CT}}
    ∇c
    max_it::Int
    tol::T
    δX::AT

    #=
    function SinkhornBarycenter(ω::Vector{T}, 
                                Ss::Vector{SinkhornDivergence{LOG, SAFE, SYM, ACC, T, d, AT, VT, CT}}, 
                                CCs::Vector{CostCollection{T,CT}},
                                ∇c,
                                max_it::Int,
                                tol::T,
                                δX::AT) where {LOG, SAFE, SYM, ACC, T, d, AT, VT, CT}
        new{LOG, SAFE, SYM, ACC, T, d, AT, VT, CT}(ω, Ss, CCs, ∇c, max_it, tol, δX)
    end
    =#
end

function SinkhornBarycenter(ω,
                            Xμ::AT,
                            μ::VT,
                            Vs::Vector{SinkhornVariable{T,d,AT,VT}},
                            c,
                            ∇c,
                            params::SinkhornParameters{SAFE, SYM, ACC, T},
                            max_it::Int,
                            tol,
                            islog) where {SAFE, SYM, ACC, T, d, AT, VT}

    log_μ = log.(μ)

    CCs = [ CostCollection(Xμ, Vs[i].X, c) for i in eachindex(Vs) ]
    Ss = [ SinkhornDivergence(SinkhornVariable(Xμ, μ, log_μ), Vs[i], CCs[i], params, islog) for i in eachindex(Vs) ]
    SinkhornBarycenter(ω, Ss, CCs, ∇c, max_it, tol, zero(Xμ))
end

const SafeSinkhornBarycenter = SinkhornBarycenter{T, true} where {T}
const UnsafeSinkhornBarycenter = SinkhornBarycenter{T, false} where {T}
const SymmetricSinkhornBarycenter = SinkhornBarycenter{T, SAFE, true} where {T, SAFE}
const AcceleratedSinkhornBarycenter = SinkhornBarycenter{T, SAFE, SYM, true} where {T, SAFE, SYM}

function compute!(B::SinkhornBarycenter)

    Xμ = B.Ss[begin].V1.X
    μ = B.Ss[begin].V1.α
    N = size(B.δX, 1)
    for it in 1:B.max_it
        B.δX .= 0
        # the SinkhornDivergences B.Ss each hold their own representation of Vμ, sharing only positions, weights, and logarithmic weights. Hence, the following can be done fully in parallel.
        for k in eachindex(B.Ss)
            S = B.Ss[k]
            # S.V1 is Vμ (the kth representation), S.V2 is the kth input density
            initialize_potentials!(S) 
            compute!(S)
            # now, the k B.Ss element holds W²(μ,α[k])
            # calculate δX[i] = ∂/∂Xμ[i] ( ∑ₖ ω[k] W²(μ,α[k]) ) = ∑ₖ ω[k] ∂W²(μ,α[k])/∂Xμ[i]
            # note that the gradient already comes divided by the μ weights
            B.δX .+= B.ω[k] .* x_gradient!(S, B.∇c)
        end

        if norm(B.δX,2)/N < B.tol || it == B.max_it
            # if converged
            # note that tol > norm(δX,2)/N = 1/N √ ∑ᵢVᵢ² ≥ 1/N ∑ᵢ|Vᵢ|
            for S in B.Ss
                # compute the "other" gradients, i.e. those with respect to α[k] positions
                # y_gradient!(S, B.∇c)
            end
            # return the value of the barycenter problem, i.e. the distance.
            return B.ω' * [value(S) for S in B.Ss]
        else
            # if not converged, change Xμ accordingly
            Xμ .-= B.δX
        end
    end
end

#=
function barycenter_sinkhorn!(ω, Vμ ,V_vec, CC_vec, ∇c, params, maxit, tol)

    S_vec = [SinkhornDivergence(Vμ, V_vec[s], CC_vec[s]; ε=params.ε, q=params.q, Δ=params.Δ, tol=params.tol) 
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
=#