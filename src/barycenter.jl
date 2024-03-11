"""
    SinkhornBarycenter

    Represents a Sinkhorn barycenter problem.

    Fields:
    - `ω::Vector{T}`: weights of the input distributions.
    - `Ss::Vector{SinkhornDivergence}`: Sinkhorn divergences between the input distributions and the barycenter.
    - `CCs::Vector{CostCollection}`: cost collections for the Sinkhorn divergences.
    - `∇c`: gradient of the cost function.
    - `max_it::Int`: maximum number of iterations. This can be different from the maximum number of iterations of the Sinkhorn divergences, typically it is much smaller.
    - `tol::T`: tolerance for the stopping criterion. This can be different from the tolerance of the Sinkhorn divergences.
    - `δX::AT`: gradient of the barycenter problem with respect to the positions of the particles representing the barycenter.

    Type parameters:
    - `LOG, SAFE, SYM, ACC`: As in `SinkhornDivergence` and in fact identical to those values of the contained `SinkhornDivergence` objects.
"""
struct SinkhornBarycenter{LOG, SAFE, SYM, ACC, T, d, AT, VT, CT}
    ω::Vector{T}
    Ss::Vector{SinkhornDivergence{LOG, SAFE, SYM, ACC, T, d, AT, VT, CT}}
    CCs::Vector{CostCollection{T,CT}}
    ∇c
    max_it::Int
    tol::T
    δX::AT
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