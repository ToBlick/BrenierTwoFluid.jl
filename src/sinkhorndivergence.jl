mutable struct SinkhornParameters{T, SAFE}
    Δ::T #diameter
    q::T #scaling parameter
    s::T #current scale
    ε::T #minimum scale
    p::T #ε ∼ x^p
    tol::T #tolerance
    maxit::Int

    function SinkhornParameters(Δ::T, q::T, s::T, ε::T, p::T, tol::T, maxit) where {T}
        if tol == Inf
            SAFE = false
        else
            SAFE = true
        end
        new{T, SAFE}(Δ, q, s, ε, p, tol, maxit)
    end
end

struct SinkhornDivergence{T, d, AT, VT, CT, PT <: SinkhornParameters{T}}
    V1::SinkhornVariable{T,d,AT,VT} #X,α,log(α),f,h₁
    V2::SinkhornVariable{T,d,AT,VT} #Y,β,log(β),g,h₂
    CC::CostCollection{T,CT}       #C_xy,C_yx,C_xx,C_yy
    params::PT
    f₋::VT
    g₋::VT
end

issafe(SP::SinkhornParameters{T, SAFE}) where {T,SAFE} = SAFE

function SinkhornDivergence(V1::SinkhornVariable{T,d,AT,VT}, 
                            V2::SinkhornVariable{T,d,AT,VT}, 
                            CC::CostCollection{T,CT};
                            Δ = maximum(CC.C_xy),
                            q = 0.9,
                            ε = maximum(CC.C_xy) * 1e-3,
                            p = 2.0,
                            tol = Inf,
                            maxit = Int(ceil(Δ/ε))) where {T,d,AT <: AbstractArray{T,d},VT <: AbstractVector{T},CT}
                            SinkhornDivergence(V1,V2,CC,SinkhornParameters(Δ,q,Δ,ε,p,tol,maxit),zero(V1.f),zero(V2.f))
end

scale(S::SinkhornDivergence) = S.params.s
maxit(S::SinkhornDivergence) = S.params.maxit
tol(S::SinkhornDivergence) = S.params.tol
minscale(S::SinkhornDivergence) = S.params.ε

function SinkhornDivergence(X::AT,α::VT,
                            Y::AT,β::VT,
                            CC::CostCollection{T,CT};
                            Δ = maximum(CC.C_xy),
                            q = 0.9,
                            ε = maximum(CC.C_xy) * 1e-3,
                            p = 2.0,
                            tol = Inf,
                            maxit = Int(ceil(Δ/ε))) where {T,d,AT <: AbstractArray{T,d},VT <: AbstractVector{T},CT}
    SinkhornDivergence(SinkhornVariable(X,α),SinkhornVariable(Y,β),CC,SinkhornParameters(Δ,q,Δ,ε,p,tol,maxit),zero(V1.f),zero(V2.f))
end

function softmin(j, C::AbstractMatrix{T}, f::AbstractVector{T}, log_α::AbstractVector{T}, ε::T) where T
    M = -Inf
    r = 0.0
    @inbounds for i in eachindex(f)
        v = (f[i] - C[i,j])/ε + log_α[i]
        if v <= M
            r += exp(v-M)
        else
            r *= exp(M-v)
            r += 1
            M = v
        end
    end
    (- ε * (log(r) + M))
end

function sinkhorn_step!(S::SinkhornDivergence{T}) where T
    @inbounds for i in eachindex(S.V1.f)
        S.V1.f[i] = 1/2 * S.f₋[i] + 1/2 * softmin(i, S.CC.C_yx, S.g₋, S.V2.log_α, scale(S))
        S.V1.h[i] = 1/2 * S.V1.h[i] + 1/2 * softmin(i, S.CC.C_xx, S.V1.h, S.V1.log_α, scale(S))
    end
    @inbounds for j in eachindex(S.V2.f)
        S.V2.f[j] = 1/2 * S.g₋[j] + 1/2 * softmin(j, S.CC.C_xy, S.f₋, S.V1.log_α, scale(S))
        S.V2.h[j] = 1/2 * S.V2.h[j] + 1/2 * softmin(j, S.CC.C_yy, S.V2.h, S.V2.log_α, scale(S))
    end
    #=
    @inbounds for i in eachindex(S.V1.f)
        S.V1.f[i] = softmin(i, S.CC.C_yx, S.V2.f, S.V2.log_α, scale(S))
        S.V1.h[i] = 1/2 * S.V1.h[i] + 1/2 * softmin(i, S.CC.C_xx, S.V1.h, S.V1.log_α, scale(S))
    end
    @inbounds for j in eachindex(S.V2.f)
        S.V2.f[j] = softmin(j, S.CC.C_xy, S.V1.f, S.V1.log_α, scale(S))
        S.V2.h[j] = 1/2 * S.V2.h[j] + 1/2 * softmin(j, S.CC.C_yy, S.V2.h, S.V2.log_α, scale(S))
    end
    =#
end

function sinkhorn_step!(f, f₋, g₋, h, log_α, log_β, C_yx, C_xx, ε) where T
    @inbounds for i in eachindex(f)
        f[i] = 1/2 * f₋[i] + 1/2 * softmin(i, C_yx, g₋, log_β, ε)
        h[i] = 1/2 * h[i] + 1/2 * softmin(i, C_xx, h, log_α, ε)
    end
end

function compute!(S::SinkhornDivergence)
    if !issafe(S.params)
        while scale(S) >= minscale(S)
            sinkhorn_step!(S)
            S.f₋ .= S.V1.f
            S.g₋ .= S.V2.f
            S.params.s = scale(S) * S.params.q
        end
    else
        it = 0
        while it < maxit(S)
            sinkhorn_step!(S)
            if norm(S.f₋ - S.V1.f, 1)/norm(S.f₋,1) + norm(S.g₋ - S.V2.f, 1)/norm(S.g₋,1) < tol(S)
                break
            end
            S.f₋ .= S.V1.f
            S.g₋ .= S.V2.f
            S.params.s = maximum((scale(S) * S.params.q, minscale(S)))
        end
    end
    value(S)
end

function value(S::SinkhornDivergence)
    (S.V1.f - S.V1.h)' * S.V1.α + (S.V2.f - S.V2.h)' * S.V2.α
end

function x_gradient!(∇S, S::SinkhornDivergence{T,d}, ∇c) where {T,d}
    C_xy = S.CC.C_xy
    C_xx = S.CC.C_xx
    X = S.V1.X
    Y = S.V2.X
    g = S.V2.f
    f = S.V1.f
    h = S.V1.h
    β = S.V2.α
    α = S.V1.α
    ε = scale(S)
    ∇S .= 0
    for i in eachindex(f)
        for j in eachindex(g)
            v = exp((g[j] + f[i] - C_xy[i,j])/ε) * β[j]
            ∇S[i,:] .+= v .* ∇c(X[i,:],Y[j,:]) .* α[i]
        end
        for k in eachindex(h)
            v = exp((h[k] + h[i] - C_xx[i,k])/ε) * α[k]
            ∇S[i,:] .-= v .* ∇c(X[i,:],X[k,:]) .* α[i]
        end
    end
    ∇S
end

function x_gradient(S::SinkhornDivergence, ∇c)
    ∇S = zero(S.V1.X)
    x_gradient!(∇S, S, ∇c)
    ∇S
end

function y_gradient!(∇S, S::SinkhornDivergence{T,d}, ∇c) where {T,d}
    C_yx = S.CC.C_yx
    C_yy = S.CC.C_yy
    X = S.V1.X
    Y = S.V2.X
    g = S.V2.f
    f = S.V1.f
    h = S.V2.h
    β = S.V2.α
    α = S.V1.α
    ε = scale(S)
    ∇S .= 0
    for j in eachindex(g)
        for i in eachindex(f)
            v = exp((g[j] + f[i] - C_yx[j,i])/ε) * α[i]
            ∇S[j,:] .+= v .* ∇c(Y[j,:],X[i,:]) .* β[j]
        end
        for k in eachindex(h)
            v = exp((h[k] + h[j] - C_yy[j,k])/ε) * β[k]
            ∇S[j,:] .-= v .* ∇c(Y[j,:],Y[k,:]) .* β[j]
        end
    end
    ∇S
end

function y_gradient(S::SinkhornDivergence, ∇c)
    ∇S = zero(S.V1.X)
    y_gradient!(∇S, S, ∇c)
    ∇S
end