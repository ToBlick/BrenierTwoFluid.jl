struct SinkhornDivergence{T,d}
    V1::SinkhornVariable{T,d}  #X,α,log(α),f,h₁
    V2::SinkhornVariable{T,d}  #Y,β,log(β),g,h₂
    CC::CostCollection{T}       #C_xy,C_yx,C_xx,C_yy

    f₋::AbstractVector{T}
    g₋::AbstractVector{T}
    
    Δ::T #diameter
    q::T #scaling parameter
    s::Vector{T} #current scale
    ε::T #minimum scale
    p::T #ε ∼ x^p

    function SinkhornDivergence(V1::SinkhornVariable{T,d}, 
                                V2::SinkhornVariable{T,d}, 
                                CC::CostCollection{CT};
                                Δ = maximum(CC.C_xy),
                                q = 0.9,
                                ε = maximum(CC.C_xy) * 1e-3,
                                p = 2) where {T,d,CT}
        new{T,d}(V1,V2,CC,zero(V1.f),zero(V2.f),Δ,q,[Δ],ε,p)
    end
end

scale(S::SinkhornDivergence) = S.s[1] 

function SinkhornDivergence(X::AbstractArray{T,d},α::AbstractVector{T},
                            Y::AbstractArray{T,d},β::AbstractVector{T},
                            CC::CostCollection{CT};
                            Δ = maximum(CC.C_xy),
                            q = 0.9,
                            ε = maximum(CC.C_xy) * 1e-2,
                            p = 2) where {T,d,CT}
    SinkhornDivergence(SinkhornVariable(X,α),SinkhornVariable(Y,β),CC,Δ,q,ε,p)
end

function softmin(j, C::AbstractMatrix, f::AbstractVector{T}, log_α::AbstractVector{T}, ε::T) where T
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
    (- ε * (log(r) + M))::T
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
    S.f₋ .= S.V1.f
    S.g₋ .= S.V2.f
end

function compute!(S::SinkhornDivergence)
    while scale(S) > S.ε
        sinkhorn_step!(S)
        S.s[1] = scale(S) * S.q
    end
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
    ∇Wαβ_i = zero(∇S[begin,:])
    ∇Wαα_i = zero(∇S[begin,:])
    for i in eachindex(f)
        ∇Wαβ_i .= 0
        denom_αβ = 0
        for j in eachindex(g)
            v = exp((g[j] + f[i] - C_xy[i,j])/ε) * β[j]
            ∇Wαβ_i .+= v .* ∇c(X[i,:],Y[j,:])
            denom_αβ += v
        end
        ∇Wαα_i .= 0
        denom_αα = 0
        for k in eachindex(h)
            v = exp((h[k] + h[i] - C_xx[i,k])/ε) * α[k]
            ∇Wαα_i .+= v .* ∇c(X[i,:],X[k,:])
            denom_αα += v
        end
        ∇S[i,:] .= (∇Wαβ_i ./ denom_αβ .- ∇Wαα_i ./ denom_αα) .* α[i]
    end
    ∇S
end

function x_gradient(S::SinkhornDivergence, ∇c)
    ∇S = zero(S.V1.X)
    x_gradient!(∇S, S, ∇c)
    ∇S
end


function SinkhornDivergence(X, α,
                            Y, β,
                            c,
                            f, g, h_x, h_y,
                            Δ = 1,
                            σ = 1e-2,
                            q = 0.9)

    log_α = log.(α)
    log_β = log.(β)

    C_xy = LazyCost(X, Y, c)
    C_yx = LazyCost(Y, X, c)
    C_xx = LazyCost(X, X, c)
    C_yy = LazyCost(Y, Y, c)

    if Δ == 1
        f = C_xy * β
        g = C_yx * α

        h_x = C_xx * α
        h_y = C_yy * β
    end

    f₋ = copy(f)
    g₋ = copy(g)

    s = Δ

    while s > σ
        ε = s^2

        f₋ .= f
        g₋ .= g

        for i in eachindex(f)
            f[i] = 1/2 * f₋[i] + 1/2 * softmin(i, g₋, C_yx, log_β, ε)
            h_x[i] = 1/2 * h_x[i] + 1/2 * softmin(i, h_x, C_xx, log_α, ε)
        end
        for j in eachindex(g)
            g[j] = 1/2 * g₋[j] + 1/2 * softmin(j, f₋, C_xy, log_α, ε)
            h_y[j] = 1/2 * h_y[j] + 1/2 * softmin(j, h_y, C_yy, log_β, ε)
        end

        s *= q
    end

    ε = s^2
    S_ε = (f - h_x)' * α + (g - h_y)' * β

    grad_x = zero(X)
    for i in eachindex(f)
        nom1 = zero(X[1,:])
        denom1 = 0
        nom2 = zero(X[1,:])
        denom2 = 0

        gvals = [ g[j] - C_xy[i,j] for j in eachindex(g) ]
        mg = maximum(gvals)
        for j in eachindex(g)
            nom1 += exp((g[j] - C_xy[i,j] - mg)/ε) * (X[i,:] - Y[j,:])
            denom1 += exp((g[j] - C_xy[i,j] - mg)/ε)
        end
        hvals = [ h_x[k] - C_xx[i,k] for k in eachindex(f) ]
        mh = maximum(hvals)
        for k in eachindex(f)
            nom2 += exp((h_x[k] - C_xx[i,k] -mh )/ε) * (X[i,:] - X[k,:])
            denom2 += exp((h_x[k] - C_xx[i,k] -mh)/ε)
        end
        grad_x[i,:] .= nom1/denom1 - nom2/denom2
    end

    # π_αβ = [ α[i] * β[j] * exp((f[i] + g[j] - C_xy[i,j]) / ε) for i in eachindex(α), j in eachindex(β) ]

    return (f = f, 
            g = g, 
            h_x = h_x, 
            h_y = h_y, 
            ε = ε, 
            S_ε = S_ε,
            grad_x = grad_x)
end

