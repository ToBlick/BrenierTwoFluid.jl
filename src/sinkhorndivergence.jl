mutable struct SinkhornParameters{T, SAFE, SYM, ACC}
    Δ::T #diameter
    q::T #scaling parameter

    ω::T #acceleration parameter
    crit_it::Int #iteration number when ω is inferred
    p_ω::Int # ω is inferred comparing the residual at crit_it to that at crit_it - p_ω

    s::T #current scale
    ε::T #minimum scale

    p::T #ε ∼ x^p
    tol::T #tolerance
    maxit::Int

    function SinkhornParameters(Δ::T, q::T, ω::T, crit_it::Int, p_ω::Int, s::T, ε::T, p::T, tol::T, maxit, sym, acc) where {T}
        if tol == Inf
            SAFE = false
        else
            SAFE = true
        end
        new{T, SAFE, sym, acc}(Δ, q, ω, crit_it, p_ω, s, ε, p, tol, maxit)
    end
end

function SinkhornParameters(CC::CostCollection;
                            Δ = maximum(CC.C_xy),
                            q = 1.0,
                            ω = 1.0,
                            crit_it = 20,
                            p_ω = 2,
                            ε = maximum(CC.C_xy) * 1e-3,
                            s = ε,
                            p = 2.0,
                            tol = 1e-3,
                            maxit = Int(ceil(Δ/ε)),
                            sym = false,
                            acc = true)
    SinkhornParameters(Δ, q, ω, crit_it, p_ω, s, ε, p, tol, maxit, sym, acc)
end

struct SinkhornDivergence{T, d, AT, VT, CT, PT <: SinkhornParameters{T}}
    V1::SinkhornVariable{T,d,AT,VT} #X,α,log(α),f,h₁
    V2::SinkhornVariable{T,d,AT,VT} #Y,β,log(β),g,h₂
    CC::CostCollection{T,CT}        #C_xy,C_yx,C_xx,C_yy
    params::PT
    f₋::VT
    g₋::VT
end

issafe(SP::SinkhornParameters{T, SAFE}) where {T,SAFE} = SAFE
issymmetric(SP::SinkhornParameters{T, SAFE, SYM}) where {T,SAFE, SYM} = SYM
isaccelerated(SP::SinkhornParameters{T, SAFE, SYM, ACC}) where {T, SAFE, SYM, ACC} = ACC

#=
function SinkhornDivergence(V1::SinkhornVariable{T,d,AT,VT}, 
                            V2::SinkhornVariable{T,d,AT,VT}, 
                            CC::CostCollection{T,CT};
                            Δ = maximum(CC.C_xy),
                            q = 0.9,
                            ω = 1.0,
                            s = Δ,
                            ε = maximum(CC.C_xy) * 1e-3,
                            p = 2.0,
                            tol = Inf,
                            maxit = Int(ceil(Δ/ε)),
                            symmetric = false) where {T,d,AT <: AbstractArray{T,d},VT <: AbstractVector{T},CT}
                            SinkhornDivergence(V1,V2,CC,SinkhornParameters(Δ,q,ω,s,ε,p,tol,maxit,symmetric),zero(V1.f),zero(V2.f))
end

function SinkhornDivergence(X::AT,α::VT,
    Y::AT,β::VT,
    CC::CostCollection{T,CT};
    Δ = maximum(CC.C_xy),
    q = 0.9,
    ω = 1.0,
    s = Δ,
    ε = maximum(CC.C_xy) * 1e-3,
    p = 2.0,
    tol = Inf,
    maxit = Int(ceil(Δ/ε)),
    symmetric = false) where {T,d,AT <: AbstractArray{T,d},VT <: AbstractVector{T},CT}
    SinkhornDivergence(SinkhornVariable(X,α),SinkhornVariable(Y,β),CC,SinkhornParameters(Δ,q,ω,s,ε,p,tol,maxit,symmetric),zero(V1.f),zero(V2.f))
end
=#

function SinkhornDivergence(V1::SinkhornVariable{T,d,AT,VT}, 
    V2::SinkhornVariable{T,d,AT,VT}, 
    CC::CostCollection{T,CT},
    params::SinkhornParameters) where {T,d,AT <: AbstractArray{T,d},VT <: AbstractVector{T},CT}
    SinkhornDivergence(V1,V2,CC,params,zero(V1.f),zero(V2.f))
end

scale(S::SinkhornDivergence) = S.params.s
maxit(S::SinkhornDivergence) = S.params.maxit
tol(S::SinkhornDivergence) = S.params.tol
minscale(S::SinkhornDivergence) = S.params.ε
acceleration(S::SinkhornDivergence) = S.params.ω

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
    if issymmetric(S.params)
        @inbounds for i in eachindex(S.V1.f)
            S.V1.f[i] = 1/2 * S.f₋[i] + 1/2 * softmin(i, S.CC.C_yx, S.g₋, S.V2.log_α, scale(S))
            S.V1.h[i] = 1/2 * S.V1.h[i] + 1/2 * softmin(i, S.CC.C_xx, S.V1.h, S.V1.log_α, scale(S))
            if isaccelerated(S.params)
                S.V1.f[i] = (1 - acceleration(S)) * S.f₋[i] + acceleration(S) * S.V1.f[i]
            end
        end
        @inbounds for j in eachindex(S.V2.f)
            S.V2.f[j] = 1/2 * S.g₋[j] + 1/2 * softmin(j, S.CC.C_xy, S.f₋, S.V1.log_α, scale(S))
            S.V2.h[j] = 1/2 * S.V2.h[j] + 1/2 * softmin(j, S.CC.C_yy, S.V2.h, S.V2.log_α, scale(S))
            # no acceleration is done for the symmetric problem
            if isaccelerated(S.params)
                S.V2.f[j] = (1 - acceleration(S)) * S.g₋[j] + acceleration(S) * S.V2.f[j]
            end
        end
    else
        @inbounds for i in eachindex(S.V1.f)
            if isaccelerated(S.params)
                S.V1.f[i] = (1 - acceleration(S)) * S.V1.f[i] +  acceleration(S) * softmin(i, S.CC.C_yx, S.V2.f, S.V2.log_α, scale(S))
            else
                S.V1.f[i] = softmin(i, S.CC.C_yx, S.V2.f, S.V2.log_α, scale(S))
            end
            S.V1.h[i] = 1/2 * S.V1.h[i] + 1/2 * softmin(i, S.CC.C_xx, S.V1.h, S.V1.log_α, scale(S))
        end
        @inbounds for j in eachindex(S.V2.f)
            if isaccelerated(S.params)
                S.V2.f[j] = (1 - acceleration(S)) * S.V2.f[j] + acceleration(S) * softmin(j, S.CC.C_xy, S.V1.f, S.V1.log_α, scale(S))
            else
                S.V2.f[j] = softmin(j, S.CC.C_xy, S.V1.f, S.V1.log_α, scale(S))
            end
            S.V2.h[j] = 1/2 * S.V2.h[j] + 1/2 * softmin(j, S.CC.C_yy, S.V2.h, S.V2.log_α, scale(S))
        end
    end
end

#=
function sinkhorn_step!(f, f₋, g₋, h, log_α, log_β, C_yx, C_xx, ε) where T
    @inbounds for i in eachindex(f)
        f[i] = 1/2 * f₋[i] + 1/2 * softmin(i, C_yx, g₋, log_β, ε)
        h[i] = 1/2 * h[i] + 1/2 * softmin(i, C_xx, h, log_α, ε)
    end
end
=#

function compute!(S::SinkhornDivergence)
    it = 1
    r_p = 0
    #trace = []
    if !issafe(S.params)
        while scale(S) >= minscale(S)
            if it == 2
                S.params.ω = ω_opt
            end
            sinkhorn_step!(S)
            if issymmetric(S.params)
                S.f₋ .= S.V1.f
                S.g₋ .= S.V2.f
            end

            # lower the scale
            S.params.s = scale(S) * S.params.q

            # calculate ω
            if it == S.params.crit_it - S.params.p_ω
                r_p = π2_err
            elseif it == S.params.crit_it
                θ²_est = (π2_err/r_p)^(1/S.params.p_ω)
                S.params.ω = Real(2 / (1 + sqrt(1 - θ²_est)))
            end

            #π1_err, π2_err = marginal_errors(S)
            #push!(trace, [π1_err, π2_err, value(S)])

            it += 1
        end
    else
        while it <= maxit(S)
            sinkhorn_step!(S)
            # tolerance criterion: relative change of potentials
            #if norm(S.f₋ - S.V1.f, 1)/norm(S.f₋,1) + norm(S.g₋ - S.V2.f, 1)/norm(S.g₋,1) < tol(S)
            #    break
            #end
            if it % 5 == 0
                # tolerance criterion: marginal violation
                #π1_err, π2_err = marginal_errors(S)
                π2_err = marginal_error(S)
                #push!(trace, [π1_err, π2_err, value(S)])
                #println("marginal errors: $π1_err and $π2_err at scale ε = $(scale(S)).")
                if π2_err < tol(S)
                    break
                end
            end
            if issymmetric(S.params)
                S.f₋ .= S.V1.f
                S.g₋ .= S.V2.f
            end

            # lower the scale
            if S.params.q != 1
                S.params.s = maximum((scale(S) * S.params.q, minscale(S)))
            end

            # calculate ω
            if it == S.params.crit_it - S.params.p_ω
                r_p = marginal_error(S)
            elseif it == S.params.crit_it
                π2_err = marginal_error(S)
                θ²_est = (π2_err/r_p)^(1/S.params.p_ω)
                S.params.ω = Real(2 / (1 + sqrt(1 - θ²_est)))
            end

            it += 1
        end
    end
    #println("iterations: $it.")
    return value(S)
end

function marginal_errors(S::SinkhornDivergence)
    π1_err = 0
    π2_err = 0
    @inbounds for i in eachindex(S.V1.f)
        sm_i = softmin(i, S.CC.C_yx, S.V2.f, S.V2.log_α, scale(S))
        π2_err += abs( exp( (S.V1.f[i] - sm_i) / scale(S) ) - 1 ) * S.V1.α[i]
    end
    @inbounds for j in eachindex(S.V2.f)
        sm_j = softmin(j, S.CC.C_xy, S.V1.f, S.V1.log_α, scale(S))
        π1_err += abs( exp( (S.V2.f[j] - sm_j) / scale(S) ) - 1 ) * S.V2.α[j]
    end
    return π1_err, π2_err
end

function marginal_error(S::SinkhornDivergence)
    π2_err = 0
    @inbounds for i in eachindex(S.V1.f)
        sm_i = softmin(i, S.CC.C_yx, S.V2.f, S.V2.log_α, scale(S))
        π2_err += abs( exp( (S.V1.f[i] - sm_i) / scale(S) ) - 1 ) * S.V1.α[i]
    end
    return π2_err
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
        # W(α,β) - term
        for i in eachindex(f)
            v = exp((g[j] + f[i] - C_yx[j,i])/ε) * α[i]
            ∇S[j,:] .+= v .* ∇c(Y[j,:],X[i,:]) .* β[j]
        end
        # W(β,β) - term
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