"""
SinkhornVariable

holds particle positions, weights, logarithm of the weights, potential and de-biasing potential, as well as those from the previous iteration
"""
struct SinkhornVariable{T, d, AT <: AbstractArray{T,d}, VT <: AbstractVector{T}}
    X::AT
    α::VT
    log_α::VT
    f::VT
    f₋::VT
    h::VT
    h₋::VT
    ∇fh::AT
end

function SinkhornVariable(X::AT, α::VT) where {T, d, AT <: AbstractArray{T,d}, VT <: AbstractVector{T}}
    log_α = log.(α)
    f     = zero(α)
    f₋    = zero(α)
    h     = zero(α)
    h₋    = zero(α)
    ∇fh   = zero(X)
    SinkhornVariable(X, α, log_α, f, f₋, h, h₋, ∇fh)
end

function SinkhornVariable(X::AT, α::VT, log_α::VT) where {T, d, AT <: AbstractArray{T,d}, VT <: AbstractVector{T}}
    f     = zero(α)
    f₋    = zero(α)
    h     = zero(α)
    h₋    = zero(α)
    ∇fh   = zero(X)
    SinkhornVariable(X, α, log_α, f, f₋, h, h₋, ∇fh)
end

positions(V::SinkhornVariable) = V.X
logdensity(V::SinkhornVariable) = V.log_α
density(V::SinkhornVariable) = V.log_α
potential(V::SinkhornVariable) = V.f
debiasing_potential(V::SinkhornVariable) = V.h

function initialize_potentials!(V1::SinkhornVariable, V2::SinkhornVariable, CC::CostCollection)
    V1.f   .= 0.0 # CC.C_xy * V2.α
    V1.f₋  .= 0.0 # V1.f
    V2.f   .= 0.0 # CC.C_yx * V1.α
    V2.f₋  .= 0.0 # V2.f
    V1.h   .= 0.0 # CC.C_xx * V1.α
    V1.h₋  .= 0.0 # V1.h
    V2.h   .= 0.0 # CC.C_yy * V2.α
    V2.h₋  .= 0.0 # V2.h
    V1.∇fh .= 0.0
    V2.∇fh .= 0.0
    # the latter are good for entropic OT with large ε.
end