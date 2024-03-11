"""
    SinkhornVariable

    Probability distribution, represented as a collection of weighted particles, and its associated transport potentials.

    Fields:
    - `X::AT`: particle positions. `AT` is an N×d AbstractArray with elements of type `T`.
    - `α::VT`: particle weights. `VT` is an N×1 AbstractVector with elements of type `T`.
    - `log_α::VT`: logarithm of the particle weights.
    - `f::VT`: transport potential, i.e. the dual variable associated with the probability distribution ∑ᵢαᵢδ(Xᵢ).
    - `f₋::VT`: cache-like array to store the previous value of `f`.
    - `h::VT`: transport potential for the de-biasing step.
    - `h₋::VT`: cache-like array to store the previous value of `h`.
    - `∇fh::AT`: gradient of `f+h` evaluated at `X`.

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

function initialize_potentials_nolog!(V1::SinkhornVariable, V2::SinkhornVariable, CC::CostCollection)
    V1.f   .= V1.α # CC.C_xy * V2.α
    V1.f₋  .= V1.α # V1.f
    V2.f   .= V2.α # CC.C_yx * V1.α
    V2.f₋  .= V2.α # V2.f
    V1.h   .= V1.α # CC.C_xx * V1.α
    V1.h₋  .= V1.α # V1.h
    V2.h   .= V2.α # CC.C_yy * V2.α
    V2.h₋  .= V2.α # V2.h
    V1.∇fh .= 0.0
    V2.∇fh .= 0.0
    # the latter are good for entropic OT with large ε.
end

function initialize_potentials_log!(V1::SinkhornVariable, V2::SinkhornVariable, CC::CostCollection)
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