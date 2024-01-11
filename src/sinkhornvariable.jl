struct SinkhornVariable{T, d, AT <: AbstractArray{T,d}, VT <: AbstractVector{T}}
    X::AT
    α::VT
    log_α::VT
    f::VT
    h::VT
end

function SinkhornVariable(X::AT, α::VT) where {T, d, AT <: AbstractArray{T,d}, VT <: AbstractVector{T}}
    log_α = log.(α)
    f = zero(α)
    h = zero(α)
    SinkhornVariable(X,α,log_α,f,h)
end

positions(V::SinkhornVariable) = V.X
logdensity(V::SinkhornVariable) = V.log_α
density(V::SinkhornVariable) = V.log_α
potential(V::SinkhornVariable) = V.f
debiasing_potential(V::SinkhornVariable) = V.h

function initialize_potentials!(V1::SinkhornVariable, V2::SinkhornVariable, CC::CostCollection)
    V1.f .= CC.C_xy * V2.α
    V2.f .= CC.C_yx * V1.α
    V1.h .= CC.C_xx * V1.α
    V2.h .= CC.C_yy * V2.α
end