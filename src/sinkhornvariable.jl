struct SinkhornVariable{T,d}
    X::AbstractArray{T,d}
    α::AbstractVector{T}
    log_α::AbstractVector{T}
    f::AbstractVector{T}
    h::AbstractVector{T}

    function SinkhornVariable(X::AbstractArray{T,d}, α::AbstractVector{T}) where {T, d}
        log_α = log.(α)
        f = zero(α)
        h = zero(α)
        new{T,d}(X,α,log_α,f,h)
    end
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