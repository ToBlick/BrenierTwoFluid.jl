struct TransportPlan{T, d, AT <: AbstractArray{T,d}, VT <: AbstractVector{T}, CT <: AbstractMatrix{T}}
    Mat::Matrix{T}
    V1::SinkhornVariable{T,d,AT,VT}
    V2::SinkhornVariable{T,d,AT,VT}
    C::CT
    ε::T

    function TransportPlan(S::SinkhornDivergence{T,d,AT,VT,CT}) where {T,d,AT,VT,CT}
        new{T,d,AT,VT,CT}(transportmatrix(S),S.V1,S.V2,S.CC.C_xy,scale(S))
    end
end

Base.Matrix(Π::TransportPlan) = Π.Mat

function transportmatrix(S::SinkhornDivergence)
    [ S.V1.α[i] * S.V2.α[j] * exp((S.V1.f[i] + S.V2.f[j] - S.CC.C_xy[i,j]) / scale(S)) for i in eachindex(S.V1.α), j in eachindex(S.V2.α) ]
end