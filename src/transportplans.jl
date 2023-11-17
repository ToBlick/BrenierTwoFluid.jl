struct TransportPlan{T,d}
    Mat::Matrix{T}
    V1::SinkhornVariable{T,d}
    V2::SinkhornVariable{T,d}
    C::AbstractMatrix{T}
    ε::T

    function TransportPlan(S::SinkhornDivergence{T,d}) where {T,d}
        new{T,d}(transportmatrix(S),S.V1,S.V2,S.CC.C_xy,scale(S))
    end
end

Base.Matrix(Π::TransportPlan) = Π.Mat

function transportmatrix(S::SinkhornDivergence)
    [ S.V1.α[i] * S.V2.α[j] * exp((S.V1.f[i] + S.V2.f[j] - S.CC.C_xy[i,j]) / scale(S)) for i in eachindex(S.V1.α), j in eachindex(S.V2.α) ]
end