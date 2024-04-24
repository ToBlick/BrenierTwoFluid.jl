struct TransportPlan{T, PLT <: ParticleList{T}, CT <: AbstractMatrix{T}}
    Mat::Matrix{T}
    V1::SinkhornVariable{T,PLT}
    V2::SinkhornVariable{T,PLT}
    C::CT
    ε::T
end

function TransportPlan(S::SinkhornDivergence)
    TransportPlan(transportmatrix(S),S.V1,S.V2,S.CC.C_xy,scale(S))
end

Base.Matrix(Π::TransportPlan) = Π.Mat

function transportmatrix(S::LogSinkhornDivergence)
    [ S.V1.α[i] * S.V2.α[j] * exp((S.V1.f[i] + S.V2.f[j] - S.CC.C_xy[i,j]) / scale(S)) for i in eachindex(S.V1.α), j in eachindex(S.V2.α) ]
end

function apply_M!(Mv, log_v, t1, t2, S)
    ε = scale(S)
    t1 .= S.V1.log_α .+ log_v
    for j in eachindex(S.V2.f)
            # t2 = LSE_k((f_k - C_kj)/ε + log α_k + log v_k) + log β_j
            t2[j] = - 1/ε * softmin(j, S.CC.C_xy, S.V1.f, t1, ε) + S.V2.log_α[j] + S.V2.f[j]/ε
    end
    for i in eachindex(Mv)
            # Mv = exp ( f_i/ε + LSE_j((2g_j - C_ij)/ε + t2_j) )
            Mv[i] = exp( S.V1.f[i]/ε - 1/ε * softmin(i, S.CC.C_yx, S.V2.f, t2, ε) )
    end
    return Mv
end