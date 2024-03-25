struct LowRankMatrix{T} <: AbstractMatrix{T}
    u_vecs::Vector{Vector{T}}
    σ::Vector{T}
    v_vecs::Vector{Vector{T}}
    temp::Vector{T}
end

Base.size(K::LowRankMatrix) = (length(K.u_vecs[1]), length(K.v_vecs[1]))
Base.getindex(K::LowRankMatrix, i, j) = sum([ K.u_vecs[r][i] * K.σ[r] * K.v_vecs[r][j] for r in eachindex(K.σ) ])

function LinearAlgebra.mul!(c::AbstractVector, K::LowRankMatrix, b::AbstractVector)
    c .= 0
    r = K.temp
    @threads for i in eachindex(K.σ)
        r[i] = dot(K.v_vecs[i], b)
        r[i] *= K.σ[i]
    end
    @threads for i in eachindex(K.u_vecs)
        for j in eachindex(K.u_vecs[i])
            c[j] += K.u_vecs[i][j] .* r[i]
        end
    end
    return c
end