struct LowRankMatrix{T} <: AbstractMatrix{T}
    u_vecs::Vector{Vector{T}}
    σ::Vector{T}
    v_vecs::Vector{Vector{T}}
end

Base.size(K::LowRankMatrix) = (length(K.u_vecs[1]), length(K.v_vecs[1]))
Base.getindex(K::LowRankMatrix, i, j) = sum([ K.u_vecs[r][i] * K.σ[r] * K.v_vecs[r][j] for r in eachindex(K.σ) ])

function LinearAlgebra.mul!(c::AbstractVector, K::LowRankMatrix, b::AbstractVector)
    c .= 0
    r = zero(K.σ)
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

#=
function recursive_rls_nyström(X::AT, Y::AT, k::Base.Callable, λ, δ) where {AT}

    X = rand(10,2)
    m = size(X,1)
    if m < 192 * log(1/δ)
        return diagm(ones(m))
    end
    mask = bitrand(m)
    S_bar = zeros(m,m)
    
    X̄ = X[mask,:]
    Ȳ = Y[mask,:]
    S_tilde = recursive_rls_nyström(X̄, Ȳ, k, λ, δ/3)
    S_tilde = S_bar .* S_tilde

    K = [ k(X[i,:], Y[i,:]) for i in 1:m, j in 1:m ]

    l_λ = 3/(2λ) .* [ (K - K * Ŝ * inv(Ŝ'*K*Ŝ) * Ŝ' * K)[i,i] for i in 1:m ]

    f = 16 * log( sum(l_λ) / δ)
    p = [ min(1, l_λ[i] * f ) for i in 1:m ]

    S = []

end
=#