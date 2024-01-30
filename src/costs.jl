
struct LazyCost{T, d, AT <: AbstractArray{T,d}, FT <: Base.Callable} <: AbstractMatrix{T}
    x::AT # N × d
    y::AT
    c::FT

    function LazyCost(x::AT, y::AT, c::FT = (x,y) -> 0.5 * sqeuclidean(x,y) ) where {T, d, AT <: AbstractArray{T,d}, FT <: Base.Callable}
        #t = c(x[begin], y[begin])
        new{T,d,AT,FT}(x,y,c)
    end
end

Base.getindex(C::LazyCost{T,1}, i, j) where {T} = C.c(C.x[i], C.y[j])::T
Base.getindex(C::LazyCost{T}, i, j) where {T} = @views C.c(C.x[i,:], C.y[j,:])::T
Base.size(C::LazyCost) = (size(C.x,1), size(C.y,1))

#=struct LazySlice{T,CT} <: AbstractVector{T}
    j::Int
    C::AbstractMatrix{CT}
    f::AbstractVector{T}
    log_α::AbstractVector{T}
    ε::T

    function LazySlice( j::Int, C::AbstractMatrix{CT}, f::AbstractVector{T}, log_α::AbstractVector{T}, ε::Real) where {T, CT}
        new{T,CT}(j, C, f, log_α, ε)
    end
end
Base.getindex(S::LazySlice{T}, i) where {T} = S.f[i] / S.ε - S.C[i, S.j] / S.ε + S.log_α[i]
Base.size(S::LazySlice) = size(S.f)
=#

struct CostCollection{T, CT <: AbstractMatrix{T}}
    C_xy::CT
    C_yx::CT
    C_xx::CT
    C_yy::CT

    function CostCollection(X::AT, Y::AT, c::FT) where {T, AT <: AbstractArray{T}, FT <: Base.Callable}
        ct = typeof(LazyCost(X,Y,c))
        new{T,ct}(LazyCost(X,Y,c),LazyCost(Y,X,c),LazyCost(X,X,c),LazyCost(Y,Y,c))
    end
end

function c_periodic(x::VT,y::VT,D) where {T,VT <: AbstractVector{T}}
    d = 0
    for i in eachindex(x)
        if x[i] - y[i] > D[i]/2
            d += (x[i] - y[i] - D[i])^2
        elseif x[i] - y[i] < -D[i]/2
            d += (x[i] - y[i] + D[i])^2
        else
            d += (x[i] - y[i])^2
        end
    end
    0.5 * d
end

function ∇c_periodic(x,y,D)
    ∇c = zero(x)
    for i in eachindex(x)
        if x[i] - y[i] > D[i]/2
            ∇c[i] = x[i] - y[i] - D[i]
        elseif x[i] - y[i] < -D[i]/2
            ∇c[i] = (x[i] - y[i] + D[i])
        else
            ∇c[i] = x[i] - y[i]
        end
    end
    ∇c
end