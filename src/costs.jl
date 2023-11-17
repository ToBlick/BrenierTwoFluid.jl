
struct LazyCost{T,XT,d,FT} <: AbstractMatrix{T}
    x::AbstractArray{XT,d} # N × d
    y::AbstractArray{XT,d}
    c::FT

    function LazyCost(x::Array{XT,d}, y::Array{XT,d}, c::FT = (x,y) -> 0.5 * sqeuclidean(x,y) ) where {XT, d, FT <: Base.Callable}
        t = c(x[begin], y[begin])
        new{typeof(t),XT,d,FT}(x,y,c)
    end
end

Base.getindex(C::LazyCost{T,1}, i, j) where {T} = C.c(C.x[i], C.y[j])::T
Base.getindex(C::LazyCost{T}, i, j) where {T} = @views C.c(C.x[i,:], C.y[j,:])::T
Base.size(C::LazyCost) = (size(C.x,1), size(C.y,1))

struct LazySlice{T,CT} <: AbstractVector{T}
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

struct CostCollection{CT}
    C_xy::AbstractMatrix{CT}
    C_yx::AbstractMatrix{CT}
    C_xx::AbstractMatrix{CT}
    C_yy::AbstractMatrix{CT}

    function CostCollection(X::AbstractArray{T,d}, Y::AbstractArray{T,d}, c::FT) where {T,d,FT <: Base.Callable}
        t = c(X[begin], Y[begin])
        new{typeof(t)}(LazyCost(X,Y,c),LazyCost(Y,X,c),LazyCost(X,X,c),LazyCost(Y,Y,c))
    end
end