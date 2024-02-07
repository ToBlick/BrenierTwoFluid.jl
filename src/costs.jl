
struct LazyCost{T, d, AT <: AbstractArray{T,d}, FT <: Base.Callable} <: AbstractMatrix{T}
    x::AT # N × d
    y::AT
    c::FT

    function LazyCost(x::AT, y::AT, c::FT = (x,y) -> 0.5 * sqeuclidean(x,y) ) where {T, d, AT <: AbstractArray{T,d}, FT <: Base.Callable}
        new{T,d,AT,FT}(x,y,c)
    end
end

Base.getindex(C::LazyCost{T,1}, i, j) where {T} = C.c(C.x[i], C.y[j])::T
Base.getindex(C::LazyCost{T}, i, j) where {T} = @views C.c(C.x[i,:], C.y[j,:])::T
Base.size(C::LazyCost) = (size(C.x,1), size(C.y,1))

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

        #reflecting boundary
        #=
        for i in axes(X,1)
            for j in 1:2
                if X[i,j] > 0.5
                    X[i,j] = 1 - X[i,j]
                    V[i,j] *= -1
                elseif X[i,j] < -0.5
                    X[i,j] = - 1 - X[i,j]
                    V[i,j] *= -1
                end
            end
        end
        =#