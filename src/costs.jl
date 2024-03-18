"""
    LazyCost

    AbstractMatrix type that represents a cost matrix between two point clouds `x` and `y`. The cost matrix is not computed until it is indexed.

    Fields:
    - `x::AT`: N × d `AbstractArray` of points.
    - `y::AT`: M × d `AbstractArray` of points.
    - `c::FT`: `Callable` scalar cost function `x,y -> c(x,y)`. The default is 1/2 times the squared Euclidean distance.
"""
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

"""
    LazyGibbsKernel

    AbstractMatrix type that represents a Gibbs kernel between two point clouds `x` and `y`. The Gibbs kernel is not computed until it is indexed.

    Fields:
    - `C::LazyCost`: N × M `LazyCost` matrix.
    - `ε::Base.RefValue`: reference of the Gibbs kernel scale.
"""
struct LazyGibbsKernel{T, CT <: AbstractMatrix{T}} <: AbstractMatrix{T}
    C::CT
    ε::Base.RefValue{T} # mutable
end

function LazyGibbsKernel(C::LazyCost{T}, ε::T) where {T}
    LazyGibbsKernel(C, Ref(ε))
end

scale(K::LazyGibbsKernel{T}) where T = K.ε[]::T
function set_scale!(K::LazyGibbsKernel{T}, ε::T) where T
    K.ε[] = ε
end
Base.getindex(K::LazyGibbsKernel{T}, i, j) where {T} = exp(-K.C[i,j]/scale(K))::T
Base.size(K::LazyGibbsKernel) = size(K.C)

"""
    CostCollection

    Collection of cost matrices involving two point clouds `X` and `Y`. 

    Fields:
    - `C_xy::CT`: N × M `AbstractMatrix`.
    - `C_yx::CT`: M × N `AbstractMatrix`.
    - `C_xx::CT`: N × N `AbstractMatrix`.
    - `C_yy::CT`: M × M `AbstractMatrix`.
"""
struct CostCollection{T, CT <: AbstractMatrix{T}, KT <: AbstractMatrix{T}}
    C_xy::CT
    C_yx::CT
    C_xx::CT
    C_yy::CT    # typically lazy costs

    K_xy::KT
    K_yx::KT
    K_xx::KT
    K_yy::KT    # typically lazy Gibbs kernels

    function CostCollection(X::AT, Y::AT, c::FT, ε) where {T, AT <: AbstractArray{T}, FT <: Base.Callable}
        C_xy = LazyCost(X,Y,c)
        C_yx = LazyCost(Y,X,c)
        C_xx = LazyCost(X,X,c)
        C_yy = LazyCost(Y,Y,c)

        K_xy = LazyGibbsKernel(C_xy, ε)
        K_yx = LazyGibbsKernel(C_yx, ε)
        K_xx = LazyGibbsKernel(C_xx, ε)
        K_yy = LazyGibbsKernel(C_yy, ε)

        new{T, typeof(C_xy), typeof(K_xy)}( C_xy, C_yx, C_xx, C_yy, K_xy, K_yx, K_xx, K_yy )
    end
end

scale(C::CostCollection) = scale(C.K_xy)
function set_scale!(C::CostCollection, ε)
    set_scale!(C.K_xy, ε)
    set_scale!(C.K_yx, ε)
    set_scale!(C.K_xx, ε)
    set_scale!(C.K_yy, ε)
end

"""
    c_periodic

    Periodic cost function between two points `x` and `y` with periodicity `D`.

    # Arguments
    - `x::VT`: N × d `AbstractVector` of points.
    - `y::VT`: M × d `AbstractVector` of points.
    - `D::VT`: d `AbstractVector` of periodicity.

    # Returns
    - `d::T`: Scalar cost.
"""

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