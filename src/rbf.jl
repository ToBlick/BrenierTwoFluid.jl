function bspline(r, i, p)
    if p == 0
        return i <= r < i+1 ? 1.0 : 0.0
    else
        return (r - i) * bspline(r, i, p - 1) / p + (i + p + 1 - r) * bspline(r, i+1, p-1) / p
    end
end

using Plots

function kde(y,X,α,h,p)
    ρ = 0.0
    for i in axes(X,1)
        ρ += α[i] * bspline(norm(y-X[i,:])/h, -p/2-0.5, p)
    end
    return ρ
end

function kde(y,X,α,h,p,s,species)
    ρ = 0.0
    for i in axes(X,1)
        if species[i] == s
            ρ += α[i] * bspline(norm(y-X[i,:])/h, -p/2-0.5, p)
        end
    end
    return ρ
end