"""
    SinkhornParameters

    Fields:
    - `Δ::T`: maximum value we expect the cost function to take
    - `q::T`: scaling parameter. If ε-scaling is employed, the scale s is reduced by a factor q at each iteration.
    - `η::T`: acceleration parameter. At every iteration, f ← (1-η)f₋ + ηf₊.
    - `crit_it::Int`: after `crit_it` iterations, η is inferred from the change in of the marginal error and updated.
    - `p_η::Int`: the change in marginal from `crit_it - p_η` to `crit_it` is used to infer η.
    - `s::T`: current scale. When no ε-scaling is employed, this is constant and equal to ε.
    - `ε::T`: minimum scale.
    - `p::T`: the proportionality of ε to Δ (e.g. p = 2 for the quadratic cost)
    - `tol::T`: tolerance for the stopping criterion, which is either the marginal error or the change in the dual potentials.
    - `max_it::Int`: maximum number of iterations.
    - `tol_it::Int`: the stopping criterion is checked every `tol_it` iterations.

    Type parameters:
    - `SAFE`: whether or not the stopping criterion is checked rather than just doing a certain amount of iterations. This is set to `false` only if `tol == Inf`.
    - `SYM`: whether or not the Sinkhorn algorithm is symmetrized, i.e. f ← 0.5f₋ + 0.5f₊.
    - `ACC`: whether or not the Sinkhorn algorithm is accelerated, i.e. f ← (1-η)f₋ + ηf₊.
    - `DEB`: whether or not the Sinkhorn algorithm is debiased, i.e. computing W₂(μ,ν) - (W₂(μ,μ) + W₂(ν,ν))/2.
"""
mutable struct SinkhornParameters{SAFE, SYM, ACC, DEB, T}
    Δ::T
    q::T

    η::T
    crit_it::Int
    p_η::Int

    s::T
    ε::T

    p::T
    tol::T
    max_it::Int
    tol_it::Int

    function SinkhornParameters(Δ::T, q::T, η::T, crit_it::Int, p_η::Int, s::T, ε::T, p::T, tol::T, max_it, tol_it, safe, sym, acc, deb) where {T}
        new{safe, sym, acc, deb, T}(Δ, q, η, crit_it, p_η, s, ε, p, tol, max_it, tol_it)
    end
end

function SinkhornParameters(;
        Δ = 1.0,
        q = 1.0,
        η = 1.0,
        crit_it = 20,
        p_η = 2,
        ε = 1e-2,
        s = ε,
        p = 2.0,
        tol = 1e-6,
        max_it = Int(ceil(10 * Δ/ε)),
        tol_it = 2,
        safe = true,
        sym = false,
        acc = true,
        deb = true)
    SinkhornParameters(Δ, q, η, crit_it, p_η, s, ε, p, tol, max_it, tol_it, safe, sym, acc, deb)
end

scale(p::SinkhornParameters) = p.s