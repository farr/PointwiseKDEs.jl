module PointwiseKDEs

export PointwiseKDE

using Distributions
using LinearAlgebra
using StatsFuns
using Random

struct PointwiseKDE{T <: Real} <: ContinuousMultivariateDistribution
    pts::Matrix{T}
    CF::Cholesky{T, Matrix{T}}
end

"""
    PointwiseKDE(points::Matrix{T <: Real})

Construct a Gaussian KDE from the given matrix of points.

`points` should have size `(ndim, npts)`.  Bandwidth selection is Scott's rule.

The result is a `MultivariateDistribution` object suitable for use with `rand`
or `logpdf` or ....
"""
function PointwiseKDE(pts::Matrix{T}) where T <: Real
    nd, np = size(pts)
    C = cov(pts, dims=2) ./ np^(2/(4+nd))
    CF = cholesky(C)

    PointwiseKDE(pts, CF)
end

function Distributions.length(d::PointwiseKDE)
    size(d.pts, 1)
end

function Distributions.sampler(d::PointwiseKDE)
    d
end

function Distributions.eltype(d::PointwiseKDE{T}) where T <: Number
    T
end

function Distributions._rand!(rng::AbstractRNG, d::PointwiseKDE{T}, x::AbstractArray) where T <: Real
    nd, np = size(d.pts)
    i = rand(rng, 1:np)

    pt = vec(d.pts[:,i])
    cpt = d.CF.L*randn(rng, nd)

    for i in eachindex(x)
        x[i] = pt[i] + cpt[i]
    end
    x
end

function Distributions._rand!(rng::AbstractRNG, d::PointwiseKDE{T}, x::AbstractMatrix) where T <: Real
    nd, np = size(x)

    for j in 1:np
        x[:,j] = Distributions._rand!(rng, d, x[:,j])
    end

    x
end

function Distributions._logpdf(d::PointwiseKDE{T}, x::AbstractArray) where T <: Real
    nd, np = size(d.pts)

    logp::T = zero(T) - Inf
    for i in 1:np
        r = x .- vec(d.pts[:,i])
        ru = d.CF.L \ r

        logp = logaddexp(logp, -0.5*dot(ru, ru))
    end

    logp - sum(0.5*log(2*pi) .+ log.(diag(d.CF.L))) - log(np)
end

end # module
