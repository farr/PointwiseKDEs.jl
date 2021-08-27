# `PointwiseKDEs`

One thing lacking currently in Julia is a generic Gaussian KDE in multiple dimensions, similar to `scipy`'s `gaussian_kde`.  This package provides exactly that: a constructor that takes a matrix of points in arbitrary dimension and constructs the Gaussian KDE density as a `MultivariateDistribution` from [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl).  Bandwidth is chosen following Scott's rule.

Unlike [KernelDensity.jl](https://github.com/JuliaStats/KernelDensity.jl), the cost to evaluate the density from a `PointwiseKDE` scales linearly with the number of input points; the cost to draw a point from the KDE is constant.

## Construct a KDE

```juliarepl
> using PointwiseKDE
> ndim = 3
> npts = 1000
> pts = randn(ndim, npts)
> kde = PointwiseKDE(pts)
```

## Draw From the KDE

```juliarepl
> draw_pt = rand(kde) # draw one sample
> draw_pts = rand(kde, 100) # A (ndim, 100) matrix of draws
```

## Compute KDE densities:

```juliarepl
> rand_pt = randn(3)
> loglike = logpdf(kde, rand_pt)
```
