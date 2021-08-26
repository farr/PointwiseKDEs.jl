using Distributions
using LinearAlgebra
using PointwiseKDEs
using Statistics
using Test

@testset "PointwiseKDE Tests" begin
    @testset "3D Gaussian" begin
        mu = randn(3)
        Sigma = randn(3,3)
        Sigma = Sigma' * Sigma

        CSigma = cholesky(Sigma)

        pts = mu .+ CSigma.L*randn(3, 1000)

        kde = PointwiseKDE(pts)
        draw_pts = rand(kde, 1000)

        mu_draw = mean(draw_pts, dims=2)
        sigma_draw = cov(draw_pts, dims=2)

        @testset "Mean Agrees" begin
            @test all(abs.(CSigma.L \ (mu .- mu_draw)) .< 0.1)
        end

        @testset "Average density is close" begin
            dist = MvNormal(mu, Sigma)
            logp_kde = [logpdf(kde, draw_pts[:,i]) for i in 1:size(draw_pts,2)]
            logp_mvn = [logpdf(dist, draw_pts[:,i]) for i in 1:size(draw_pts,2)]
            @test abs(mean(logp_kde .- logp_mvn)) < 0.1
        end
    end
end
