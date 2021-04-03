using BayesModels
using Turing
using Statistics
using RDatasets
using Test

@testset "blm" begin
    @testset "Air Quality dataset" begin
        data = dataset("robustbase", "airmay") |>
        dropmissing |>
        df -> mapcols(col -> float.(col), df) |>
        df -> mapcols(col -> col./std(col), df)
        chain = blm(@formula(Y ~ X1 + X2 + X3), data)
        summary,_ = describe(chain)
        params = summary[:,:parameters]
        # check parameter names
        @test Set(params) == Set([:α,Symbol("β[X1]"),Symbol("β[X2]"),Symbol("β[X3]"),:σ])
        rhat = summary[:,:rhat]
        # check convergence statistic
        @test maximum(abs.(1.0 .- rhat)) <= 0.01
    end
    @testset "Earthquakes dataset" begin
        data = dataset("datasets", "quakes") |>
        df -> mapcols(col -> float.(col), df) |>
        df -> mapcols(col -> col./std(col), df)
        chain = blm(@formula(Mag ~ Lat + Long + Depth + Stations), data, MCMC(NUTS(),500))
        summary,_ = describe(chain)
        params = summary[:,:parameters]
        # check parameter names
        @test Set(params) == Set([:α,Symbol("β[Lat]"),Symbol("β[Long]"),Symbol("β[Depth]"),Symbol("β[Stations]"),:σ])
        rhat = summary[:,:rhat]
        # check convergence statistic
        @test maximum(abs.(1.0 .- rhat)) <= 0.01
    end
end
