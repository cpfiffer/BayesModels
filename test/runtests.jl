using BayesModels
using Turing
using Statistics
using DataFrames
using RDatasets
using Test

@testset "blm" begin
    @testset "Air Quality dataset" begin
        data = dataset("robustbase", "airmay") |>
        dropmissing |>
        df -> mapcols(col -> float.(col), df) |>
        df -> mapcols(col -> (col .- mean(col))./std(col), df)
        chain = blm(@formula(Y ~ X1 + X2 + X3), data)
        summary,_ = describe(chain)
        params = summary[:,:parameters]
        # check parameter names
        @test Set(params) == Set([:α,Symbol("β[X1]"),Symbol("β[X2]"),Symbol("β[X3]"),:σ])
        rhat = summary[:,:rhat]
        # check convergence statistic
        @test maximum(abs.(1.0 .- rhat)) <= 0.05
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
        @test maximum(abs.(1.0 .- rhat)) <= 0.05
    end
    @testset "Wages dataset" begin
        # Load the wages dataframe. 
        # I'm thinning out the dataframe to make it quicker,
        # Bayesian methods don't scale as well with bigger data.
        df = RDatasets.dataset("Ecdat", "Wages")[1:10:end, :]
        # Estimate the model;
        # Defaults to 1000 MCMC samples using NUTS
        chain = blm(@formula(LWage ~ 1 + Married + Ed + Married*Ed), df)
        @test all(isfinite.(mean(chain)[:,:mean]))
        summary,_ = describe(chain)
        rhat = summary[:,:rhat]
        @test maximum(abs.(1.0 .- rhat)) <= 0.05
        α_prior = Normal(0,2)
        β_prior = (X,y) -> MvNormal(zeros(size(X,2)),2)
        # use truncated Normal for observation variance
        σ_prior = truncated(Normal(0,5),0,Inf)
        chain = blm(@formula(LWage ~ 1 + Married + Ed + Married*Ed), df; α_prior=α_prior, β_prior=β_prior, σ_prior=σ_prior)
        # TODO: maybe use importance sampling to verify that priors were actually used?
        @test all(isfinite.(mean(chain)[:,:mean]))
        summary,_ = describe(chain)
        rhat = summary[:,:rhat]
        @test maximum(abs.(1.0 .- rhat)) <= 0.05
    end
    @testset "HI dataset (logistic regression)" begin
        df = dataset("Ecdat","HI")[sample(1:22265,500),:] # subsample
        chain = blm(@formula(HHI ~ Whi + Education + Race + Region), df, MCMC(HMC(0.05,10),1000))
        @test all(isfinite.(mean(chain)[:,:mean]))
        summary,_ = describe(chain)
        rhat = summary[:,:rhat]
        @test maximum(abs.(1.0 .- rhat)) <= 0.05
    end
    @testset "Affairs dataset (poisson regression)" begin
        df = dataset("COUNT","affairs")[sample(1:601,300),:] # subsample
        chain = blm(@formula(NAffairs ~ Kids + VryUnhap + Unhap + VryHap + HapAvg + AvgMarr), df, MCMC(HMC(0.05,10),1000))
        @test all(isfinite.(mean(chain)[:,:mean]))
        summary,_ = describe(chain)
        rhat = summary[:,:rhat]
        @test maximum(abs.(1.0 .- rhat)) <= 0.05
    end
    @testset "Airplane dataset (exponential regression)" begin
        df = RDatasets.dataset("Ecdat","Airline")
        chain = blm(Exponential, @formula(Output ~ 1 + Cost + LF*PF), df)
        @test all(isfinite.(mean(chain)[:,:mean]))
        # test with custom link function
        chain = blm(Exponential, @formula(Output ~ 1 + Cost + LF*PF), df, MCMC(NUTS(),1000), x -> 1e-8 + softplus(x))
        @test all(isfinite.(mean(chain)[:,:mean]))
        # this is a stupid model, so we won't bother with convergence tests
    end
    @testset "Custom 3-stage model" begin
        # Simple 3-stage hierarchical regression model
        @model function hlm(X,y,priors)
            N,K = size(X)
            σ_β ~ filldist(InverseGamma(3,2),K)
            σ_α ~ InverseGamma(3,2)
            σ ~ InverseGamma(2,3)
            α ~ Normal(0,σ_α)
            β ~ MvNormal(zeros(K),σ_β) # diagonal pooled variance
            η = α .+ X*β
            y ~ MvNormal(η, σ)
        end
        df = RDatasets.dataset("Ecdat", "Wages")[1:10:end, :]
        chain = bayesreg(@formula(LWage ~ 1 + Married + Ed + Married*Ed), df, MCMC(NUTS(0.65),500), hlm)
        @test all(isfinite.(mean(chain)[:,:mean]))
        summary,_ = describe(chain)
        rhat = summary[:,:rhat]
        @test maximum(abs.(1.0 .- rhat)) <= 0.05
    end
end
