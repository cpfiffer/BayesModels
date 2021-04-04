# BayesModels.jl

BayesModels.jl is a package that uses [Turing](https://github.com/TuringLang/Turing.jl) for 
Bayesian inference and [StatsModels](https://github.com/JuliaStats/StatsModels.jl) for its easy-to-use 
formula API.

Currently only Bayesian linear regression is supported, but I plan at some point to extend it to 
additional model types.

# How to

You can run a Bayesian linear regression using `blm(formula, dataframe)`.

```julia
# Import packages.
using BayesModels
using RDatasets
using DataFrames

# Load the wages dataframe. 
# I'm thinning out the dataframe to make it quicker,
# Bayesian methods don't scale as well with bigger data.
df = RDatasets.dataset("Ecdat", "Wages")[1:10:end, :]

# Estimate the model;
# Defaults to 1000 MCMC samples using NUTS
chain = blm(@formula(LWage ~ 1 + Married + Ed + Married*Ed), df)

# Display the results.
display(chain)
```

Custom priors for the intercept `α` and coefficients `β`, as well as any likelihood specific parameters, can be optionally passed as keyword arguments. Priors can be specified as either a `Distribution` or a function of the data `(X,y) -> Distribution`:

```julia
# use higher prior variance on intercept and coefficients
α_prior = Normal(0,2)
β_prior = (X,y) -> MvNormal(zeros(size(X,2)),2)
# use truncated Normal for observation variance
σ_prior = Truncated(Normal(0,5),0,Inf)
chain = blm(@formula(LWage ~ 1 + Married + Ed + Married*Ed), df; α_prior=α_prior, β_prior=β_prior, σ_prior=σ_prior)
```

You can customize inference by explicitly setting the `InferenceStrategy`. The `MCMC` strategy takes a
sampler algorithm as its first argument and passes all other arguments directly to `sample`:

```julia
df = RDatasets.dataset("Ecdat", "Wages")[1:10:end, :]
# Draw 500 samples from 4 chains in parallel using HMC instead of NUTS.
chain = blm(@formula(LWage ~ 1 + Married + Ed + Married*Ed), df, MCMC(HMC(0.05,10),MCMCThreads(),500,n_chains=4))
```

`blm` also supports `Bernoulli`, `Exponential`, and `Poisson` likelihoods for non-Normal response variables. By default, `blm` chooses from one of `Normal`, `Bernoulli`, or `Poisson` depending on whether the response variable is, respectively, real, binary-categorical, or integer typed:

```julia
df = dataset("Ecdat","HI")[sample(1:22265,500),:]
# will use Bernoulli likelihood with logistic link function by default (i.e. logistic regression)
# because HHI is a binary variable.
chain = blm(@formula(HHI ~ Whi + Education + Race + Region), df, MCMC(HMC(0.05,10),500))
```

You can also specify the likelihood type manually to override this behavior:

```julia
df = RDatasets.dataset("Ecdat","Airline")
# Note that this is for the sake of demonstration only... this model doesn't necessarily make sense.
chain = blm(Exponential, @formula(Output ~ 1 + Cost + LF*PF), df)
```

`blm` by default uses the canonical link functions for `Normal`, `Poisson`, and `Bernoulli` (i.e. identity, log, and logit) and a non-canonical absolute inverse `abs(η)^-1` for `Exponential`. You can provide your own choice of link function by simply passing it as an additional argument in the trailing `modelargs...`:

```julia
# Uses softplus as the link function for the Exponential model;
# Need to add a small epsilon to ensure that the rate parameter > 0
g(x) = 1e-8 + softplus(x)
chain = blm(Exponential, @formula(Output ~ 1 + Cost + LF*PF), df, MCMC(NUTS(),1000), g)
```