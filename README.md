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

# Estimate the model.
chain = blm(@formula(LWage ~ 1 + Married + Ed + Married*Ed), df, 1000)

# Display the results.
display(chain)
```