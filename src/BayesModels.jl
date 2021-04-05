module BayesModels

using Turing
using Statistics
using LinearAlgebra
using DataFrames
using Reexport
using Parameters
import MCMCChains
@reexport using StatsModels
@reexport using StatsFuns

function get_nlogp(model)
    # Set up the model call, sample from the prior.
    vi = Turing.VarInfo(model)

    # Define a function to optimize.
    function nlogp(sm)
        spl = Turing.SampleFromPrior()
        new_vi = Turing.VarInfo(vi, spl, sm)
        model(new_vi, spl)
        -new_vi.logp
    end

    return nlogp
end

"""
Base type for different inference methods (e.g. MCMC vs VI).
"""
abstract type InferenceStrategy end
"""
Markov-chain Monte-Carlo inference via the Turing.jl sampling interface.
"""
struct MCMC{TAlg,TArgs,TKwArgs} <: InferenceStrategy
    alg::TAlg
    sampleargs::TArgs
    samplekwargs::TKwArgs
    MCMC(alg::TAlg, args...;kwargs...) where {TAlg<:Turing.InferenceAlgorithm} =
        new{TAlg,typeof(args),typeof(kwargs)}(alg,args,kwargs)
end
# TODO add VI support
# struct VI <: InferenceStrategy end

# Default interface methods
"""
    inference(::DynamicPPL.Model, strat::InferenceStrategy)

Run probabilistic inference for the given model using the given InferenceStrategy (e.g. MCMC or variational inference).
The type of the result depends on the implementation. MCMC implementations should return the sampled chains, whereas VI
should return a fitted variational distribution.
"""
inference(::DynamicPPL.Model, strat::InferenceStrategy) = error("no inference implementation for $(typeof(strat))")
"""
    postprocess(::InferenceStrategy, inferenceresult, args...)

Applies postprocessing to the given inference result. Returns the (possibly modified) inference result.
"""
postprocess(::InferenceStrategy, inferenceresult, args...) = inferenceresult

# MCMC inference
inference(model::DynamicPPL.Model, mcmc::MCMC) = sample(model, mcmc.alg, mcmc.sampleargs...;mcmc.samplekwargs...)

export InferenceStrategy, MCMC

# Convenience constants
const Prior = Union{Function,<:Distribution}
asfunc(prior::Prior) = prior
asfunc(prior::Distribution) = (X,y) -> prior

include("blm.jl")

end # module
