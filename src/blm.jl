abstract type InferenceStrategy end
struct MCMC{TAlg,TArgs,TKwArgs} <: InferenceStrategy
    nsamples::Int
    alg::TAlg
    algargs::TArgs
    algkwargs::TKwArgs
    MCMC(alg::TAlg, nsamples::Int, args...;kwargs...) where {TAlg<:Turing.InferenceAlgorithm} =
        new{TAlg,typeof(args),typeof(kwargs)}(nsamples,alg,args,kwargs)
end
# TODO add VI support
# struct VI <: InferenceStrategy end

export InferenceStrategy, MCMC

# Default interface methods
"""
    inference(::DynamicPPL.Model, strat::InferenceStrategy)

Run probabilistic inference for the given model using the given InferenceStrategy (e.g. MCMC or variational inference).
The type of the result depends on the implementation. MCMC implementations should return a ChainDataFrame, whereas VI
should return a fitted variational distribution.
"""
inference(::DynamicPPL.Model, strat::InferenceStrategy) = error("no inference implementation for $(typeof(strat))")
"""
    postprocess(::InferenceStrategy, inferenceresult, f::FormulaTerm, df::DataFrame)

Applies postprocessing to the given inference result. Returns the (possibly modified) inference result.
"""
postprocess(::InferenceStrategy, inferenceresult, f::FormulaTerm, df::DataFrame) = inferenceresult

# MCMC inference
inference(model::DynamicPPL.Model, mcmc::MCMC) = sample(model, mcmc.alg, mcmc.nsamples, mcmc.algargs...;mcmc.algkwargs...)

function postprocess(::MCMC, chain::MCMCChains.Chains, f::FormulaTerm, df::DataFrame)
    # Get formula names.
    nms = coefnames(f.rhs)
    nms_dict = Dict(["β[$i]" => "β[$(nms[i])]" for i in eachindex(nms)])
    # Overwrite the default names.
    chain = MCMCChains.replacenames(chain, nms_dict)
end

"""
    blm(f::FormulaTerm, df::DataFrame, inferencestrat::InferenceStrategy=MCMC(NUTS(),1000); alphaprior, betaprior, sigmaprior)

Fits a Bayesian linear model specified by `f` to the given data. Return value depends on the specified inference strategy.
MCMC will return a ChainDataFrame, whereas VI will return a fitted variational distribution.
"""
function blm(
    f::FormulaTerm,
    df::DataFrame, 
    inferencestrat::InferenceStrategy=MCMC(NUTS(), 1000);
    alphaprior=nothing, 
    betaprior=nothing, 
    sigmaprior=nothing,
    schemahints::Dict{Symbol}=Dict{Symbol,Any}(),
)
    F = apply_schema(f, schema(f, df, schemahints))
    y, X = modelcols(F, df)
    # set priors
	alphaprior = isnothing(alphaprior) ? Normal() : alphaprior
	betaprior = isnothing(betaprior) ? MvNormal(zeros(size(X,2)), 1) : betaprior
	sigmaprior = isnothing(sigmaprior) ? InverseGamma(2,3) : sigmaprior
	@model function bayesreg(
		x, 
		y, 
		alphaprior, 
		betaprior, 
		sigmaprior,
	)
        N,K = size(x)
		σ ~ sigmaprior
		α ~ alphaprior
		β ~ betaprior
		# beta is K x 1
		# y is N x 1
		# x is N x K
		ŷ = α .+ x*β
		y ~ MvNormal(ŷ, σ)
	end
	# Model
	model = bayesreg(X, y, alphaprior, betaprior, sigmaprior)
    # Sample
    res = inference(model, inferencestrat)
    res = postprocess(inferencestrat,res,F,df)
end

export blm
