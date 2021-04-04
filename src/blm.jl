"""
Renames chain variables to include covariate names.
"""
function postprocess(::MCMC, chain::MCMCChains.Chains, f::FormulaTerm, df::DataFrame)
    # Get formula names.
    nms = coefnames(f.rhs)
    nms_dict = Dict(["β[$i]" => "β[$(nms[i])]" for i in eachindex(nms)])
    # Overwrite the default names.
    chain = MCMCChains.replacenames(chain, nms_dict)
end

"""
Internal function that specifies default priors for non-Normal likelihood BLMs. Additional priors for custom model
definitions may be passed in `kwargs`.
"""
blmpriors(::Type{<:Distribution}; α = Normal(), β = (X,y) -> MvNormal(zeros(size(X,2)),1), kwargs::KW{<:Prior}...) = (α=α,β=β,kwargs...)
"""
Internal function that specifies default priors for Normal likelihood BLMs.
"""
blmpriors(
    ::Type{Normal};
    α = Normal(),
    β = (X,y) -> MvNormal(zeros(size(X,2)),1),
    σ = InverseGamma(2,3),
    kwargs::KW{<:Prior}...
) = (α=α,β=β,σ=σ,kwargs...)

"""
    bayesreg(f::FormulaTerm,df::DataFrame,inferencestrat::InferenceStrategy,modelfn::F,modelargs...;schemahints::Dict{Symbol}=Dict{Symbol,Any}())

Fits a generic two-stage Bayesian regression model specified by `f` and implemented by `modelfn` to the given data. `modelfn` must be of the form
`F(X,y,priors,modelargs...)` where `X` is the design matrix, `y` is the response vector, `priors` is a named tuple of prior distributions and
`modelargs` may be any number of additional arguments passed directly to the model. Return value depends on the specified inference strategy.
MCMC will return a Chains object, whereas VI will return a fitted variational distribution.
"""
function bayesreg(
    f::FormulaTerm,
    df::DataFrame,
    inferencestrat::InferenceStrategy,
    modelfn::F,
    modelargs...;
    schemahints::Dict{Symbol}=Dict{Symbol,Any}(),
    priors::KW{<:Prior}...
) where {F<:Function}
    # Set-up
    spec = apply_schema(f, schema(f, df, schemahints))
    y, X = modelcols(spec, df)
    # Priors
    priordists = map(p -> p(X,y), map(asfunc, priors)) # convert to functional form and initialize
    # Model
    model = modelfn(X, y, priordists, modelargs...)
    # Inference
    res = inference(model, inferencestrat)
    # Postprocessing
    res = postprocess(inferencestrat, res, spec, df)
end

"""
    blm([::Type{L}], f::FormulaTerm, df::DataFrame, inferencestrat::InferenceStrategy=MCMC(NUTS(),1000), modelargs...; schemahints, priors...)

Fits a two-stage Bayesian linear model with likelihood distribution type `L` specified by `f` to the given data. `L` defaults to `Normal` (Gaussian response)
if not specified. Supported likelihoods are `Normal`, `Bernoulli`, `Exponential`, and `Poisson`. Return value depends on the specified inference strategy.
MCMC will return a Chains object, whereas VI will return a fitted variational distribution. Hints for building the schema from `f` can be provided as a Dict
via `schemahints`. Additional model arguments, e.g. a custom link function for provided linear models, can be provided in `modelargs`.
"""
function blm(
    ::Type{L},
    f::FormulaTerm,
    df::DataFrame, 
    inferencestrat::InferenceStrategy=MCMC(NUTS(), 1000),
    modelargs...;
    schemahints::Dict{Symbol}=Dict{Symbol,Any}(),
    priors::KW{<:Prior}...
) where {L<:Distribution}
    # model function; specifies distribution type for lm
    modelfn(args...) = lm(L, args...)
    # run bayes regression on the model
    bayesreg(f, df, inferencestrat, modelfn, modelargs...; schemahints=schemahints, blmpriors(priors...)...)
end
# Default to Normal likelihood
blm(f::FormulaTerm, args...;kwargs...) = blm(Normal,f,args...;kwargs...)

# default link (inverse) functions
defaultlink(::Type{Normal}) = η -> η
defaultlink(::Type{Bernoulli}) = η -> logistic(η)
defaultlink(::Type{Exponential}) = η -> 1/abs(η)
defaultlink(::Type{Poisson}) = η -> exp(η)

"""
Linear model with Normal response.
"""
@model function lm(::Type{Normal}, x, y, priors, g=defaultlink(Normal))
    N,K = size(x)
    @unpack α_prior, β_prior, σ_prior = priors
    σ ~ σ_prior
    α ~ α_prior
    β ~ β_prior
    # beta is K x 1
    # y is N x 1
    # x is N x K
    η = α .+ x*β
    y ~ MvNormal(g(η), σ)
end

"""
Linear model with non-Normal (Bernoulli, Exponential, or Poisson) response.
"""
@model function lm(::Type{L}, x, y, priors, g=defaultlink(L)) where {L<:Union{Bernoulli,Exponential,Poisson}}
    N,K = size(x)
    @unpack α_prior, β_prior = priors
    α ~ α_prior
    β ~ β_prior
    η = α .+ x*β
    for i in 1:N
        let μ = g(η)
            y[i] ~ L(μ)
        end
    end
end

export bayesreg, blm
