module BayesModels

using Turing
using Statistics
using Reexport
using LinearAlgebra
using DataFrames
using Optim
import DynamicPPL
@reexport using StatsModels

export blm

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


function blm(f::FormulaTerm, df::DataFrame, N::Int=1000, alpha_prior=nothing, beta_prior=nothing, sigma_prior=nothing)
    f = apply_schema(f, schema(f, df))

    # Define the default value for x when missing
    y, X = modelcols(f, df)
    K = size(X, 2)
    defaults = (y = Vector{Real}(undef, 2),priors=zeros(K))
    data = (y = y,
            x = X)

	if isnothing(alpha_prior)
		alpha_prior = Normal()
	end

	if isnothing(beta_prior)
		beta_prior = MvNormal(K, 1)
	end

	if isnothing(sigma_prior)
		sigma_prior = InverseGamma(2,3)
	end
	
	@model function bayesreg(
		x, 
		y, 
		alpha_prior, 
		beta_prior, 
		sigma_prior,
		::Type{TV}=Vector{Float64}
	) where {TV}
		N, K = size(x)

		sigma ~ sigma_prior
		intercept ~ alpha_prior
		beta = TV(undef, K)
		beta ~ beta_prior

		# beta is K x 1
		# y is N x 1
		# x is N x K
		yhat = intercept .+ x * beta

		y ~ MvNormal(yhat, sigma)	
	end

	# Model
	model = bayesreg(X, y, alpha_prior, beta_prior, sigma_prior)

    # Sample
    chain = sample(model, NUTS(), N)
    
    # Get formula names.
    nms = coefnames(f.rhs)
    nms_dict = Dict(["beta[$i]" => nms[i] for i in eachindex(nms)])

    # Overwrite the default names.
    chain = set_names(chain, nms_dict, sorted=false)

    return chain
end

end # module
