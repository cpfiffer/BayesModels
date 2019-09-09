module BayesModels

using Turing
using Statistics
using Reexport
using LinearAlgebra
using DataFrames
using Optim
@reexport using StatsModels

export blm

function formula_names(f::FormulaTerm)
    names = String[]
    terms = f.rhs isa Tuple ?
        [z for z in f.rhs] :
        [f.rhs]

    for term in terms
        if term isa StatsModels.ConstantTerm
            push!(names, "Constant")
        else
            push!(names, string(term))
        end
    end

    return names
end

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

function BayesLinear(vi, sampler, model)
    # Set the accumulated logp to zero.
    vi.logp = 0

    # Retrieve the response variables.
    if isdefined(model.data, :y)
        y = model.data.y
    else # x is a parameter
        y = model.defaults.y
    end

    # Retrieve the response variables.
    if isdefined(model.data, :x)
        x = model.data.x
    else # x is a parameter
        x = model.defaults.x
    end

    # Retrieve the MAP priors.
    if isdefined(model.data, :priors)
        priors = model.data.priors
        varprior = 1
    else
        priors = model.defaults.priors
        varprior = 100
    end

    # Draw parameters from a MvNormal.
    β, lp = Turing.assume(
        sampler,
        MvNormal(priors, varprior),
        Turing.VarName(:c_β, :β, ""),
        vi
    )

    vi.logp += lp
    
    # Calcuate predicted values.
    μs = x*β

    # Observe the expected values.
    vi.logp = Turing.observe(
        sampler,
        MvNormal(μs, 1),
        y, 
        vi
    ) |> sum
end

function blm(f::FormulaTerm, df::DataFrame, N::Int=1000)
    apply_schema(f, schema(f, df))

    # Define the default value for x when missing
    X = modelmatrix(f.rhs, df)
    K = size(X, 2)
    defaults = (y = Vector{Real}(undef, 2),priors=zeros(K))
    data = (y = df[:, f.lhs.sym],
            x = X)

    # Instantiate a Model object.
    model1 = Turing.Model{Tuple{:α, :β}, Tuple{:X}}(BayesLinear, data, defaults)

    # Create a starting point, call the optimizer.
    sm_0 = repeat([1.0], K)
    lb = repeat([-Inf], K)
    ub = repeat([Inf], K)
    nlogp = get_nlogp(model1)
    result = optimize(nlogp, lb, ub, sm_0, Fminbox())

    newdata = merge(data, (priors=result.minimizer,))

    model = Turing.Model{Tuple{:α, :β}, Tuple{:x}}(BayesLinear, newdata, defaults)

    # Sample
    chain = sample(model, NUTS(), N)
    
    # Get formula names.
    nms = formula_names(f)
    nms_dict = Dict(["β[$i]" => nms[i] for i in eachindex(nms)])

    # Overwrite the default names.
    chain = set_names(chain, nms_dict, sorted=false)

    return chain
end

end # module
