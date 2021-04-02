module BayesModels

using Turing
using Statistics
using LinearAlgebra
using DataFrames
using Reexport
import MCMCChains
@reexport using StatsModels

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

include("blm.jl")

end # module
