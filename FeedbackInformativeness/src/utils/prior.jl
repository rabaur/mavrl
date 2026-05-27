using Distributions
using Turing: filldist

"""
Shared priors.
"""
function shared_priors(num_r_params::Int, true_β::Union{Real,Nothing}=nothing)
    return (
        r = filldist(Normal(0.0, 1.0), num_r_params),
        β = isnothing(true_β) ? Uniform(0.0, 5.0) : Normal(true_β, 0.05)
    )
end