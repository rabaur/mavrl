using ..MDPs: MDP
using ForwardDiff

"""
Computes state-state transition probabilities P_π ∈ ℝ^{nS × nS} induced by policy π.
"""
function π_marginalized_transitions(P::Array{Float64, 3}, π::AbstractMatrix)
    nS, nA, _ = size(P)
    T = eltype(π)  # Get the element type of π (Float64 or Dual)
    P_π = zeros(T, nS, nS)
    @inbounds for s in 1:nS, s′ in 1:nS
        P_π[s, s′] = sum(π[s, a] * P[s, a, s′] for a in 1:nA)
    end
    return P_π
end

"""
Computes the stationary distribution of a MDP given a policy.
"""
function stationary_dist_exact(mdp::MDP, π::AbstractMatrix)

    # build π-marginalized transition matrix
    Pπ = π_marginalized_transitions(mdp.P, π)

    return stationary_dist_exact(Pπ)
end

function stationary_dist_exact(Pπ::AbstractMatrix)

   # the left eigenvector associated with the real eigenvalue 1 is the (unnormalized) stationary distribution
   _, evecs = eigen(Pπ', sortby=λ -> -abs(λ))

   # unnormalized stationary distribution
   μ = real.(evecs[:, 1])

   return μ ./ sum(μ)
end

function stationary_dist_solve(Pπ::AbstractMatrix)

    nS = size(Pπ, 1)
    M = I - Pπ' + fill(one(eltype(Pπ)), nS, nS)
    b = fill(one(eltype(Pπ)), nS)
    μ = M \ b
    return μ ./ sum(μ)
end


function stationary_dist_solve(mdp::MDP, π::AbstractMatrix)

    # build π-marginalized transition matrix
    Pπ = π_marginalized_transitions(mdp.P, π)

    # solve for the stationary distribution
    return stationary_dist_solve(Pπ)
end

