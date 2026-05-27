using ..MDPs: MDP
using ..Utils: Q_opt, softmax
using ForwardDiff
using Optim

"""
Computes the Distribution Mismatch Coefficient (DMC) of an MDP. It is defined as
DMC = sup_π ∑_{s ∈ S} μ*(s) / μπ(s)
where μ* and μπ are the stationary distribution of the optimal policy and π respectively.
As the search over π is generally intractable, we parameterize π as a softmax policy
and use a gradient-based optimization to find the policy that maximizes the DMC.
"""
function dmc(mdp::MDP, R::Matrix)

    # Compute the stationary distribution of the optimal policy
    Q_star = Q_opt(mdp, R; max_iter=10_000)
    π_star = softmax(Q_star)
    μ_star = stationary_dist_solve(mdp, π_star)

    # Objective
    function J(θ)
        π = softmax(θ)
        μ = stationary_dist_solve(mdp, π)

        # Negative because we want to maximize the DMC
        -sum(μ_star ./ μ)
    end

    nS, nA = mdp.nS, mdp.nA

    θ0 = zeros(nS, nA)
    result = optimize(J, θ0, LBFGS(); autodiff=:forward)
    θ_min = result.minimizer
    π_min = softmax(θ_min)

    return π_min, -J(θ_min) / mdp.nS
end