using Distributions

"""
Q-value iteration.
Q(s, a) = R(s, a) + γ Σ_{s'} P(s,a,s') max_{a'} Q(s',a')
"""
function Q_opt_iteration(
    mdp::MDP,
    Q::Matrix,
    R::Matrix
)
    Q_new = zeros(eltype(Q), size(Q))
    Q_max = maximum(Q, dims=2)
    @inbounds for s in 1:mdp.nS
        for a in 1:mdp.nA
                update = mdp.γ * sum(mdp.P[s, a, s′] * Q_max[s′] for s′ in 1:mdp.nS)
                Q_new[s, a] = R[s, a] + update[1]
        end
    end
    return Q_new
end

"""
Replaces the max with a softmax with temperature α for differentiability.
"""
function soft_Q_opt_iteration(
    mdp::MDP,
    Q::Matrix,
    R::Matrix,
    α::Float64
)
    Q_new = zeros(eltype(Q), size(Q))
    V_α = α * log.(sum(exp.(Q ./ α), dims=2))
    @inbounds for s in 1:mdp.nS, a in 1:mdp.nA
            update = mdp.γ * sum(mdp.P[s, a, s_] .* V_α[s_, :] for s_ in 1:mdp.nS)
            Q_new[s, a] = R[s, a] + update[1]
    end
    return Q_new
end

function Q_opt(mdp::MDP, R::Matrix; max_iter::Int=1000)
    Q = zeros(eltype(R), size(R))
    for _ in 1:max_iter
        Q_new = Q_opt_iteration(mdp, Q, R)
        Δ = maximum(abs.(Q_new - Q))
        Q = Q_new
        if Δ < 1e-6
            break
        end
    end
    return Q
end

"""
Finite horizon Q-value iteration.

Returns a matrix of size (nS, nA, T) where the last dimension is the time step.
"""
function Q_opt_finite_horizon(mdp::MDP, R::Matrix, T::Int)
    Q = zeros(eltype(R), size(R, 1), size(R, 2), T)
    @inbounds for t in 1:T
        if t == 1
            Q[:, :, t] = R
        else
            Q[:, :, t] = Q_opt_iteration(mdp, Q[:, :, t-1], R)
        end
    end
    return Q
end

"""
Q-value iteration with softmax temperature.

Disclaimer: This has proven to be slower and less accurate than the true
Q-value iteration above. We recommend using `Q_opt` instead.
"""
function soft_Q_opt(mdp::MDP, R::Matrix; max_iter::Int=1000, α::Float64=0.1)
    Q = zeros(eltype(R), size(R))
    for i in 1:max_iter
        Q_new = soft_Q_opt_iteration(mdp, Q, R, α)
        Δ = maximum(abs.(Q_new - Q))
        Q = Q_new
        if Δ < 1e-6
            break
        end
    end
    return Q
end

"""
Computes P̄ matrix where P̄(s,a,s',a') = P(s'|s,a) ⋅ π(a'|s')
"""
function compute_P̄(mdp::MDP, Q)
    nS, nA = mdp.nS, mdp.nA
    P̄ = zeros(eltype(Q), (nS, nA, nS, nA))  # Use similar to preserve element type
    
    # Compute the (greedy) optimal policy
    π = Utils.softmax(Q, dims=2)
    
    @inbounds for s in 1:nS, a in 1:nA, s′ in 1:nS, a′ in 1:nA
        P̄[s, a, s′, a′] = mdp.P[s, a, s′] * π[s′, a′]
    end

    P̄ = reshape(P̄, nS * nA, nS * nA)
    
    return P̄
end

"""
Converts Q-values to rewards using the Bellman equation
R = Q - γPπQ
"""
function Q_to_R(Q, mdp::MDP, β::Real=10.0)
    P̄ = compute_P̄(mdp, Q)
    Q_vec = vec(Q)
    R_vec = (I - mdp.γ * P̄)* Q_vec
    return reshape(R_vec, size(Q))
end

function Q_to_R(Q, mdp::MDP, P̄::AbstractMatrix)
    Q_vec = vec(Q)
    R_vec = (I - mdp.γ * P̄)* Q_vec
    return reshape(R_vec, size(Q))
end

"""
Computes the Q-value prior density given a reward prior density.
pQ(Q) = pR((I - γP̄)Q) * det(I - γP̄)
where P̄ is the state-action transition matrix incorporating the policy.
"""
function Q_prior_density(
    Q,
    mdp::MDP,
    reward_prior::Distribution
)
    # Compute P̄ matrix
    P̄ = compute_P̄(mdp, Q)
    
    # Compute the determinant term
    logdet_term = logdet(I - mdp.γ * P̄)
    
    # Convert Q to implied rewards
    R = Q_to_R(Q, mdp, P̄)
    
    # Compute the reward prior density at the implied rewards
    reward_density = logpdf(reward_prior, vec(R))
    
    # Return the Q prior density
    return reward_density + logdet_term
end