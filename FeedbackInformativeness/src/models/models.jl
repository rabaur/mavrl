using Turing
using LinearAlgebra
using StatsBase: mean

"""
Feeback model for the episodic task, demonstration feedback type.
"""
@model function feedback_model(
    ::NaiveDemonstrationFeedback,
    ::EpisodicTask,
    choice_sets::Nothing,
    choices::Vector{Trajectory},
    mdp::MDP;
    num_steps::Union{Int,Nothing}=nothing, # Unused for this model
    true_β::Union{Real,Nothing}=nothing
)
    
    # Sample from shared priors model
    priors = Utils.shared_priors(mdp.num_r_params, true_β)

    r ~ priors.r
    β ~ priors.β
    
    # Initialize the reward matrix according to the param_idx_mapping
    R = similar(r, mdp.nS, mdp.nA)
    fill!(R, zero(eltype(r)))
    for i in 1:mdp.num_r_params
        for (s, a) in mdp.param_idx_mapping[i]
            R[s, a] = r[i]
        end
    end
    
    Q = Utils.Q_opt(mdp, R)
    
    # 1) precompute logs
    π = Utils.softmax(β .* Q; dims=2) # (nS,nA)
    logπ = log.(π) # (nS,nA)
    logP = log.(mdp.P) # (nS,nA,nS)

    # 2) accumulate once
    ll = 0.0

    for traj in choices, t in 1:length(traj)-1
        s  = traj[t  ][1]
        a  = traj[t  ][2]
        s′ = traj[t+1][1]

        ll += logπ[s, a]      # action log-prob
        ll += logP[s, a, s′]  # transition log-prob
    end

    # 3) tack the whole thing onto the log‐density
    Turing.@addlogprob! ll
end

"""
Feedback model for finite horizon task, demonstration feedback type.
"""
@model function feedback_model(
    ::NaiveDemonstrationFeedback,
    ::FiniteHorizonTask,
    choice_sets::Nothing,
    choices::Vector{Trajectory},
    mdp::MDP;
    num_steps::Union{Int,Nothing}=nothing,
    true_β::Union{Real,Nothing}=nothing
)
    # Sample from shared priors model
    priors = Utils.shared_priors(mdp.num_r_params, true_β)
    r ~ priors.r
    β ~ priors.β

    # Initialize the reward matrix according to the param_idx_mapping
    R = similar(r, mdp.nS, mdp.nA)  # Use similar to preserve element type
    fill!(R, zero(eltype(r)))  # Initialize with zeros of the correct type
    for i in 1:mdp.num_r_params
        for (s, a) in mdp.param_idx_mapping[i]
            R[s, a] = r[i]
        end
    end

    # Get Q-values from rewards
    Q = Utils.Q_opt_finite_horizon(mdp, R, num_steps)
    
    # Compute policy using softmax
    π = Utils.softmax(β .* Q; dims=2)
    logπ = log.(π)
    logP = log.(mdp.P)
    
    # For each demonstration
    num_trajectories = size(choices, 1)
    num_steps = size(choices[1], 1)

    ll = 0.0
    
    for i in 1:num_trajectories, t in 1:num_steps - 1
        s  = choices[i][t  ][1]
        a  = choices[i][t  ][2]
        s′ = choices[i][t+1][1]

        ll += logπ[s, a, num_steps - t + 1]
        ll += logP[s, a, s′]
    end

    Turing.@addlogprob! ll
end

"""
Feedback model for the preference feedback type.
"""
@model function feedback_model(
    ::PreferenceFeedback,
    ::TaskType, # EpisodicTask or FiniteHorizonTask does not matter for this model
    choice_sets::Vector{Tuple{Trajectory, Trajectory}},
    choices::Vector{Int},
    mdp::MDP;
    num_steps::Union{Int,Nothing}=nothing, # Unused for this model
    true_β::Union{Real,Nothing}=nothing
)
        
    # Sample from shared priors model
    priors = Utils.shared_priors(mdp.num_r_params, true_β)
    r ~ priors.r
    β ~ priors.β

    # Initialize the reward matrix according to the param_idx_mapping
    R = similar(r, mdp.nS, mdp.nA)  # Use similar to preserve element type
    fill!(R, zero(eltype(r)))  # Initialize with zeros of the correct type
    for i in 1:mdp.num_r_params
        for (s, a) in mdp.param_idx_mapping[i]
            R[s, a] = r[i]
        end
    end

    # For each preference pair
    for (i, (traj1, traj2)) in enumerate(choice_sets)
        # Compute returns for both trajectories
        r1 = sum(R[s,a] for (s,a) in traj1)
        r2 = sum(R[s,a] for (s,a) in traj2)
        
        # Use stable_sigmoid function to get probability
        p = Utils.stable_sigmoid((r2 - r1) * β)
        
        # Check if p is NaN or Inf
        if isinf(p) || isnan(p)
            Turing.@addlogprob! -Inf
            return
        end

        p = clamp(p, 1e-6, 1 - 1e-6)
        
        # Observe actual choice
        choices[i] ~ Bernoulli(p)
    end
end

"""
Feedback model for the QValueWalkDemonstrationFeedback type.
Instead of performing inference directly on rewards, we perform inference
on Q-values and convert them to rewards using the Bellman equation.
The prior over Q-values is derived from a prior over rewards.
"""
@model function feedback_model(
    ::QValueWalkDemonstrationFeedback,
    ::TaskType, # EpisodicTask or FiniteHorizonTask does not matter for this model
    choice_sets::Nothing,
    choices::Vector{Trajectory},
    mdp::MDP;
    num_steps::Union{Int,Nothing}=nothing, # Unused for this model
    true_β::Union{Real,Nothing}=nothing
)
    nS, nA = mdp.nS, mdp.nA
    
    # Sample from shared priors model
    priors = Utils.shared_priors(mdp.num_r_params, true_β)
    β ~ priors.β

    # Sample Q-values from over-dispersed normal (this is only a placeholder, the true log likelihood is computed below)
    base_prior = filldist(Normal(0.0, 10.0), nS * nA)
    q ~ base_prior
    
    # Initialize full Q-value matrix with zeros, preserving element type
    Q = similar(q, (nS, nA))
    Q .= reshape(q, (nS, nA))
    
    # Add the log probability adjustment for the Q-value prior
    wrong_prior_log_prob = logpdf(base_prior, q)
    right_prior_log_prob = Utils.Q_prior_density(Q, mdp, priors.r)
    Turing.@addlogprob! right_prior_log_prob - wrong_prior_log_prob
    
    # Compute policy using softmax
    π = Utils.softmax(β .* Q; dims=2)
    logπ = log.(π)
    logP = log.(mdp.P)
    
    # For each demonstration
    num_trajectories = size(choices, 1)
    num_steps = size(choices[1], 1)

    ll = 0.0
    
    for i in 1:num_trajectories, t in 1:num_steps - 1
        s  = choices[i][t  ][1]
        a  = choices[i][t  ][2]
        s′ = choices[i][t+1][1]

        ll += logπ[s, a]
        ll += logP[s, a, s′]
    end

    Turing.@addlogprob! ll
end

"""
Stop feedback model using a discrete-time hazard model.

Hazard at step t:  h_t = 1 - exp(-λ · Cum_t)
where Cum_t = ρ·Cum_{t-1} + max(0, max_a' Q[s_t,a'] - Q[s_t,a_t]).

λ (sensitivity) and ρ (regret discount) are fixed to the values used during
data generation and passed as keyword arguments.
"""
@model function feedback_model(
    ::StopFeedback,
    ::TaskType,
    segments::Vector{Trajectory},
    stop_data::NamedTuple,
    mdp::MDP;
    num_steps::Union{Int,Nothing}=nothing,
    true_β::Union{Real,Nothing}=nothing
)
    stop_times = stop_data.stop_times
    λ_fixed = stop_data.λ
    ρ = stop_data.ρ

    priors = Utils.shared_priors(mdp.num_r_params, true_β)
    r ~ priors.r
    β ~ priors.β

    R = similar(r, mdp.nS, mdp.nA)
    fill!(R, zero(eltype(r)))
    for i in 1:mdp.num_r_params
        for (s, a) in mdp.param_idx_mapping[i]
            R[s, a] = r[i]
        end
    end

    Q = Utils.Q_opt(mdp, R)

    ll = 0.0

    for (idx, seg) in enumerate(segments)
        τ = stop_times[idx]
        cum = zero(eltype(Q))

        for (t, (s, a)) in enumerate(seg)
            instant = max(zero(eltype(Q)), maximum(Q[s, :]) - Q[s, a])
            cum = ρ * cum + instant
            h = 1.0 - exp(-λ_fixed * cum)
            h = clamp(h, 1e-8, 1.0 - 1e-8)

            if τ >= 1 && t == τ
                ll += log(h)
                break
            else
                ll += log(1.0 - h)
            end
        end
    end

    Turing.@addlogprob! ll
end

"""
Rating feedback model using an ordered logit (cumulative link) model.

P(y = k | u, θ) = σ(θ_k - u) - σ(θ_{k-1} - u)
where u = mean segment reward and θ are ordered cutpoints.

Cutpoints are given a prior that enforces ordering via positive increments.
"""
@model function feedback_model(
    ::RatingFeedback,
    ::TaskType,
    segments::Vector{Trajectory},
    rating_data::NamedTuple,
    mdp::MDP;
    num_steps::Union{Int,Nothing}=nothing,
    true_β::Union{Real,Nothing}=nothing
)
    ratings = rating_data.ratings
    K = rating_data.K

    priors = Utils.shared_priors(mdp.num_r_params, true_β)
    r ~ priors.r

    R = similar(r, mdp.nS, mdp.nA)
    fill!(R, zero(eltype(r)))
    for i in 1:mdp.num_r_params
        for (s, a) in mdp.param_idx_mapping[i]
            R[s, a] = r[i]
        end
    end

    # Ordered cutpoints: θ_1 free, remaining via positive increments
    θ₁ ~ Normal(0.0, 2.0)
    if K > 2
        deltas ~ filldist(truncated(Normal(0.0, 1.0), 0.0, Inf), K - 2)
        θ = Vector{eltype(θ₁)}(undef, K - 1)
        θ[1] = θ₁
        for k in 2:(K - 1)
            θ[k] = θ[k - 1] + deltas[k - 1]
        end
    else
        θ = [θ₁]
    end

    ll = 0.0

    for (idx, seg) in enumerate(segments)
        u = mean(R[s, a] for (s, a) in seg)
        k = ratings[idx]  # 1-indexed: 1..K

        if k == 1
            p = Utils.stable_sigmoid(θ[1] - u)
        elseif k == K
            p = 1.0 - Utils.stable_sigmoid(θ[K - 1] - u)
        else
            p = Utils.stable_sigmoid(θ[k] - u) - Utils.stable_sigmoid(θ[k - 1] - u)
        end
        p = clamp(p, 1e-8, 1.0 - 1e-8)
        ll += log(p)
    end

    Turing.@addlogprob! ll
end

"""
Combined feedback model: jointly infers reward from preference, demonstration,
stop, and rating data sharing a single reward posterior.

`choice_sets` is unused (pass `nothing`); all data is in `choices`, which is a
NamedTuple with fields:
  pref_choice_sets, pref_choices, demo_choices,
  stop_segments, stop_data, rating_segments, rating_data
"""
@model function feedback_model(
    ::CombinedFeedback,
    ::TaskType,
    choice_sets::Nothing,
    choices::NamedTuple,
    mdp::MDP;
    num_steps::Union{Int,Nothing}=nothing,
    true_β::Union{Real,Nothing}=nothing
)
    priors = Utils.shared_priors(mdp.num_r_params, true_β)
    r ~ priors.r
    β ~ priors.β

    R = similar(r, mdp.nS, mdp.nA)
    fill!(R, zero(eltype(r)))
    for i in 1:mdp.num_r_params
        for (s, a) in mdp.param_idx_mapping[i]
            R[s, a] = r[i]
        end
    end

    Q = Utils.Q_opt(mdp, R)

    ll = 0.0

    # ── Preference likelihood ─────────────────────────────────────────
    for (i, (traj1, traj2)) in enumerate(choices.pref_choice_sets)
        r1 = sum(R[s, a] for (s, a) in traj1)
        r2 = sum(R[s, a] for (s, a) in traj2)
        p = Utils.stable_sigmoid((r2 - r1) * β)
        if isinf(p) || isnan(p)
            Turing.@addlogprob! -Inf
            return
        end
        p = clamp(p, 1e-6, 1.0 - 1e-6)
        choices.pref_choices[i] ~ Bernoulli(p)
    end

    # ── Demonstration likelihood ──────────────────────────────────────
    π = Utils.softmax(β .* Q; dims=2)
    logπ = log.(π)
    logP = log.(mdp.P)

    for traj in choices.demo_choices, t in 1:length(traj)-1
        s  = traj[t  ][1]
        a  = traj[t  ][2]
        s′ = traj[t+1][1]
        ll += logπ[s, a]
        ll += logP[s, a, s′]
    end

    # ── Stop likelihood ───────────────────────────────────────────────
    stop_times = choices.stop_data.stop_times
    λ_fixed    = choices.stop_data.λ
    ρ          = choices.stop_data.ρ

    for (idx, seg) in enumerate(choices.stop_segments)
        τ = stop_times[idx]
        cum = zero(eltype(Q))
        for (t, (s, a)) in enumerate(seg)
            instant = max(zero(eltype(Q)), maximum(Q[s, :]) - Q[s, a])
            cum = ρ * cum + instant
            h = 1.0 - exp(-λ_fixed * cum)
            h = clamp(h, 1e-8, 1.0 - 1e-8)
            if τ >= 1 && t == τ
                ll += log(h)
                break
            else
                ll += log(1.0 - h)
            end
        end
    end

    # ── Rating likelihood ─────────────────────────────────────────────
    ratings = choices.rating_data.ratings
    K = choices.rating_data.K

    θ₁ ~ Normal(0.0, 2.0)
    if K > 2
        deltas ~ filldist(truncated(Normal(0.0, 1.0), 0.0, Inf), K - 2)
        θ = Vector{eltype(θ₁)}(undef, K - 1)
        θ[1] = θ₁
        for k in 2:(K - 1)
            θ[k] = θ[k - 1] + deltas[k - 1]
        end
    else
        θ = [θ₁]
    end

    for (idx, seg) in enumerate(choices.rating_segments)
        u = mean(R[s, a] for (s, a) in seg)
        k = ratings[idx]
        if k == 1
            p = Utils.stable_sigmoid(θ[1] - u)
        elseif k == K
            p = 1.0 - Utils.stable_sigmoid(θ[K - 1] - u)
        else
            p = Utils.stable_sigmoid(θ[k] - u) - Utils.stable_sigmoid(θ[k - 1] - u)
        end
        p = clamp(p, 1e-8, 1.0 - 1e-8)
        ll += log(p)
    end

    Turing.@addlogprob! ll
end
