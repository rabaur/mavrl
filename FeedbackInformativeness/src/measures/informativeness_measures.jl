using ComplexityMeasures
using LinearAlgebra
using Associations
using Turing

"""
Estimate the mutual information using the formula I(X; Y) = H(X) + H(Y) - H(X, Y)
between two sets of samples `X_samples` and `Y_samples` from random variables X and Y respectively.
Uses the Kraskov estimator with `kraskov_k` nearest neighbors.
"""
function mutual_information(X_samples, Y_samples; kraskov_k=10)
    X_ssset = StateSpaceSet(X_samples)
    Y_ssset = StateSpaceSet(Y_samples)
    return association(KSG1(k=kraskov_k), X_ssset, Y_ssset)
end

"""
Estimate the information gain between two sets of samples `prior_samples` and `posterior_samples`
"""
function information_gain(prior_samples, posterior_samples; kraskov_k=10)

    # Set up kNN estimator
    prior_ssset = StateSpaceSet(prior_samples)
    post_ssset = StateSpaceSet(posterior_samples)
    estimator = Kraskov(Shannon(base=2), k=kraskov_k)

    # Compute entropy of prior and posterior
    H_P = information(estimator, prior_ssset)
    H_Q = information(estimator, post_ssset)

    # Information gain is the difference in entropy
    return H_P - H_Q
end

"""
Computes the L2 error between an estimated reward function `R_est` and the true reward function `R_true`.
"""
function l2_error(R_est_vec, R_true_vec)
    return norm(R_est_vec - R_true_vec)
end

"""
Computes the cosine similarity between an estimated reward function `R_est` and the true reward function `R_true`.
The cosine similarity is defined as the dot product of the normalized vectors, which ranges from -1 to 1.
Note: Only pass the learnable parameters, not all.
"""
function cosine_similarity(R_est_vec, R_true_vec)
    R_est_vec_centered = R_est_vec .- minimum(R_est_vec)
    R_true_vec_centered = R_true_vec .- minimum(R_true_vec)
    # Compute dot product and norms
    dot_product = dot(R_est_vec_centered, R_true_vec_centered)
    norm_est = norm(R_est_vec_centered)
    norm_true = norm(R_true_vec_centered)
    
    # Handle case where either norm is zero
    if norm_est == 0 || norm_true == 0
        return 0.0
    end
    
    return dot_product / (norm_est * norm_true)
end

"""
Computes the expected regret weighted by the initial-state distribution,
matching `regret_tabular` in `mavrl/evaluation/regret.py`.

Returns `(regret, discounted_value)` where both are scalars weighted by the
initial-state distribution of the MDP.

`V_true` and `init_weight` can be precomputed once and reused across samples
to avoid redundant Q-value iteration on the (fixed) true reward.
"""
function expected_regret(R_est, R_true, mdp; V_true::Union{Vector{Float64},Nothing}=nothing,
                         init_weight::Union{Vector{Float64},Nothing}=nothing)
    if V_true === nothing
        Q_true = Utils.Q_opt(mdp, R_true; max_iter=1_000)
        V_true = vec(maximum(Q_true, dims=2))
    end
    if init_weight === nothing
        init_weight = zeros(mdp.nS)
        for s in mdp.S₀
            init_weight[s] = 1.0 / length(mdp.S₀)
        end
    end

    Q_est = Utils.Q_opt(mdp, R_est; max_iter=1_000)
    policy_actions = argmax.(eachrow(Q_est))
    policy_est = zeros(mdp.nS, mdp.nA)
    for s in 1:mdp.nS
        policy_est[s, policy_actions[s]] = 1
    end

    R_pi = vec(sum(R_true .* policy_est, dims=2))
    P_pi = zeros(mdp.nS, mdp.nS)
    for s in 1:mdp.nS
        for s_prime in 1:mdp.nS
            P_pi[s, s_prime] = sum(mdp.P[s, :, s_prime] .* policy_est[s, :])
        end
    end

    ϵ = mdp.γ == 1.0 ? 1e-10 : 0.0
    V_est = vec((I - (mdp.γ - ϵ) * P_pi) \ R_pi)

    regret = dot(init_weight, V_true .- V_est)
    discounted_value = dot(init_weight, V_est)
    return regret, discounted_value
end

"""
Compute measures of interest for a given chain.

When `per_sample_regret=true` (default `false`), regret is computed for every
posterior R sample and then averaged — accurate but expensive (one Q-value
iteration per sample).  When `false`, regret is computed once from the
posterior mean reward, which is much cheaper.
"""
function compute_posterior_measures(r_samples::Array{Float64, 2}, β_samples::Array{Float64, 1}, true_β::Float64, mdp::MDPs.MDP;
                                    compute_info_gain::Bool=true,
                                    per_sample_regret::Bool=false)
    
    num_posterior_samples = size(r_samples, 1)

    r_information_gain = NaN
    joint_information_gain = NaN
    if compute_info_gain
        prior_reward_samples = rand(Utils.shared_priors(mdp.num_r_params, true_β).r, num_posterior_samples)'
        prior_β_samples = rand(Utils.shared_priors(mdp.num_r_params, true_β).β, num_posterior_samples)
        r_information_gain = information_gain(prior_reward_samples, r_samples)
        joint_prior_samples = hcat(prior_reward_samples, prior_β_samples)
        joint_posterior_samples = hcat(r_samples, β_samples)
        joint_information_gain = information_gain(joint_prior_samples, joint_posterior_samples)
    end
    
    # Compute posterior mean reward
    post_mean_r = vec(mean(r_samples, dims=1))
    post_mean_R = zeros(mdp.nS, mdp.nA)
    for i in 1:mdp.num_r_params
        for (s, a) in mdp.param_idx_mapping[i]
            post_mean_R[s, a] = post_mean_r[i]
        end
    end
    
    # Compute posterior median reward
    post_median_r = vec(median(r_samples, dims=1))
    post_median_R = zeros(mdp.nS, mdp.nA)
    for i in 1:mdp.num_r_params
        for (s, a) in mdp.param_idx_mapping[i]
            post_median_R[s, a] = post_median_r[i]
        end
    end
    
    # Precompute V_true and init_weight once
    Q_true = Utils.Q_opt(mdp, mdp.R; max_iter=10_000)
    V_true = vec(maximum(Q_true, dims=2))
    init_weight = zeros(mdp.nS)
    for s in mdp.S₀
        init_weight[s] = 1.0 / length(mdp.S₀)
    end

    R_true_vec = zeros(mdp.num_r_params)
    for i in 1:mdp.num_r_params
        (s, a) = first(mdp.param_idx_mapping[i])
        R_true_vec[i] = mdp.R[s, a]
    end

    # Per-sample L2 and cosine (cheap — no Q-value iteration)
    l2_errors = zeros(num_posterior_samples)
    cos_sims = zeros(num_posterior_samples)
    for i in 1:num_posterior_samples
        r_sample_vec = r_samples[i, :]
        l2_errors[i] = l2_error(r_sample_vec, R_true_vec)
        cos_sims[i] = cosine_similarity(r_sample_vec, R_true_vec)
    end

    # Regret & discounted value
    if per_sample_regret
        regrets = zeros(num_posterior_samples)
        disc_values = zeros(num_posterior_samples)
        for i in 1:num_posterior_samples
            R_sample = zeros(mdp.nS, mdp.nA)
            for j in 1:mdp.num_r_params
                for (s, a) in mdp.param_idx_mapping[j]
                    R_sample[s, a] = r_samples[i, j]
                end
            end
            regrets[i], disc_values[i] = expected_regret(R_sample, mdp.R, mdp; V_true=V_true, init_weight=init_weight)
        end
        regret_mean = mean(regrets)
        regret_std = std(regrets)
        disc_value_mean = mean(disc_values)
    else
        regret_mean, disc_value_mean = expected_regret(post_mean_R, mdp.R, mdp; V_true=V_true, init_weight=init_weight)
        regret_std = NaN
    end

    l2_err_mean = mean(l2_errors)
    l2_err_std = std(l2_errors)
    cos_sim_mean = mean(cos_sims)
    cos_sim_std = std(cos_sims)

    β_est = mean(β_samples)
    β_err = abs(β_est - true_β)

    # Gaussian fit to the posterior: N(μ, Σ) over reward parameters
    post_std_r = vec(std(r_samples, dims=1))
    post_cov_r = cov(r_samples)
    
    return Dict(
        "r_information_gain" => r_information_gain,
        "joint_information_gain" => joint_information_gain,
        "l2_error_mean" => l2_err_mean,
        "l2_error_std" => l2_err_std,
        "cosine_similarity_mean" => cos_sim_mean,
        "cosine_similarity_std" => cos_sim_std,
        "regret_mean" => regret_mean,
        "regret_std" => regret_std,
        "discounted_value_mean" => disc_value_mean,
        "beta_est" => β_est,
        "beta_err" => β_err,
        "posterior_mean" => post_mean_r,
        "posterior_std" => post_std_r,
        "posterior_cov" => post_cov_r,
        "posterior_median" => post_median_r
    )
end
