using MCMCChains

"""
Extract reward and β samples from a chain.
Returns a tuple of (reward_samples, β_samples).
"""
function extract_samples(chain::MCMCChains.Chains, feedback_type::PreferenceFeedback, mdp::MDP)

    # Group returns a #params x #samples x #chains array
    reward_samples = Array(group(chain, :r))
    β_samples = vec(Array(group(chain, :β)))
    return reward_samples, β_samples
end

function extract_samples(chain::MCMCChains.Chains, feedback_type::DemonstrationFeedback, mdp::MDP)
    # For demonstrations, directly extract rewards and β
    reward_samples = Array(group(chain, :r))
    β_samples = vec(Array(group(chain, :β)))
    return reward_samples, β_samples
end

function extract_samples(chain::MCMCChains.Chains, feedback_type::QValueWalkDemonstrationFeedback, mdp::MDP)
    # For Q-value walk, we need to:
    # 1. Extract Q-values and β
    # 2. Convert Q-values to rewards using the Bellman equation
    q_samples = Array(group(chain, :q))
    β_samples = vec(Array(group(chain, :β)))
    
    # Initialize array for reward samples
    num_samples = size(q_samples, 1)
    num_states = mdp.nS
    num_actions = mdp.nA
    reward_samples = zeros(num_samples, num_states * num_actions)
    
    # Convert each Q sample to rewards
    for i in 1:num_samples
        Q = reshape(q_samples[i, :], num_states, num_actions)
        R = Utils.Q_to_R(Q, mdp, 10.0)
        reward_samples[i, :] = vec(R)
    end
    
    return reward_samples, β_samples
end

function extract_samples(chain::MCMCChains.Chains, feedback_type::StopFeedback, mdp::MDP)
    reward_samples = Array(group(chain, :r))
    β_samples = vec(Array(group(chain, :β)))
    return reward_samples, β_samples
end

function extract_samples(chain::MCMCChains.Chains, feedback_type::RatingFeedback, mdp::MDP)
    reward_samples = Array(group(chain, :r))
    # Rating model does not sample β; return zeros as placeholder
    β_samples = zeros(size(reward_samples, 1))
    return reward_samples, β_samples
end

function extract_samples(chain::MCMCChains.Chains, feedback_type::CombinedFeedback, mdp::MDP)
    reward_samples = Array(group(chain, :r))
    β_samples = vec(Array(group(chain, :β)))
    return reward_samples, β_samples
end

"""
Helper function to extract samples with appropriate feedback type and MDP.
"""
function extract_samples(chain, feedback_type::FeedbackType, mdp::MDP)
    error("extract_samples not implemented for feedback type: ", typeof(feedback_type))
end

"""
Extract reward and β samples from raw variational inference samples and sym2range mapping.
Returns a tuple of (reward_samples, β_samples).
"""
function extract_samples(samples::Matrix{Float64}, sym2range, feedback_type::PreferenceFeedback, mdp::MDP)
    # Extract r and β indices
    r_ranges = sym2range[:r]
    β_ranges = sym2range[:β]
    
    # Flatten ranges and extract samples
    r_indices = union(r_ranges...)
    β_indices = union(β_ranges...)
    
    reward_samples = samples[r_indices, :]'
    β_samples = vec(samples[β_indices, :])

    # Ensure that each row of reward_samples is a sample from the posterior
    @assert size(reward_samples)[2] == mdp.nS * mdp.nA
    
    return reward_samples, β_samples
end

function extract_samples(samples::Matrix{Float64}, sym2range, feedback_type::DemonstrationFeedback, mdp::MDP)
    # Extract r and β indices
    r_ranges = sym2range[:r]
    β_ranges = sym2range[:β]
    
    # Flatten ranges and extract samples
    r_indices = union(r_ranges...)
    β_indices = union(β_ranges...)
    
    reward_samples = samples[r_indices, :]'
    β_samples = vec(samples[β_indices, :])
    
    return reward_samples, β_samples
end

function extract_samples(samples::Matrix{Float64}, sym2range, feedback_type::QValueWalkDemonstrationFeedback, mdp::MDP)
    # Extract q and β indices
    q_ranges = sym2range[:q]
    β_ranges = sym2range[:β]
    
    # Flatten ranges and extract samples
    q_indices = union(q_ranges...)
    β_indices = union(β_ranges...)
    
    q_samples = samples[q_indices, :]'
    β_samples = vec(samples[β_indices, :])
    
    # Initialize array for reward samples
    num_samples = size(q_samples, 1)
    num_states = mdp.nS
    num_actions = mdp.nA
    reward_samples = zeros(num_samples, num_states * num_actions)
    
    # Convert each Q sample to rewards
    for i in 1:num_samples
        Q = reshape(q_samples[i, :], num_states, num_actions)
        R = Utils.Q_to_R(Q, mdp, 10.0)
        reward_samples[i, :] = vec(R)
    end
    
    return reward_samples, β_samples
end

"""
Helper function to extract samples with appropriate feedback type and MDP.
"""
function extract_samples(samples::Matrix{Float64}, sym2range, feedback_type::FeedbackType, mdp::MDP)
    error("extract_samples not implemented for feedback type: ", typeof(feedback_type))
end
