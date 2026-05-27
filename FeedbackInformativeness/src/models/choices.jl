using StatsBase: mean, quantile

_rand_start(mdp::MDP, s::Int) = s
_rand_start(mdp::MDP, ::Nothing) = rand(1:mdp.nS)

"""
Generates choice-sets and choices for demonstration feedback.
Note that for demonstrations, the choice-sets are implicit, thus we return nothing.
The choices are the actions taken in the demonstrations (and the steps taken).

When `goal_state` is provided the demonstrator starts at `initial_state` (required)
and runs until the goal state is reached (included in the trajectory) instead of for
a fixed `num_steps`.
"""
function generate_choices(
    ::DemonstrationFeedback,
    mdp::MDP,
    num_feedback_samples::Int,
    num_steps::Int,
    β_true::Float64;
    initial_state::Union{Int, Nothing} = nothing,
    task_type::TaskType = EpisodicTask,
    goal_state::Union{Int, Nothing} = nothing
)
    # Get Q-values from true rewards
    if task_type isa EpisodicTask
        Q = Utils.Q_opt(mdp, mdp.R; max_iter=10000)
    else
        Q = Utils.Q_opt_finite_horizon(mdp, mdp.R, num_steps)
    end
    
    # Compute the policy
    π = Utils.softmax(β_true * Q)
    
    # Simulate demonstrations
    choices::Vector{Trajectory} = []
    
    if goal_state !== nothing
        start = _rand_start(mdp, initial_state)
        for _ in 1:num_feedback_samples
            trajectory, _ = simulate_trajectory_to_goal(mdp, π, start, goal_state)
            push!(choices, trajectory)
        end
    else
        for _ in 1:num_feedback_samples
            trajectory, _ = simulate_trajectory(mdp, π, num_steps, _rand_start(mdp, initial_state))
            push!(choices, trajectory)
        end
    end
    return nothing, choices
end


"""
Generates choice-sets and choices for preference feedback.
The choice-sets are the pairs of trajectories.
The choices are the preference between the two trajectories.

"""
function generate_choices(
    ::PreferenceFeedback,
    mdp::MDP,
    num_feedback_samples::Int,
    num_steps::Int,
    β_true::Float64;
    initial_state::Union{Int, Nothing} = nothing,
    task_type::TaskType = EpisodicTask  # Unused for preference feedback
)::Tuple{Vector{Tuple{Trajectory, Trajectory}}, Vector{Int}}
    # Generate trajectories with random policy
    uniform_policy = ones(mdp.nS, mdp.nA) ./ mdp.nA
    trajectories::Vector{Trajectory} = []
    
    choice_sets::Vector{Tuple{Trajectory, Trajectory}} = []
    for _ in 1:num_feedback_samples
        traj1, _ = simulate_trajectory(mdp, uniform_policy, num_steps, _rand_start(mdp, initial_state))
        traj2, _ = simulate_trajectory(mdp, uniform_policy, num_steps, _rand_start(mdp, initial_state))
        push!(choice_sets, (traj1, traj2))
    end

    choices = Int[]
    
    for (traj1, traj2) in choice_sets
        r1 = sum(mdp.R[s,a] for (s,a) in traj1)
        r2 = sum(mdp.R[s,a] for (s,a) in traj2)
        
        p = Utils.stable_sigmoid((r2 - r1) * β_true)
        choice = rand(Bernoulli(p))
        push!(choices, choice)
    end
    
    return choice_sets, choices
end


"""
Generates segments and simulated stop times for stop feedback.

Returns `(segments, stop_data)` where:
- `segments::Vector{Trajectory}` — each segment is a short trajectory
- `stop_data::NamedTuple{(:stop_times, :λ, :ρ)}` — stop time per segment (-1 = censored),
  calibrated sensitivity λ, and the regret discount ρ used during generation
"""
function generate_choices(
    ::StopFeedback,
    mdp::MDP,
    num_feedback_samples::Int,
    num_steps::Int,
    β_true::Float64;
    initial_state::Union{Int, Nothing} = nothing,
    task_type::TaskType = EpisodicTask(),
    c::Float64 = 2.0,
    regret_percentile::Float64 = 75.0,
    regret_discount::Float64 = 0.1
)
    Q = Utils.Q_opt(mdp, mdp.R; max_iter=10000)
    π = Utils.softmax(β_true * Q)

    segments = Vector{Trajectory}()
    for _ in 1:num_feedback_samples
        traj, _ = simulate_trajectory(mdp, π, num_steps, _rand_start(mdp, initial_state))
        push!(segments, traj)
    end

    # Compute max cumulative regret per segment for λ calibration
    max_regrets = Float64[]
    for seg in segments
        cum = 0.0
        max_cum = 0.0
        for (s, a) in seg
            instant = max(0.0, maximum(Q[s, :]) - Q[s, a])
            cum = regret_discount * cum + instant
            max_cum = max(max_cum, cum)
        end
        push!(max_regrets, max_cum)
    end

    ref = length(max_regrets) > 0 ? quantile(max_regrets, regret_percentile / 100.0) : 1.0
    λ = ref > 1e-8 ? c / ref : c

    # Simulate stop times
    stop_times = Int[]
    for seg in segments
        cum = 0.0
        stopped = false
        for (t, (s, a)) in enumerate(seg)
            instant = max(0.0, maximum(Q[s, :]) - Q[s, a])
            cum = regret_discount * cum + instant
            h = 1.0 - exp(-λ * cum)
            if rand() < h
                push!(stop_times, t)
                stopped = true
                break
            end
        end
        if !stopped
            push!(stop_times, -1)
        end
    end

    stop_data = (stop_times=stop_times, λ=λ, ρ=regret_discount)
    return segments, stop_data
end


"""
Generates segments and ordinal ratings for rating feedback.

Returns `(segments, rating_data)` where:
- `segments::Vector{Trajectory}` — each segment is a short trajectory
- `rating_data::NamedTuple{(:ratings, :cutpoints, :K)}` — ordinal label per segment (1-indexed),
  the cutpoints used for binning, and number of categories K
"""
function generate_choices(
    ::RatingFeedback,
    mdp::MDP,
    num_feedback_samples::Int,
    num_steps::Int,
    β_true::Float64;
    initial_state::Union{Int, Nothing} = nothing,
    task_type::TaskType = EpisodicTask(),
    num_categories::Int = 5,
    noise_std::Float64 = 0.0
)
    # Use a uniform-random policy so ratings span the full quality range
    uniform_policy = ones(mdp.nS, mdp.nA) ./ mdp.nA

    segments = Vector{Trajectory}()
    for _ in 1:num_feedback_samples
        traj, _ = simulate_trajectory(mdp, uniform_policy, num_steps, _rand_start(mdp, initial_state))
        push!(segments, traj)
    end

    # Mean per-step reward for each segment
    returns = Float64[]
    for seg in segments
        u = mean(mdp.R[s, a] for (s, a) in seg)
        if noise_std > 0.0
            u += randn() * noise_std
        end
        push!(returns, u)
    end

    # Quantile cutpoints
    K = num_categories
    cutpoints = Float64[]
    for k in 1:(K - 1)
        push!(cutpoints, quantile(returns, k / K))
    end

    # Assign ratings (1-indexed: 1 .. K)
    ratings = Int[]
    for u in returns
        r = 1
        for θ in cutpoints
            if u > θ
                r += 1
            end
        end
        push!(ratings, r)
    end

    rating_data = (ratings=ratings, cutpoints=cutpoints, K=K)
    return segments, rating_data
end


"""
Generates combined data for all four feedback modalities.

Returns `(nothing, combined_data)` where `combined_data` is a NamedTuple
bundling each single-modality dataset.
"""
function generate_choices(
    ::CombinedFeedback,
    mdp::MDP,
    num_feedback_samples::Int,   # unused — each modality uses its own count
    num_steps::Int,
    β_true::Float64;
    initial_state::Union{Int, Nothing} = nothing,
    task_type::TaskType = EpisodicTask(),
    num_pref::Int   = 128,
    num_demo::Int   = 1,
    num_stop::Int   = 256,
    num_rating::Int = 64,
    goal_state::Union{Int, Nothing} = nothing
)
    pref_cs, pref_ch = generate_choices(
        PreferenceFeedback(), mdp, num_pref, num_steps, β_true;
        initial_state=initial_state, task_type=task_type)

    demo_start = goal_state !== nothing ? first(mdp.S₀) : initial_state
    _, demo_ch = generate_choices(
        NaiveDemonstrationFeedback(), mdp, num_demo, num_steps, β_true;
        initial_state=demo_start, task_type=task_type, goal_state=goal_state)

    stop_segs, stop_d = generate_choices(
        StopFeedback(), mdp, num_stop, num_steps, 1.0;
        initial_state=initial_state, task_type=task_type) # hardcoded β=1.0

    rating_segs, rating_d = generate_choices(
        RatingFeedback(), mdp, num_rating, num_steps, β_true;
        initial_state=initial_state, task_type=task_type)

    combined = (
        pref_choice_sets = pref_cs,
        pref_choices     = pref_ch,
        demo_choices     = demo_ch,
        stop_segments    = stop_segs,
        stop_data        = stop_d,
        rating_segments  = rating_segs,
        rating_data      = rating_d,
    )
    return nothing, combined
end