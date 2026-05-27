using Distributions
using LinearAlgebra

struct MDP
    nS::Int  # Number of states
    nA::Int  # Number of actions
    mdp_class::String
    P::Array{Float64, 3}  # Transition probabilities (num_states, num_actions, num_states) P(s' | s, a) = P[s, a, s']
    R::Array{Float64, 2}  # Rewards (num_states, num_actions)
    num_r_params::Int  # Number of learnable parameters in the reward matrix
    param_idx_mapping::Dict{Int, Vector{Tuple{Int, Int}}}  # Dictionary mapping reward parameter indices to lists of (state, action) pairs
    S₀::Vector{Int}  # Initial states
    γ::Float64
end

function mdp_factory(args::Dict{String, Any})
    if args["mdp_type"] == "random"
        return create_random_MDP(
            args["num_states"],
            args["num_actions"],
            args["num_clusters"],
            args["cluster_props_decay"],
            args["min_cluster_prop"],
            args["within_cluster_concentration"],
            args["between_cluster_concentration"],
            args["reward_cluster_mean_decay"],
            args["reward_cluster_mean_min"],
            args["reward_within_cluster_variance"];
            has_self_loops=args["has_self_loops"]
        )
    elseif args["mdp_type"] == "deepsea"
        return create_deepsea_MDP(args["size"], args["discount_factor"], args["p_rand"])
    elseif args["mdp_type"] == "circular"
        return create_circular_MDP(args["size"], args["cycle_length"], args["cost_cycle"], args["discount_factor"])
    elseif args["mdp_type"] == "tree"
        return create_tree_MDP(
            args["n"],
            args["k"],
            args["reward_decay"],
            args["min_base_reward"],
            args["p_undo"],
            args["reward_std"],
            args["discount_factor"]
        )
    elseif args["mdp_type"] == "grid"
        return create_grid_MDP(
            args["grid_size"],
            args["reward_type"],
            args["discount_factor"],
            get(args, "p_rand", 0.0)
        )
    else
        error("Unknown MDP type: $(args["mdp_type"])")
    end
end

"""
Helper function to ensure cluster sizes sum to num_states.
Uses the largest remainder method (also known as Hare-Niemeyer method) for proportional rounding.
"""
function get_cluster_sizes(proportions::Vector{Float64}, num_states::Int)
    # Calculate initial sizes using floor
    sizes = floor.(Int, proportions .* num_states)
    
    # Calculate remainders
    remainders = proportions .* num_states .- sizes
    
    # Calculate how many states we still need to distribute
    remaining = num_states - sum(sizes)
    
    # Sort indices by remainder in descending order
    sorted_indices = sortperm(remainders, rev=true)
    
    # Distribute remaining states to clusters with largest remainders
    for i in 1:remaining
        sizes[sorted_indices[i]] += 1
    end
    
    return sizes
end

"""
Finds the largest beta value such that the probability of the smallest class is at least lowest_class_prob through binary search.
"""
function find_max_beta(
    num_clusters::Int,
    lowest_class_prob::Float64;
    init_beta_range::Tuple{Float64, Float64}=(0.0, 10.0),
    max_iter::Int=1000,
    tol::Float64=1e-6
)
    @assert num_clusters > 1 "Number of clusters must be greater than 1"
    @assert lowest_class_prob ≤ 1.0 / num_clusters "Lowest class probability must be less than or equal to 1 / num_clusters"

    # Binary search
    low = init_beta_range[1]
    high = init_beta_range[2]
    mid = (low + high) / 2
    iter = 0
    while iter < max_iter && high - low > tol
        iter += 1
        mid = (low + high) / 2
        if prob_of_smallest_class(mid, num_clusters) >= lowest_class_prob
            low = mid
        else
            high = mid
        end
    end
    return mid
end

"""
Computes the probability of the smallest class for a given beta value.
"""
function prob_of_smallest_class(beta::Float64, num_clusters::Int)
    potential = exp.(-beta .* (0:num_clusters-1))
    return potential[end] / sum(potential)
end

"""
Gets cluster proportions using exponential decay.
"""
function get_cluster_proportions(
    num_clusters::Int,
    cluster_size_decay::Float64,
    min_cluster_prop::Float64
)
    beta = cluster_size_decay
    if prob_of_smallest_class(beta, num_clusters) ≤ min_cluster_prop
        beta = find_max_beta(num_clusters, min_cluster_prop)
        @info "Decreasing `cluster_prop_decay` from $cluster_size_decay to $beta to ensure that smallest cluster size is ≥ $min_cluster_prop"
    end
    potential = exp.(-beta .* (0:num_clusters-1))
    return potential ./ sum(potential)
end

"""
Creates a random MDP with clustered states.
Parameters:
- num_states: total number of states
- num_actions: number of actions
- num_clusters: number of clusters
- cluster_props_decay: decay factor for cluster sizes (larger = more uneven)
- min_cluster_prop: minimum proportion of states in any cluster
- within_cluster_concentration: concentration parameter for Dirichlet distribution within clusters
- between_cluster_concentration: concentration parameter for Dirichlet distribution between clusters
- reward_cluster_mean_decay: decay factor for reward means across clusters
- reward_cluster_mean_min: minimum mean reward in any cluster
- reward_within_cluster_variance: variance of rewards within clusters
- self_loops: whether to allow self-loops in transitions
"""
function create_random_MDP(
    num_states::Int,
    num_actions::Int,
    num_clusters::Int,
    cluster_props_decay::Float64,
    min_cluster_prop::Float64,
    within_cluster_concentration::Float64,
    between_cluster_concentration::Float64,
    reward_cluster_mean_decay::Float64,
    reward_cluster_mean_min::Float64,
    reward_within_cluster_variance::Float64;
    has_self_loops::Bool=false
)
    # Generate the cluster proportions
    cls_props = get_cluster_proportions(num_clusters, cluster_props_decay, min_cluster_prop)

    # Generate cluster sizes
    cls_sizes = get_cluster_sizes(cls_props, num_states)

    # Verify cluster sizes sum to num_states
    @assert sum(cls_sizes) == num_states "Cluster sizes do not sum to num_states: $(sum(cls_sizes)) != $num_states"

    # Create array of cluster indices
    cumsum_cls_sizes = cumsum(cls_sizes)
    cls_idxs = ones(Int, num_states)
    for cls_idx in 2:num_clusters
        cls_idxs[cumsum_cls_sizes[cls_idx-1]+1:cumsum_cls_sizes[cls_idx]] .= cls_idx
    end

    # Create transition matrix by sampling from Dirichlet distribution
    P = zeros(Float64, num_states, num_actions, num_states)
    for s in 1:num_states
        for a in 1:num_actions
            concentration_params = between_cluster_concentration * ones(num_states)
            concentration_params[cls_idxs .== cls_idxs[s]] .= within_cluster_concentration
            transition_probs = rand(Dirichlet(concentration_params))
            if !has_self_loops
                transition_probs[s] = 0
                transition_probs ./= sum(transition_probs)
            end
            P[s, a, :] = transition_probs
        end
    end

    # Find rewards means per cluster (decaying exponentially, smallest cluster has highest mean to simulate sparsity)
    reward_means = reverse(get_cluster_proportions(num_clusters, reward_cluster_mean_decay, reward_cluster_mean_min))

    # Create reward matrix
    R = zeros(Float64, num_states, num_actions)
    for s in 1:num_states
        for a in 1:num_actions
            R[s, a] = rand(Normal(reward_means[cls_idxs[s]], reward_within_cluster_variance))
        end
    end

    num_r_params = num_states * num_actions
    param_idx_mapping = Dict{Int, Vector{Tuple{Int, Int}}}()
    for i in 1:num_states
        for j in 1:num_actions
            flat_idx = (i - 1) * num_actions + j
            if !haskey(param_idx_mapping, flat_idx)
                param_idx_mapping[flat_idx] = []
            end
            push!(param_idx_mapping[flat_idx], (i, j))
        end
    end

    # Create MDP
    return MDP(
        num_states,
        num_actions,
        "random",
        P,
        R,
        num_r_params,
        param_idx_mapping,
        collect(1:num_states),  # All states can be initial
        0.9  # Default discount factor
    )
end

"""
For a row-major `n_rows` x `n_cols` grid, return the flat index of the cell at `(row, col)`.
"""
function flat_idx(n_rows::Int, n_cols::Int, row::Int, col::Int)
    flat_index = (row - 1) * n_cols + col
    @assert flat_index > 0 "Flat index must be > 0"
    @assert flat_index <= n_rows * n_cols "Flat index must be ≤ n_rows * n_cols, but got $(row - 1) * $(n_cols) + $(col) = $(flat_index) > $(n_rows * n_cols)"
    return flat_index
end

"""
DeepSea Exploration MDP, adapted from Osband et al. (2019)

Imagine a `size` x `size` grid, with `size` > 2, where the vertical axis represents the depth of the water column,
and the horizontal axis represents the distance from the left edge of the grid.

A diver starts as state (1, 1), and its main goal is to find the treasure, which can only be found if dives right in (nS, nS).
At each steps the diver dives deeper, but at each step, can decide to dive down-left (a=1) or down-right (a=2).
Optionally, you can provide a parameter 0 <= `p_rand` <= 0.5, which is the probability that the diver does not execute the intended action.
If the diver is already at the left-most or right-most columns and chooses to move down-left or down-right respectively, it will remain in the same column (boundary).
If the diver is at the bottom of the grid, any action will teleport the diver to the initial state (1, 1). (continuous)

Even though the original MDP is formulated on a grid, the states are effectively in the lower-triangular part of the grid.
We thus only represent the lower-triangular part of the grid.

Fix R₁ >> R₂ > R₃. The diver receives a reward of R₃ if it dives right, and a reward of R₂ if it dives left unless
the right action leads to the treasure, in which case it receives a reward of R₁.

`γ` is the discount factor.
"""
function create_deepsea_MDP(
    size::Int,
    γ::Float64,
    p_rand::Float64 = 0.0
)

    @assert size >= 2 "Size must be ≥ 2"
    @assert p_rand >= 0.0 && p_rand <= 0.5 "p_rand must be between 0 and 0.5"

    # Define rewards
    Rᵣ = 0.0 # Reward for moving left
    Rₜ = 1.0 # Reward for finding treasure. Must be at least size times larger than Rₗ
    Rₗ = Rₜ / (2.0 * size) # Reward for moving right

    nS = size * (size + 1) ÷ 2
    nA = 2
    P = zeros(Float64, nS, nA, nS)
    R = zeros(Float64, nS, nA)
    
    # Create vector that holds the "beginning of level-index" for each level in the lower-triangular matrix,
    # i.e., for a DeepSea MDP of size 4, this would be [1, 2, 4, 7],
    # since the rows start at these elements were the full matrix flattened row-wise
    level_idxs = [1]
    for i in 2:size
        push!(level_idxs, level_idxs[end] + i - 1)
    end

    # Define the dynamics for all states except the last level
    # (which we treat separately due to the teleporting behavior)
    for level in 1:size
        for offset in 0:level-1
            # Each state is connected to states in the next level.
            # Either is is connected to the state corresponding to the left action (offset -1) or right action (offset +1)
            # This needs to be clamped within the boundaries of a level.
            # A left action gets Rₗ reward, a right action Rₜ, unless the right action leads to the treasure chest.
            # Then the reward of a right action is Rₜ
            curr_state = level_idxs[level] + offset
            if level == size
                next_left_state = 1
                next_right_state = 1
                P[curr_state, 1, next_left_state] = 1.0
                P[curr_state, 2, next_right_state] = 1.0
            else
                next_level_idx = level_idxs[level + 1]
                base_state = next_level_idx + offset
                next_left_state = max(base_state - 1, next_level_idx)
                next_right_state = min(base_state + 1, next_level_idx + level + 1)
                P[curr_state, 1, next_left_state] = 1.0 - p_rand
                P[curr_state, 2, next_left_state] = p_rand
                P[curr_state, 1, next_right_state] = p_rand
                P[curr_state, 2, next_right_state] = 1.0 - p_rand
            end
            R[curr_state, 1] = Rₗ
            if level == size && offset == level-1
                R[curr_state, 2] = Rₜ
            else
                R[curr_state, 2] = Rᵣ
            end
        end
    end

    # Initial states: Only the top-left state
    S₀ = [flat_idx(size, size, 1, 1)]

    # Number of learnable parameters: Number of states * number of actions
    num_r_params = nS * nA

    # The parameter index mapping is trivial, since we learn the complete set of rewards
    param_idx_mapping = Dict{Int, Vector{Tuple{Int, Int}}}()
    for i in 1:nS
        for j in 1:nA
            flat_idx = (i - 1) * nA + j
            if !haskey(param_idx_mapping, flat_idx)
                param_idx_mapping[flat_idx] = []
            end
            push!(param_idx_mapping[flat_idx], (i, j))
        end
    end

    # Create the MDP
    mdp = MDP(nS, nA, "deepsea", P, R, num_r_params, param_idx_mapping, S₀, γ)
    return mdp
end

"""
This MDP is intended to show the effect of aperiodicity.

Parameters:
- `n`: number of states in the inner cycle
- `k`: The length of the outer cycle (i.e., the number of edges, thus the number of states in the outer cycle is k-1)
- `cycle_cost`: cost of entering the outer cycle
- `γ`: discount factor

Let sᵢ be the i-th state in the inner cycle.
Let cᵢʲ be the j-th state, j=1,..,k-1, in the outer cycle starting at sᵢ.
We use the following (linear) layout for the states:
[s₁, s₂, ..., sₙ, c₁¹, c₁², ..., c₁ᵏ⁻¹, c₂¹, c₂², ..., c₂ᵏ⁻¹, ..., cₙ¹, cₙ², ..., cₙᵏ⁻¹]
i.e., inner cycle, followed by n times k-1 states corresponding to the outer cycles.
The transitions are as follows:
- action 1 in sᵢ moves to sᵢ₊₁ (continue inner cycle)
- action 2 in sᵢ moves to cᵢ¹ (enter outer cycle)
- actions 1, 2 in cᵢʲ move to cᵢʲ⁺¹ if j < k - 1 (continue cycle)
- actions 1, 2 in cᵢʲ move to sᵢ if j = k - 1 (cycle complete, back to inner cycle)
Special case: if k = 1, then the outer cycle is a self-loop to sᵢ.

Rewards for inner transitions R(sᵢ → sᵢ₊₁) are sampled from standard normals. Only these rewards are learnable.
Rewards for entering outer cycle is R(sᵢ → cᵢ¹) = `cycle_cost`
"""
function create_circular_MDP(
    n::Int,
    k::Int,
    cycle_cost::Float64,
    γ::Float64
)
    @assert cycle_cost <= 0.0 "Cost of cycle must be ≤ 0"
    @assert 1 <= k <= n "Cycle length must be ≥ 1 and ≤ size"

    # size states for the inner cycle, and size * (cycle_length - 1) states for each outer cycle
    nS = n + n * (k - 1)
    println("n: $n, k: $k, nS: $nS")
    nA = 2
    P = zeros(Float64, nS, nA, nS)
    R = zeros(Float64, nS, nA)

    # Number of learnable params (only inner cycle)
    num_r_params = n
    
    # Initialize the transitions for the inner cycle
    for sᵢ in 1:n

        # Closing inner cycle
        if sᵢ == n
            sᵢ₊₁ = 1
        else
            sᵢ₊₁ = sᵢ + 1
        end
        P[sᵢ, 1, sᵢ₊₁] = 1.0 # continue inner cycle
        R[sᵢ, 1] = rand(Normal(0.0, 1.0)) # Standard normal reward for inner transitions
        if k == 1
            # Self loop
            P[sᵢ, 2, sᵢ] = 1.0
            R[sᵢ, 2] = cycle_cost # Cost of entering outer cycle
        else
            # True cycle with k-1 states
            cᵢ¹ = n + (sᵢ - 1) * (k - 1) + 1
            println("sᵢ: $sᵢ, cᵢ¹: $cᵢ¹")
            P[sᵢ, 2, cᵢ¹] = 1.0
            R[sᵢ, 2] = cycle_cost # Cost of entering outer cycle

            for cᵢʲ⁺¹ in cᵢ¹+1:cᵢ¹+k-2
                # Within cycle
                cᵢʲ = cᵢʲ⁺¹ - 1
                P[cᵢʲ, 1, cᵢʲ⁺¹] = 1.0
                P[cᵢʲ, 2, cᵢʲ⁺¹] = 1.0
            end
            cᵢᵏ⁻¹ = cᵢ¹ + k - 2

            # Enter inner cycle again
            P[cᵢᵏ⁻¹, 1, sᵢ] = 1.0
            P[cᵢᵏ⁻¹, 2, sᵢ] = 1.0
        end
    end

    # Create the parameter index mapping
    param_idx_mapping = Dict{Int, Vector{Tuple{Int, Int}}}()
    for i in 1:n
        param_idx_mapping[i] = [(i, 1)]
    end

    # Only states of inner cycles can be starting states
    S₀ = collect(1:n)

    # Create the MDP
    mdp = MDP(nS, nA, "circular", P, R, num_r_params, param_idx_mapping, S₀, γ)
    return mdp
end

"""
Finds the maximum reward decay such that the minimum base reward is met.
"""
function find_max_reward_decay(num_leaves::Int, min_base_reward::Float64)
    @assert min_base_reward > 0 "Minimum base reward must be positive"
    @assert min_base_reward ≤ 1.0 "Minimum base reward must be ≤ 1.0"
    
    # Binary search for the maximum decay
    low = 0.0
    high = 10.0  # Start with a reasonable upper bound
    mid = (low + high) / 2
    max_iter = 1000
    tol = 1e-6
    
    for _ in 1:max_iter
        if high - low < tol
            break
        end
        
        mid = (low + high) / 2
        # Calculate the reward for the last leaf
        last_leaf_reward = exp(-mid * (num_leaves - 1))
        
        if last_leaf_reward >= min_base_reward
            low = mid
        else
            high = mid
        end
    end
    
    return mid
end

"""
Sequential Decision Making MDP (tree-structured MDP).

The goal of this MDP is to showcase the effect of an exponentially growing state-space and sparse (terminal) rewards.

The MDP is structered as a tree, with the root being the only initial state s₀, at which the agent can select from `k` decisions.
The three consists of `n` levels, with the last level being the leaves l of the tree. With that, the environment has N = kⁿ possible trajectories (ignoring the "undo" action).
The leaves are absorbing.

We assume a sparse-reward function, where agents only receive rewards on the final action. All other rewards are zero.
We vary the sparsity of the rewards by exponentially decaying them - wlog, the base reward of a trajectory ending in leaf lᵢ for i ∈ {1, ..., N} is ∝ exp(-`reward_decay`⋅(i-1))
With `min_base_reward` ≤ 1 / N you can ensure that transitioning to leaf N yields at least `min_base_reward` (the decay parameter is adjusted accordingly).
On top of the base reward, we add Gaussian noise with variance `reward_variance` to all learnable rewards.
Note that the only possible action when reaching a leaf is staying in that leaf, but this action does not yield a reward (only reaching the leaf yields the reward).

In addition to the k forward actions, there is a "undo" action that transfers you to the parent node (in case of the root, it is a self-loop to the root) with probability `p_undo`.
With probability 1 - p_undo a random forward action is chosen.
"""
function create_tree_MDP(
    n::Int,
    k::Int,
    reward_decay::Float64,
    min_base_reward::Float64,
    p_undo::Float64,
    reward_std::Float64,
    discount_factor::Float64
)
    @assert n > 0 "Number of levels must be positive"
    @assert k > 0 "Number of decisions must be positive"
    @assert 0 ≤ p_undo ≤ 1 "Undo probability must be between 0 and 1"
    @assert reward_std ≥ 0 "Reward standard deviation must be non-negative"
    @assert min_base_reward ≤ 1.0 / (k^n - 1) "Minimum base reward must be ≤ 1 / (k^n - 1)"
    
    # Calculate total number of states
    num_states = (k^n - 1) ÷ (k - 1)  # Sum of geometric series for complete k-ary tree
    num_leaves = k^(n-1)  # Number of leaves is k^(n-1) for a complete k-ary tree
    nS = num_states
    nA = k + 1  # k forward actions + 1 undo action
    
    # Adjust reward decay if needed
    if exp(-reward_decay * (num_leaves - 1)) < min_base_reward
        reward_decay = find_max_reward_decay(num_leaves, min_base_reward)
        @info "Decreasing `reward_decay` to $reward_decay to ensure minimum base reward of $min_base_reward"
    end
    
    # Initialize transition and reward matrices
    P = zeros(Float64, nS, nA, nS)
    R = zeros(Float64, nS, nA)
    
    # Create state mapping for easy parent/child lookup
    state_to_level = Dict{Int, Int}()
    state_to_parent = Dict{Int, Int}()
    state_to_children = Dict{Int, Vector{Int}}()
    
    # Initialize children arrays for all non-leaf nodes
    for s in 1:((num_states - num_leaves))
        state_to_children[s] = Int[]
    end
    
    # Build the tree structure
    current_state = 1
    for level in 1:n
        states_in_level = k^(level-1)
        for state_in_level in 1:states_in_level
            state_to_level[current_state] = level
            
            if level > 1
                # For a k-ary tree, parent of node i is floor((i-2)/k) + 1
                parent = floor(Int, (current_state - 2) / k) + 1
                state_to_parent[current_state] = parent
                push!(state_to_children[parent], current_state)
            end
            
            current_state += 1
        end
    end
    
    # Set up transitions and rewards
    for s in 1:nS
        level = state_to_level[s]
        
        if level == n
            # Leaf node - self-loop for all actions with no reward
            for a in 1:nA
                P[s, a, s] = 1.0
            end
        else
            # Non-leaf node
            children = state_to_children[s]
            
            # Forward actions
            for a in 1:k
                P[s, a, children[a]] = 1.0
                
                # If this is a transition to a leaf node, assign the reward
                if level == n-1
                    # Number of nodes up to the leaf level is (k^(n-1) - 1) / (k-1)
                    up_to_leaf_level = (k^(n-1) - 1) ÷ (k-1)
                    leaf_idx = children[a] - up_to_leaf_level
                    base_reward = exp(-reward_decay * (leaf_idx - 1))
                    R[s, a] = base_reward + rand(Normal(0, reward_std))
                end
            end
            
            # Undo action
            if level == 1
                # Root node - self-loop
                P[s, k+1, s] = 1.0
            else
                parent = state_to_parent[s]
                P[s, k+1, parent] = p_undo
                # Distribute remaining probability among forward actions
                for a in 1:k
                    P[s, k+1, children[a]] = (1 - p_undo) / k
                end
            end
        end
    end
    
    # Number of learnable parameters (only leaf rewards)
    num_r_params = num_leaves
    
    # Create parameter index mapping
    param_idx_mapping = Dict{Int, Vector{Tuple{Int, Int}}}()
    before_leaf_states = filter(s -> state_to_level[s] == n-1, 1:nS)
    for (i, s) in enumerate(before_leaf_states)
        for a in 1:k
            flat_idx = (i - 1) * k + a
            if !haskey(param_idx_mapping, flat_idx)
                param_idx_mapping[flat_idx] = []
            end
            push!(param_idx_mapping[flat_idx], (s, a))
        end
    end
    
    # Only root state can be initial
    S₀ = [1]
    
    # Create the MDP
    mdp = MDP(nS, nA, "tree", P, R, num_r_params, param_idx_mapping, S₀, discount_factor)
    return mdp
end

"""
Grid successor state with boundary checks (1-indexed).
Actions: RIGHT=1, UP=2, LEFT=3, DOWN=4, STAY=5.
"""
function grid_succ_state(i::Int, j::Int, a::Int, grid_size::Int)
    deltas = [(0, 1), (-1, 0), (0, -1), (1, 0), (0, 0)]  # RIGHT, UP, LEFT, DOWN, STAY
    di, dj = deltas[a]
    i_new = i + di
    j_new = j + dj
    if i_new < 1 || i_new > grid_size || j_new < 1 || j_new > grid_size
        return i, j
    end
    return i_new, j_new
end

function grid_reward_sparse(grid_size::Int; goal_val::Float64=1.0)
    R1d = fill(-0.1, grid_size, grid_size)
    R1d[grid_size, grid_size] = goal_val
    return reshape(R1d, grid_size^2)
end

function grid_reward_cliff(grid_size::Int;
    base_val::Float64=0.0,
    top_path_val::Float64=-1.0,
    bottom_path_val::Float64=-4.0,
    goal_val::Float64=4.0
)
    R1d = fill(base_val, grid_size, grid_size)
    R1d[1, 2:end] .= top_path_val
    R1d[grid_size, 1:end-1] .= bottom_path_val
    R1d[grid_size, grid_size] = goal_val
    return reshape(R1d, grid_size^2)
end

function grid_reward_trap(grid_size::Int; goal_val::Float64=1.0, trap_val::Float64=-4.0)
    R1d = zeros(grid_size, grid_size)
    R1d[grid_size, grid_size] = goal_val

    trap_size = max(1, grid_size ÷ 5)
    # +1 to convert Python 0-indexed center positions to Julia 1-indexed
    trap_centers = [grid_size ÷ 4 + 2, 3 * grid_size ÷ 4 + 1]
    half = trap_size ÷ 2

    for cr in trap_centers, cc in trap_centers
        for dr in -half:(trap_size - half - 1), dc in -half:(trap_size - half - 1)
            row, col = cr + dr, cc + dc
            if 1 <= row <= grid_size && 1 <= col <= grid_size
                if (row, col) != (1, 1) && (row, col) != (grid_size, grid_size)
                    R1d[row, col] = trap_val
                end
            end
        end
    end
    return reshape(R1d, grid_size^2)
end

"""
Creates a 2D grid MDP with 5 actions (RIGHT, UP, LEFT, DOWN, STAY).

Rewards depend only on the departure state, so `num_r_params = nS` with parameter
tying across all actions for the same state.

Arguments:
- `grid_size`: number of rows/columns
- `reward_type`: one of "sparse", "cliff", "trap"
- `γ`: discount factor
- `p_rand`: probability of executing a uniformly random *other* action instead of the
  intended one (0.0 = deterministic)
"""
function create_grid_MDP(
    grid_size::Int,
    reward_type::String,
    γ::Float64,
    p_rand::Float64=0.0
)
    @assert grid_size >= 2 "grid_size must be >= 2"
    @assert 0.0 <= p_rand <= 1.0 "p_rand must be in [0, 1]"

    nS = grid_size^2
    nA = 5
    P = zeros(Float64, nS, nA, nS)

    for i in 1:grid_size, j in 1:grid_size
        s = (i - 1) * grid_size + j
        for a in 1:nA
            for a_prime in 1:nA
                i2, j2 = grid_succ_state(i, j, a_prime, grid_size)
                s_prime = (i2 - 1) * grid_size + j2
                if a == a_prime
                    P[s, a, s_prime] += 1.0 - p_rand
                else
                    P[s, a, s_prime] += p_rand / (nA - 1)
                end
            end
        end
    end

    # State-only reward vector
    if reward_type == "sparse"
        Rs = grid_reward_sparse(grid_size)
    elseif reward_type == "cliff"
        Rs = grid_reward_cliff(grid_size)
    elseif reward_type == "trap"
        Rs = grid_reward_trap(grid_size)
    else
        error("Unknown grid reward type: $reward_type")
    end

    # Broadcast to (nS, nA) — reward depends only on departure state
    R = zeros(Float64, nS, nA)
    for s in 1:nS, a in 1:nA
        R[s, a] = Rs[s]
    end

    # Parameter tying: one learnable param per state, shared across all actions
    num_r_params = nS
    param_idx_mapping = Dict{Int, Vector{Tuple{Int, Int}}}()
    for s in 1:nS
        param_idx_mapping[s] = [(s, a) for a in 1:nA]
    end

    S₀ = [1]
    return MDP(nS, nA, "grid", P, R, num_r_params, param_idx_mapping, S₀, γ)
end

function sample_next_state(mdp::MDP, s::Int, a::Int)
    return rand(Categorical(mdp.P[s, a, :]))
end

function get_reward(mdp::MDP, s::Int, a::Int)
    return mdp.R[s, a]
end

function simulate_trajectory(
    mdp::MDP,
    policy::Array{Float64, 2},
    num_steps::Int,
    initial_state::Union{Int, Nothing} = nothing
)
    if initial_state === nothing
        s₀_probs = ones(length(mdp.S₀)) ./ length(mdp.S₀)
        s = rand(Categorical(s₀_probs)) # Start from a random state in the initial set
    else
        s = initial_state
    end
    trajectory = []
    total_reward = 0.0

    for t in 1:num_steps
        a = rand(Categorical(policy[s, :]))
        push!(trajectory, (s, a))
        total_reward += mdp.γ^(t-1) * mdp.R[s, a]
        s = sample_next_state(mdp, s, a)
    end

    return trajectory, total_reward
end

"""
Simulate until one step AFTER the agent enters `goal_state`, so that the
likelihood model observes `logπ[goal, a]` (the action choice at the goal).

The trajectory ends with `[..., (goal, a_goal), (s_after, a_after)]`.
The model loop `t = 1:length(traj)-1` then processes the goal entry as a
full `(s, a, s')` triple rather than only as a next-state.
"""
function simulate_trajectory_to_goal(
    mdp::MDP,
    policy::Array{Float64, 2},
    initial_state::Int,
    goal_state::Int;
    max_steps::Int = 500
)
    s = initial_state
    trajectory = Tuple{Int,Int}[]
    total_reward = 0.0
    reached_goal = false

    for t in 1:max_steps
        a = rand(Categorical(policy[s, :]))
        push!(trajectory, (s, a))
        total_reward += mdp.γ^(t-1) * mdp.R[s, a]

        if reached_goal
            break
        end
        if s == goal_state
            reached_goal = true
        end
        s = sample_next_state(mdp, s, a)
    end

    return trajectory, total_reward
end

"""
Simulates a trajectory for a finite-horizon task.
"""
function simulate_trajectory(
    mdp::MDP,
    policy::Array{Float64, 3},
    num_steps::Int,
    initial_state::Union{Int, Nothing} = nothing
)
    if initial_state === nothing
        s₀_probs = ones(length(mdp.S₀)) ./ length(mdp.S₀)
        s = rand(Categorical(s₀_probs)) # Start from a random state in the initial set
    else
        s = initial_state
    end
    trajectory = []
    total_reward = 0.0

    for t in 1:num_steps
        a = rand(Categorical(policy[s, :, num_steps - t + 1]))
        push!(trajectory, (s, a))
        total_reward += mdp.γ^(t-1) * mdp.R[s, a]
        s = sample_next_state(mdp, s, a)
    end

    return trajectory, total_reward
end

"""
Returns the set of states that are reachable from the initial set of states in `mdp`
"""
function get_reachable_states(mdp::MDP)
    reachable = Set{Int}(mdp.S₀)  # Start with initial states
    frontier = Set{Int}(mdp.S₀)   # States to explore next
    
    while !isempty(frontier)
        new_frontier = Set{Int}()
        for s in frontier
            for a in 1:mdp.nA
                # Find all states reachable from s with action a
                for s_next in 1:mdp.nS
                    if mdp.P[s, a, s_next] > 0 && !(s_next in reachable)
                        push!(reachable, s_next)
                        push!(new_frontier, s_next)
                    end
                end
            end
        end
        frontier = new_frontier
    end
    
    return sort(collect(reachable))
end

"""
Returns a list of (state, action) pairs that are reachable in the MDP.
"""
function get_reachable_state_action_pairs(mdp::MDP)
    reachable_states = get_reachable_states(mdp)
    pairs = []
    for s in reachable_states
        for a in 1:mdp.nA
            # Check if there's any non-zero transition probability from this state-action pair
            if any(mdp.P[s, a, :] .> 0)
                push!(pairs, (s, a))
            end
        end
    end
    return pairs
end