using NPZ
using Random

include("mdp.jl")

function generate_and_save_circular_mdp(
    n::Int=10,  # number of states in inner cycle
    k::Int=2,  # length of outer cycle
    cycle_cost::Float64=-1.0,  # cost of entering outer cycle
    γ::Float64=0.9,  # discount factor
    output_dir::String="mdp_data"
)
    # Create the MDP
    mdp = create_circular_MDP(n, k, cycle_cost, γ)
    
    # Create output directory if it doesn't exist
    mkpath(output_dir)
    
    # Save transition matrix and rewards as .npz file
    npzwrite(joinpath(output_dir, "mdp_data.npz"), Dict(
        "transitions" => mdp.P,
        "rewards" => mdp.R,
        "feasible_initial_states" => mdp.S₀
    ))
    
    println("Generated circular MDP with $(mdp.nS) states and $(mdp.nA) actions")
    println("Saved to $(joinpath(output_dir, "mdp_data.npz"))")
    
    return mdp
end


function generate_and_save_random_mdp(
    num_states::Int=30,
    num_actions::Int=2,
    num_clusters::Int=3,
    cluster_props_decay::Float64=0.5,
    min_cluster_prop::Float64=0.05,
    within_cluster_concentration::Float64=0.3,
    between_cluster_concentration::Float64=0.01,
    reward_cluster_mean_decay::Float64=0.5,
    reward_cluster_mean_min::Float64=0.0,
    reward_within_cluster_variance::Float64=0.1,
    has_self_loops::Bool=false,
    output_dir::String="mdp_data"
)
    mdp = create_random_MDP(
        num_states,
        num_actions,
        num_clusters,
        cluster_props_decay,
        min_cluster_prop,
        within_cluster_concentration,
        between_cluster_concentration,
        reward_cluster_mean_decay,
        reward_cluster_mean_min,
        reward_within_cluster_variance;
        has_self_loops=has_self_loops
    )

    # Create output directory if it doesn't exist
    mkpath(output_dir)
    
    # Save transition matrix and rewards as .npz file
    npzwrite(joinpath(output_dir, "mdp_data.npz"), Dict(
        "transitions" => mdp.P,
        "rewards" => mdp.R,
        "feasible_initial_states" => mdp.S₀
    ))
    
    println("Generated random MDP with $(mdp.nS) states and $(mdp.nA) actions")
    println("Saved to $(joinpath(output_dir, "mdp_data.npz"))")
    
    return mdp
end

function generate_and_save_tree_mdp(
    n::Int=4,  # number of levels
    k::Int=2,  # number of decisions per node
    reward_decay::Float64=0.5,  # decay factor for rewards
    min_base_reward::Float64=0.1,  # minimum base reward
    p_undo::Float64=0.1,  # probability of successful undo
    reward_variance::Float64=0.00,  # variance of reward noise
    output_dir::String="mdp_data"
)
    # Create the MDP
    mdp = create_tree_MPD(n, k, reward_decay, min_base_reward, p_undo, reward_variance)
    
    # Create output directory if it doesn't exist
    mkpath(output_dir)
    
    # Save transition matrix and rewards as .npz file
    npzwrite(joinpath(output_dir, "mdp_data.npz"), Dict(
        "transitions" => mdp.P,
        "rewards" => mdp.R,
        "feasible_initial_states" => mdp.S₀
    ))
    
    println("Generated tree MDP with $(mdp.nS) states and $(mdp.nA) actions")
    println("Saved to $(joinpath(output_dir, "mdp_data.npz"))")
    
    return mdp
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    # Default to tree MDP with 2 levels
    mdp = generate_and_save_tree_mdp()
end
