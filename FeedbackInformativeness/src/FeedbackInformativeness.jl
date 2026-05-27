module FeedbackInformativeness

# Import all required packages
using Flux
using Distributed
using Turing
using Turing.Variational
using Turing.ADTypes
using TuringBenchmarking
using Plots
using StatsPlots
using LinearAlgebra
using Random
using Distributions
using StatsFuns
using ArgParse
using UUIDs
using MCMCChains
using Base.Threads
using Bijectors: bijector, inverse, transformed, Shift, Scale
using Bijectors

# Define submodules
module Types
    include("types.jl")
    
    # Feedback types
    export FeedbackType, PreferenceFeedback, DemonstrationFeedback, NaiveDemonstrationFeedback, QValueWalkDemonstrationFeedback, StopFeedback, RatingFeedback, CombinedFeedback
    
    # Remaining types
    export EpisodicTask, FiniteHorizonTask, Trajectory, TaskType

    # Mapping from strings to types
    export string_to_feedback_type, string_to_task_type
end

module MDPs
    using ..Types
    include("mdp.jl")
    export MDP, mdp_factory, create_random_MPD, create_simple_MDP, create_deepsea_MDP, create_circular_MDP, create_grid_MDP
    export sample_next_state, get_reward, simulate_trajectory, simulate_trajectory_to_goal, get_reachable_states, get_reachable_state_action_pairs
end

module Utils
    using ..Types  # Import Types module to access its types
    using ..MDPs   # Import MDPs module to access its types
    include("utils/visualization.jl")
    include("utils/db.jl")
    include("utils/postprocessing.jl")
    include("utils/logging.jl")
    include("utils/math.jl")
    include("utils/qvalue.jl")
    include("utils/prior.jl")
end

module Measures
    using ..Types  # Import Types module to access its types
    using ..MDPs   # Import MDPs module to access its types
    using ..Utils
    include("measures/utils.jl")
    include("measures/distribution_mismatch_coefficient.jl")
    include("measures/diameter.jl")
    include("measures/informativeness_measures.jl")
end

module Models
    using ..Types  # Import Types module to access its types
    using ..Utils
    using ..MDPs
    include("models/models.jl")
    include("models/choices.jl")
end

# Re-export everything from submodules
using .Types
using .MDPs
using .Models
using .Utils
using .Measures

# Export main function
export main

end # module 