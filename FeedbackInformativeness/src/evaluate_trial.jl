using Distributed
using ArgParse
using UUIDs

# Add workers and make sure they have access to the module
addprocs(16, exeflags="--project=$(Base.active_project())")
@everywhere using FeedbackInformativeness: Types, Utils, MDPs, Measures, Models, Turing, Turing.ADTypes, Bijectors

"""
Parse command-line arguments.
"""
function parse_arguments()
    argparse_settings = ArgParseSettings()

    # Parameters common among all MDPs
    @add_arg_table argparse_settings begin
        "--num_steps"
            help = "Number of steps in the trajectory"
            default = 5
            arg_type = Int64
        "--num_feedback_samples"
            help = "Number of feedback samples"
            default = 1
            arg_type = Int64
        "--num_mcmc_samples"
            help = "Number of MCMC samples"
            default = 10000
            arg_type = Int64
        "--mdp_type"
            help = "Type of MDP to use (random, deepsea, tree)"
            default = "random"
            arg_type = String
        "--database_path"
            help = "Path to the database"
            default = "data/results.db"
            arg_type = String
        "--trial_id"
            help = "Trial ID"
            default = string(uuid4())
            arg_type = String
        "--run_feedback_types"
            help = "Feedback types to run (preference, demonstration, qwalk)"
            nargs = '+'  # Accept one or more arguments
            arg_type = String
            default = ["preference", "demonstration-naive"]
        "--plot"
            help = "Whether to plot results"
            action = :store_true
        "--true_beta"
            help = "True rationality coefficient"
            default = 1.0
            arg_type = Float64
        "--infer_beta"
            help = "Whether to infer the rationality coefficient"
            action = :store_true
        "--discount_factor"
            help = "Discount factor"
            default = 0.9
            arg_type = Float64
        "--task_type"
            help = "Type of task to use (episodic, finite)"
            default = "episodic"
            arg_type = String
    end

    # Add MDP-specific arguments
    @add_arg_table argparse_settings begin
        # Common parameters for random and simple MDPs
        "--num_states"
            help = "Number of states (for random and simple MDPs)"
            default = 4
            arg_type = Int64
        "--num_actions"
            help = "Number of actions (for random and simple MDPs)"
            default = 2
            arg_type = Int64
        "--transition_alpha"
            help = "Transition factor (for random MDPs)"
            default = 1.0
            arg_type = Float64
        "--R_sparseness"
            help = "Sparseness parameter for rewards (for random MDPs)"
            default = 0.5
            arg_type = Float64
    end

    # Add DeepSea-specific parameters
    @add_arg_table argparse_settings begin
        "--size"
            help = "Size of the DeepSea MDP"
            default = 4
            arg_type = Int64
        "--p_rand"
            help = "Random action probability for DeepSea MDP"
            default = 0.1
            arg_type = Float64
    end

    # Add "circular"-specific parameters
    @add_arg_table argparse_settings begin
        "--cost_cycle"
            help = "Cost of cycle in the circular MDP"
            default = -0.1
            arg_type = Float64
        "--cycle_length"
            help = "Length of the cycle in the circular MDP"
            default = 2
            arg_type = Int64
    end

    # Add "random"-specific parameters
    @add_arg_table argparse_settings begin
        "--num_clusters"
            help = "Number of clusters in the random MDP"
            default = 5
            arg_type = Int64
        "--cluster_props_decay"
            help = "Decay factor for cluster proportions in the random MDP"
            default = 0.5
            arg_type = Float64
        "--min_cluster_prop"
            help = "Minimum cluster proportion in the random MDP"
            default = 0.1
            arg_type = Float64
        "--within_cluster_concentration"
            help = "Concentration parameter for within-cluster Dirichlet distribution in the random MDP"
            default = 0.2
            arg_type = Float64
        "--between_cluster_concentration"
            help = "Concentration parameter for between-cluster Dirichlet distribution in the random MDP"
            default = 0.01
            arg_type = Float64
        "--reward_cluster_mean_decay"
            help = "Decay factor for reward means across clusters in the random MDP"
            default = 0.5
            arg_type = Float64
        "--reward_cluster_mean_min"
            help = "Minimum mean reward in any cluster in the random MDP"
            default = 0.0
            arg_type = Float64
        "--reward_within_cluster_variance"
            help = "Variance of rewards within clusters in the random MDP"
            default = 0.05
            arg_type = Float64
        "--has_self_loops"
            help = "Whether to allow self-loops in the random MDP"
            action = :store_true
    end

    # Add "tree"-specific parameters
    @add_arg_table argparse_settings begin
        "--n"
            help = "Number of levels in the tree MDP"
            default = 3
            arg_type = Int64
        "--k"
            help = "Number of actions per level in the tree MDP"
            default = 2
            arg_type = Int64
        "--p_undo"
            help = "Probability of undoing an action in the tree MDP"
            default = 0.1
            arg_type = Float64
        "--reward_decay"
            help = "Discount factor for rewards in the tree MDP"
            default = 0.5
            arg_type = Float64
        "--min_base_reward"
            help = "Minimum base reward in the tree MDP"
            default = 0.0
            arg_type = Float64
        "--reward_std"
            help = "Standard deviation of rewards in the tree MDP"
            default = 0.01
            arg_type = Float64
    end

    return parse_args(argparse_settings)
end

"""
Run inference using Variational Inference.
"""
function run_inference(model, num_samples=1000)
    # Use MCMC sampling
    samples = Turing.sample(model, Turing.NUTS(0.65), Turing.MCMCDistributed(), num_samples ÷ 16, 16)
    return samples
end

function main()
    args = parse_arguments()
    run_feedback_types = [Types.string_to_feedback_type[s] for s in args["run_feedback_types"]]

    # Initialize database if it doesn't exist
    db = Utils.init_database(args["database_path"])

    # Start timing
    start_time = time()

    # Create MDP based on type
    mdp = MDPs.mdp_factory(args)

    results = Dict{String, Dict{String, Any}}()
    reward_samples = Dict{String, Array{Float64, 2}}()
    β_samples_dict = Dict{String, Vector{Float64}}()

    for feedback_type in run_feedback_types
        @info "Running feedback type: $feedback_type"
        
        # Generate feedback data and choices
        choice_sets, choices = Models.generate_choices(
            feedback_type,
            mdp,
            args["num_feedback_samples"],
            args["num_steps"],
            args["true_beta"];
            task_type=Types.string_to_task_type[args["task_type"]]
        )

        # Build and run the Turing model for feedback
        model = Models.feedback_model(
                feedback_type,
                Types.string_to_task_type[args["task_type"]],
                choice_sets,
                choices,
                mdp,
                num_steps = args["num_steps"],
                true_β = args["infer_beta"] ? nothing : args["true_beta"]
        )

        inference_results = run_inference(
            model, 
            args["num_mcmc_samples"]
        )
        samples = inference_results
        rewards, β_samples = Utils.extract_samples(samples, feedback_type, mdp)

        reward_samples[string(feedback_type)] = rewards
        β_samples_dict[string(feedback_type)] = β_samples

        # Store metrics
        measures = Measures.compute_posterior_measures(rewards, β_samples, args["true_beta"], mdp)
        results[string(feedback_type)] = measures
    end

    # Plot comparison of all feedback types
    if args["plot"]
        Utils.plot_feedback_type_comparison(
            reward_samples,
            β_samples_dict,
            mdp,
            args["true_beta"];
            prefix="comparison"
        )
    end

    # For each feedback type, compute the mutual information between posteriors.
    # This is stored in the experiment table.
    mutual_info_posterior = Dict{String, Float64}()
    for (i, feedback_type) in enumerate(run_feedback_types)
        for (j, other_feedback_type) in enumerate(run_feedback_types)
            if j > i
                mi = Measures.mutual_information(reward_samples[string(feedback_type)], reward_samples[string(other_feedback_type)])
                mutual_info_posterior["$(string(feedback_type))_$(string(other_feedback_type))"] = mi
            end
        end
    end
    
    # Calculate and add runtime to args
    args["runtime_seconds"] = time() - start_time

    # Save results to database
    @info "Saving results to database"
    Utils.save_results(db, args, mdp, results, mutual_info_posterior)

    Utils.print_metrics(results, mutual_info_posterior)
end

main()