"""
MCMC baseline experiments on 10×10 grid MDPs.

Runs NUTS inference for each (reward_type, feedback_type) combination and
saves posterior metrics (L2 error, cosine similarity, regret) to a JSON file.

Usage (full grid, local):
    julia -O3 --project=. scripts/run_grid_experiments.jl [--num_chains 4] [--samples_per_chain 500] [--seed 42]

    (-O3 is optional; pass it on the `julia` command for aggressive compiler optimizations. With
    num_chains>1, worker processes are spawned from the same Julia binary and typically inherit -O3.)

Single task (cluster — one job per environment × feedback type):
    julia --project=. scripts/run_grid_experiments.jl --task_index 0 [--num_chains 1] [--output results/shard_0.json]

    julia --project=. scripts/run_grid_experiments.jl --reward_type sparse --feedback_type preference -o results/shard.json

List task indices (0-based):
    julia --project=. scripts/run_grid_experiments.jl --list_tasks

Merge shards:
    julia --project=. scripts/merge_grid_mcmc_results.jl results/shards/*.json -o results/grid_mcmc_merged.json
"""

using Distributed

function parse_int_flag(name::String, default::Int)
    idx = findfirst(==("--$name"), ARGS)
    idx !== nothing && idx < length(ARGS) ? parse(Int, ARGS[idx + 1]) : default
end

function parse_string_flag(name::String)
    idx = findfirst(==("--$name"), ARGS)
    idx !== nothing && idx < length(ARGS) ? String(ARGS[idx + 1]) : nothing
end

has_flag(name::String) = findfirst(==("--$name"), ARGS) !== nothing

const NUM_CHAINS = parse_int_flag("num_chains", 4)
const SAMPLES_PER_CHAIN = parse_int_flag("samples_per_chain", 500)
const SEED = parse_int_flag("seed", 42)

if nworkers() < NUM_CHAINS
    addprocs(NUM_CHAINS - nworkers(), exeflags="--project=$(Base.active_project())")
end
@everywhere using FeedbackInformativeness: Types, Utils, MDPs, Measures, Models, Turing

using Random
using JSON
using Dates
using Printf

# ── Normalization values ──────────────────────────────────────────────────
const NORM_VALUES_PATH = joinpath(@__DIR__, "..", "..", "results", "normalization_values.json")
const NORM_VALUES = let
    open(NORM_VALUES_PATH, "r") do io
        JSON.parse(io)
    end
end

function normalize_value(raw::Float64, optimal::Float64, uniform::Float64)::Float64
    100.0 * (raw - uniform) / (optimal - uniform)
end

# ── Configuration ──────────────────────────────────────────────────────────

const REWARD_TYPES = ["sparse", "cliff", "trap"]
const FEEDBACK_TYPES = [
    Types.PreferenceFeedback(),
    Types.NaiveDemonstrationFeedback(),
    Types.StopFeedback(),
    Types.RatingFeedback(),
    Types.CombinedFeedback(),
]

const GRID_SIZE        = 10
const GAMMA            = 0.95
const P_RAND           = 0.0
const NUM_STEPS        = 10   # segment / trajectory length
const TRUE_BETA        = 5.0
const NUM_PREF_SAMPLES = 64
const NUM_DEMO_SAMPLES = 1
const NUM_STOP_SAMPLES = 64
const NUM_RATING_SAMPLES = 64

function num_feedback_samples(ft::Types.FeedbackType)
    ft isa Types.PreferenceFeedback          && return NUM_PREF_SAMPLES
    ft isa Types.NaiveDemonstrationFeedback  && return NUM_DEMO_SAMPLES
    ft isa Types.StopFeedback                && return NUM_STOP_SAMPLES
    ft isa Types.RatingFeedback              && return NUM_RATING_SAMPLES
    ft isa Types.CombinedFeedback            && return 0  # each modality uses its own count
    error("Unknown feedback type: $ft")
end

const TASK_SPECS = [(r, f) for r in REWARD_TYPES for f in FEEDBACK_TYPES]
const NUM_TASKS = length(TASK_SPECS)

function parse_feedback_type_arg(s::String)::Types.FeedbackType
    haskey(Types.string_to_feedback_type, s) || error("Unknown --feedback_type $(repr(s)). Valid: $(collect(keys(Types.string_to_feedback_type)))")
    return Types.string_to_feedback_type[s]
end

function resolve_single_task()
    if has_flag("list_tasks")
        return nothing
    end
    ti = parse_string_flag("task_index")
    if ti !== nothing
        idx = parse(Int, ti)
        (0 <= idx < NUM_TASKS) || error("--task_index must be in 0:$(NUM_TASKS-1)")
        return TASK_SPECS[idx + 1]
    end
    rr = parse_string_flag("reward_type")
    ff = parse_string_flag("feedback_type")
    if rr !== nothing && ff !== nothing
        rr in REWARD_TYPES || error("--reward_type must be one of: $(join(REWARD_TYPES, ", "))")
        return (rr, parse_feedback_type_arg(ff))
    end
    if rr !== nothing || ff !== nothing
        error("Provide both --reward_type and --feedback_type, or use --task_index")
    end
    return nothing
end

# ── Inference helper ──────────────────────────────────────────────────────

const MH_SAMPLES_PER_CHAIN = 5000

function needs_likelihood_only_sampler(ft::Types.FeedbackType)
    ft isa Types.NaiveDemonstrationFeedback ||
    ft isa Types.StopFeedback ||
    ft isa Types.CombinedFeedback
end

function run_inference(model, feedback_type::Types.FeedbackType;
                       num_chains=NUM_CHAINS, samples_per_chain=SAMPLES_PER_CHAIN)
    if needs_likelihood_only_sampler(feedback_type)
        sampler = Turing.MH()
        n = MH_SAMPLES_PER_CHAIN
        @info "Using MH (gradient-free) — Q-value iteration in likelihood"
    else
        sampler = Turing.NUTS(0.65)
        n = samples_per_chain
    end
    if num_chains > 1
        return Turing.sample(model, sampler, Turing.MCMCDistributed(),
                             n, num_chains)
    else
        return Turing.sample(model, sampler, n)
    end
end

# ── State visitation diagnostic ──────────────────────────────────────────

const HEAT_CHARS = [' ', '░', '▒', '▓', '█']

function extract_all_trajectories(feedback_type::Types.FeedbackType, choice_sets, choices)
    trajs = Vector{Vector{Tuple{Int,Int}}}()
    if feedback_type isa Types.CombinedFeedback
        for (t1, t2) in choices.pref_choice_sets
            push!(trajs, t1); push!(trajs, t2)
        end
        append!(trajs, choices.demo_choices)
        append!(trajs, choices.stop_segments)
        append!(trajs, choices.rating_segments)
    elseif feedback_type isa Types.PreferenceFeedback
        for (t1, t2) in choice_sets
            push!(trajs, t1); push!(trajs, t2)
        end
    elseif feedback_type isa Types.NaiveDemonstrationFeedback
        append!(trajs, choices)
    elseif feedback_type isa Types.StopFeedback || feedback_type isa Types.RatingFeedback
        append!(trajs, choice_sets)
    end
    return trajs
end

function print_state_visitation(trajs, grid_size::Int)
    counts = zeros(Int, grid_size, grid_size)
    for traj in trajs
        for (s, _) in traj
            i = div(s - 1, grid_size) + 1
            j = mod(s - 1, grid_size) + 1
            counts[i, j] += 1
        end
    end
    max_c = maximum(counts)
    max_c == 0 && return
    io = IOBuffer()
    println(io, "    State visitation (max=$max_c):")
    for i in 1:grid_size
        print(io, "    ")
        for j in 1:grid_size
            level = clamp(round(Int, counts[i, j] / max_c * (length(HEAT_CHARS) - 1)) + 1, 1, length(HEAT_CHARS))
            print(io, HEAT_CHARS[level], HEAT_CHARS[level])
        end
        println(io)
    end
    @info String(take!(io))
end

# ── One experiment ───────────────────────────────────────────────────────

function run_single_experiment(reward_type::String, feedback_type::Types.FeedbackType)
    ft_str = string(feedback_type)
    key = "$(reward_type)_$(ft_str)"
    @info "━━━ Building grid MDP: $reward_type  |  feedback: $ft_str ━━━"

    mdp = MDPs.create_grid_MDP(GRID_SIZE, reward_type, GAMMA, P_RAND)
    n_samples = num_feedback_samples(feedback_type)
    goal_state = GRID_SIZE^2

    demo_kwargs = if feedback_type isa Types.DemonstrationFeedback
        (initial_state=1, goal_state=goal_state)
    elseif feedback_type isa Types.CombinedFeedback
        (goal_state=goal_state,)
    else
        (;)
    end

    t_data = @elapsed begin
        choice_sets, choices = Models.generate_choices(
            feedback_type,
            mdp,
            n_samples,
            NUM_STEPS,
            TRUE_BETA;
            task_type=Types.EpisodicTask(),
            demo_kwargs...
        )
    end
    @info "    Data generated in $(round(t_data; digits=1))s"
    trajs = extract_all_trajectories(feedback_type, choice_sets, choices)
    print_state_visitation(trajs, GRID_SIZE)

    true_β_for_model = feedback_type isa Types.RatingFeedback ? nothing : TRUE_BETA
    model = Models.feedback_model(
        feedback_type,
        Types.EpisodicTask(),
        choice_sets,
        choices,
        mdp;
        num_steps=NUM_STEPS,
        true_β=true_β_for_model
    )

    t_mcmc = @elapsed begin
        chain = run_inference(model, feedback_type)
    end
    @info "    MCMC finished in $(round(t_mcmc; digits=1))s"

    rewards, β_samples = Utils.extract_samples(chain, feedback_type, mdp)
    measures = Measures.compute_posterior_measures(rewards, β_samples, TRUE_BETA, mdp; compute_info_gain=false)

    env_id = "grid_$(reward_type)"
    nv = NORM_VALUES[env_id]
    norm_regret = normalize_value(
        measures["regret_mean"],
        nv["optimal_regret"],
        nv["uniform_regret"],
    )
    norm_disc_val = normalize_value(
        measures["discounted_value_mean"],
        nv["optimal_discounted_value"],
        nv["uniform_discounted_value"],
    )

    cov_matrix = measures["posterior_cov"]
    cov_lists = [cov_matrix[i, :] |> collect for i in 1:size(cov_matrix, 1)]

    entry = Dict(
        "reward_type"              => reward_type,
        "feedback_type"            => ft_str,
        "l2_error_mean"            => measures["l2_error_mean"],
        "l2_error_std"             => measures["l2_error_std"],
        "cosine_sim_mean"          => measures["cosine_similarity_mean"],
        "cosine_sim_std"           => measures["cosine_similarity_std"],
        "regret_mean"              => measures["regret_mean"],
        "regret_std"               => measures["regret_std"],
        "regret_normalized"        => norm_regret,
        "discounted_value_mean"    => measures["discounted_value_mean"],
        "discounted_value_normalized" => norm_disc_val,
        "beta_est"                 => measures["beta_est"],
        "beta_err"                 => measures["beta_err"],
        "r_info_gain"              => measures["r_information_gain"],
        "posterior_mean"           => collect(measures["posterior_mean"]),
        "posterior_std"            => collect(measures["posterior_std"]),
        "posterior_cov"            => cov_lists,
        "mcmc_time_s"              => t_mcmc,
        "num_chains"               => NUM_CHAINS,
        "samples_per_chain"        => SAMPLES_PER_CHAIN,
    )

    @info "    L2=$(round(measures["l2_error_mean"]; digits=3))  " *
          "cos=$(round(measures["cosine_similarity_mean"]; digits=3))  " *
          "regret=$(round(measures["regret_mean"]; digits=3))  " *
          "regret_norm=$(round(norm_regret; digits=1))%  " *
          "disc_val_norm=$(round(norm_disc_val; digits=1))%"

    return key => entry
end

sanitize_json(x) = x
sanitize_json(x::AbstractFloat) = isfinite(x) ? x : nothing
sanitize_json(v::AbstractVector) = [sanitize_json(e) for e in v]
sanitize_json(d::AbstractDict) = Dict(k => sanitize_json(v) for (k, v) in d)

function save_results(all_results, outfile)
    open(outfile, "w") do io
        JSON.print(io, sanitize_json(all_results), 2)
    end
end

function default_outfile(single::Bool)
    outdir = joinpath(@__DIR__, "..", "..", "results_camera_ready", "mcmc")
    mkpath(outdir)
    tag = Dates.format(now(), "yyyymmdd_HHMMSS")
    if single
        return joinpath(outdir, "grid_mcmc_shard_$(tag).json")
    else
        return joinpath(outdir, "grid_mcmc_results_$(tag).json")
    end
end

function print_task_list()
    print(stderr, "# NUM_TASKS=$(NUM_TASKS) (0-based --task_index)\n")
    for (i, (r, f)) in enumerate(TASK_SPECS)
        println("$(i-1)\t$(r)\t$(string(f))")
    end
end

function print_summary_table(all_results)
    println("\n", "="^110)
    println("SUMMARY  (regret_norm / value_norm: 100% = optimal, 0% = uniform random)")
    println("="^110)
    header = @sprintf("%-15s %-15s %8s %8s %10s %8s %8s %8s",
        "Reward", "Feedback", "L2↓", "Cos↑", "Regret↓", "Reg%↑", "Val%↑", "Time(s)")
    println(header)
    println("-"^110)
    for reward_type in REWARD_TYPES
        for feedback_type in FEEDBACK_TYPES
            key = "$(reward_type)_$(string(feedback_type))"
            r = all_results[key]
            println(@sprintf("%-15s %-15s %8.3f %8.3f %10.3f %7.1f%% %7.1f%% %8.1f",
                reward_type, r["feedback_type"],
                r["l2_error_mean"], r["cosine_sim_mean"],
                r["regret_mean"], r["regret_normalized"],
                r["discounted_value_normalized"], r["mcmc_time_s"]))
        end
    end
    println("="^110)
end

function main()
    if has_flag("list_tasks")
        print_task_list()
        return
    end

    Random.seed!(SEED)
    single = resolve_single_task()
    out_path = parse_string_flag("output")
    outfile = something(out_path, default_outfile(single !== nothing))

    if single === nothing
        all_results = Dict{String, Any}()
        for reward_type in REWARD_TYPES
            for feedback_type in FEEDBACK_TYPES
                key, entry = run_single_experiment(reward_type, feedback_type)
                all_results[key] = entry
                save_results(all_results, outfile)
                @info "    Intermediate results saved to $outfile"
            end
        end
        @info "All results saved to $outfile"
        print_summary_table(all_results)
    else
        reward_type, feedback_type = single
        key, entry = run_single_experiment(reward_type, feedback_type)
        save_results(Dict(key => entry), outfile)
        @info "Single-task result saved to $outfile"
    end
end

main()
