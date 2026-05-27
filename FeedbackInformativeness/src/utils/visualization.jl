using Plots
using StatsPlots

"""
Plot posterior distributions for rewards and β.
Saves figures to disk with a specified prefix (e.g., "preferences" or "demonstrations").

Parameters:
- rewards: Matrix of size [num_samples, num_reward_parameters]
- β_samples: Vector of size num_samples
- true_reward: True reward vector/matrix
- true_β: True rationality coefficient
- prefix: String prefix for saved files
"""
function plot_results(rewards, β_samples, true_reward, true_β; prefix="results", path="plots")
    # Combine rewards and β_samples into one matrix for plotting
    params = hcat(rewards, β_samples)
    n_params = size(params, 2)  # Total number of parameters (rewards + β)
    
    # Grid of subplots
    ps = []
    for i in 1:n_params
        for j in 1:n_params
            if i == j
                # Diagonal: 1D histogram
                p = histogram(
                    params[:, i],
                    title=i == n_params ? "β" : "r$i",
                    legend=false
                )
                # Draw vertical line at true value
                if i < n_params
                    # i-th reward dimension
                    true_val = vec(true_reward)[i]
                    vline!([true_val], color=:red, linestyle=:dash)
                else
                    # β
                    vline!([true_β], color=:red, linestyle=:dash)
                end
            else
                # Off-diagonal: 2D histogram
                p = histogram2d(
                    params[:, i],
                    params[:, j],
                    xlabel=i == n_params ? "β" : "r$i",
                    ylabel=j == n_params ? "β" : "r$j",
                    nbins=30,
                    legend=false,
                    color=:viridis
                )
            end
            push!(ps, p)
        end
    end
    
    # Combine all pairwise plots and save
    p_pairs = plot(
        ps...,
        layout=(n_params, n_params),
        size=(200*n_params, 200*n_params),
        plot_title="Posterior Distributions - $(prefix)"
    )
    savefig(p_pairs, "$(path)/posterior_pairs_$(prefix).png")
    
    return p_pairs
end

"""
Plot KDE distributions for each parameter, overlaying different feedback types.
Saves figures to disk with a specified prefix.

Parameters:
- rewards_dict: Dictionary mapping feedback type names to reward matrices
- β_dict: Dictionary mapping feedback type names to β samples
- true_reward: True reward vector/matrix
- true_β: True rationality coefficient
- prefix: String prefix for saved files
"""
function plot_feedback_type_comparison(r_dict, β_dict, mdp, true_β; prefix="results", path="plots")
    plots = []
    color_mapping = Dict(
        "preference" => "#2E54D3",
        "demonstration-naive" => "#F2A34E",
        "demonstration-qwalk" => "#4D9C6A"
    )
    r_reshaped_dict = Dict()
    num_samples = 0
    try
        num_samples = size(r_dict["preference"], 1)
    catch
        num_samples = size(r_dict["demonstration-naive"], 1)
    end
    for (feedback_type, rewards) in r_dict
        r_reshaped = zeros(num_samples, mdp.nS, mdp.nA)
        for i in 1:mdp.num_r_params
            for (s, a) in mdp.param_idx_mapping[i]
                r_reshaped[:, s, a] = rewards[:, i]
            end
        end
        r_reshaped_dict[feedback_type] = r_reshaped
    end
    for s in 1:mdp.nS
        for a in 1:mdp.nA
            p = vline([mdp.R[s, a]], color="black", linewidth=5)
            for (feedback_type, r_reshaped) in r_reshaped_dict
                histogram!(
                    p,
                    r_reshaped[:, s, a],
                    color=color_mapping[feedback_type],
                    alpha=0.5,
                    legend=false,
                    normalize=:pdf
                )
                post_mean = mean(r_reshaped[:, s, a])
                vline!(p, [post_mean], label="μ$(first(feedback_type))", color=color_mapping[feedback_type], linewidth=5)
                ylims!(p, (0, 2))
            end
            push!(plots, p)
        end
    end
    plot_all = plot(plots..., layout=(mdp.nS, mdp.nA), size=(300*mdp.nA, 200*mdp.nS))
    savefig(plot_all, "$(path)/feedback_type_comparison_$(prefix).png")
end