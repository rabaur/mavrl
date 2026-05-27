using Printf

"""
Print metrics for a specific feedback type.
"""
function print_feedback_metrics(metrics)
    @printf("%-30s %.4f\n", "Info Gain r (Prior-Posterior):", metrics["r_information_gain"])
    @printf("%-30s %.4f\n", "Info Gain joint (Prior-Posterior):", metrics["joint_information_gain"])
    @printf("%-30s %.4f ± %.4f\n", "L2 Error:", metrics["l2_error_mean"], metrics["l2_error_std"])
    @printf("%-30s %.4f ± %.4f\n", "Cosine Similarity:", metrics["cosine_similarity_mean"], metrics["cosine_similarity_std"])
    @printf("%-30s %.4f ± %.4f\n", "Expected Regret:", metrics["regret_mean"], metrics["regret_std"])
    @printf("%-30s %.4f\n", "β Estimate:", metrics["beta_est"])
    @printf("%-30s %.4f\n", "β Error:", metrics["beta_err"])
end

"""
Print all metrics from an experiment.

Parameters:
- results: Dictionary mapping feedback type strings to their metrics
- mutual_info_posterior: Dictionary mapping pairs of feedback types to their mutual information
- spec_gap: Spectral gap of the MDP
"""
function print_metrics(results::Dict{String, Dict{String, Any}}, mutual_info_posterior::Dict{String, Float64})
    println("\n" * "="^50)
    println(" " * "EVALUATION METRICS")
    println("="^50)
    
    # Print metrics for each feedback type
    for (feedback_type, metrics) in results
        println("\n$(titlecase(feedback_type)) Metrics:")
        println("-"^(length(feedback_type) + 8))
        print_feedback_metrics(metrics)
    end
    
    # Print mutual information between posteriors
    if !isempty(mutual_info_posterior)
        println("\nMutual Information Between Posteriors:")
        println("-"^35)
        for (pair, mi) in mutual_info_posterior
            type1, type2 = split(pair, "_")
            @printf("%-30s %.4f\n", "$(titlecase(type1)) vs $(titlecase(type2)):", mi)
        end
    end
end
