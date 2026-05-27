using ..MDPs: MDP
using ..Utils: Q_opt

"""
Computes the hitting time to a set of target states.
"""
function expected_hitting_time(P::Array{Float64, 3}, π::Matrix{Float64}, target_set::Vector{Int})
    nS, _, _ = size(P)
    @assert length(target_set) == length(Set(target_set)) "Indicies in target_set must be unique."
    @assert all(target_set .<= nS) "Indicies in target_set must be less than or equal to the number of states."
    target_set = sort(target_set)

    # Construct permutation of columns such that target_set is the last columns.
    perm = collect(1:nS)
    for target_state in target_set
        filter!(s -> s ≠ target_state, perm)
        push!(perm, target_state)
    end

    # Compute state-state transition kernel based on π.
    T = π_marginalized_transitions(P, π)

    # Permute columns of P_π such that target_set is the last columns.
    T_perm = T[perm, perm]

    # Number of target states.
    n_targ = length(target_set)
    n_from = nS - n_targ

    # Make the last states absorbing:
    T_perm[end-n_targ+1:end, :] .= 0  # Set all columns to 0 for last num_to rows
    T_perm[end-n_targ+1:end, end-n_targ+1:end] .= Matrix{Float64}(I, n_targ, n_targ)  # Set identity matrix

    # Extract submatrix of T_perm corresponding to non-target states.
    T_from = T_perm[1:n_from, 1:n_from]

    # Solve (I - T_from)h = 1 for hitting times h.
    h_sub = (I - T_from) \ ones(n_from)

    # Build full hitting time vector, setting last n_targ entries to 0
    # (since hitting time to target states is 0).
    h_perm = vcat(h_sub, zeros(n_targ))

    # Revert permutation of hitting times.
    inv_perm = sortperm(perm)
    h_full = h_perm[inv_perm]

    return h_full
end

"""
Computes the diameter of the MDP.
The diameter is defined as:
D ≔ sup_{s₁ ≠ s₂} inf_π 𝔼[H(s₁, s₂)]
where H(s₁, s₂) is the hitting time from s₁ to s₂ when following policy π.

We use a trick suggested in Conserva and Rauber (2022).
For each target state sₜ, we adapt the reward of the MDP to be:
R(s, a) = -𝟙{s ≠ sₜ}
i.e., the reward is -1 if the state is not the target state, and 0 otherwise.

We solve the MDP with the adapted rewards for each target state sₜ
and compute the hitting time to the target state, then take the maximum.
"""
function diameter(mdp::MDP)
    nS, nA = mdp.nS, mdp.nA
    P = mdp.P
    diameter = 0

    for sₜ in 1:nS
        # Create augmented reward matrix.
        R = fill(-1.0, nS, nA)
        R[sₜ, :] .= 0.0

        # Get optimal Q-values
        Q = Q_opt(mdp, R; max_iter=10000)
        π = zeros(nS, nA)
    
        # Compute optimal policy from Q values
        for s in 1:nS
            π[s, argmax(Q[s, :])] = 1.0
        end

        # Get hitting times
        H = expected_hitting_time(P, π, [sₜ])

        # Worst-case hitting time
        max_H = maximum(H)

        # Update diameter.
        diameter = max(diameter, max_H)
    end

    return diameter
end