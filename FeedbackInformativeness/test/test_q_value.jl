using Test

# Import the functions we need from the module
using FeedbackInformativeness.MDPs: MDP
using FeedbackInformativeness.Utils: Q_opt, soft_Q_opt

@testset "Q-value function tests" begin
    @testset "Trivial single-state MDP" begin
        # Create a trivial single-state MDP
        nS = 1
        nA = 3
        γ = 0.9
        
        # Transition probabilities: stay in the same state
        P = zeros(nS, nA, nS)
        for a in 1:nA
            P[1, a, 1] = 1.0
        end
        
        # Rewards for each action
        R = zeros(1, 3)
        R[1, 1] = 1
        R[1, 2] = 2
        R[1, 3] = 3 # Optimal action
        
        # Some parameters are not meaningful for this MDP
        mdp = MDP(nS, nA, "trivial", P, R, -1, Dict{Int, Vector{Tuple{Int, Int}}}(), [1], γ)
        
        # Compute Q-values
        our_Q_opt = Q_opt(mdp, R; max_iter=1000)
        our_Q_soft = soft_Q_opt(mdp, R, α=0.05)
        
        Q_opt_expected = zeros(1, 3)

        # For the optimal action, it is simply 3 / (1 - γ) (by infinite geometric series)
        Q_opt_expected[1, 3] = 3 / (1 - γ)

        # For the other actions, we loose out on the first optimal reward, but then act optimally (action=3) after
        Q_opt_expected[1, 1] = (1 - 3) + Q_opt_expected[1, 3]
        Q_opt_expected[1, 2] = (2 - 3) + Q_opt_expected[1, 3]

        @test size(our_Q_opt) == size(R)
        @test size(our_Q_soft) == size(R)
        @test all(isapprox.(our_Q_opt, Q_opt_expected, atol=1e-5))
        @test all(isapprox.(our_Q_soft, Q_opt_expected, atol=1e-5))
    end

    @testset "Two-state deterministic MDP 1" begin
        # Create a two-state deterministic MDP
        nS = 2
        nA = 1
        γ = 0.9
        r = 1.0  # Reward for transitioning from s0 to s1
        
        # Transition probabilities
        P = zeros(nS, nA, nS)
        P[1, 1, 2] = 1.0  # From s0, action moves to s1
        P[2, 1, 2] = 1.0  # s1 is absorbing
        
        # Rewards
        R = zeros(nS, nA)
        R[1, 1] = r  # Only reward when transitioning from s0 to s1
        
        # Some parameters are not meaningful for this MDP
        mdp = MDP(nS, nA, "two-state", P, R, -1, Dict{Int, Vector{Tuple{Int, Int}}}(), [1], γ)
        
        # Compute Q-values
        Q = Q_opt(mdp, R)
        Q_soft = soft_Q_opt(mdp, R, α=0.05)
        
        # Expected Q-values:
        # Q*(s0,a) = r
        # Q*(s1,a) = 0
        expected_Q = zeros(nS, nA)
        expected_Q[1, 1] = r
        
        @test size(Q) == size(R)
        @test size(Q_soft) == size(R)
        @test all(isapprox.(Q, expected_Q, atol=1e-6))
        @test all(isapprox.(Q_soft, expected_Q, atol=1e-6))
    end

    @testset "Two-state deterministic MDP 2" begin
        # Create a two-state deterministic MDP
        nS = 2
        nA = 1
        γ = 0.9
        r1 = 1.0
        r2 = 2.0

        P = zeros(nS, nA, nS)
        P[1, 1, 2] = 1.0  # From s0, action moves to s1
        P[2, 1, 1] = 1.0  # From s1, action moves to s0
        
        R = zeros(nS, nA)
        R[1, 1] = r1
        R[2, 1] = r2
        
        # Some parameters are not meaningful for this MDP
        mdp = MDP(nS, nA, "two-state", P, R, -1, Dict{Int, Vector{Tuple{Int, Int}}}(), [1], γ)
        
        # Compute Q-values
        Q = Q_opt(mdp, R)
        Q_soft = soft_Q_opt(mdp, R, α=0.05)

        # Expected Q-values:
        # Q*(s0,a) = r1 + γ⋅r2 + γ²⋅r1 + ... = ∑ γ^(2i)⋅r1 + ∑ γ^(2i+1)⋅r2 = (-r1 / (y² - 1)) + (-r2⋅γ / (y² - 1))
        # Q*(s1,a) = r2 + γ⋅r1 + γ²⋅r2 + ... = ∑ γ^(2i)⋅r2 + ∑ γ^(2i+1)⋅r1 = (-r2 / (y² - 1)) + (-r1⋅γ / (y² - 1))
        expected_Q = zeros(nS, nA)
        expected_Q[1, 1] = (-r1 / (γ^2 - 1)) + (-r2*γ / (γ^2 - 1))
        expected_Q[2, 1] = (-r2 / (γ^2 - 1)) + (-r1*γ / (γ^2 - 1))
        
        @test size(Q) == size(R)
        @test size(Q_soft) == size(R)
        @test all(isapprox.(Q, expected_Q, atol=1e-5))
        @test all(isapprox.(Q_soft, expected_Q, atol=1e-5))
    end
    
    
end 