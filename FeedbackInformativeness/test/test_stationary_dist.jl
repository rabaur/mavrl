using Test

# Import the functions we need from the module
using FeedbackInformativeness.MDPs: MDP
using FeedbackInformativeness.Measures: stationary_dist_exact, stationary_dist_solve

@testset "Stationary distribution tests" begin
    @testset "Circular MDP with uniform policy" begin
        # Create a circular MDP with 5 states and 2 actions
        nS = 5
        nA = 2
        γ = 0.9
        
        # Transition probabilities
        P = zeros(nS, nA, nS)
        
        # Action 1: clockwise (state i -> state i+1 mod nS)
        for i in 1:nS
            next_state = mod(i, nS) + 1
            P[i, 1, next_state] = 1.0
        end
        
        # Action 2: counterclockwise (state i -> state i-1 mod nS)
        for i in 1:nS
            prev_state = mod(i - 2, nS) + 1
            P[i, 2, prev_state] = 1.0
        end
        
        # Rewards (not important for stationary distribution)
        R = zeros(nS, nA)
        
        # Create MDP
        mdp = MDP(nS, nA, "circular", P, R, -1, Dict{Int, Vector{Tuple{Int, Int}}}(), [1], γ)
        
        # Create uniform policy (equal probability for both actions)
        π = fill(0.5, nS, nA)
        
        # Compute stationary distribution
        μ_exact = stationary_dist_exact(mdp, π)
        μ_solve = stationary_dist_solve(mdp, π)
        
        # Expected stationary distribution: uniform [0.2, 0.2, 0.2, 0.2, 0.2]
        expected_μ = fill(0.2, nS)
        
        @test size(μ_exact) == (nS,)
        @test size(μ_solve) == (nS,)
        @test all(isapprox.(μ_exact, expected_μ, atol=1e-6))
        @test all(isapprox.(μ_solve, expected_μ, atol=1e-6))
        @test isapprox(sum(μ_exact), 1.0, atol=1e-6)  # Should sum to 1
        @test isapprox(sum(μ_solve), 1.0, atol=1e-6)  # Should sum to 1
    end

    @testset "complex MDP" begin
        # Create MDP from Seabrook and Wiskott (2023)
        Pπ = zeros(6, 6)

        # Outgoing transitions from state 1
        Pπ[1, 4] = 0.3
        Pπ[1, 2] = 0.4
        Pπ[1, 3] = 0.3

        # Outgoing transitions from state 2
        Pπ[2, 1] = 0.5
        Pπ[2, 3] = 0.5

        # Outgoing transitions from state 3
        Pπ[3, 1] = 0.6
        Pπ[3, 2] = 0.4

        # Outgoing transitions from state 4
        Pπ[4, 5] = 0.5
        Pπ[4, 6] = 0.5

        # Outgoing transitions from state 5
        Pπ[5, 4] = 0.4
        Pπ[5, 6] = 0.6

        # Outgoing transitions from state 6
        Pπ[6, 4] = 0.8
        Pπ[6, 2] = 0.1
        Pπ[6, 5] = 0.1

        μ_exact = round.(stationary_dist_exact(Pπ), digits=2)
        μ_solve = round.(stationary_dist_solve(Pπ), digits=2)

        @test size(μ_exact) == (6,)
        @test size(μ_solve) == (6,)
        @test all(isapprox.(μ_exact, [0.09, 0.09, 0.07, 0.31, 0.18, 0.26], atol=1e-6))
        @test all(isapprox.(μ_solve, [0.09, 0.09, 0.07, 0.31, 0.18, 0.26], atol=1e-6))
        @test isapprox(sum(μ_exact), 1.0, atol=1e-6)  # Should sum to 1
        @test isapprox(sum(μ_solve), 1.0, atol=1e-6)  # Should sum to 1
    end
end 