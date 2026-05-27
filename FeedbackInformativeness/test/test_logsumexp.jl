using Test

# Import the functions we need from the module
using FeedbackInformativeness.Utils: softmax_logsumexp, softmax

@testset "Softmax" begin
    @testset "Log-sum-exp trick" begin
        x = randn(10, 10)
        β = 1.0
        @test all(isapprox.(softmax_logsumexp(x, β), softmax(x), atol=1e-6))
    end
end