using Test

# Load the main module which includes all submodules
include("../src/FeedbackInformativeness.jl")
using .FeedbackInformativeness

# Include all test files
include("test_q_value.jl")
include("test_logsumexp.jl")
include("test_stationary_dist.jl")