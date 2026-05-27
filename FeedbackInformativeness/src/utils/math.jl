"""
Numerically stable sigmoid function that works with ReverseDiff.TrackedReal values.
"""
function stable_sigmoid(x)
    if x >= 0
        z = exp(-x)
        return 1 / (1 + z)
    else
        z = exp(x)
        return z / (1 + z)
    end
end

"""
Computes the softmax of a matrix.
"""
function softmax(x; dims=2)
    x = x .- maximum(x, dims=dims)
    return exp.(x) ./ sum(exp.(x), dims=dims)
end

"""
Computes the log-sum-exp trick.
"""
function softmax_logsumexp(x, β)
    x_β = x .* β
    m = maximum(x_β, dims=2)
    x_β_m = x_β .- m
    return exp.(x_β_m .- log.(sum(exp.(x_β_m), dims=2)))
end