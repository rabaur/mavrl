"""
Merge shard JSON files from distributed `run_grid_experiments.jl` runs into one object.

Usage:
    julia --project=. scripts/merge_grid_mcmc_results.jl shard1.json shard2.json ... -o combined.json
    julia --project=. scripts/merge_grid_mcmc_results.jl results/grid_mcmc_shards/*.json -o combined.json
"""

using JSON

function main()
    args = ARGS
    out_idx = findfirst(x -> x == "-o" || x == "--output", args)
    out_idx === nothing && error("Pass output path with -o or --output combined.json")
    out_path = args[out_idx + 1]
    input_paths = collect(args[1:out_idx-1])
    isempty(input_paths) && error("Pass at least one input JSON shard")

    merged = Dict{String, Any}()
    for path in input_paths
        data = open(JSON.parse, path)
        data isa Dict{String, Any} || error("Expected object at top level in $path")
        for (k, v) in data
            if haskey(merged, k)
                @warn "Duplicate key $k — keeping first occurrence" path
            else
                merged[k] = v
            end
        end
    end

    open(out_path, "w") do io
        JSON.print(io, merged, 2)
    end
    @info "Wrote $(length(merged)) entries to $out_path"
end

main()
