using SQLite
using DataFrames
using JSON
using LinearAlgebra
using UUIDs
using Dates

"""
Retry a function with exponential backoff.

Parameters:
- f: Function to retry
- max_attempts: Maximum number of retry attempts (default: 5)
- initial_delay: Initial delay in seconds before first retry (default: 0.1)
- max_delay: Maximum delay in seconds between retries (default: 2.0)
"""
function retry_with_backoff(f::Function; max_attempts::Int=5, initial_delay::Float64=0.1, max_delay::Float64=2.0)
    attempt = 1
    delay = initial_delay
    
    while attempt <= max_attempts
        try
            return f()
        catch e
            if !(e isa SQLite.SQLiteException && e.msg == "database is locked") || attempt == max_attempts
                rethrow(e)
            end
            
            @warn "Database locked, retrying in $(delay) seconds (attempt $(attempt)/$(max_attempts))"
            sleep(delay)
            delay = min(delay * 2, max_delay)  # Exponential backoff with cap
            attempt += 1
        end
    end
    
    error("Failed after $(max_attempts) attempts")
end

"""
Generic upsert: given a table name and a Dict or NamedTuple of column=>value,
build an `INSERT … ON CONFLICT(key) DO UPDATE` automatically.
"""
function upsert!(db::SQLite.DB, table::AbstractString, data::AbstractDict,
                 conflict_key::AbstractString)
    retry_with_backoff() do
        cols = collect(keys(data))
        placeholders = fill("?", length(cols))
        placeholders_str = join(placeholders, ", ")
        col_list    = join(cols, ", ")
        # build the "col = excluded.col" list for the DO UPDATE
        update_list = join([c * " = excluded." * c for c in cols], ", ")
        sql = """
        INSERT INTO $table ($col_list)
             VALUES ($placeholders_str)
        ON CONFLICT($conflict_key) DO UPDATE SET
             $update_list
        """
        stmt = SQLite.Stmt(db, sql)
        SQLite.execute(stmt, Tuple(values(data)))
        return nothing
    end
end

"""
Initialize the SQLite database with necessary tables if it doesn't exist.
Otherwise, just return a connection to the existing database.
"""
function init_database(db_path)
    # Create directory if it doesn't exist
    mkpath(dirname(db_path))
    
    # Check if database exists
    db_exists = isfile(db_path)
    db = SQLite.DB(db_path)

    # Only create tables if database is new
    if !db_exists
        # Read experiment table schema from file
        experiment_table_schema = read(joinpath(dirname(db_path), "schema", "create_experiments.sql"), String)
        
        # Create experiments table with MDP fields and UUID as primary key
        SQLite.execute(db, experiment_table_schema)

        # Read results table schema from file
        results_table_schema = read(joinpath(dirname(db_path), "schema", "create_results.sql"), String)
        
        # Create results table with foreign key to UUID
        SQLite.execute(db, results_table_schema)
    end
    
    return db
end

"""
Save experiment results to database.

Parameters:
- db: SQLite database connection
- args: Dictionary of command-line arguments
- mdp: MDP instance
- metrics: For each feedback type, a dictionary of metrics
"""
function save_results(db,
    args::Dict{String, Any},
    mdp::MDP,
    metrics::Dict{String, Dict{String, Any}},
    mutual_info_posterior::Dict{String, Float64}
)
    # Calculate runtime
    runtime_seconds = args["runtime_seconds"]

    # Pack up all experiment fields into a single Dict
    # Start with a copy of args and add the computed fields
    exp_data = copy(args)
    exp_data["timestamp"] = string(now())
    exp_data["runtime_seconds"] = runtime_seconds
    exp_data["transition_matrix"] = JSON.json(Array(mdp.P))
    exp_data["true_reward_matrix"] = JSON.json(Array(mdp.R))
    exp_data["mutual_info_posterior"] = JSON.json(mutual_info_posterior)
    # Delete run_feedback_types from exp_data
    delete!(exp_data, "run_feedback_types")
    delete!(exp_data, "database_path")
    delete!(exp_data, "plot")

    # Upsert experiment data
    @info "Upserting experiment data"
    upsert!(db, "experiments", exp_data, "trial_id")
    
    # Insert preference results
    for (feedback_type_string, metrics) in metrics
        res_data = copy(metrics)
        res_data["trial_id"] = args["trial_id"]
        res_data["feedback_type"] = feedback_type_string
        res_data["posterior_mean"] = JSON.json(Array(metrics["posterior_mean"]))
        res_data["posterior_median"] = JSON.json(Array(metrics["posterior_median"]))
        upsert!(db, "results", res_data, "trial_id,feedback_type")
    end
end
