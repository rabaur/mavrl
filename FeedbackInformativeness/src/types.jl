const Action = Int  # Only consider integer actions for now
const State = Int  # Only consider integer states for now
# Type alias for a trajectory (sequence of state-action pairs)
const Trajectory = Vector{Tuple{State, Action}}

# Abstract type for different feedback types
abstract type FeedbackType end
abstract type DemonstrationFeedback <: FeedbackType end

# Concrete feedback types
struct PreferenceFeedback <: FeedbackType end
struct QValueWalkDemonstrationFeedback <: DemonstrationFeedback end
struct NaiveDemonstrationFeedback <: DemonstrationFeedback end
struct StopFeedback <: FeedbackType end
struct RatingFeedback <: FeedbackType end
struct CombinedFeedback <: FeedbackType end

# Define string conversion methods
Base.string(::PreferenceFeedback) = "preference"
Base.string(::QValueWalkDemonstrationFeedback) = "demonstration-qwalk"
Base.string(::NaiveDemonstrationFeedback) = "demonstration-naive"
Base.string(::StopFeedback) = "stop"
Base.string(::RatingFeedback) = "rating"
Base.string(::CombinedFeedback) = "combined"

# Optional: Define show methods for prettier printing
Base.show(io::IO, ::PreferenceFeedback) = print(io, "PreferenceFeedback()")
Base.show(io::IO, ::QValueWalkDemonstrationFeedback) = print(io, "QValueWalkDemonstrationFeedback()")
Base.show(io::IO, ::NaiveDemonstrationFeedback) = print(io, "NaiveDemonstrationFeedback()")
Base.show(io::IO, ::StopFeedback) = print(io, "StopFeedback()")
Base.show(io::IO, ::RatingFeedback) = print(io, "RatingFeedback()")
Base.show(io::IO, ::CombinedFeedback) = print(io, "CombinedFeedback()")

# Types for task type
abstract type TaskType end
struct EpisodicTask <: TaskType end
struct FiniteHorizonTask <: TaskType end

# Optional: Define show methods for prettier printing
Base.show(io::IO, ::EpisodicTask) = print(io, "EpisodicTask()")
Base.show(io::IO, ::FiniteHorizonTask) = print(io, "FiniteHorizonTask()")


# String to feedback type
string_to_feedback_type = Dict(
    "preference" => PreferenceFeedback(),
    "demonstration-naive" => NaiveDemonstrationFeedback(),
    "demonstration-qwalk" => QValueWalkDemonstrationFeedback(),
    "stop" => StopFeedback(),
    "rating" => RatingFeedback(),
    "combined" => CombinedFeedback()
)

# String to task type
string_to_task_type = Dict(
    "episodic" => EpisodicTask(),
    "finite" => FiniteHorizonTask()
)