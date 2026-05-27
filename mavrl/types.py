
from enum import Enum

from numpy.typing import NDArray

class DataKey(str, Enum):

    # Core data
    OBS = "obs"
    ACTS = "acts"
    REWS = "rewards"
    NEXT_OBS = "next_obs"
    NEXT_ACTS = "next_acts"
    
    # Optional: States and obs may differ after transformation
    STATES = "states"
    NEXT_STATES = "next_states"
    ACT_FEATS = "action_features"
    NEXT_ACT_FEATS = "next_action_features"

    # Validity flags
    VALID = "valid"  # Always discard this data
    TERMINAL = "terminal"  # This flag is needed to compute TD-error for terminal states

    # Feedback specific data
    FEEDBACK_TYPE = "feedback_type"
    PREFERENCE = "preference"
    RATING = "rating"
    RANKING = "ranking"
    STOP_TIME = "stop_time"
    RATIONALITY = "rationality"
    LAMBDA = "lambda"  # Stop sensitivity parameter
    REGRET_DISCOUNT = "regret_discount"  # Discount factor for old regret
    GAMMA = "gamma"
    TD_ERROR_WEIGHT = "td_error_weight"


Trajectory = dict[DataKey, NDArray]

class FeedbackType(str, Enum):
    """Types of feedback for multi-feedback learning."""
    PREF = "pref"
    DEMO = "demo"
    RATE = "rating"
    STOP = "stop"