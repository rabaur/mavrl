import numpy as np
from mavrl.utils.feature_transforms import to_one_hot


def action_feature_factory(feature_type: str, n_actions: int):
    """
    Creates action features.

    Returns:
        (n_actions, action_feature_dim) tensor where row i corresponds to features of action i.
    """
    if feature_type == "one_hot":
        actions = np.arange(n_actions)
        feats = np.vectorize(to_one_hot)(actions, n_actions)
    else:
        raise ValueError(f"Invalid feature type: {feature_type}")
    assert feats.shape[0] == n_actions, f"Action-feature matrix has {feats.shape[0]} instead of {n_actions=} rows"
    return feats

