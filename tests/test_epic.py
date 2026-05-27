import numpy as np
from mavrl.evaluation.epic import canonically_shaped_reward, epic_distance
from mavrl.envs.grid_env.rewards import reward_factory
from mavrl.utils.reward import shape

def test_random_potential():

    # Generate random reward function
    n_S, n_A = 100, 50
    gamma = 0.99
    shaping_factor = 0.75
    R_true = np.random.randn(n_S, n_A, n_S)

    # Create state-wise bounded function f(s)
    f = np.random.randn(n_S)

    # Create shaped reward
    R_shaped = shape(R_true, f, gamma)

    # Compute the canonical rewards
    R_true_canon = canonically_shaped_reward(R_true, gamma)
    R_shaped_canon = canonically_shaped_reward(R_shaped, gamma)

    # Canonicalized versions should be equivalent
    assert np.allclose(R_true_canon, R_shaped_canon), "Canonicalized reward are not the same"

    # Shape and scale reward
    R_shaped_and_scaled = shape(R_true, f, gamma, shaping_factor)

    # Epic distance should be 0
    dist = epic_distance(R_true, R_shaped_and_scaled, gamma)
    assert np.allclose(dist, 0, atol=1e-6), f"Epic distance {dist} != 0"


def test_grid_envs():
    """Tests the pairwise distances reported in table A.2 of https://arxiv.org/pdf/2006.13900"""

    reward_types = ["sparse", "dense", "center", "penalty", "path", "cliff"]
    gamma = 0.99

    distances = {}

    def set_distance(rtype_A: str, rtype_B: str, val: float):
        distances[frozenset({rtype_A, rtype_B})] = val
    
    def get_distance(rtype_A: str, rtype_B: str):
        return distances[frozenset({rtype_A, rtype_B})]

    # The distance to yourself is 0
    for rtype in reward_types:
        set_distance(rtype, rtype, 0.0)
    
    # Set distances between distinct rewards

    # sparse
    set_distance("sparse", "dense", 0.0)
    set_distance("sparse", "center", 0.75)
    set_distance("sparse", "penalty", 1.0)
    set_distance("sparse", "path", 0.1602)
    set_distance("sparse", "cliff", 0.3676)

    # dense
    set_distance("dense", "center", 0.75)
    set_distance("dense", "penalty", 1.0)
    set_distance("dense", "path", 0.1602)
    set_distance("dense", "cliff", 0.3676)

    # center
    set_distance("center", "penalty", 0.6614)
    set_distance("center", "path", 0.7071)
    set_distance("center", "cliff", 0.6692)

    # penalty
    set_distance("penalty", "path", 0.9871)
    set_distance("penalty", "cliff", 0.9300)

    # path
    set_distance("path", "cliff", 0.2672)

    grid_size = 3
    for rtype_A in reward_types:
        for rtype_B in reward_types:
            try:
                R_A = reward_factory(grid_size, rtype_A, gamma)
            except NotImplementedError:
                continue
            try:
                R_B = reward_factory(grid_size, rtype_B, gamma)
            except NotImplementedError:
                continue
            dist = epic_distance(R_A, R_B, gamma)
            gt = get_distance(rtype_A, rtype_B)
            assert np.allclose(dist, gt, atol=1e-4), f"Dist {dist} between {rtype_A} and {rtype_B} != {gt}"

if __name__ == "__main__":
    test_grid_envs()