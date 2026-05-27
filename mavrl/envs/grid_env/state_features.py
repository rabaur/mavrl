import numpy as np
from mavrl.utils.feature_transforms import to_one_hot


def state_feature_factory(feature_type: str, grid_size: int, n_dct_basis_fns: int, **kwargs) -> np.ndarray:
    """
    Creates state features.

    Returns:
        (n_states, state_feature_dim) tensor where row i corresponds to features of state i.
    """
    n_states = grid_size**2
    if feature_type == "one_hot":
        feats = one_hot_features(n_states)
    elif feature_type == "continuous_coordinate":
        feats = coordinate_features(grid_size)
    elif feature_type == "dct":
        n_dct_basis_fns = kwargs.get("n_dct_basis_fns", 8)
        feats = dct_features(grid_size, n_dct_basis_fns)
    elif feature_type == "embedding":
        feats = discrete_features(n_states)
    else:
        raise ValueError(f"Invalid feature type: {feature_type}")
    assert feats.shape[0] == n_states, f"State-feature matrix has {feats.shape[0]} instead of {n_states=} rows"
    return feats

def dct_features(grid_size: int, n_dct_basis_fns: int) -> np.ndarray:
    """
    Creates 2D DCT-II features on an N×N grid over [0,1]^2
    using the first n_dct_basis_fns in each dimension.
    """
    N = grid_size
    K = n_dct_basis_fns

    # grid coordinates
    x = np.arange(N)
    y = np.arange(N)
    xv, yv = np.meshgrid(x, y, indexing='ij')

    # DCT-II normalization factors
    def alpha(p):
        return np.where(p == 0, np.sqrt(1.0 / N), np.sqrt(2.0 / N))

    # compute basis functions
    feats = []
    for u in range(K):
        au = alpha(u)
        cos_u = np.cos(np.pi * (2 * xv + 1) * u / (2 * N))
        for v in range(K):
            av = alpha(v)
            cos_v = np.cos(np.pi * (2 * yv + 1) * v / (2 * N))
            feats.append(au * av * cos_u * cos_v)

    # stack into (N, N, K*K)
    feats = np.stack(feats, axis=-1)
    return feats.reshape((N*N, -1)).astype(np.float32)

def coordinate_features(grid_size: int):
    """
    Creates coordinate features on an N×N grid scaled within [0,1]^2
    """
    N = grid_size

    # grid coordinates
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    feats = np.stack([xv, yv], axis=-1)
    return feats.reshape((N*N, -1)).astype(np.float32)