from numpy.typing import NDArray
import numpy as np
from hilbert import encode

def hilbert_encode(coords, num_dims, resolution):
    return encode(coords, num_dims, int(np.log2(resolution)))

def morton_encode(coords):
    """
    Encode N-dimensional coordinates to Morton codes (Z-order curve).
    
    Args:
        coords: array of shape (n, num_dims) with integer coordinates
        
    Returns:
        array of shape (n,) with Morton codes
    """
    n_points, num_dims = coords.shape
    coords_uint = coords.astype(np.uint64)
    
    # Initialize Morton codes
    morton = np.zeros(n_points, dtype=np.uint64)
    
    # Interleave bits from all dimensions
    for bit_idx in range(32):  # Support up to 32 bits per coordinate
        for dim in range(num_dims):
            # Extract bit at position bit_idx from dimension dim
            bit = (coords_uint[:, dim] & (1 << bit_idx)) >> bit_idx
            # Place it at position (bit_idx * num_dims + dim) in the Morton code
            morton |= bit << (bit_idx * num_dims + dim)
    
    return morton

def unfold_tensor(tensor: NDArray, curve_type: str, flatten_axes: list[int], unfold_axis: int) -> NDArray:
    """
    Unfolds a N-dimensional tensor by flattening the specified dimensions and then unfolding along the specified dimension.

    Args:
        tensor: The tensor to unfold.
        curve_type: The type of curve to use for unfolding. Options: "hilbert", "morton"
        flatten_axes: The axes to flatten.
        unfold_axis: The axis to unfold along.

    Returns:
        The unfolded tensor.
    """
    flatten_dims = [tensor.shape[i] for i in flatten_axes]
    n_flatten = len(flatten_dims)
    n_unfold = tensor.shape[unfold_axis]
    assert len(set(flatten_dims)) == 1, "All flattened dimensions must have the same size"
    assert np.log2(flatten_dims[0]) == int(np.log2(flatten_dims[0])), "Flattened dimension must be a power of 2"
    assert unfold_axis < tensor.ndim, "Unfold axis must be less than the number of dimensions of the tensor"
    assert unfold_axis not in flatten_axes, "Unfold axis must not be in the list of flattened axes"

    resolution = flatten_dims[0]
    idxs_1d = np.arange(resolution)
    idxs = np.meshgrid(*([idxs_1d] * n_flatten), indexing='ij')
    idxs_stacked = np.stack(idxs, axis=-1)  # (resolution, resolution, ..., resolution, n_flatten)
    idxs_flat = np.reshape(idxs_stacked, (-1, n_flatten))  # (resolution ** n_flatten, n_flatten)

    # Get ordering using selected curve
    if curve_type == "hilbert":
        idxs_flat_encoded = hilbert_encode(idxs_flat, n_flatten, resolution)  # (resolution ** n_flatten,)
        idxs_shuffled = idxs_flat[idxs_flat_encoded]  # (resolution ** n_flatten, n_flatten)
    elif curve_type == "morton":
        morton_codes = morton_encode(idxs_flat)  # (resolution ** n_flatten,)
        idxs_shuffled = idxs_flat[np.argsort(morton_codes)]  # (resolution ** n_flatten, n_flatten)
    else:
        raise ValueError(f"Unknown curve type: {curve_type}. Use 'hilbert' or 'morton'.")

    # move the flatten axes to the front
    tensor = np.moveaxis(tensor, flatten_axes, range(len(flatten_axes)))
    # aggregate the remaining axes
    remaining_axes = [i for i in range(tensor.ndim) if i not in flatten_axes]
    tensor = np.max(tensor, axis=remaining_axes[0])
    result = tensor[(*idxs_shuffled.T, ...)]  # (resolution ** n_flatten, n_unfold)

    return result
