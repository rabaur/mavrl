import numpy as np
from typing import Optional
from numpy.typing import NDArray

def Rs_to_Rsas(Rs: NDArray, na: int):
    ns = Rs.shape[0]
    return np.broadcast_to(Rs[:, None, None], (ns, na, ns))

def Rsa_to_Rsas(Rsa: NDArray) -> NDArray:
    """
    Converts a reward function R(s,a) to a reward function R(s,a,s').
    """
    R_sa = np.asarray(Rsa)
    S, A = R_sa.shape
    R_sas = np.broadcast_to(R_sa[..., None], (S, A, S))
    return R_sas

def shape(R: NDArray, f: NDArray, gamma: float, scale: Optional[float] = None):
    assert R.ndim == 3, f"Reward needs to be defined on s,a,s, but {R.shape}"
    assert f.ndim == 1 and f.shape[0] == R.shape[0], f"f needs to be defined on s, but shape does match with R"
    if scale:
        R *= scale
    return R + gamma * f[None, None, :] - f[:, None, None]