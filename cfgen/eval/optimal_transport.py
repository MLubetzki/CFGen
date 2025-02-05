# From https://github.com/atong01/conditional-flow-matching/blob/v0/src/models/components/optimal_transport.py

from typing import Optional
from functools import partial
import ot
import numpy as np

def wasserstein(
    x0: np.array,
    x1: np.array,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    """
    Compute the Wasserstein distance between two distributions.

    Args:
        x0 (torch.Tensor): The first distribution.
        x1 (torch.Tensor): The second distribution.
        method (Optional[str], optional): The method for computing Wasserstein distance.
            Options are "exact", "sinkhorn". Defaults to None.
        reg (float, optional): Regularization parameter for the Sinkhorn method. Defaults to 0.05.
        power (int, optional): Power for the distance computation, can be 1 or 2. Defaults to 2.
        **kwargs: Additional keyword arguments.

    Raises:
        ValueError: If an unknown method is provided.

    Returns:
        float: The computed Wasserstein distance.
    """
    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments where a, b are marginals and
    # M is a cost matrix
    if method == "exact" or method is None:
        ot_fn = ot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(ot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = ot.unif(x0.shape[0]), ot.unif(x1.shape[0])
    if x0.ndim > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.ndim > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = np.sqrt(np.sum((x0[:, None, :] - x1[None, :, :]) ** 2, axis=-1)) # TODO check if this is correct
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M, numItermax=1e7)
    if power == 2:
        ret = np.sqrt(ret)
    return ret
