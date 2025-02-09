from typing import Callable, List, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn

# TODO unused, remove?
def unsqueeze_right(x, num_dims=1):
    """
    Unsqueezes the last `num_dims` dimensions of `x`.

    Args:
        x (torch.Tensor): Input tensor.
        num_dims (int, optional): Number of dimensions to unsqueeze. Defaults to 1.

    Returns:
        torch.Tensor: Tensor with unsqueezed dimensions.
    """
    return x.view(x.shape + (1,) * num_dims)

# TODO not tested yet. Adapt documentation
def pad_t_like_x(t, x):
    """Function to reshape the time vector t by the number of dimensions of x.

    Parameters
    ----------
    x : Tensor, shape (bs, *dim)
        represents the source minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    t : Tensor, shape (bs, number of x dimensions)

    Example
    -------
    x: Tensor (bs, C, W, H)
    t: Vector (bs)
    pad_t_like_x(t, x): Tensor (bs, 1, 1, 1)
    """
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.ndim - 1)))

# TODO unused, remove?
def kl_std_normal(mean_squared, var):
    """
    Computes Gaussian KL divergence.

    Args:
        mean_squared (torch.Tensor): Mean squared values.
        var (torch.Tensor): Variance values.

    Returns:
        torch.Tensor: Gaussian KL divergence.
    """
    return 0.5 * (var + mean_squared - jnp.log(var.clamp(min=1e-15)) - 1.0)

def split_rng_dict(rng_dict, num : int=2):
    """
    Splits a dictionary of random number generators into `num` parts.

    Args:
        rng_dict (dict): Dictionary of random number generators.
        num (int): Number of parts to split the dictionary into.

    Returns:
        List[dict]: List of dictionaries of random number generators.
    """
    split_dict = jax.tree.map(lambda x: jax.random.split(x, num), rng_dict)
    return [{k: split_dict[k][i] for k in rng_dict} for i in range(num)]

class MLP(nn.Module):
    dims: List[int]
    batch_norm: bool
    dropout: bool
    dropout_p: float
    activation: Optional[Callable] = nn.elu
    final_activation: Optional[str] = None

    # We're passing the final activation as a string from hydra. This returns the correct function
    def get_final_activation(self):
        if self.final_activation == "tanh":
            return nn.tanh
        elif self.final_activation == "sigmoid":
            return nn.sigmoid
        elif self.final_activation == "elu":
            return nn.elu

        return None     

    @nn.compact
    def __call__(self, x, train: bool=False):
        """
        Forward pass of the MLP.

        Args:
            x (numpy.ndarray): Input array.
            train (bool): training vs test mode. If True, updates BatchNorm average and applies dropout

        Returns:
            numpy.ndarray: Output of the MLP.
        """
        for i in range(len(self.dims[:-1])):
            x = nn.Dense(self.dims[i])(x)
            if self.batch_norm:
                x = nn.BatchNorm(use_running_average=not train)(x)
            if self.dropout:
                x = nn.Dropout(dropout_p, deterministic=not train)(x)
            x = self.activation(x)
        
        x = nn.Dense(self.dims[-1])(x) # final layer
        final_activation = self.get_final_activation()
        if not final_activation:
            return x
        else:
            return final_activation(x)
