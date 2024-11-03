from typing import Callable, List, Optional

import jax.numpy as jnp
import flax.linen as nn

# TODO fix documentation

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
    return t.reshape(-1, *([1] * (x.dim() - 1)))

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

class MLP(nn.Module):
    dims: List[int]
    batch_norm: bool
    dropout: bool
    dropout_p: float
    activation: Optional[Callable] = nn.elu
    final_activation: Optional[str] = None

    def setup(self):
        """
        Multi-Layer Perceptron (MLP) model.

        Args:
            dims (List[int]): List of dimensions for each layer.
            batch_norm (bool): Whether to use batch normalization.
            dropout (bool): Whether to use dropout.
            dropout_p (float): Dropout probability.
            activation (Optional[Callable], optional): Activation function. Defaults to torch.nn.SELU.
            final_activation (Optional[str], optional): Final activation function ("tanh", "sigmoid", or None). Defaults to None.
        """
        # MLP 
        layers = []
        for i in range(len(self.dims[:-1])):
            block = []
            block.append(nn.Dense(self.dims[i]))
            if self.batch_norm: 
                block.append(nn.BatchNorm(use_running_average=True)) # TODO adapt during train/test
            block.append(self.activation)
            if self.dropout:
                block.append(nn.Dropout(dropout_p))
            layers.append(nn.Sequential(block))
        
        # Last layer without activation 
        layers.append(nn.Dense(self.dims[-1]))
        # Compile the neural net
        self.net = nn.Sequential(layers)
        
        if self.final_activation == "tanh":
            self.final_activation_func = nn.tanh
        elif self.final_activation == "sigmoid":
            self.final_activation_func = nn.sigmoid
        elif self.final_activation == "elu":
            self.final_activation_func = nn.elu
        else:
            self.final_activation_func = None

    def __call__(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.net(x)
        if not self.final_activation_func:
            return x
        else:
            return self.final_activation_func(x)
