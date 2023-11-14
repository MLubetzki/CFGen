from typing import Callable, List, Optional
import torch
import torch.nn as nn

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

def kl_std_normal(mean_squared, var):
    """
    Computes Gaussian KL divergence.

    Args:
        mean_squared (torch.Tensor): Mean squared values.
        var (torch.Tensor): Variance values.

    Returns:
        torch.Tensor: Gaussian KL divergence.
    """
    return 0.5 * (var + mean_squared - torch.log(var.clamp(min=1e-15)) - 1.0)

class MLP(torch.nn.Module):
    def __init__(self, 
                 dims: List[int],
                 batch_norm: bool, 
                 dropout: bool, 
                 dropout_p: float, 
                 activation: Optional[Callable] = torch.nn.SELU, 
                 final_activation: Optional[str] = None):
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
        super(MLP, self).__init__()

        # Attributes 
        self.dims = dims
        self.batch_norm = batch_norm
        self.activation = activation

        # MLP 
        layers = []
        for i in range(len(self.dims[:-1])):
            block = []
            block.append(torch.nn.Linear(self.dims[i], self.dims[i+1]))
            if batch_norm: 
                block.append(torch.nn.BatchNorm1d(self.dims[i+1]))
            block.append(self.activation())
            if dropout:
                block.append(torch.nn.Dropout(dropout_p))
            layers.append(torch.nn.Sequential(*block))
        self.net = torch.nn.Sequential(*layers)
        
        if final_activation == "tanh":
            self.final_activation = torch.nn.Tanh()
        elif final_activation == "sigmoid":
            self.final_activation = torch.nn.Sigmoid()
        else:
            self.final_activation = None

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.net(x)
        if not self.final_activation:
            return x
        else:
            return self.final_activation(x)
