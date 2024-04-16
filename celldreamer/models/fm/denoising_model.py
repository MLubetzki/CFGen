import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

from celldreamer.models.fm.layer_utils import Linear
from celldreamer.models.base.utils import unsqueeze_right

# Util functions
def zero_init(module):
    """
    Initializes the weights and biases of a PyTorch module with zero values.

    Args:
        module (torch.nn.Module): PyTorch module for weight and bias initialization.

    Returns:
        torch.nn.Module: The input module with weights and biases initialized to zero.
    """
    nn.init.constant_(module.weight.data, 0)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias.data, 0)
    return module

def get_timestep_embedding(
    timesteps,
    embedding_dim: int,
    dtype=torch.float32,
    max_timescale=10_000,
    min_timescale=1,
    ):
    """
    Generates a sinusoidal embedding for a sequence of timesteps.

    Args:
        timesteps (torch.Tensor): 1-dimensional tensor representing the input timesteps.
        embedding_dim (int): Dimensionality of the embedding. It must be an even number.
        dtype (torch.dtype, optional): Data type for the resulting tensor. Default is torch.float32.
        max_timescale (float, optional): Maximum timescale value for the sinusoidal embedding. Default is 10,000.
        min_timescale (float, optional): Minimum timescale value for the sinusoidal embedding. Default is 1.

    Returns:
        torch.Tensor: Sinusoidal embedding tensor for the input timesteps with the specified embedding_dim.
    """
    # Adapted from tensor2tensor and VDM codebase.
    assert timesteps.ndim == 1
    assert embedding_dim % 2 == 0
    timesteps *= 1000.0  # In DDPM the time step is in [0, 1000], here [0, 1]
    num_timescales = embedding_dim // 2
    inv_timescales = torch.logspace(  # or exp(-linspace(log(min), log(max), n))
        -np.log10(min_timescale),
        -np.log10(max_timescale),
        num_timescales,
        device=timesteps.device,
    )
    emb = timesteps.to(dtype)[:, None] * inv_timescales[None, :]  # (T, D/2)
    return torch.cat([emb.sin(), emb.cos()], dim=1)  # (T, D)

# ResNet MLP 
class MLPTimeStep(pl.LightningModule):
    def __init__(self, 
                 in_dim: int,
                 hidden_dim: int,
                 dropout_prob: int,
                 n_blocks: int, 
                 model_type: str,
                 embed_time: bool,
                 size_factor_min: float, 
                 size_factor_max: float,
                 embed_size_factor: bool,
                 covariate_list: list,
                 embedding_dim=128, 
                 normalization="layer", 
                 conditional=False,
                 embed_condition=False, 
                 n_cond=None,
                 multimodal=False, 
                 is_binarized=False, 
                 modality_list=None, 
                 conditioning_probability=0.8
                 ):
        
        super().__init__()
        
        # Gene expression dimension 
        self.in_dim = in_dim
        
        # The network downsizes the input multiple times 
        self.hidden_dim = hidden_dim 
        
        # Initialize attributes 
        self.model_type = model_type
        self.embed_time = embed_time
        self.size_factor_min = size_factor_min
        self.size_factor_max = size_factor_max
        self.embed_size_factor = embed_size_factor 
        self.embedding_dim = embedding_dim
        self.embed_condition = embed_condition
        self.conditional = conditional
        self.multimodal = multimodal
        self.is_binarized = is_binarized
        self.covariate_list = covariate_list
        self.modality_list = modality_list
        self.conditioning_probability = conditioning_probability
        
        added_dimensions = 0  # incremented if not embedding conditioning variables 
        
        # Time embedding network
        if embed_time:
            self.time_embedder = nn.Sequential(
                Linear(embedding_dim, embedding_dim * 4),  # Upsample embedding
                nn.SiLU(),
                Linear(embedding_dim * 4, embedding_dim * 4))
        else:
            added_dimensions += 1
            
        # Size factor embeddings 
        if model_type=="conditional_latent":
            if embed_size_factor:
                self.size_factor_embedder = nn.Sequential(
                    Linear(embedding_dim, embedding_dim * 4),  # Upsample embedding
                    nn.SiLU(),
                    Linear(embedding_dim * 4, embedding_dim * 4))
            else:
                added_dimensions += 1
            
        # Covariate embedding
        if conditional:
            self.condition_embedder = {}
            for covariate in covariate_list:
                if embed_condition:
                    self.condition_embedder[covariate] = nn.Sequential(
                        Linear(n_cond[covariate], embedding_dim * 4),  # Upsample embedding
                        nn.SiLU(),
                        Linear(embedding_dim * 4, embedding_dim * 4))
                else:
                    added_dimensions += 1
            self.condition_embedder = torch.nn.ModuleDict(self.condition_embedder)
        
        # Initial convolution
        self.net_in = Linear(in_dim, self.hidden_dim)

        # Down path: n_blocks blocks with a resnet block and maybe attention.
        self.blocks = []

        # Dimensionality preserving Resnet in the bottleneck 
        for _ in range(n_blocks):
            self.blocks.append(ResnetBlock(in_dim=self.hidden_dim,
                                                out_dim=self.hidden_dim,
                                                added_dimensions=added_dimensions,
                                                dropout_prob=dropout_prob,
                                                embedding_dim=embedding_dim * 4,  
                                                normalization=normalization, 
                                                embed_condition=embed_condition))
        
        # Set up blocks
        self.blocks = nn.ModuleList(self.blocks)
        
        if normalization not in ["layer", "batch"]:
            self.net_out = nn.Sequential(
                nn.SiLU(),
                Linear(self.hidden_dim, in_dim))
        else:
            self.net_out = nn.Sequential(
                nn.LayerNorm(self.hidden_dim) if normalization=="layer" else nn.BatchNorm1d(num_features=self.hidden_dim),
                nn.SiLU(),
                Linear(self.hidden_dim, in_dim))

    def forward(self, x, t, l, y, inference=False, unconditional=False, covariate=None):
        # If time is unique (e.g., during sampling) for all batch observations, repeat over the batch dimension
        if t.shape[0] == 1:
            t = t.repeat((x.shape[0],) + (1,) * (t.ndim-1))
        
        # Make a copy of time for using in time embeddings
        t_for_embeddings = t.clone().detach()
                
        # Embed time
        if self.embed_time:
            t_for_embeddings = t_for_embeddings.squeeze()
            emb = self.time_embedder(get_timestep_embedding(t_for_embeddings, self.embedding_dim))
        else:
            if t_for_embeddings.ndim != x.ndim:
                emb = unsqueeze_right(t_for_embeddings, x.ndim - t_for_embeddings.ndim)
                
        # Embed condition
        is_conditioned = torch.bernoulli(torch.tensor([self.conditioning_probability])).item() if not inference else 1. # Bernoulli variable to decide whether to condition or not
        if self.conditional and is_conditioned and not unconditional:
            if self.embed_condition:
                if covariate == None:
                    covariate = np.random.choice(self.covariate_list)
                y = self.condition_embedder[covariate](y[covariate])
                emb = emb + y
            else:
                raise NotImplementedError
    
        # Embed size factor
        if self.model_type == "conditional_latent":
            if self.multimodal and not self.is_binarized:
                for mod in self.modality_list:
                    if self.embed_size_factor:
                        l_mod = l[mod].squeeze()
                        l_mod = (l_mod - self.size_factor_min[mod]) / (self.size_factor_max[mod] - self.size_factor_min[mod])
                        l_mod = self.size_factor_embedder(get_timestep_embedding(l_mod, self.embedding_dim))
                        emb = emb + l_mod
                    else:
                        if l[mod].ndim != x.ndim:
                            l = unsqueeze_right(l[mod], x.ndim-l[mod].ndim)  
                        emb = torch.cat([emb, l[mod]], dim=1)
            else:
                if self.embed_size_factor:
                    l = l.squeeze()
                    l = (l - self.size_factor_min) / (self.size_factor_max - self.size_factor_min)
                    l = self.size_factor_embedder(get_timestep_embedding(l, self.embedding_dim))
                    emb = emb + l
                else:
                    if l.ndim != x.ndim:
                        l = unsqueeze_right(l, x.ndim-l.ndim)  
                    emb = torch.cat([emb, l], dim=1)                

        # Compute prediction
        h = self.net_in(x)  
        for block in self.blocks:  # n_blocks times
            h = block(h, emb)
        pred = self.net_out(h)
        return pred 

class ResnetBlock(nn.Module):
    """
    A block for a Multi-Layer Perceptron (MLP) with skip connection.

    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int, optional): Dimension of the output features. Defaults to None, in which case it's set equal to input_dim.
        condition_dim (int, optional): Dimension of the conditional input. Defaults to None.
        dropout_prob (float, optional): Dropout probability. Defaults to 0.0.
        norm_groups (int, optional): Number of groups for layer normalization. Defaults to 32.
    """
    def __init__(
        self,
        in_dim,
        out_dim=None,
        added_dimensions=0,
        dropout_prob=0.0,
        embedding_dim=None, 
        normalization="batch", 
        embed_condition=True):
        
        super().__init__()
                
        # Variables controlling if time and size factor should be embedded
        self.embed_condition = embed_condition    
        self.embedding_dim = embedding_dim
    
        # Set output_dim to input_dim if not provided
        out_dim = in_dim if out_dim is None else out_dim
        self.out_dim = out_dim

        # First linear block with LayerNorm and SiLU activation
        if normalization not in ["layer", "batch"]:
            self.net1 = nn.Sequential(
                nn.SiLU(),
                Linear(in_dim, out_dim))          
        else:
            self.net1 = nn.Sequential(
                nn.LayerNorm(in_dim) if normalization=="layer" else nn.BatchNorm1d(num_features=in_dim),
                nn.SiLU(),
                Linear(in_dim, out_dim))
        
        # Projections for conditions 
        if embed_condition:
            self.cond_proj = nn.Sequential(nn.SiLU(), 
                                            Linear(self.embedding_dim, out_dim))
            
        # Second linear block with LayerNorm, SiLU activation, and optional dropout
        if normalization not in ["layer", "batch"]:
            self.net2 = nn.Sequential(
                nn.SiLU(),
                *([nn.Dropout(dropout_prob)] * (dropout_prob > 0.0)),
                zero_init(Linear(out_dim + added_dimensions, out_dim)))
        else:
            self.net2 = nn.Sequential(
                nn.LayerNorm(out_dim + added_dimensions) if normalization=="layer" else nn.BatchNorm1d(num_features=out_dim + added_dimensions),
                nn.SiLU(),
                *([nn.Dropout(dropout_prob)] * (dropout_prob > 0.0)),
                zero_init(Linear(out_dim + added_dimensions, out_dim)))

        # Linear projection for skip connection if input_dim and output_dim differ
        if in_dim != out_dim:
            self.skip_proj = Linear(in_dim, out_dim)

    def forward(self, x, emb):
        """
        Forward pass of the MLP block.

        Args:
            x (torch.Tensor): Input features.
            condition (torch.Tensor, optional): Conditional input. Defaults to None.

        Returns:
            torch.Tensor: Output features.
        """
        # Forward pass through the first linear block
        h = self.net1(x)

        # Condition time and library size 
        if self.embed_condition:
            emb = self.cond_proj(emb)           
            h = h + emb
        else:
            h = torch.cat([h, emb], dim=1)
                
        # Forward pass through the second linear block
        h = self.net2(h)

        # Linear projection for skip connection if input_dim and output_dim differ
        if x.shape[1] != self.out_dim:
            x = self.skip_proj(x)

        # Add skip connection to the output
        assert x.shape == h.shape
        
        return x + h
