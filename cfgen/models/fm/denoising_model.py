import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
from cfgen.models.fm.layer_utils import Linear

# Util functions
# def zero_init(module):
#     """
#     Initializes the weights and biases of a PyTorch module with zero values.

#     Args:
#         module (torch.nn.Module): PyTorch module for weight and bias initialization.

#     Returns:
#         torch.nn.Module: The input module with weights and biases initialized to zero.
#     """
#     nn.init.constant_(module.weight.data, 0)
#     if hasattr(module, 'bias') and module.bias is not None:
#         nn.init.constant_(module.bias.data, 0)
#     return module

def get_timestep_embedding(
    timesteps,
    embedding_dim: int,
    max_timescale=10_000,
    min_timescale=1,
    ):
    """
    Generates a sinusoidal embedding for a sequence of timesteps.

    Args:
        timesteps (torch.Tensor): 1-dimensional tensor representing the input timesteps.
        embedding_dim (int): Dimensionality of the embedding. It must be an even number.
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
    inv_timescales = jnp.logspace(  # or exp(-linspace(log(min), log(max), n))
        -jnp.log10(min_timescale),
        -jnp.log10(max_timescale),
        num_timescales
    )
    emb = timesteps[:, None] * inv_timescales[None, :]  # (T, D/2)
    return jnp.concat([jnp.sin(emb), jnp.cos(emb)], axis=1)  # (T, D)


class GenericEmbedder(nn.Module):
    dims: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.dims)(x)
        x = nn.silu(x)
        x = nn.Dense(self.dims)(x)
        return x
    
class NormedSiluBlock(nn.Module):
    dims: int
    normalization: str=None
    dropout: float=0.0 # TODO
    zero_init: bool=False # whether to zero_init the linear layer
    # TODO implement zero_init

    @nn.compact
    def __call__(self, x, train: bool=False):
        if self.normalization in ["layer", "batch"]:
            x = nn.LayerNorm()(x) if self.normalization == "layer" else nn.BatchNorm(use_running_average=not train)(x)
        x = nn.silu(x)
        if self.dropout > 0.0:
            x = nn.Dropout(rate=self.dropout, deterministic=not train)(x)
        x = nn.Dense(self.dims)(x)

        return x

# ResNet MLP 
class MLPTimeStep(nn.Module):
    in_dim: int
    hidden_dim: int
    dropout_prob: int
    n_blocks: int
    size_factor_min: float
    size_factor_max: float
    embed_size_factor: bool
    covariate_list: list
    embedding_dim: int=128
    normalization: str="layer"
    conditional: bool=False
    is_binarized: bool=False
    modality_list: list=None
    conditioning_probability: float=0.8 
    guided_conditioning: bool=True


    def setup(self):
        # Time embedding network
        self.time_embedder = GenericEmbedder(dims=self.embedding_dim)
            
        # Size factor embeddings 
        if self.embed_size_factor:
            self.size_factor_embedder = GenericEmbedder(dims=self.embedding_dim)
        
        # Initial convolution
        self.net_in = nn.Dense(self.hidden_dim)

        # Down path: n_blocks blocks with a resnet block and maybe attention.
        # Dimensionality preserving Resnet in the bottleneck 
        self.blocks = [ResnetBlock(in_dim=self.hidden_dim,
                                                out_dim=self.hidden_dim,
                                                dropout_prob=self.dropout_prob,
                                                embedding_dim=self.embedding_dim,  
                                                normalization=self.normalization)
                        for _ in range(self.n_blocks)]
        
        self.net_out = NormedSiluBlock(dims=self.in_dim, normalization=self.normalization)

    def __call__(self, x, t, l, y, inference=False, unconditional=False, covariate=None):
        guiding_prngkey = self.make_rng("guiding")     
        # Make a copy of time for using in time embeddings
        t_for_embeddings = t.squeeze() # TODO do we have to make a copy?
        
        # Time embedding   
        emb = self.time_embedder(get_timestep_embedding(t_for_embeddings, self.embedding_dim))
                
        # Embed condition
        if self.guided_conditioning:
            guiding_prngkey, subkey = random.split(guiding_prngkey) # TODO check if this works as intended
            is_conditioned = random.bernoulli(subkey, self.conditioning_probability) if not inference else 1 # Bernoulli variable to decide whether to condition or not
            if self.conditional and is_conditioned and not unconditional:
                if covariate == None:  
                    guiding_prngkey, subkey = random.split(guiding_prngkey) # TODO check if this works as intended
                    covariate = self.covariate_list[int(random.choice(subkey, len(self.covariate_list)))]
                emb = emb + y[covariate]
        else:
            # Normal conditioning 
            for covariate in y:
                emb = emb + y[covariate]
    
        # Embed size factor
        if self.embed_size_factor:
            if not self.is_binarized:
                for mod in self.modality_list:
                    l_mod = l[mod].squeeze()
                    l_mod = (l_mod - self.size_factor_min[mod]) / (self.size_factor_max[mod] - self.size_factor_min[mod])
                    l_mod = self.size_factor_embedder(get_timestep_embedding(l_mod, self.embedding_dim))
                    emb = emb + l_mod
            else:
                l = l.squeeze()
                l = (l - self.size_factor_min) / (self.size_factor_max - self.size_factor_min)
                l = self.size_factor_embedder(get_timestep_embedding(l, self.embedding_dim))
                emb = emb + l        

        # Compute prediction
        h = self.net_in(x)  
        for block in self.blocks:  # n_blocks times
            h = block(h, emb)
        pred = self.net_out(h)
        return pred 

class ResnetBlock(nn.Module):
    in_dim: int
    out_dim: int=None
    dropout_prob: float=0.0
    embedding_dim: int=None
    normalization: str="batch"

    """
    A block for a Multi-Layer Perceptron (MLP) with skip connection.

    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int, optional): Dimension of the output features. Defaults to None, in which case it's set equal to input_dim.
        condition_dim (int, optional): Dimension of the conditional input. Defaults to None.
        dropout_prob (float, optional): Dropout probability. Defaults to 0.0.
        norm_groups (int, optional): Number of groups for layer normalization. Defaults to 32.
    """
    def setup(self): 
        # First linear block with LayerNorm and SiLU activation
        self.net1 = NormedSiluBlock(dims=self.out_dim, normalization=self.normalization)
        
        # Projections for conditions 
        self.cond_proj = NormedSiluBlock(self.out_dim)
        
        # Second linear block with LayerNorm, SiLU activation, and optional dropout
        self.net2 = NormedSiluBlock(dims=self.out_dim, normalization=self.normalization, dropout=self.dropout_prob, zero_init=True)

        # Linear projection for skip connection if input_dim and output_dim differ
        if self.out_dim != None and self.out_dim != self.in_dim:
            self.skip_proj = nn.Dense(self.out_dim)

    def __call__(self, x, emb):
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
        emb = self.cond_proj(emb)           
        h = h + emb
                
        # Forward pass through the second linear block
        h = self.net2(h)

        # Linear projection for skip connection if input_dim and output_dim differ
        if self.out_dim != None and x.shape[1] != self.out_dim:
            x = self.skip_proj(x)

        # Add skip connection to the output
        assert x.shape == h.shape
        
        return x + h
    