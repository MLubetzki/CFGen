import flax.linen as nn

class CategoricalFeaturizer(nn.Module):
    n_cat: int
    one_hot_encode_features: bool
    embedding_dimensions: int=None

    @nn.compact
    def __call__(self, obs):
        """Extract features 

        Args:
            obs (torch.Tensor): The batch of observations 

        Returns:
            torch.Tensor: Extracted embeddings 
        """
        if self.one_hot_encode_features: 
            return nn.one_hot(obs, num_classes=self.n_cat)
        else:
            return nn.Embed(self.n_cat, self.embedding_dimensions)(obs.astype(int))
        