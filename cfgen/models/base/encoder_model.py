import jax
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn

from scvi.distributions import JaxNegativeBinomialMeanDisp as NegativeBinomial
from torch.distributions import Poisson, Bernoulli
from cfgen.models.base.utils import MLP


# TODO fix documentation

class EncoderModel(nn.Module):
    """
    PyTorch Lightning Module for an encoder-decoder model.

    Args:
        in_dim (dict): Dictionary specifying the input dimensions for each modality.
        encoder_kwargs (dict): If multimodal, dictionary with arguments, one per effect.
        scaler (dict): If multimodal, dictionary with one scaler per modality.
        learning_rate (float): Learning rate for optimization.
        weight_decay (float): Weight decay for optimization.
        covariate_specific_theta (bool): Flag indicating whether theta is specific to covariates.
        conditioning_covariate (str): Covariate used for conditioning.
        n_cat (int): Number of categories for the theta parameter.
        is_binarized (bool): Flag indicating whether the input data is binarized.

    Methods:
        training_step: Executes a training step.
        validation_step: Executes a validation step.
        configure_optimizers: Configures the optimizer for training.
        encode: Encodes input data.
        decode: Decodes encoded data.

    """
    in_dim : dict
    encoder_kwargs : dict # TODO ?
    learning_rate : float
    weight_decay : float
    covariate_specific_theta : bool
    conditioning_covariate : str # TODO ?
    n_cat : int=None
    is_binarized : bool=False
    encoder_multimodal_joint_layers: list=None

    def setup(self):
        """
        Initializes the EncoderModel.
        """

        # Joint into a single latent space or not 
        if self.encoder_multimodal_joint_layers:
            self.encoder_joint = None # Initialize another layer 

        # List of modalities present in the data 
        self.modality_list = list(self.encoder_kwargs.keys())

        # Theta for the negative binomial parameterization of scRNA-seq
        in_dim_rna = self.in_dim["rna"]
        # Inverse dispersion
        if self.covariate_specific_theta:
            self.theta = self.param("theta", nn.initializers.normal(), (n_cat, in_dim_rna))
        else:
            self.theta = self.param("theta", nn.initializers.normal(), (in_dim_rna,))

        # Modality specific part 
        encoder = {}
        decoder = {}
        for mod in self.modality_list:
            encoder[mod] = MLP(**self.encoder_kwargs[mod])
            if self.encoder_multimodal_joint_layers:
                self.encoder_kwargs[mod]["dims"].append(self.encoder_multimodal_joint_layers["dims"][-1]) # TODO don't modify dict, it's frozen
            decoder_dims = {"dims": [*self.encoder_kwargs[mod]["dims"][::-1], self.in_dim[mod]]}
            decoder_kwargs = { key: value for key, value in self.encoder_kwargs[mod].items() if key != "dims" }
            decoder[mod] = MLP(**(decoder_dims | decoder_kwargs))
        
        # Shared modality part in the encoder 
        if self.encoder_multimodal_joint_layers:
            joint_inputs = sum([self.encoder_kwargs[mod]["dims"][0] for mod in self.modality_list])
            self.encoder_multimodal_joint_layers["dims"] = [joint_inputs, *self.encoder_multimodal_joint_layers["dims"]]
            self.encoder_joint = MLP(**self.encoder_multimodal_joint_layers)

        self.encoder = encoder
        self.decoder = decoder
        # TODO save hyperparameters

    def __call__(self, batch):
        """
        Executes a single step of training or validation.

        Args:
            batch (dict): Batch of input data.
            dataset_type (str): Type of dataset, either 'train' or 'valid'.

        Returns:
            loss (tensor): Loss value for the step.

        """
        X = batch["X"]
        size_factor = {mod: jnp.expand_dims(X[mod].sum(1), 1) for mod in X}

        # Conditioning covariate encodings
        y = batch["y"][self.conditioning_covariate]

        z = self.encode(batch)
        mu_hat = self.decode(z, size_factor)
        return self.calc_loss(X, mu_hat)


    def calc_loss(self, X, mu_hat):
        # Compute the negative log-likelihood of the data under the model
        loss = 0
        for mod in mu_hat:
            if mod == "rna":
                # Negative Binomial log-likelihood
                if not self.covariate_specific_theta:
                    px = NegativeBinomial(mu_hat[mod], jnp.exp(self.theta))
                else:
                    px = NegativeBinomial(mu_hat[mod], jnp.exp(self.theta[y]))
            elif mod == "atac":
                if not self.is_binarized:
                    px = Poisson(rate=mu_hat[mod]) # TODO fix
                else:
                    px = Bernoulli(probs=mu_hat[mod]) # TODO fix
            else:
                raise NotImplementedError
            loss -= px.log_prob(X[mod]).sum(1).mean()

        return loss


    def encode(self, batch):
        """
        Encodes input data.

        Args:
            batch (dict): Batch of input data.

        Returns:
            z (tensor or dict): Encoded data.

        """
        z = {}
        for mod in self.modality_list:
            z[mod] = self.encoder[mod](batch["X_norm"][mod])
            
        # Implement joint layers if defined
        if self.encoder_multimodal_joint_layers:
            z_joint = jnp.concatenate([z[mod] for mod in z], dim=-1)
            z = self.encoder_joint(z_joint)     
        return z

    def decode(self, x, size_factor):
        """
        Decodes encoded data.

        Args:
            x (tensor or dict): Encoded data.
            size_factor (tensor or dict): Size factor.

        Returns:
            mu_hat (tensor or dict): Decoded data.

        """
        mu_hat = {}
        for mod in self.modality_list:
            if not self.encoder_multimodal_joint_layers:
                x_mod = self.decoder[mod](x[mod])
            else:
                x_mod = self.decoder[mod](x)

            if mod != "atac" or (mod == "atac" and not self.is_binarized):
                mu_hat_mod = nn.softmax(x_mod, axis=1)  # for Poisson counts the parameterization is similar to RNA 
                mu_hat_mod = mu_hat_mod * size_factor[mod]
            else:
                mu_hat_mod = nn.sigmoid(x_mod)
            mu_hat[mod] = mu_hat_mod
        return mu_hat
