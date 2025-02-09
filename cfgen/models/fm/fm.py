from typing import Literal
# import numpy as np
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
import optax
from diffrax import ODETerm, diffeqsolve, Dopri5

# from scvi.distributions import NegativeBinomial
# from torch.distributions import Poisson, Bernoulli
from scvi.distributions import JaxNegativeBinomialMeanDisp as NegativeBinomial
from numpyro.distributions import Bernoulli, Poisson
from cfgen.eval.evaluate import compute_umap_and_wasserstein
from cfgen.models.base.utils import pad_t_like_x
from cfgen.models.fm.ot_sampler import OTPlanSampler

# from torchdyn.core import NeuralODE

class FM(nn.Module):
    encoder_model: nn.Module
    denoising_model: nn.Module
    feature_embeddings: dict
    plotting_folder: Path
    in_dim: int
    size_factor_statistics: dict
    covariate_list: str
    theta_covariate: str
    size_factor_covariate: str
    encoder_type: str = "fixed"
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    antithetic_time_sampling: bool = True
    scaling_method: str = "log_normalization" # Change int to str
    sigma: float = 0.1
    covariate_specific_theta: float = False
    plot_and_eval_every: int=100 
    use_ot: bool=True
    is_binarized: bool=False
    modality_list: list=None
    guidance_weights: dict=None

    def setup(self):
        """
        Flow matching for single-cell model. 
        
        Args:
            denoising_model (nn.Module): Denoising model.
            feature_embeddings (dict): Feature embeddings for covariates.
            x0_from_x_kwargs (dict): Arguments for the x0_from_x MLP.
            plotting_folder (Path): Folder for saving plots.
            in_dim (int): Number of genes.
            conditioning_covariates (str): Covariate controlling the size factor sampling.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
            weight_decay (float, optional): Weight decay. Defaults to 0.0001.
            antithetic_time_sampling (bool, optional): Use antithetic time sampling. Defaults to True.
            scaling_method (str, optional): Scaling method for input data. Defaults to "log_normalization".
            sigma (float, optional): variance around straight path for flow matching objective.
        """
        self.criterion = optax.losses.squared_error
        # OT sampler
        if self.use_ot:
            assert False # no u
            self.ot_sampler = OTPlanSampler(method="exact")

    def __call__(self, batch, dataset: Literal['train', 'valid'], train: bool=False):
        """
        Common step for training and validation.

        Args:
            batch: Batch data.
            dataset (Literal['train', 'valid']): Dataset type.

        Returns:
            torch.Tensor: Loss value.
        """
        x = batch["X"]  # counts
        
        # Collect labels 
        y_fea = self._featurize_batch_y(batch)

        # Encode observations into the latent space
        x0 = self.encoder_model.encode(batch)
        if not hasattr(self.encoder_model, "encoder_joint"):
            x0 = jnp.concatenate([x0[mod] for mod in self.modality_list], axis=1)  # concatenate ordered by the modality list 

        # Quantify size factor 
        if self.is_binarized:
            # If binarized, the size factor is not required for atac 
            size_factor = x["rna"].sum(1).unsqueeze(1)
            log_size_factor = jnp.log(size_factor)            
        else:
            size_factor = {mod: jnp.expand_dims(x[mod].sum(1), 1) for mod in self.modality_list}
            log_size_factor = {mod: jnp.log(size_factor[mod]) for mod in self.modality_list}
        
        # Sample time 
        t = self._sample_times(x0.shape[0])  # B
        
        # Sample noise 
        z = self.sample_noise_like(x0)  # B x G
        
        # Get objective and perturbed observation
        t, x_t, u_t = self.sample_location_and_conditional_flow(z, x0, t)

        # Forward through the model 
        v_t = self.denoising_model(x_t, t, log_size_factor, y_fea)
        loss = self.criterion(u_t, v_t)  # (B, )
        
        # Save results
        # metrics = {
        #     "batch_size": z.shape[0],
        #     f"{dataset}/loss": loss.mean()}
        # self.log_dict(metrics, prog_bar=True)
        
        return loss.mean()
    
    # Private methods
    def _featurize_batch_y(self, batch):
        """
        Featurize all the covariates 

        Args:
            batch: Batch data.

        Returns:
            torch.Tensor: Featurized covariates.
        """
        y = {}     
        for feature_cat in batch["y"]:
            y_cat = self.feature_embeddings[feature_cat](batch["y"][feature_cat])
            y[feature_cat] = y_cat
        return y
    
    def _sample_times(self, batch_size):
        """
        Sample times, can be sampled to cover the 

        Args:
            batch_size (int): Batch size.

        Returns:
            torch.Tensor: Sampled times.
        """
        key = self.make_rng("distr")
        if self.antithetic_time_sampling:
            t0 = random.uniform(key, minval=0, maxval=1/batch_size)
            times = jnp.arange(t0, 1.0, 1.0 / batch_size)
        else:
            times = random.uniform(key, shape=batch_size)
        return times
    

    def _conditioning_wrapper(self,
                                t: jnp.ndarray,  # Time tensor
                                x: jnp.ndarray,  # Input tensor
                                diff_args: any,  # Additional arguments from diffrax
                                l: dict,  # Log library size
                                y: dict,  # Conditioning variable
                                guidance_weights: dict,  # Weights for attribute-based guiding
                                conditioning_covariates: list,  # Covariate names for conditioning
                                unconditional: bool, # Flag for unconditional generation
    ):
        """
        Forward pass of the torch_wrapper.

        Args:
            t: Time tensor, will be repeated for each sample in the batch.
            x: Input tensor.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: The output of the model after applying conditioning.
        """
        # Repeat and concatenate time tensor to match the batch size
        t = jnp.repeat(t, x.shape[0])[:, None]

        # Unconditional generation or guided conditioning
        if unconditional or self.denoising_model.guided_conditioning:
            m_uncond = self.denoising_model(x, t, l, y, inference=True, unconditional=True, covariate=None)
            m = jnp.copy(m_uncond)
        
        # Conditional generation
        if not unconditional:
            if self.denoising_model.guided_conditioning:
                # Apply guided conditioning using provided weights
                for cov in conditioning_covariates:
                    m += guidance_weights[cov] * \
                            (self.denoising_model(x, t, l, y, inference=True, unconditional=False, covariate=cov) - m_uncond)
            else:
                # Normal conditioning without guidance
                m = self.denoising_model(x, t, l, y, inference=True, unconditional=False, covariate=None)
        
        return m
    

    # @torch.no_grad()
    def sample(self,
               batch_size, 
               n_sample_steps,
               theta_covariate, 
               size_factor_covariate,
               conditioning_covariates,
               covariate_indices=None, 
               log_size_factor=None,
               unconditional=False, 
               guidance_weights=None):
        
        if guidance_weights==None:
            guidance_weights=self.guidance_weights
            
        # Sample random noise 
        z = random.normal(self.make_rng("distr"), (batch_size, self.denoising_model.in_dim))

        # Sample random classes from the sampling covariates
        if covariate_indices==None:
            covariate_indices = {}
            for covariate in conditioning_covariates:  # for the covariates we decide to condition on 
                covariate_indices[covariate] = random.randint(self.make_rng("distr"), shape=(batch_size,), minval=0, maxval=self.feature_embeddings[covariate].n_cat)
             
        # TODO fix this in the dataloader, then adjust the calculation to work on self.size_factor_statistics
        size_factor_statistics = jax.tree.map(lambda tensor: tensor.numpy().astype(jnp.float32), self.size_factor_statistics) # TODO this is hacky

        # Sample size factor from the associated distribution
        if log_size_factor==None:
            # If size factor conditions the denoising, sample from the log-norm distribution. Else the size factor is None
            if not self.is_binarized:
                log_size_factor = {}
                for mod in self.modality_list:
                    mean_size_factor, sd_size_factor = size_factor_statistics["mean"][mod][size_factor_covariate], size_factor_statistics["sd"][mod][size_factor_covariate]
                    mean_size_factor, sd_size_factor = mean_size_factor[covariate_indices[size_factor_covariate]], sd_size_factor[covariate_indices[size_factor_covariate]]
                    size_factor_dist = mean_size_factor + sd_size_factor*random.normal(self.make_rng("distr"), shape=mean_size_factor.shape)
                    log_size_factor_mod = size_factor_dist.reshape(-1, 1)
                    log_size_factor[mod] = log_size_factor_mod
            else:
                mean_size_factor, sd_size_factor = size_factor_statistics["mean"][size_factor_covariate], size_factor_statistics["sd"][size_factor_covariate]
                mean_size_factor, sd_size_factor = mean_size_factor[covariate_indices[size_factor_covariate]], sd_size_factor[covariate_indices[size_factor_covariate]]
                size_factor_dist = mean_size_factor + sd_size_factor*random.normal(self.make_rng("distr"), shape=mean_size_factor.shape)
                log_size_factor = size_factor_dist.reshape(-1, 1)
        
        # Featurize the covariate
        if not unconditional:
            y = {}
            for covariate in covariate_indices:
                y[covariate] = self.feature_embeddings[covariate](covariate_indices[covariate])
        else: 
            y = None

        # Generate 
        denoising_model_ode = partial(self._conditioning_wrapper,
                                      l=log_size_factor,
                                      y=y,
                                      guidance_weights=guidance_weights,
                                      conditioning_covariates=conditioning_covariates,
                                      unconditional=unconditional)

        term = ODETerm(denoising_model_ode)
        solver = Dopri5()
        x0 = diffeqsolve(term, solver, t0=0.0, t1=1.0, y0=z, dt0=1.0/n_sample_steps).ys[-1].squeeze()
        
        # If we use joint layers, split the output to get separate z's
        if not self.encoder_model.encoder_multimodal_joint_layers:
            x0 = jnp.split(x0, [self.in_dim[d] for d in self.modality_list], axis=1)
            x0 = {mod: x0[i] for i, mod in enumerate(self.modality_list)}

        # Exponentiate log-size factor for decoding  
        if not self.is_binarized:
            size_factor = {mod: jnp.exp(log_size_factor[mod]) for mod in self.modality_list}
        else:
            size_factor = jnp.exp(log_size_factor)
            
        # Decode to parameterize sampling distributions
        x = self._decode(x0, size_factor)

        # Sample from noise model
        sample = {}  # containing final samples 
        for mod in x:
            if mod=="rna":  
                if not self.covariate_specific_theta:
                    distr = NegativeBinomial(mean=x[mod], inverse_dispersion=jnp.exp(self.encoder_model.theta))
                else:
                    distr = NegativeBinomial(mean=x[mod], inverse_dispersion=jnp.exp(self.encoder_model.theta[covariate_indices[theta_covariate]]))
            else:  # if mod is atac
                if not self.encoder_model.is_binarized:
                    distr = Poisson(rate=x[mod])
                else:
                    distr = Bernoulli(probs=x[mod])
            sample[mod] = distr.sample(self.make_rng("distr"))
        return sample
    
    # @torch.no_grad()
    def batched_sample(self, 
                       batch_size, 
                       repetitions,
                       n_sample_steps, 
                       theta_covariate, 
                       size_factor_covariate,
                       conditioning_covariates, 
                       covariate_indices=None, 
                       log_size_factor=None, 
                       unconditional=False):
        
        total_samples = {mod:[] for mod in self.modality_list}
            
        # Covariate is same for all modalities 
        for i in range(repetitions):
            if covariate_indices != None:
                covariate_indices_batch = {}
                for covariate in covariate_indices:
                    covariate_indices_batch[covariate] = covariate_indices[covariate][(i*batch_size):((i+1)*batch_size)] 
            else:
                covariate_indices_batch = None
                
            # Input to the sampling pre-defined size factors if provided to the function 
            if self.is_binarized:
                log_size_factor_batch = log_size_factor[(i*batch_size):((i+1)*batch_size)] if log_size_factor != None else None 
            else:
                if log_size_factor != None:
                    log_size_factor_batch = {} 
                    for mod in self.modality_list:
                        log_size_factor_batch[mod] = log_size_factor[mod][(i*batch_size):((i+1)*batch_size)] 
                else:
                    log_size_factor_batch = None
            
            # Sample batch 
            X_samples = self.sample(batch_size,
                                    n_sample_steps,
                                    theta_covariate, 
                                    size_factor_covariate,
                                    conditioning_covariates,
                                    covariate_indices_batch, 
                                    log_size_factor_batch, 
                                    unconditional)
                
            for mod in X_samples:
                total_samples[mod].append(X_samples[mod])                
        
        # Concatenate observations in the samples 
        return {mod: jnp.concat(total_samples[mod], axis=0) for mod in self.modality_list}                

    def _decode(self, z, size_factor):
        # Decode the rescaled z
        if self.is_binarized:
            size_factor = {"rna": size_factor}  # Compatibility with the decoder implementation for multimodal data 
        z = self.encoder_model.decode(z, size_factor)
        return z
    
    def sample_noise_like(self, x):
        return random.normal(self.make_rng("distr"), x.shape)

    def sample_location_and_conditional_flow(self, x0, x1, t=None):
        """
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1]
        with respect to the minibatch OT plan $\\Pi$.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]

        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        # Resample from OT coupling 
        if self.use_ot:
            assert False # NO U
            x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        # Sample time 
        if t is None:
            t = random.uniform(self.make_rng("distr"), x0.shape[0])
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        # Sample noise along straight line
        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        return t, xt, ut

    def sample_xt(self, x0, x1, t, epsilon):
        """
        Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        epsilon : Tensor, shape (bs, *dim)
            noise sample from N(0, 1)

        Returns
        -------
        xt : Tensor, shape (bs, *dim)

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def compute_mu_t(self, x0, x1, t):
        """
        Compute the mean of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: t * x1 + (1 - t) * x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        t = pad_t_like_x(t, x1)
        if self.use_ot:
            mu_t = t * x1 + (1 - t) * x0
        else:
            mu_t = t * x1
        return mu_t
    
    def compute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        standard deviation sigma

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        if self.use_ot:
            return self.sigma
        else:
            return 1 - (1 - self.sigma) * t
    
    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        if self.use_ot:
            return x1 - x0
        else:
            t = jnp.expand_dims(t, 1)
            return (x1 - (1 - self.sigma) * xt) / (1 - (1 - self.sigma) * t)


    def validation_step(self, batch, batch_idx):
        """
        Validation step for VDM.

        Args:
            batch: Batch data.
            batch_idx: Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        return self._step(batch, dataset='valid')
    
    def test_step(self, batch, batch_idx):
        """
        Training step for VDM.

        Args:
            batch: Batch data.
            batch_idx: Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        pass

    # @torch.no_grad()
    def compute_metrics_and_plots(self, data, dataset_type, *arg, **kwargs):
        """
        Concatenates all observations from the test data loader in a single dataset.

        Args:
            outputs: List of outputs from the test step.

        Returns:
            None
        """
        batch_size = 1000
        repetitions = batch_size // 100
        X_generated_dict = self.batched_sample(100, repetitions, 20, self.theta_covariate, self.size_factor_covariate, self.covariate_list) # TODO change sample steps back to 2?
        # Plot UMAP of generated cells and real test cells
        wd = compute_umap_and_wasserstein(X_generated_dict,
                                          X_real=data, plotting_folder=self.plotting_folder,
                                          epoch=0, # TODO fix
                                          modality_list=self.modality_list)
        
        metric_dict = {}
        for key in wd:
            metric_dict[f"{dataset_type}_{key}"] = wd[key]

        # Compute Wasserstein distance between real test set and generated data 
        # TODO self.log_dict(wd)
        return wd
    
