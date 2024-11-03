import os
from pathlib import Path
import uuid
import torch
from torch.utils.data import random_split

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.training import train_state
import optax

from cfgen.paths import TRAINING_FOLDER
from cfgen.data.scrnaseq_loader import RNAseqLoader
from cfgen.models.base.encoder_model import EncoderModel

# Helper class to include batch stats in flax train_state
class TrainState(train_state.TrainState):
  batch_stats: dict

class EncoderEstimator:
    """Class for training and using the cfgen model."""
    
    def __init__(self, args):
        """
        Initialize encoder Estimator.

        Args:
            args (Args): Configuration hyperparameters for the model.
        """
        # args is a dictionary containing the configuration hyperparameters 
        self.args = args
        
        # date and time to name run 
        self.unique_id = str(uuid.uuid4())
        
        # dataset path as Path object 
        self.data_path = Path(self.args.dataset.dataset_path)
        
        # Initialize training directory         
        self.training_dir = TRAINING_FOLDER / self.args.logger.project / self.unique_id
        print("Create the training folders...")
        self.training_dir.mkdir(parents=True, exist_ok=True)

        print("Initialize data module...")
        self.init_datamodule()  # Initialize the data module  
        self.get_fixed_rna_model_params()  # Initialize the data derived model parameters 
        self.init_trainer()
        
        print("Initialize model...")
        self.init_model()  # Initialize the model

    def init_datamodule(self):
        """
        Initialization of the data module.
        """        
        # Initialize the dataset using RNAseqLoader
        self.dataset = RNAseqLoader(self.data_path,
                                    layer_key=self.args.dataset.layer_key,
                                    covariate_keys=self.args.dataset.covariate_keys,
                                    subsample_frac=self.args.dataset.subsample_frac, 
                                    normalization_type=self.args.dataset.normalization_type,
                                    is_binarized=self.args.dataset.is_binarized)
        
        # Determine the number of categories for covariate-specific theta
        if self.args.encoder.covariate_specific_theta:
            self.n_cat = len(self.dataset.id2cov[self.args.dataset.theta_covariate])
        else:
            self.n_cat = None

        # Split the dataset into training and validation sets
        self.train_data, self.valid_data = random_split(self.dataset,
                                                        lengths=self.args.dataset.split_rates)   
        
        # TODO do we want to keep torch for dataloading?
        # Initialize the data loaders for training and validation
        self.train_dataloader = torch.utils.data.DataLoader(self.train_data,
                                                            batch_size=self.args.training_config.batch_size,
                                                            shuffle=True,
#                                                            num_workers=4, 
                                                            drop_last=True)
        
        self.valid_dataloader = torch.utils.data.DataLoader(self.valid_data,
                                                            batch_size=self.args.training_config.batch_size,
                                                            shuffle=False,
 #                                                           num_workers=4, 
                                                            drop_last=True)
    
    def get_fixed_rna_model_params(self):
        """Set the model parameters extracted from the data loader object.
        """
        # get the gene dimensions for each modality
        self.gene_dim = {mod: self.dataset.X[mod].shape[1] for mod in self.dataset.X}

    def init_trainer(self):
        """
        Initialize Trainer.
        """
        # TODO implement checkpointing / early stopping
        
        # TODO implement logging

    def init_model(self):
        """Initialize the encoder model.
        """
        # Initialize the model using the provided arguments and data-derived parameters
        self.encoder_model = EncoderModel(in_dim=self.gene_dim,
                                          n_cat=self.n_cat,
                                          conditioning_covariate=self.args.dataset.theta_covariate, 
                                          **self.args.encoder)
        print("Encoder architecture", self.encoder_model)

    def train(self):
        """
        Train the generative model using the provided trainer.
        """
        # Define the training step
        @jax.jit # TODO check if the model is really jit-able and make it if not
        def train_step(state, x):
            # Compute gradients
            loss, grads = jax.value_and_grad(state.apply_fn)({"params": state.params, "batch_stats": state.batch_stats}, x)
            # Apply gradients
            state = state.apply_gradients(grads=grads["params"])
            return state, loss

        # Initialize model parameters
        key = jax.random.PRNGKey(0)
        x = next(iter(self.train_dataloader))  # First batch for shape inference
        x = jax.tree.map(lambda tensor: tensor.numpy().astype(np.float32), x) # TODO this is hacky
        variables = self.encoder_model.init(key, x)
        params = variables["params"]
        batch_stats = variables["batch_stats"]

        # Set up the optimizer and training state
        optimizer = optax.adam(self.args.encoder.learning_rate)
        state = TrainState.create(
            apply_fn=self.encoder_model.apply,
            params=params,
            batch_stats=batch_stats,
            tx=optimizer
        )

        # Training loop
        for epoch in range(self.args.trainer.max_epochs):
            for batch in self.train_dataloader:
                batch = jax.tree.map(lambda tensor: tensor.numpy().astype(np.float32), batch) # TODO this is hacky
                state, loss = train_step(state, batch)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
            self.test({"params": state.params, "batch_stats": state.batch_stats})

        self.final_checkpoint = {"params": state.params, "batch_stats": state.batch_stats}

    def test(self, variables=None):
        """
        Test the generative model.
        """

        if not variables:
            if not hasattr(self, "final_activation"):
                raise ValueError("You need to train the model or suppy a checkpoint")
            else:
                variables = self.final_checkpoint

        loss = 0.0
        for batch in self.valid_dataloader:
            batch = jax.tree.map(lambda tensor: tensor.numpy().astype(np.float32), batch) # TODO this is hacky
            loss += self.encoder_model.apply(variables, batch) # TODO jit

        print("Test Error %f" % loss)


    # TODO temporary helper, remove
    def reconstruct_dataset(self):
        decoded = []
        for batch in self.valid_dataloader:
            batch = jax.tree.map(lambda tensor: tensor.numpy().astype(np.float32), batch) # TODO this is hacky
            X = batch["X"]
            size_factor = {mod: jnp.expand_dims(X[mod].sum(1), 1) for mod in X}
            encoded = self.encoder_model.apply(self.final_checkpoint, batch, method=self.encoder_model.encode)
            decoded.append(self.encoder_model.apply(self.final_checkpoint, encoded, size_factor, method=self.encoder_model.decode)["rna"])
        return np.concatenate(decoded, axis=0)

    # TODO temporary helper, remove
    def umaps(self):
        import scanpy as sc
        from scvi.distributions import JaxNegativeBinomialMeanDisp as NegativeBinomial
        from matplotlib import pyplot as plt
        import pandas as pd

        orig = self.valid_data.dataset[self.valid_data.indices]["X"]["rna"]
        gen_mu = self.reconstruct_dataset()
        sampled = np.array(NegativeBinomial(gen_mu, jnp.exp(self.final_checkpoint["params"]["theta"])).sample(jax.random.PRNGKey(0)))

        adata_original_rna = sc.AnnData(X=orig)
        adata_generated_rna = sc.AnnData(X=sampled)

        obs = pd.DataFrame(["real" for _ in range(len(orig))]+["generated" for _ in range(len(sampled))])
        obs.columns = ["dataset_type"]
        adata_rna = sc.AnnData(np.concatenate([orig, sampled], axis=0), obs=obs)

        sc.pp.log1p(adata_rna)
        sc.tl.pca(adata_rna)
        sc.pp.neighbors(adata_rna)
        sc.tl.umap(adata_rna)
        sc.pl.pca(adata_rna, color="dataset_type")
        sc.pl.umap(adata_rna, color="dataset_type")
        plt.show()
        
