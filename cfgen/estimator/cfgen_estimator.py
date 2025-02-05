from pathlib import Path
import uuid
import math
import logging
import numpy as np
import torch
from torch.utils.data import random_split
from cfgen.paths import TRAINING_FOLDER
from cfgen.data.scrnaseq_loader import RNAseqLoader
from cfgen.models.featurizers.category_featurizer import CategoricalFeaturizer
from cfgen.models.fm.denoising_model import MLPTimeStep
from cfgen.models.fm.fm import FM
from cfgen.models.base.encoder_model import EncoderModel

import jax
import flax.linen as nn
from flax.training import train_state, orbax_utils
import optax
from tqdm import tqdm

import orbax.checkpoint as ocp

# Helper class to include batch stats in flax train_state
class TrainState(train_state.TrainState):
  batch_stats: dict


class CfgenEstimator:
    """Class for training and using the cfgen model."""
    def __init__(self, args):
        """
        Initialize the CFGgen Estimator.

        Args:
            args (Args): Configuration hyperparameters for the model.
        """
        # args is a dictionary containing the configuration hyperparameters 
        self.args = args
        
        # date and time to name run 
        self.unique_id = str(uuid.uuid4())
        
        # dataset path as Path object 
        self.data_path = Path(self.args.dataset.dataset_path)
        self.is_binarized = self.args.encoder.is_binarized
        
        # Initialize training directory         
        self.training_dir = TRAINING_FOLDER / self.args.logger.project / self.unique_id
        self.plotting_dir = self.training_dir / "plots"
        print("Create the training folders...")
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.plotting_dir.mkdir(exist_ok=True)

        print("Initialize data module...")
        self.init_datamodule()  # Initialize the data module  
        self.get_fixed_rna_model_params()  # Initialize the data derived model params 
        self.init_trainer()
        
        print("Initialize feature embeddings...")
        self.init_feature_embeddings()  # Initialize the feature embeddings 
        
        print("Initialize model...")
        self.init_model()  # Initialize

    def init_datamodule(self):
        """
        Initialization of the data module
        """        
        # Initialize dataloaders for the different tasks 
        self.dataset = RNAseqLoader(self.data_path,
                                    layer_key=self.args.dataset.layer_key,
                                    covariate_keys=self.args.dataset.covariate_keys,
                                    subsample_frac=self.args.dataset.subsample_frac, 
                                    normalization_type=self.args.dataset.normalization_type,
                                    is_binarized=self.is_binarized)

        # Initialize the data loaders 
        self.train_data, self.valid_data = random_split(self.dataset,
                                                        lengths=self.args.dataset.split_rates)   
        
        self.train_dataloader = torch.utils.data.DataLoader(self.train_data,
                                                            batch_size=self.args.training_config.batch_size,
                                                            shuffle=True,
                                                            num_workers=4, 
                                                            drop_last=True)
        
        self.valid_dataloader = torch.utils.data.DataLoader(self.valid_data,
                                                            batch_size=self.args.training_config.batch_size,
                                                            shuffle=False,
                                                            num_workers=4, 
                                                            drop_last=True)
    
    def get_fixed_rna_model_params(self):
        """Set the model parameters extracted from the data loader object
        """
        self.gene_dim = {mod: self.dataset.X[mod].shape[1] for mod in self.dataset.X}
        self.modality_list = list(self.gene_dim.keys())
        self.in_dim = {}
        if not hasattr(self.args.encoder, "encoder_multimodal_joint_layers") or not self.args.encoder.encoder_multimodal_joint_layers:  # Optional latent space shared between modalities
            for mod in self.dataset.X:
                self.in_dim[mod] = self.args.encoder.encoder_kwargs[mod]["dims"][-1]
        else:
            self.in_dim = self.args.encoder.encoder_multimodal_joint_layers["dims"][-1]

    def init_trainer(self):
        """
        Initialize Trainer
        """
        self.checkpointer = ocp.PyTreeCheckpointer()
        logging.getLogger("absl").setLevel(logging.WARNING) # silence orbax logs
            
    def init_feature_embeddings(self):
        """
        Initialize feature embeddings either for drugs or covariates 
        """
        # Contains the embedding class of multiple feature types
        self.feature_embeddings = {}  
        self.num_classes = {}
                
        for cov, cov_names in self.dataset.id2cov.items():
            self.feature_embeddings[cov] = CategoricalFeaturizer(len(cov_names), 
                                                                    self.args.dataset.one_hot_encode_features, 
                                                                    embedding_dimensions=self.args.denoising_module.embedding_dim)
            if self.args.dataset.one_hot_encode_features:
                self.num_classes[cov] = len(cov_names)
            else:
                self.num_classes[cov] = self.args.denoising_module.embedding_dim

    def init_model(self):
        """Initialize the (optional) autoencoder and generative model 
        """
        # Initialize denoising model 
        if self.is_binarized:
            size_factor_statistics = {"mean": self.dataset.log_size_factor_mu, 
                                        "sd": self.dataset.log_size_factor_sd}
        else:
            size_factor_statistics = {"mean": {mod: self.dataset.log_size_factor_mu[mod] for mod in self.dataset.log_size_factor_mu}, 
                                        "sd": {mod: self.dataset.log_size_factor_sd[mod] for mod in self.dataset.log_size_factor_sd}}
                

        # Initialize the deoising model 
        denoising_model = MLPTimeStep(in_dim=sum(self.in_dim.values()) if type(self.in_dim) == dict else self.in_dim, 
                                        hidden_dim=self.args.denoising_module.hidden_dim,
                                        dropout_prob=self.args.denoising_module.dropout_prob,
                                        n_blocks=self.args.denoising_module.n_blocks, 
                                        size_factor_min=self.dataset.min_size_factor, 
                                        size_factor_max=self.dataset.max_size_factor,
                                        embed_size_factor=self.args.denoising_module.embed_size_factor, 
                                        covariate_list=self.args.dataset.covariate_keys,
                                        embedding_dim=self.args.denoising_module.embedding_dim,
                                        normalization=self.args.denoising_module.normalization,
                                        conditional=self.args.denoising_module.conditional, 
                                        is_binarized=self.is_binarized, 
                                        modality_list=self.modality_list, 
                                        guided_conditioning=self.args.denoising_module.guided_conditioning)
        
        print("Denoising model", denoising_model)
        
        # Initialize encoder
        self.encoder_model = EncoderModel(in_dim=self.gene_dim,
                                          n_cat=self.feature_embeddings[self.args.dataset.theta_covariate].n_cat,
                                          conditioning_covariate=self.args.dataset.theta_covariate, 
                                          **self.args.encoder)
        print("Encoder architecture", self.encoder_model)
        print(self.encoder_model)
    
        # If model is pre-trained, load weights
        if self.args.training_config.encoder_ckpt != None:
            self.encoder_checkpointer = ocp.PyTreeCheckpointer()
            print(f"Load checkpoints from {self.args.training_config.encoder_ckpt}")
            self.encoder_checkpoint = self.encoder_checkpointer.restore(self.args.training_config.encoder_ckpt)

            
        # Flow matching model
        self.generative_model = FM(
            encoder_model=self.encoder_model,
            denoising_model=denoising_model,
            feature_embeddings=self.feature_embeddings,
            plotting_folder=self.plotting_dir,
            in_dim=self.in_dim,
            size_factor_statistics=size_factor_statistics,
            covariate_list=self.args.dataset.covariate_keys, 
            theta_covariate=self.args.dataset.theta_covariate,
            size_factor_covariate=self.args.dataset.size_factor_covariate,
            is_binarized=self.is_binarized,
            modality_list=self.modality_list,
            guidance_weights=self.args.dataset.guidance_weights,
            **self.args.generative_model  # model_kwargs should contain the rest of the arguments
            )        

    def train(self):
        """
        Train the generative model using the provided trainer.
        """
        # self.trainer_generative.fit(
        #     self.generative_model,
        #     train_dataloaders=self.train_dataloader,
        #     val_dataloaders=self.valid_dataloader)

        # Initialize model parameters
        key = jax.random.PRNGKey(42) # TODO allow setting a seed (to be reproducible, dataloader must be taken into consideration)
        x = next(iter(self.train_dataloader))  # First batch for shape inference
        x = jax.tree.map(lambda tensor: tensor.numpy().astype(np.float32), x) # TODO this is hacky
        variables = self.generative_model.init(key, x, "train", train=True)
        params = variables["params"]
        batch_stats = variables["batch_stats"]

        # Restore encoder checkpoint
        params["encoder_model"] = self.encoder_checkpoint["model"]["params"] # TODO it can't be intended that we have to do it this way. Read orbax docs!
        batch_stats["encoder_model"] = self.encoder_checkpoint["model"]["batch_stats"]

        # Set up the optimizer and training state
        optimizer = optax.adam(self.args.generative_model.learning_rate)
        state = TrainState.create(
            apply_fn=self.generative_model.apply,
            params=params,
            batch_stats=batch_stats,
            tx=optimizer
        )

        # Prepare checkpointing
        ckpt = {"model": state}
        self.orbax_save_args = orbax_utils.save_args_from_target(ckpt)

        lowest_test_loss = math.inf
        # Training loop
        for epoch in range(self.args.trainer.max_epochs):
            for batch in tqdm(self.train_dataloader):
                batch = jax.tree.map(lambda tensor: tensor.numpy().astype(np.float32), batch) # TODO this is hacky
                state, loss = self._train_step(state, batch)

            test_loss = self.test({"params": state.params, "batch_stats": state.batch_stats})
            print(f"Epoch {epoch}, train error: {loss:.4f}, test error: {test_loss:.4f}")

            if test_loss < lowest_test_loss:
                lowest_test_loss = test_loss
                ckpt = {"model": state}
                self.checkpointer.save(self.training_dir / "checkpoints" / "fm" / "early_stopping_checkpoint", ckpt, save_args=self.orbax_save_args, force=True)


        self.final_model = {"params": state.params, "batch_stats": state.batch_stats}
        final_checkpoint = {"model": state}
        self.checkpointer.save(self.training_dir / "checkpoints" /"fm" / "final_checkpoint", final_checkpoint, save_args=self.orbax_save_args)

    
#    @partial(jax.jit, static_argnums=0)
    def _train_step(self, state, x):

        def loss_fn(params, batch_stats, x):
            return state.apply_fn(
                {"params": params, "batch_stats": batch_stats},
                x,
                dataset="train",
                train=True,
                rngs={"time_sampling": jax.random.key(42), "noise": jax.random.key(1337), "guiding": jax.random.key(69)}, # TODO fix seeding
                mutable="batch_stats")


        # TODO make certain that the autoencoder parameters are not updated
        (loss, updates), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, state.batch_stats, x)
        
        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=updates['batch_stats'])

        return state, loss
    
    # @partial(jax.jit, static_argnums=0)
    def _valid_step(self, variables, batch):
        return self.generative_model.apply(variables, batch, dataset="test", train=False, rngs={"time_sampling": jax.random.key(42), "noise": jax.random.key(1337), "guiding": jax.random.key(69)},)

    def test(self, variables=None):
        """
        Test the generative model.
        """
        if not variables:
            if not hasattr(self, "final_model"):
                raise ValueError("You need to train the model or supply a checkpoint")
            else:
                variables = self.final_model

        loss = 0.0
        for batch in self.valid_dataloader:
            batch = jax.tree.map(lambda tensor: tensor.numpy().astype(np.float32), batch) # TODO this is hacky
            loss += self._valid_step(variables, batch)


        self.generative_model.apply(variables,
                                    self.valid_data[:]["X"],
                                    "test",
                                    method=FM.compute_metrics_and_plots,
                                    rngs={"time_sampling": jax.random.key(42), "noise": jax.random.key(1337), "guiding": jax.random.key(69)}
                                    )
 
        return loss
    

    # TODO stolen from fm.py 
    def configure_optimizers(self):
        """
        Optimizer configuration 

        Returns:
            dict: Optimizer configuration.
        """
        params = list(self.parameters())
        
        for covariate in self.feature_embeddings:
            if not self.feature_embeddings[covariate].one_hot_encode_features:
                params += list(self.feature_embeddings[covariate].parameters())
                 
        optimizer = torch.optim.AdamW(params, 
                                    self.learning_rate, 
                                    weight_decay=self.weight_decay)
        return optimizer