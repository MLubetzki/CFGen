import os
from pathlib import Path
import uuid
import torch
from torch.utils.data import random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from celldreamer.paths import TRAINING_FOLDER
from celldreamer.data.scrnaseq_loader import RNAseqLoader
from celldreamer.models.featurizers.category_featurizer import CategoricalFeaturizer
from celldreamer.models.fm.denoising_model import MLPTimeStep
from celldreamer.models.fm.fm import FM
from celldreamer.models.base.encoder_model import EncoderModel

# Some general settings for the run
os.environ["WANDB__SERVICE_WAIT"] = "300"
torch.autograd.set_detect_anomaly(True)

class CellDreamerEstimator:
    """Class for training and using the CellDreamer model."""
    def __init__(self, args):
        """
        Initialize the CellDreamerEstimator.

        Args:
            args (Args): Configuration hyperparameters for the model.
        """
        # args is a dictionary containing the configuration hyperparameters 
        self.args = args
        
        # date and time to name run 
        self.unique_id = str(uuid.uuid4())
        
        # dataset path as Path object 
        self.data_path = Path(self.args.dataset.dataset_path)
        self.multimodal = self.args.dataset.multimodal
        self.is_binarized = self.args.encoder.is_binarized
        
        # Initialize training directory         
        self.training_dir = TRAINING_FOLDER / self.args.logger.project / self.unique_id
        self.plotting_dir = self.training_dir / "plots"
        print("Create the training folders...")
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.plotting_dir.mkdir(exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
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
        self.dataset = RNAseqLoader(data_path=self.data_path,
                                    layer_key=self.args.dataset.layer_key,
                                    covariate_keys=self.args.dataset.covariate_keys,
                                    subsample_frac=self.args.dataset.subsample_frac, 
                                    encoder_type=self.args.dataset.encoder_type,
                                    multimodal=self.multimodal,
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
        if not self.dataset.multimodal:
            # If not multimodal, gene dimension and input dimension computed only for RNA
            self.gene_dim = self.dataset.X.shape[1] 
            self.in_dim = self.gene_dim if self.args.dataset.encoder_type!="learnt_autoencoder" else self.args.encoder.encoder_kwargs["dims"][-1]
            self.modality_list = None 
        else:
            self.gene_dim = {mod: self.dataset.X[mod].shape[1] for mod in self.dataset.X}
            self.modality_list = list(self.gene_dim.keys())
            self.in_dim = {}
            if not self.args.encoder.encoder_multimodal_joint_layers:
                for mod in self.dataset.X:
                    if self.args.dataset.encoder_type!="learnt_autoencoder":
                        self.in_dim[mod] = self.gene_dim[mod]
                    else:
                        self.in_dim[mod] = self.args.encoder.encoder_kwargs[mod]["dims"][-1]
            else:
                self.in_dim = self.args.encoder.encoder_multimodal_joint_layers["dims"][-1]

    def init_trainer(self):
        """
        Initialize Trainer
        """
        # Callbacks for saving checkpoints 
        checkpoint_callback = ModelCheckpoint(dirpath=self.training_dir / "checkpoints", 
                                                **self.args.checkpoints)
        callbacks = [checkpoint_callback]
        
        # Early stopping checkpoints 
        if self.args.training_config.use_early_stopping:
            early_stopping_callbacks = EarlyStopping(**self.args.early_stopping)
            callbacks.append(early_stopping_callbacks)
        
        # Logger settings 
        self.logger = WandbLogger(save_dir=self.training_dir,
                                    name=self.unique_id, 
                                    **self.args.logger)
        
        self.trainer_generative = Trainer(callbacks=callbacks, 
                                          default_root_dir=self.training_dir, 
                                          logger=self.logger,
                                          **self.args.trainer)
            
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
                                                                    self.device, 
                                                                    embedding_dimensions=self.args.denoising_module.embedding_dim)
            if self.args.dataset.one_hot_encode_features:
                self.num_classes[cov] = len(cov_names)
            else:
                self.num_classes[cov] = self.args.denoising_module.embedding_dim

    def init_model(self):
        """Initialize the (optional) autoencoder and generative model 
        """
        # Initialize denoising model 
        conditioning_cov = self.args.dataset.conditioning_covariate  
        if not self.dataset.multimodal or (self.dataset.multimodal and self.is_binarized):
            size_factor_statistics = {"mean": self.dataset.log_size_factor_mu, 
                                        "sd": self.dataset.log_size_factor_sd}
        else:
            size_factor_statistics = {"mean": {mod: self.dataset.log_size_factor_mu[mod] for mod in self.dataset.log_size_factor_mu}, 
                                        "sd": {mod: self.dataset.log_size_factor_sd[mod] for mod in self.dataset.log_size_factor_sd}}
                
        # scaler = self.dataset.get_scaler()
        
        # Initialize the deoising model 
        denoising_model = MLPTimeStep(in_dim=sum(self.in_dim.values()) if (self.multimodal and not self.args.encoder.encoder_multimodal_joint_layers) else self.in_dim, 
                                        hidden_dim=self.args.denoising_module.hidden_dim,
                                        dropout_prob=self.args.denoising_module.dropout_prob,
                                        n_blocks=self.args.denoising_module.n_blocks, 
                                        model_type=self.args.denoising_module.model_type, 
                                        size_factor_min=self.dataset.min_size_factor, 
                                        size_factor_max=self.dataset.max_size_factor,
                                        embedding_dim=self.args.denoising_module.embedding_dim,
                                        normalization=self.args.denoising_module.normalization,
                                        conditional=self.args.denoising_module.conditional, 
                                        multimodal=self.dataset.multimodal, 
                                        is_binarized=self.is_binarized, 
                                        modality_list=self.modality_list, 
                                        embed_size_factor=self.args.denoising_module.embed_size_factor).to(self.device)
        
        print("Denoising model", denoising_model)
        
        # Initialize encoder
        self.encoder_model = EncoderModel(in_dim=self.gene_dim,
                                          n_cat=self.feature_embeddings[self.args.dataset.conditioning_covariate].n_cat,
                                          conditioning_covariate=self.args.dataset.conditioning_covariate, 
                                          encoder_type=self.args.dataset.encoder_type,
                                          **self.args.encoder)
        print("Encoder architecture", self.encoder_model)
    
        # If model is pre-trained, load weights
        if self.args.training_config.encoder_ckpt != None:
            # Load weights 
            print(f"Load checkpoints from {self.args.training_config.encoder_ckpt}")
            self.encoder_model.load_state_dict(torch.load(self.args.training_config.encoder_ckpt)["state_dict"])
            # Freeze encoder 
            for param in self.encoder_model.parameters():
                param.requires_grad = False
        self.encoder_model.eval()
            
        # Flow matching model
        self.generative_model = FM(
            encoder_model=self.encoder_model,
            denoising_model=denoising_model,
            feature_embeddings=self.feature_embeddings,
            plotting_folder=self.plotting_dir,
            in_dim=self.in_dim,
            size_factor_statistics=size_factor_statistics,
            encoder_type=self.args.dataset.encoder_type,
            conditioning_covariate=conditioning_cov,
            model_type=denoising_model.model_type, 
            multimodal=self.dataset.multimodal,
            is_binarized=self.is_binarized,
            modality_list=self.modality_list,
            **self.args.generative_model  # model_kwargs should contain the rest of the arguments
            )

    def train(self):
        """
        Train the generative model using the provided trainer.
        """
        self.trainer_generative.fit(
            self.generative_model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.valid_dataloader)
    
    def test(self):
        """
        Test the generative model.
        """
        self.trainer_generative.test(
            self.generative_model,
            dataloaders=self.valid_dataloader)
    