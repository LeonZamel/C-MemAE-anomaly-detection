from pytorch_lightning.core.datamodule import LightningDataModule

from model_transformers.model_transformer import ModelTransformer
from models.memae import Memae
from systems.memae_autoencoder_system import MemaeSystem


class ModelFreezer(ModelTransformer):
    def transform(self, model: Memae, target, datamodule: LightningDataModule):
        assert isinstance(model, Memae), "Provided model must be of type MemAE"
        #encoder freeze
        for p in model.encoder.parameters():
            p.requires_grad = False 
        #decoder freeze
        for p in model.decoder.parameters():
            p.requires_grad = False 
        #memory freeze
        for p in model.mem_rep.parameters():
            p.requires_grad = False 
        #model.mem_rep.memory.reset_parameters()
        return model
