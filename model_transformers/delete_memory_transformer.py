from pytorch_lightning.core.datamodule import LightningDataModule

from model_transformers.model_transformer import ModelTransformer
from models.memae import Memae
from systems.memae_autoencoder_system import MemaeSystem


class DeleteMemoryTransformer(ModelTransformer):
    def transform(self, model: Memae, target, datamodule: LightningDataModule):
        assert isinstance(model, Memae), "Provided model must be of type MemAE"
        model.mem_rep.memory.reset_parameters()
        return model
