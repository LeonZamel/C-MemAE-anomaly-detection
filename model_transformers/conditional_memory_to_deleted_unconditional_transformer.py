from pytorch_lightning.core.datamodule import LightningDataModule

from model_transformers.model_transformer import ModelTransformer
from models.memae import Memae
from modules.conditional_memory_module import ConditionalMemModule
from modules.memory_module import MemModule


class ConditionalMemoryToDeletedUnconditionalTransformer(ModelTransformer):
    def transform(self, model: Memae, target, datamodule: LightningDataModule):
        assert isinstance(model, Memae), "Provided model must be of type MemAE"
        assert isinstance(model.mem_rep, ConditionalMemModule), "Memory should be conditional"
        model.mem_rep = MemModule(
            mem_dim=model.mem_rep.mem_dim,
            fea_dim=model.mem_rep.fea_dim,
            shrink_thres=model.mem_rep.shrink_thres,
            use_cosine_similarity=model.mem_rep.use_cosine_similarity,
        )

        return model
