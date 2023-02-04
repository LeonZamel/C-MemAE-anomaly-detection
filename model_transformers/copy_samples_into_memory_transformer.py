import torch
from pytorch_lightning.core.datamodule import LightningDataModule

from model_transformers.model_transformer import ModelTransformer
from models.memae import Memae
from modules.memory_module import MemModule
from systems.memae_autoencoder_system import MemaeSystem


class CopySamplesIntoMemoryTransformer(ModelTransformer):
    def transform(self, model: Memae, target, datamodule: LightningDataModule):
        assert isinstance(model, Memae), "Provided model must be of type MemAE"
        dm = datamodule
        dm.prepare_data()
        dm.setup()
        dl = dm.train_dataloader()

        all_samples = []
        for x, y in dl:
            all_samples.append(x)
        all_samples = torch.cat(all_samples)

        # assert (
        #     all_samples.shape[0] == model.mem_rep.mem_dim
        # ), "Memory size isn't equal to number of training samples"

        encoded = model.encoder(all_samples).squeeze()

        model.mem_rep = MemModule(
            mem_dim=all_samples.shape[0],
            fea_dim=model.mem_rep.fea_dim,
            shrink_thres=model.mem_rep.shrink_thres,
            use_cosine_similarity=model.mem_rep.use_cosine_similarity,
        )
        model.mem_rep.memory.weight.data = encoded
        return model
