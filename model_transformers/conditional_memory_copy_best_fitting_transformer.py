import torch
from pytorch_lightning.core.datamodule import LightningDataModule

from model_transformers.model_transformer import ModelTransformer
from models.memae import Memae
from modules.conditional_memory_module import ConditionalMemModule
from modules.memory_module import MemModule, MemoryUnit


class ConditionalMemoryCopyBestFittingTransformer(ModelTransformer):
    def transform(self, model: Memae, target, datamodule: LightningDataModule):
        assert isinstance(model, Memae), "Provided model must be of type MemAE"
        assert isinstance(model.mem_rep, ConditionalMemModule), "Memory should be conditional"
        dm = datamodule
        dm.prepare_data()
        dm.setup()
        dl = dm.train_dataloader()

        all_samples = []
        for x, y in dl:
            all_samples.append(x)
        all_samples = torch.cat(all_samples)

        encoded = model.encoder(all_samples).squeeze()

        outputs = []
        mem_data = []
        mem_unit: MemoryUnit
        for mem_unit in model.mem_rep.memory:
            outputs.append(mem_unit(encoded)["pre_softmax_att"])
            mem_data.append(mem_unit.weight.data)
        outputs = torch.cat(outputs, dim=1)
        mem_data = torch.cat(mem_data)

        # Pick the best fitting memory items. We need as many as we have samples
        # i.e. all_samples.shape[0]
        best = torch.max(outputs, dim=1)

        model.mem_rep = MemModule(
            mem_dim=all_samples.shape[0],
            fea_dim=model.mem_rep.fea_dim,
            shrink_thres=model.mem_rep.shrink_thres,
            use_cosine_similarity=model.mem_rep.use_cosine_similarity,
        )
        model.mem_rep.memory.weight.data = mem_data[best.indices]

        return model
