import torch.nn as nn

from modules.conditional_memory_module import ConditionalMemModule
from modules.memory_module import MemModule


class Memae(nn.Module):
    def __init__(
        self,
        num_targets,
        fea_dim,
        mem_dim,
        encoder,
        decoder,
        use_cosine_similarity,
        shrink_thres,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.mem_rep = (
            MemModule(
                mem_dim=mem_dim,
                fea_dim=fea_dim,
                shrink_thres=shrink_thres,
                use_cosine_similarity=use_cosine_similarity,
            )
            if num_targets == 1
            else ConditionalMemModule(
                num_targets=num_targets,
                mem_dim=mem_dim,
                fea_dim=fea_dim,
                shrink_thres=shrink_thres,
                use_cosine_similarity=use_cosine_similarity,
            )
        )

    def forward(self, x, condition_int=None):
        f = self.encoder(x)
        if condition_int is not None:
            res_mem = self.mem_rep(f, condition_int)
        else:
            res_mem = self.mem_rep(f)
        f = res_mem["output"]
        att = res_mem["att"]
        output = self.decoder(f)
        return output, att
