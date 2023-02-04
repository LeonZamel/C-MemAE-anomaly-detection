import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ae_cifar import decoder, encoder
from modules.contextual_memory_module import ContextualMemModule


class ContextualMemaeCIFAR(nn.Module):
    def __init__(self, mem_dim, shrink_thres=0.0025):
        super().__init__()

        self.encoder = encoder()
        self.mem_rep = ContextualMemModule(
            mem_dim=mem_dim,
            context_dims=(2, 2),
            fea_dim=256,
            shrink_thres=shrink_thres,
        )
        self.decoder = decoder()

    def forward(self, x):
        f = self.encoder(x)
        res_mem = self.mem_rep(f)
        f = res_mem["output"]
        att = res_mem["att"]
        output = self.decoder(f)
        return output, att
