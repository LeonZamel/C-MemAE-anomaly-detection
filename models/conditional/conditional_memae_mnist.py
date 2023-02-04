# Deprecated

from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ae_mnist import decoder, encoder
from modules.conditional_memory_module import ConditionalMemModule


class ConditionalMemaeMNIST(nn.Module):
    def __init__(self, num_targets, mem_dim, use_cosine_similarity, shrink_thres=0.0025):
        super().__init__()
        assert False

        self.encoder = encoder()
        self.mem_rep = ConditionalMemModule(
            num_targets,
            mem_dim=mem_dim,
            fea_dim=8,
            shrink_thres=shrink_thres,
            use_cosine_similarity=use_cosine_similarity,
        )
        self.decoder = decoder()

    def forward(self, x, condition_int=None):
        f = self.encoder(x)
        res_mem = self.mem_rep(f, condition_int)
        f = res_mem["output"]
        att = res_mem["att"]
        output = self.decoder(f)
        return output, att
