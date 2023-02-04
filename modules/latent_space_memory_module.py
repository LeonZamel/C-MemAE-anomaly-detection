# Deprecated
# This was a test for performing anomaly detection in latent space directly by comparing the encoding to memory entries


from __future__ import absolute_import, print_function

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from modules.memory_module import (
    batchify,
    hard_shrink_relu,
    to_channel_first,
    to_channel_last,
    unbatchify,
)


class LatentSpaceMemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, use_cosine_similarity, shrink_thres):
        super().__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        self.bias = None
        self.shrink_thres = shrink_thres
        self.use_cosine_similarity = use_cosine_similarity

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        memory = self.weight.unsqueeze(0)
        z = input.unsqueeze(1)
        att_weight = F.cosine_similarity(z, memory, dim=2)  # Automatic broadcasting over first two dims

        pre_softmax_att = att_weight

        att_weight = F.softmax(att_weight, dim=1)  # BxM
        # ReLU based shrinkage, hard shrinkage for positive value
        if self.shrink_thres > 0:
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)

        mem_trans = self.weight.permute(1, 0)  # Mem^T, CxM
        output = F.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (BxM) x (MxC) = BxC
        return {
            "output": output,
            "att": att_weight,
            "pre_softmax_att": pre_softmax_att,
        }  # output, att_weight

    def extra_repr(self):
        return "mem_dim={}, fea_dim={}".format(self.mem_dim, self.fea_dim is not None)


# NxCxHxW -> (NxHxW)xC -> addressing Mem, (NxHxW)xC -> NxCxHxW
class LatentSpaceMemModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, use_cosine_similarity, shrink_thres):
        super().__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.use_cosine_similarity = use_cosine_similarity
        self.memory = LatentSpaceMemoryUnit(
            self.mem_dim,
            self.fea_dim,
            self.use_cosine_similarity,
            self.shrink_thres,
        )

    def forward(self, input):
        num_dims = len(input.shape)
        x = to_channel_last(input)
        x, batched_dims = batchify(x, num_dims - 1)

        y_and = self.memory(x)

        y = y_and["output"]
        att = y_and["att"]
        pre_softmax_att = y_and["pre_softmax_att"]

        y = unbatchify(y, batched_dims)
        att = unbatchify(att, batched_dims)
        pre_softmax_att = unbatchify(pre_softmax_att, batched_dims)
        y = to_channel_first(y)
        att = to_channel_first(att)
        pre_softmax_att = to_channel_first(pre_softmax_att)

        pre_softmax_att = pre_softmax_att.sum(dim=[*range(1, num_dims)])

        return {"output": y, "att": att, "pre_softmax_att": pre_softmax_att}
