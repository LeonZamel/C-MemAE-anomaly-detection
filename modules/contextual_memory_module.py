# Deprecated
# This memory can be used when the data is not flattened during encoding. There is a memory module for each input vector 

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from modules.memory_module import hard_shrink_relu, to_channel_first, to_channel_last


class ContextualMemoryUnit(nn.Module):
    def __init__(
        self,
        mem_dim,
        context_dims,
        fea_dim,
        use_cosine_similarity,
        shrink_thres,
    ):
        super().__init__()
        self.mem_dim = mem_dim
        self.context_dims = context_dims
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(self.mem_dim, *self.context_dims, self.fea_dim))
        self.use_cosine_similarity = use_cosine_similarity
        self.shrink_thres = shrink_thres

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        last_dim_ind = len(input.shape) - 1
        if self.use_cosine_similarity:
            # Paper uses cosine similarity
            memory = self.weight.unsqueeze(0)
            z = input.unsqueeze(1)
            att_weight = F.cosine_similarity(
                z, memory, dim=last_dim_ind + 1
            )  # Automatic broadcasting over first two dims
        else:
            att_weight = F.linear(input, self.weight)  # Fea x Mem^T, (BxC) x (CxM) = BxM

        att_weight = F.softmax(att_weight, dim=1)  # TxM
        # ReLU based shrinkage, hard shrinkage for positive value
        if self.shrink_thres > 0:
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)

        exp_att_weight = att_weight.unsqueeze(last_dim_ind + 1)
        exp_weight = self.weight.unsqueeze(0)
        prod = exp_att_weight * exp_weight
        output = prod.sum(dim=1)

        return {"output": output, "att": att_weight}  # output, att_weight

    def extra_repr(self):
        return "mem_dim={}, fea_dim={}".format(self.mem_dim, self.fea_dim is not None)


# NxCxHxW -> (NxHxW)xC -> addressing Mem, (NxHxW)xC -> NxCxHxW
class ContextualMemModule(nn.Module):
    def __init__(
        self,
        mem_dim,
        context_dims,
        fea_dim,
        use_cosine_similarity,
        shrink_thres,
    ):
        super().__init__()
        self.mem_dim = mem_dim
        self.context_dims = context_dims
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.use_cosine_similarity = use_cosine_similarity

        self.memory = ContextualMemoryUnit(self.mem_dim, self.context_dims, self.fea_dim, self.shrink_thres)

    def forward(self, input):
        x = to_channel_last(input)

        y_and = self.memory(x)

        y = y_and["output"]
        att = y_and["att"]

        y = to_channel_first(y)
        att = to_channel_first(att)

        return {"output": y, "att": att}
