import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class MemoryUnit(nn.Module):
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
        if self.use_cosine_similarity:
            # Paper uses cosine similarity
            memory = self.weight.unsqueeze(0) # shape: 1xNxC --> N=Memory size, C = vector length/feature dimension
            z = input.unsqueeze(1) # shape: Tx1xC --> T=batch size, C = vector length/feature dimension
            att_weight = F.cosine_similarity(z, memory, dim=2)  # Automatic broadcasting over first two dims
        else:
            att_weight = F.linear(
                input, self.weight
            )  # Code by Gong using inner product, Fea x Mem^T, (BxC) x (CxM) = BxM

        # Create a clone so that this can be provided as additional output
        pre_softmax_att = att_weight.clone().detach()

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
        }

    def extra_repr(self):
        return "mem_dim={}, fea_dim={}".format(self.mem_dim, self.fea_dim is not None)


# The memory module encapsules the reshaping of the input data, before handing it off to the memory unit
class MemModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, use_cosine_similarity, shrink_thres):
        super().__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.use_cosine_similarity = use_cosine_similarity
        self.memory = MemoryUnit(
            self.mem_dim,
            self.fea_dim,
            self.use_cosine_similarity,
            self.shrink_thres,
        )

    def forward(self, input):
        # All dimensions except for the channels/feature dimension are collapsed into the batch dimension.
        # This way we are agnostic to any additional dimensions like e.g. image width or height
        # If the additional dimensions are already of size 1, they are basically just flattened/squeezed

        # Batch x Channels x D1 x D2 x D3 x ... -> (Batch * D1 * D2 * D3 * ... ) x Channels -> addressing Mem,
        # (Batch x D1 x D2 x D3 x ... ) x Channels -> Batch x Channels x D1 x D2 x D3 x ...
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

        return {"output": y, "att": att, "pre_softmax_att": pre_softmax_att}


# ReLU based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output


# Helpers
def to_channel_last(tensor):
    num_dims = len(tensor.shape)
    return tensor.permute(0, *range(2, num_dims), 1)


def to_channel_first(tensor):
    num_dims = len(tensor.shape)
    return tensor.permute(0, num_dims - 1, *range(1, num_dims - 1))


def batchify(tensor, n):
    # Flattens the first n dims into one, returns new tensor and shape of collapsed dims
    return tensor.reshape(-1, *tensor.shape[n:]), tensor.shape[:n]


def unbatchify(tensor, shape):
    # Unflattens first dim into shape
    rest = tensor.shape[1:]
    return tensor.reshape(*shape, *rest)
