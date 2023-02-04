import torch
from torch import nn

from modules.memory_module import (
    MemoryUnit,
    batchify,
    to_channel_first,
    to_channel_last,
    unbatchify,
)


class ConditionalMemModule(nn.Module):
    def __init__(
        self,
        num_targets,
        mem_dim,
        fea_dim,
        use_cosine_similarity,
        shrink_thres,
    ):
        super().__init__()
        self.num_targets = num_targets
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.use_cosine_similarity = use_cosine_similarity

        self.memory = nn.ModuleList(
            [
                MemoryUnit(
                    self.mem_dim,
                    self.fea_dim,
                    self.use_cosine_similarity,
                    self.shrink_thres,
                )
                for _ in range(num_targets)
            ]
        )

    def forward(self, input, condition_int=None): #condition_int -->  to index the target and memory unit
        num_dims = len(input.shape)
        s = input.shape
        x = to_channel_last(input)
        x, batched_dims = batchify(x, num_dims - 1)

        all_ys = []
        all_atts = []
        all_psas = []

        # TODO: Just iterate list directly?
        for i in range(self.num_targets):
            y_and = self.memory[i](x)

            y = y_and["output"]
            att = y_and["att"]
            pre_softmax_att = y_and["pre_softmax_att"]

            y = unbatchify(y, batched_dims)
            att = unbatchify(att, batched_dims)
            pre_softmax_att = unbatchify(pre_softmax_att, batched_dims)

            y = to_channel_first(y)
            att = to_channel_first(att)
            pre_softmax_att = to_channel_first(pre_softmax_att)

            all_ys.append(y)
            all_atts.append(att)
            all_psas.append(pre_softmax_att)

        all_ys = torch.stack(all_ys, 1)
        all_atts = torch.stack(all_atts, 1)

        if condition_int is None:
            # Condition is unknown, we assume it is the one which maximizes the sum of cosine similarities for that sample
            # There are other methods which might be of interest, e.g. instead of using the sum using max, etc.
            all_psas = torch.stack(all_psas, 1)
            condition_int = all_psas.sum(
                dim=[*range(2, num_dims + 1)]
            )  # There is another dim from stacking, that's why we must sum up to dimension num_dims+1
            condition_int = condition_int.argmax(dim=1)

        y = all_ys[range(s[0]), condition_int]
        att = all_atts[range(s[0]), condition_int]

        return {"output": y, "att": att}
