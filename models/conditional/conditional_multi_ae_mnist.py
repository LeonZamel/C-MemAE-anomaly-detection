# Deprecated

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import MSELoss

from models.ae_mnist import AEMNIST


class ConditionalMultiAEMNIST(nn.Module):
    def __init__(self, num_targets):
        super().__init__()

        self.num_targets = num_targets
        self.aes = nn.ModuleList([AEMNIST() for _ in range(num_targets)])

    def forward(self, x, condition_int=None):
        outs = []
        for i in range(self.num_targets):
            outs.append(self.aes[i](x))
        outs = torch.stack(outs, 1)

        if self.conditional and condition_int is None:
            # Condition is unknown, we assume it is the one which minimizes mse for that sample
            # We broadcast the mse loss over the second dim here. This throws a warning but is intended here
            errors = F.mse_loss(outs, x.unsqueeze(1), reduction="none")
            errors = errors.mean((2, 3, 4))
            condition_int = torch.argmin(errors, 1)

        return outs[range(condition_int.shape[0]), condition_int]
