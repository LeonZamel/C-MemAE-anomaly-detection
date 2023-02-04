## Deprecated. Experiment of doing anomaly detection in latent space

import torch
from torch import nn

from modules.entropy_loss import EntropyLossEncap
from systems.memae_autoencoder_system import MemaeSystem


class MemoryLatentSpaceSystem(MemaeSystem):
    def __init__(
        self,
        model,
        target,
        batch_size,
        entropy_loss_weight: float,
        shrink_threshold: float,
        condition_known=True,
    ):
        super().__init__(
            model,
            target,
            batch_size,
            entropy_loss_weight,
            shrink_threshold,
            condition_known,
        )

        self.pre_softmax_att = None

    def _forward(self, x, *args):
        x_hat, att, pre_softmax_att = self.model(x, *args)
        self.last_att_weights = att
        self.pre_softmax_att = pre_softmax_att
        return x_hat

    def validation_step(self, batch, batch_idx):
        for target in self.target:
            x, y = batch
            if self.conditional and self.condition_known:
                x_hat = self(x, torch.tensor([target], device=self.device))
            else:
                x_hat = self(x)

            conf = self.pre_softmax_att

            actuals = y == target
            actuals = actuals.long()

            conf_normal = conf[actuals].mean()
            conf_anomaly = conf[~actuals].mean()

            # Fit losses to [0,1] range, then invert so we get confidence scores
            conf = (conf - conf.min()) / (conf.max() - conf.min())

            self.AUROC(conf, actuals)
            self.log(f"auroc_score_{target}", self.AUROC)
            self.log_dict(
                {
                    f"conf_normal_{target}": conf_normal,
                    f"conf_anomaly_{target}": conf_anomaly,
                }
            )
