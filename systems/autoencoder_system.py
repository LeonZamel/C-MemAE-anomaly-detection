# Systems contain training and testing logic, as well as the model; PyTorch Lightning specific 

import collections.abc

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import AUROC


class AutoencoderSystem(pl.LightningModule):
    # Base (abstract) class for testing autoencoders
    def __init__(
        self,
        model_type,
        target,
        batch_size,
        conditional: str,
        learning_rate=1e-4,
        model_transformers=[],
    ):
        super().__init__()
        self.model = None

        assert isinstance(
            target, collections.abc.Sized
        ), """A list of targets must be passed. If the model should be unconditional,
              a list with just one item should be passed."""
        self.target = target

        assert conditional in ["known", "unknown", None]
        assert not conditional or len(target) > 1  # conditional => len(target) > 1

        self.batch_size = batch_size
        self.conditional = conditional
        self.learning_rate = learning_rate

        self.save_hyperparameters(
            "model_type",
            "target",
            "batch_size",
            "conditional",
            "learning_rate",
        )

        # Do not reduce (i.e. do not take the mean) as we will do this manually.
        # This is so the batch dimension is not reduced. This will basically only take the squared error
        self.mse_loss_func = nn.MSELoss(reduction="none")
        self.AUROC = {target: AUROC() for target in self.target}

    def cond_to_int_cond(self, target):
        # Transforms a human readable condition to internal condition, i.e., index number starting from 0
        assert self.conditional and len(self.target) > 1, "Not trained conditional"

        one_hot = (torch.tensor(self.target, device=self.device) == target.unsqueeze(1)).long()
        one_hot = torch.cat(
            [one_hot, torch.ones((one_hot.shape[0], 1), device=self.device)],
            dim=1,
        )

        out = torch.argmax(one_hot, dim=1)
        assert torch.max(out) < len(self.target), "Condition not known"
        return out

    def forward(self, x, condition=None):
        args = []
        if condition is not None:
            if self.training:
                assert self.conditional
            else:
                assert self.conditional == "known"
            args.append(self.cond_to_int_cond(condition))

        return self._forward(x, *args)

    def _forward(self, x, *args):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.conditional is not None:
            # While training, the condition is equal to the label/target
            x_hat = self(x, y)
        else:
            x_hat = self(x)
        return x_hat

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        # We fix one condition and then do validation with that
        if dataloader_idx is None:
            assert self.conditional is None
            dataloader_idx = 0
        target = self.target[dataloader_idx]

        x, y = batch
        if self.conditional == "known":
            # We are now validating and set the condition to the one it should be. We must repeat this condition for each sample
            x_hat = self(x, torch.tensor([target], device=self.device).repeat(len(y)))
        else:
            x_hat = self(x)

        # Calculate the squared error
        loss: torch.Tensor = self.mse_loss_func(x_hat, x)
        num_dims = len(loss.shape)
        # Calculate the mean loss for each sample
        # Exclude dim 0 (batch dim) from mean calculation
        loss = loss.mean(list(range(1, num_dims)))

        # Get binary labels if the sample class is equal to the target we train on
        actuals = y == target
        actuals = actuals.long()

        return (loss, actuals)

    def validation_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT) -> None:
        # Unpack the output into the output for each target
        if self.conditional:
            assert len(outputs) == len(self.target)
            returns_per_target = outputs
        else:
            returns_per_target = [outputs]

        for target, ret in zip(self.target, returns_per_target):
            # Unzip output into loss and actuals
            loss, actuals = list(zip(*ret))

            loss = torch.cat(loss)
            actuals = torch.cat(actuals)

            loss_normal = loss[actuals].mean()
            loss_anomaly = loss[~actuals].mean()

            # Fit losses to [0,1] range, then invert so we get confidence scores
            conf = 1 - ((loss - loss.min()) / (loss.max() - loss.min()))

            auroc = self.AUROC[target]

            auroc(conf, actuals)
            self.log(f"auroc_score_{target}", auroc)
            self.log_dict(
                {
                    f"loss_normal_{target}": loss_normal,
                    f"loss_anomaly_{target}": loss_anomaly,
                }
            )

        auroc_scores = torch.tensor([auroc.compute() for t, auroc in self.AUROC.items()])
        self.log("auroc_score_mean", auroc_scores.mean())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
