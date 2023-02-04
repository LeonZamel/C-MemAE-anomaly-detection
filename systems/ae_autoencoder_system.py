from typing import List

from torch import nn

from model_transformers.model_transformer import ModelTransformer
from models.ae_cifar import AECIFAR
from models.ae_cifar_flat import AECIFARFlat
from models.ae_mnist import AEMNIST
from models.ae_mnist_flat import AEMNISTFlat
from models.ae_mnist_flat2 import AEMNISTFlat2
from models.cae_cifar_flat import CAECIFARFlat
from models.cae_mnist_flat import CAEMNISTFlat
from models.conditional.conditional_multi_ae_mnist import ConditionalMultiAEMNIST
from systems.autoencoder_system import AutoencoderSystem


class AESystem(AutoencoderSystem):
    def __init__(
        self,
        model_type,
        target,
        batch_size,
        conditional: str,
        learning_rate,
        model_transformers: List[ModelTransformer] = [],
    ):
        super().__init__(model_type, target, batch_size, conditional, learning_rate, model_transformers)

        if model_type == "mnist-ae-flat":
            if self.conditional:
                self.model = CAEMNISTFlat(len(self.target))
            else:
                self.model = AEMNISTFlat()
        elif model_type == "mnist-ae-flat2":
            self.model = AEMNISTFlat2()
        elif model_type == "mnist-ae":
            if self.conditional:
                self.model = ConditionalMultiAEMNIST(len(self.target))
            else:
                self.model = AEMNIST()
        elif model_type == "cifar-ae":
            self.model = AECIFAR()
        elif model_type == "cifar-ae-flat":
            if self.conditional:
                self.model = CAECIFARFlat(len(self.target))
            else:
                self.model = AECIFARFlat()

        self.mse_loss_func_mean = nn.MSELoss(reduction="mean")

    def _forward(self, x, *args):
        x_hat = self.model(x, *args)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = super().training_step(batch, batch_idx)

        loss = self.mse_loss_func_mean(x_hat, x)
        self.log("train_loss", loss)
        return loss
