from typing import List

from torch import nn

from model_transformers.model_transformer import ModelTransformer
from models.contextual_memae_mnist import ContextualMemaeMNIST
from models.memae_cifar import MemaeCIFAR
from models.memae_cifar_flat import MemaeCIFARFlat
from models.memae_mnist import MemaeMNIST
from models.memae_mnist_flat import MemaeMNISTFlat
from models.memae_mnist_flat2 import MemaeMNISTFlat2
from modules.entropy_loss import EntropyLossEncap
from systems.autoencoder_system import AutoencoderSystem


class MemaeSystem(AutoencoderSystem):
    def __init__(
        self,
        model_type,
        target,
        batch_size,
        conditional: str,
        learning_rate,
        memory_size: int,
        entropy_loss_weight: float,
        shrink_threshold: float,
        use_cosine_similarity: bool,
        model_transformers: List[ModelTransformer] = [],
    ):
        super().__init__(
            model_type,
            target,
            batch_size,
            conditional,
            learning_rate,
            model_transformers,
        )

        if model_type == "mnist-memae-contextual":
            assert not self.conditional
            self.model = ContextualMemaeMNIST(
                len(self.target),
                memory_size,
                use_cosine_similarity=use_cosine_similarity,
                shrink_thres=shrink_threshold,
            )
        elif model_type == "mnist-memae-flat":
            self.model = MemaeMNISTFlat(
                len(self.target),
                memory_size,
                use_cosine_similarity=use_cosine_similarity,
                shrink_thres=shrink_threshold,
            )
        elif model_type == "mnist-memae-flat2":
            self.model = MemaeMNISTFlat2(
                len(self.target),
                memory_size,
                use_cosine_similarity=use_cosine_similarity,
                shrink_thres=shrink_threshold,
            )
        elif model_type == "mnist-memae":
            self.model = MemaeMNIST(
                len(self.target),
                memory_size,
                use_cosine_similarity=use_cosine_similarity,
                shrink_thres=shrink_threshold,
            )
        elif model_type == "cifar-memae-contextual":
            # TODO: FIX
            # model_constructor = lambda: ContextualMemaeCIFAR(125, args.shrink_threshold)
            pass
        elif model_type == "cifar-memae":
            self.model = MemaeCIFAR(
                len(self.target),
                memory_size,
                use_cosine_similarity=use_cosine_similarity,
                shrink_thres=shrink_threshold,
            )
        elif model_type == "cifar-memae-flat":
            self.model = MemaeCIFARFlat(
                len(self.target),
                memory_size,
                use_cosine_similarity=use_cosine_similarity,
                shrink_thres=shrink_threshold,
            )

        self.mse_loss_func_mean = nn.MSELoss(reduction="mean")
        self.entropy_loss_func = EntropyLossEncap()

        self.entropy_loss_weight = entropy_loss_weight

        self.save_hyperparameters()

        self.last_att_weights = None

    def _forward(self, x, *args):
        x_hat, att = self.model(x, *args)
        self.last_att_weights = att
        return x_hat

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = super().training_step(batch, batch_idx)

        entropy_loss = self.entropy_loss_func(self.last_att_weights)
        loss = self.mse_loss_func_mean(x_hat, x) + self.hparams.entropy_loss_weight * entropy_loss
        self.log("train_loss", loss)
        self.log("entropy_loss", entropy_loss)
        return loss
