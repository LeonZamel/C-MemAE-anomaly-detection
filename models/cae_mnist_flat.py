import torch
import torch.nn as nn

from models.ae import AE
from models.ae_mnist_flat import decoder, encoder


class CAEMNISTFlat(AE):
    def __init__(self, num_targets):
        super().__init__(encoder(), decoder())
        self.num_targets = num_targets

        self.fc_layer = fc_layer(num_targets)

    def forward(self, x: torch.Tensor, condition_int):
        x = self.encoder(x)

        # Remove the size 1 contextual image dimensions (width, height)
        x = x.squeeze(dim=2).squeeze(dim=2)

        # The condition will be some number between 0 and num_targets-1, inclusively. We must turn it into a one hot vector
        one_hot_condition = torch.nn.functional.one_hot(condition_int, num_classes=self.num_targets)

        # Concat the one hot vector with the features of the sample
        data_with_condition = torch.cat([x, one_hot_condition], dim=1)

        # Run the data throught the fc layer to mix in the condition
        x = self.fc_layer(data_with_condition)

        # Add back the size 1 contextual image dimensions (width, height)
        x = x.unsqueeze(2).unsqueeze(2)

        x = self.decoder(x)
        return x


def fc_layer(num_targets):
    # Feature vector is size 64, then we need a one hot vector for the condition
    return nn.Sequential(
        nn.Linear(64 + num_targets, 64),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(64, 64),
        nn.LeakyReLU(0.2, inplace=True),
    )
