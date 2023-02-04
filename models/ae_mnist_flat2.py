# For testing purposes of different model architectures
 
import torch.nn as nn

from models.ae import AE


class AEMNISTFlat2(AE):
    def __init__(self):
        super().__init__(encoder(), decoder())


def encoder():
    return nn.Sequential(
        nn.Conv2d(1, 64, 3, 2, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 32, 3, 2, padding=1),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(32, 32, 3, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(32, 16, 3, 3, padding=0),
        nn.BatchNorm2d(16),
        nn.LeakyReLU(0.2, inplace=True),
    )


def decoder():
    return nn.Sequential(
        nn.ConvTranspose2d(16, 32, 3, 3, padding=0, output_padding=0),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(32, 32, 3, 3, padding=1, output_padding=0),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(32, 64, 3, 2, padding=1, output_padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(64, 1, 3, 2, padding=1, output_padding=1),
    )
