import torch.nn as nn

from models.ae import AE


class AECIFARFlat(AE):
    def __init__(self):
        super().__init__(encoder(), decoder())


def encoder():
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, 2, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 128, 3, 2, padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 128, 3, 2, padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 128, 3, 2, padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 256, 2, 1, padding=0),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),
    )


def decoder():
    return nn.Sequential(
        nn.ConvTranspose2d(256, 128, 2, 1, padding=0, output_padding=0),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(128, 128, 3, 2, padding=1, output_padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(128, 128, 3, 2, padding=1, output_padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(64, 3, 3, 2, padding=1, output_padding=1),
    )
