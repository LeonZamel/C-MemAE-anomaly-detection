import torch.nn as nn


class AE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        f = self.encoder(x)
        output = self.decoder(f)
        return output
