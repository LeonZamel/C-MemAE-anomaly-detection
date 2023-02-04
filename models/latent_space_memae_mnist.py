import torch.nn as nn

from models.ae_mnist_flat import decoder, encoder
from modules.latent_space_memory_module import LatentSpaceMemModule


class LatentSpaceMemae(nn.Module):
    def __init__(self, mem_dim, use_cosine_similarity, shrink_thres=0.0025):
        super().__init__()

        self.encoder = encoder()
        self.mem_rep = LatentSpaceMemModule(
            mem_dim=mem_dim,
            fea_dim=16,
            shrink_thres=shrink_thres,
            use_cosine_similarity=use_cosine_similarity,
        )
        self.decoder = decoder()

    def forward(self, x):
        f = self.encoder(x)
        res_mem = self.mem_rep(f)
        f = res_mem["output"]
        att = res_mem["att"]
        pre_softmax_att = res_mem["pre_softmax_att"]
        output = self.decoder(f)
        return output, att, pre_softmax_att
