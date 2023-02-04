from models.ae_mnist import decoder, encoder
from models.memae import Memae


class MemaeMNIST(Memae):
    def __init__(self, num_targets, mem_dim, use_cosine_similarity, shrink_thres):
        super().__init__(
            num_targets,
            8,
            mem_dim,
            encoder(),
            decoder(),
            use_cosine_similarity,
            shrink_thres,
        )
