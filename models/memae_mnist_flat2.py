# For testing purposes of different model architectures

from models.ae_mnist_flat2 import decoder, encoder
from models.memae import Memae


class MemaeMNISTFlat2(Memae):
    def __init__(self, num_targets, mem_dim, use_cosine_similarity, shrink_thres):
        super().__init__(
            num_targets,
            16,
            mem_dim,
            encoder(),
            decoder(),
            use_cosine_similarity,
            shrink_thres,
        )
