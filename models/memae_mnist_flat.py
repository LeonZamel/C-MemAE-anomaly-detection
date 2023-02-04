from models.ae_mnist_flat import decoder, encoder
from models.memae import Memae


class MemaeMNISTFlat(Memae):
    def __init__(self, num_targets, mem_dim, use_cosine_similarity, shrink_thres):
        super().__init__(
            num_targets,
            64,
            mem_dim,
            encoder(),
            decoder(),
            use_cosine_similarity,
            shrink_thres,
        )
