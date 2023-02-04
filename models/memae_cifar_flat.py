from models.ae_cifar_flat import decoder, encoder
from models.memae import Memae


class MemaeCIFARFlat(Memae):
    def __init__(self, num_targets, mem_dim, use_cosine_similarity, shrink_thres):
        super().__init__(
            num_targets,
            256,
            mem_dim,
            encoder(),
            decoder(),
            use_cosine_similarity,
            shrink_thres,
        )
