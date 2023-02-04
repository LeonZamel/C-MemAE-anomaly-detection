from models.ae_mnist import decoder, encoder
from models.memae import Memae
from modules.contextual_memory_module import ContextualMemModule


class ContextualMemaeMNIST(Memae):
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
        self.mem_rep = ContextualMemModule(
            mem_dim=mem_dim,
            context_dims=(3, 3),
            fea_dim=8,
            use_cosine_similarity=use_cosine_similarity,
            shrink_thres=shrink_thres,
        )
