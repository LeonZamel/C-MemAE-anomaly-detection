from abc import ABC


class ModelTransformer(ABC):
    """
    This is an abstract class that every modeltransformer should inherit from
    """

    def transform(self, system, target, datamodule):
        pass
