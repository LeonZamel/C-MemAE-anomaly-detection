import collections
from typing import List

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger

from datamodules.cifar_dm import CIFARDataModule
from datamodules.mnist_dm import MNISTDataModule
from model_transformers.conditional_memory_copy_best_fitting_transformer import (
    ConditionalMemoryCopyBestFittingTransformer,
)
from model_transformers.conditional_memory_to_deleted_unconditional_transformer import (
    ConditionalMemoryToDeletedUnconditionalTransformer,
)
from model_transformers.copy_samples_into_memory_transformer import (
    CopySamplesIntoMemoryTransformer,
)
from model_transformers.delete_memory_transformer import DeleteMemoryTransformer
from model_transformers.model_transformer import ModelTransformer
from model_transformers.prototype_freezer import ModelFreezer
from models.latent_space_memae_mnist import LatentSpaceMemae
from systems.ae_autoencoder_system import AESystem
from systems.memae_autoencoder_system import MemaeSystem
from systems.memory_latent_space_autoencoder_system import MemoryLatentSpaceSystem

# Model types which are known/allowed
MEMAE_SYSTEM_MODELS = [
    "mnist-memae",
    "mnist-memae-contextual",
    "mnist-memae-flat",
    "mnist-memae-flat2",
    "mnist-memae-latent-space",
    "cifar-memae",
    "cifar-memae-flat",
]
AE_SYSTEM_MODELS = [
    "mnist-ae",
    "mnist-ae-flat",
    "mnist-ae-flat2",
    "cifar-ae",
    "cifar-ae-flat",
]

# Model transformers that can be applied to a model before training
model_transformer_map = {
    "delete_memory": DeleteMemoryTransformer(),
    "copy_samples": CopySamplesIntoMemoryTransformer(),
    "conditional_copy_best_fitting": ConditionalMemoryCopyBestFittingTransformer(),
    "conditional_to_deleted": ConditionalMemoryToDeletedUnconditionalTransformer(),
    "model_freezer": ModelFreezer(),
}


def train(config: DictConfig):
    # Set seed for all kinds of random number generators (python, numpy, torch, ...)
    pl.utilities.seed.seed_everything(config.seed)

    # Sanity check for conditional training 
    if config.conditional:
        assert all(
            map(lambda x: len(x) > 1, config.datamodule.targets)
        ), "Targets should have more than one value in conditional training"

    # Get the data set
    if config.datamodule.name == "mnist":
        datamodule_constructor = lambda target: MNISTDataModule(
            targets=target,
            batch_size=config.datamodule.batch_size,
            test_batch_size=config.datamodule.test_batch_size,
            data_dir=config.datamodule.data_dir,
            num_workers=config.datamodule.num_workers,
            num_train_samples=config.datamodule.num_train_samples,
        )
    elif config.datamodule.name == "cifar":
        datamodule_constructor = lambda target: CIFARDataModule(
            targets=target,
            batch_size=config.datamodule.batch_size,
            test_batch_size=config.datamodule.test_batch_size,
            data_dir=config.datamodule.data_dir,
            num_workers=config.datamodule.num_workers,
            num_train_samples=config.datamodule.num_train_samples,
        )

    # Get the system we want to train
    # And also the system that we want to continue training from
    # Pytorch lightning: System = includes all information needed for training
    system_constructor = None
    continue_from_system_constructor = None
    if config.model.model_type in MEMAE_SYSTEM_MODELS:
        if config.model.model_type == "memae-latent-space":
            # TODO: Add this back in
            assert False
            system_constructor = lambda model, target: MemoryLatentSpaceSystem(
                model,
                target,
                args.batch_size,
                args.entropy_loss_weight,
                args.shrink_threshold,
                condition_known,
            )
            model_constructor = lambda _: LatentSpaceMemae(
                args.memory_size,
                use_cosine_similarity=use_cosine_similarity,
                shrink_thres=args.shrink_threshold,
            )
        else: #usual execution
            if config.from_checkpoint:
                continue_from_system_constructor = lambda: MemaeSystem.load_from_checkpoint(
                    config.from_checkpoint
                )

            system_constructor = lambda target: MemaeSystem(
                model_type=config.model.model_type,
                target=target,
                batch_size=config.datamodule.batch_size,
                conditional=config.conditional,
                learning_rate=config.model.learning_rate,
                memory_size=config.model.memory_size,
                entropy_loss_weight=config.model.entropy_loss_weight,
                shrink_threshold=config.model.shrink_threshold,
                use_cosine_similarity=config.model.cosine_similarity,
                model_transformers=model_transformers,
            )

    #only for autoencoder models without memory
    elif config.model.model_type in AE_SYSTEM_MODELS:
        if config.from_checkpoint:
            continue_from_system_constructor = lambda: AESystem.load_from_checkpoint(config.from_checkpoint)

        system_constructor = lambda target: AESystem(
            model_type=config.model.model_type,
            target=target,
            batch_size=config.datamodule.batch_size,
            conditional=config.conditional,
            learning_rate=config.model.learning_rate,
            model_transformers=model_transformers,
        )

    # Get the model transformers - currently only for few shot learning
    model_transformers: List[ModelTransformer] = []
    for mt in config.model.model_transformers:
        model_transformers.append(model_transformer_map[mt])

    # Build the string for the name of the run directory
    continued = ""
    if config.from_checkpoint:
        name = ",".join((str(t) for t in continue_from_system_constructor().target))
        continued = f"-continued-{name}"
    run_name = f"run-seed-{config.seed}{continued}"

    # Iterate over all targets: set that target as normal and all others as anomaly
    # train and validate
    for target in config.datamodule.targets:
        # Reset the seed to the value we set earlier
        pl.utilities.seed.reset_seed()

        # If we do conditional training, every target will just be some class-value (e.g. and int)
        # To unify the process with conditional training, we only passes lists of targets to the systems
        if not config.conditional:
            assert not isinstance(target, collections.abc.Sized)
            target = [target]
        else:
            assert isinstance(target, collections.abc.Sized)

        target_name = ",".join((str(t) for t in target))
        print(f"Target: {target_name}")

        # Construct the system
        system = system_constructor(target)
        # If we continue from some checkpoint, load that checkpoint and copy over the model
        if config.from_checkpoint:
            print(f"Continuing from {config.from_checkpoint}")
            system.model = continue_from_system_constructor().model

        # Construct the data set
        dm = datamodule_constructor(target)

        # Apply any model transformations
        for model_transformer in model_transformers:
            system.model = model_transformer.transform(system.model, target, dm)

        # Create the logger
        version_name = f"target_{target_name}"
        logger = TensorBoardLogger("", name=run_name, version=version_name)

        # We create a fake trainer which we can use to save the model state before training
        fake_trainer = pl.Trainer(max_epochs=0, logger=False, num_sanity_val_steps=0, max_steps=0)
        fake_trainer.fit(system, dm)
        fake_trainer.save_checkpoint(f"{run_name}/{version_name}/checkpoints/epoch=init.ckpt")

        # Create the actual trainer with a checkpoint callback to save the model after every epoch
        mc_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
            save_top_k=-1, every_n_val_epochs=config.save_checkpoints_every_n_epochs, filename="{epoch}"
        )
        trainer: pl.Trainer = hydra.utils.instantiate(config.trainer, callbacks=[mc_callback], logger=logger)

        # Run the training/validation loop
        trainer.fit(system, dm)
