from typing import List

import pytorch_lightning as pl
import torch
from torchvision import datasets, transforms

from datamodules.utils import process_for_anomalydetection


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        targets: List[int],
        batch_size: int,
        test_batch_size: int,
        data_dir: str = "data",
        num_workers: int = 1,
        num_train_samples: int = 0,
    ):
        super().__init__()
        self.targets = targets
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.num_train_samples = num_train_samples

        self.test_proportion_anomaly = 0.3

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    @staticmethod
    def target_name_map():
        return dict(list(enumerate(range(10))))

    def prepare_data(self):
        # Download the data
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        dataset_train = datasets.MNIST(self.data_dir, train=True, download=False, transform=self.transform)
        dataset_test = datasets.MNIST(self.data_dir, train=False, download=False, transform=self.transform)

        self.dataset_train, self.datasets_test = process_for_anomalydetection(
            dataset_train,
            dataset_test,
            self.targets,
            self.num_train_samples,
            self.test_proportion_anomaly,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        dataloaders = []
        for dataset_test in self.datasets_test:
            dataloaders.append(
                torch.utils.data.DataLoader(
                    dataset_test,
                    batch_size=self.test_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
            )
        return dataloaders

    def test_dataloader(self):
        return None
