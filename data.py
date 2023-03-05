import pytorch_lightning as pl

import torch
from typing import Tuple, Optional
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
import numpy as np


class MNISTSupervisedDataset(Dataset):
    """MNIST dataset for the simple supervised forward-forward algorithm."""

    def __init__(
        self,
        root: str = "./",
        train: bool = True,
        download: bool = False,
        seed: int = 43,
        debug: bool = False,
    ):
        self._train = train
        self._seed = seed
        transform = Compose(
            [
                ToTensor(),
                Normalize((0.1307,), (0.3081,)),
                Lambda(lambda x: torch.flatten(x)),
            ]
        )
        self._mnist = MNIST(root, train=train, download=download, transform=transform)

        if debug:
            self._mnist.data = self._mnist.data[:100, :, :]
            self._mnist.targets = self._mnist.targets[:100]

        # create positive and negative examples
        self._images_pos = torch.empty(len(self), 28 * 28)
        self._images_neg = torch.empty(len(self), 28 * 28)
        self._targets = torch.empty(len(self))

        rng = np.random.default_rng(seed=seed)
        for i in range(len(self)):
            image, target = self._mnist[i]
            image[:10] = 0.0
            self._targets[i] = target

            self._images_pos[i] = image
            self._images_pos[i, target] = 1.0

            self._images_neg[i] = image
            valid_negative_targets = [t for t in range(10) if t != target]
            target_neg = rng.choice(valid_negative_targets)
            self._images_neg[i, target_neg] = 1.0

    @property
    def train(self) -> bool:
        return self._train

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns MNIST images with the labels encoded in the first 10 pixels, see FF paper"""
        return self._images_pos[index], self._images_neg[index], self._targets[index]

    def __len__(self):
        return len(self._mnist)


class MNISTSupervisedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        split_seed: int = 43,
        val_fraction: Optional[float] = 0.2,
        batch_size: int = 16,
        batch_size_val: Optional[int] = None,
        num_workers: int = 0,
        num_workers_val: Optional[int] = None,
        debug: bool = False,
        download: bool = False,
        seed: int = 43,
    ):
        super().__init__()
        self._data_dir = data_dir
        self._split_seed = split_seed
        self._val_fraction = val_fraction
        self._batch_size = batch_size
        self._batch_size_val = batch_size_val or batch_size
        self._num_workers = num_workers
        self._num_workers_val = num_workers_val or num_workers
        self._debug = debug
        self._download = download
        self._seed = seed

        self.mnist_train, self.mnist_val, self.mnist_test = None, None, None

    def prepare_data(self):
        MNIST(self._data_dir, download=self._download, train=True)
        MNIST(self._data_dir, download=self._download, train=False)

    def setup(self, stage: str):
        if stage == "fit":
            mnist_full = MNISTSupervisedDataset(
                self._data_dir,
                train=True,
                debug=self._debug,
                seed=self._seed,
            )
            if self._val_fraction:
                val_nr = int(self._val_fraction * len(mnist_full))
                tra_nr = len(mnist_full) - val_nr
                self.mnist_train, self.mnist_val = random_split(
                    mnist_full,
                    lengths=(tra_nr, val_nr),
                    generator=torch.Generator().manual_seed(self._split_seed),
                )
            else:
                self.mnist_train = mnist_full

        if stage == "test":
            self.mnist_test = MNISTSupervisedDataset(
                self._data_dir,
                train=False,
                debug=self._debug,
                seed=42,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_train,
            shuffle=True,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.mnist_val:
            return DataLoader(
                self.mnist_val,
                shuffle=False,
                batch_size=self._batch_size_val,
                num_workers=self._num_workers_val,
            )
        return None

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_test,
            shuffle=False,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
        )
