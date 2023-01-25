#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import default_generator, Generator, randperm
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import datasets, transforms

def split_dataset(ds, size=0.8, seed=None):
    """
    randomly splits pytorch dataset into 2 dataset
    returns 2 datasets

    args:
        ds: torch.utils.data.Dataset
            dataset to split
        size: float (optional)
            ratio of total dataset that goes into first returned dataset
        seed: int (optional)
            if not None, use manual random seed
    """
    sizes = [size, 1-size]

    if seed is not None:
        generator = Generator().manual_seed(42)
    else:
        generator = default_generator

    ds_1, ds_2 = random_split(ds, sizes, generator=generator)

    return ds_1, ds_2


class DoubleLoader():
    def __init__(self, ds1, ds2, batch_size1=1, batch_size2=1, shuffle1=False, shuffle2=False):
        self.l1 = DataLoader(ds1, batch_size=batch_size1, shuffle=shuffle1, drop_last=True)
        self.l2 = DataLoader(ds2, batch_size=batch_size2, shuffle=shuffle2, drop_last=True)
        self.size1 = len(ds1)
        self.size2 = len(ds2)
        self.size = max(self.size1, self.size2)

        self.iter_1 = self._get_iter_1()
        self.iter_2 = self._get_iter_2()
        self.i = 0
        self.step = max(batch_size1, batch_size2)
        self.in_size = tuple(ds1[0][0].shape)

    def __iter__(self):
        self.iter_1 = self._get_iter_1()
        self.iter_2 = self._get_iter_2()
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.size:
            raise StopIteration
        else:
            self.i += self.step
        try:
            data_1 = next(self.iter_1)
        except StopIteration:
            self.iter_1 = self._get_iter_1()  # reset iter
            data_1 = next(self.iter_1)

        try:
            data_2 = next(self.iter_2)
        except StopIteration:
            self.iter_2 = self._get_iter_2()  # reset iter
            data_2 = next(self.iter_2)

        # make batches be the same size
        if len(data_1) != len(data_2):
            size = min(len(data_1), len(data_2))
            data_1 = data_1[:size]
            data_2 = data_2[:size]

        return data_1, data_2

    def _get_iter_1(self):
        return iter(self.l1)

    def _get_iter_2(self):
        return iter(self.l2)

    def get_in_size(self):
        return self.in_size

    def __len__(self):
        return self.size


class Grayscale2RGBDataset(Dataset):
    """
    Dataset wrapper that converts image to RGB.

    args:
        ds: Dataset
            original dataset
        transform: callable (optional)
            transformation to apply
    """

    def __init__(self, ds, transform=None):
        super().__init__()
        self.ds = ds
        if transform is None:
            transform = transforms.Compose(
                    [transforms.Pad(2),
                     transforms.ToTensor(),
                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
                    )
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.ds[index]
        x = x.convert("RGB")
        return self.transform(x), y

    def __len__(self):
        return len(self.ds)


def get_mnist_data(datadir='_DATA', train_size=0.8, seed=None,
                   transform_train=None, transform_test=None):
    """
    return train, dev and test splits for MNIST

    args:
        datadir: str (optional)
            location of data
        train_size: float (optional)
            ratio of data that goes into train set
        seed: int (optional)
            if not None, random seed to use
        transform_train: callable (optional)
            transform to use on training data
        transform_test: callable (optional)
            transform to use on test data
    """
    ds_train = datasets.MNIST(root=datadir, download=True, train=True,
                              transform=transform_train)
    ds_test = datasets.MNIST(root=datadir, download=True, train=False,
                             transform=transform_test)
    ds_train, ds_dev = split_dataset(ds_train, size=train_size, seed=seed)

    ds_train = Grayscale2RGBDataset(ds_train, transform_train)
    ds_dev = Grayscale2RGBDataset(ds_dev, transform_test)
    ds_test = Grayscale2RGBDataset(ds_train, transform_test)

    return ds_train, ds_dev, ds_test


def get_svhn_data(datadir='_DATA', train_size=0.8, seed=None,
                  transform_train=None, transform_test=None):
    """
    return train, dev and test splits for SVHN

    args:
        datadir: str (optional)
            location of data
        train_size: float (optional)
            ratio of data that goes into train set
        seed: int (optional)
            if not None, random seed to use
        transform_train: callable (optional)
            transform to use on training data
        transform_test: callable (optional)
            transform to use on test data
    """
    if transform_train is None:
        transform_train = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
                )
    if transform_test is None:
        transform_test = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
                )

    ds_train = datasets.SVHN(root=datadir, download=True, split='train',
                             transform=transform_train)
    ds_test = datasets.SVHN(root=datadir, download=True, split='test',
                            transform=transform_test)
    ds_train, ds_dev = split_dataset(ds_train, size=train_size, seed=seed)

    return ds_train, ds_dev, ds_test


def get_loaders(datasets, batch_size):
    """
    return 5 loaders:
        DoubleLoader of training source and target data
        DataLoader of dev source data
        DataLoader of test source data
        DataLoader of dev target data
        DataLoader of test target data

    args:
        datasets: List
            list of 6 datasets -> [src_train, src_dev, src_test, trg_train, trg_dev, trg_test]
        batch_size: int
            batch size to use in dataloaders
    """

    ds_src_train, ds_src_dev, ds_src_test, ds_trg_train, ds_trg_dev, ds_trg_test = datasets

    loader_train = DoubleLoader(ds_src_train, ds_trg_train,
                                batch_size, batch_size,
                                True, True)

    loader_src_dev = DataLoader(ds_src_dev, batch_size*2, shuffle=False)
    loader_src_test = DataLoader(ds_src_test, batch_size*2, shuffle=False)
    loader_trg_dev = DataLoader(ds_trg_dev, batch_size*2, shuffle=False)
    loader_trg_test = DataLoader(ds_trg_test, batch_size*2, shuffle=False)

    return loader_train, loader_src_dev, loader_src_test, loader_trg_dev, loader_trg_test
