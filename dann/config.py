#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.optim import SGD
from .data import get_mnist_data, get_svhn_data
from .data import get_loaders
from .model import DANN_SVHN

class Config():
    """
    Default Config class
    """
    # data
    DATA_DIR = '_DATA'

    # LAMBDA
    GAMMA = 10

    # OPTIMIZER
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    ALPHA = 10
    BETA = 0.75

    def get_lr(self, p):
        """
        args:
            p: float
                training progress
        """
        return self.LEARNING_RATE / (1 + self.ALPHA * p) ** self.BETA

    def get_optimizer(self):
        return self.optimizer

    def get_model(self):
        return self.model

    def get_iters(self):
        return self.n_iters

    def get_log_settings(self):
        return self.log_every

    def get_lambda(self, p):
        """
        args:
            p: float
                training progress
        """
        return 2 / (1 + np.exp(- self.GAMMA * p)) - 1

    def get_device(self):
        return self.device


class ConfigSvhnMnist(Config):
    """
    Config for SVHN -> MNIST experiment
    """
    def __init__(self, n_iters=10000, batch_size=32, seed=None, log_every=500):
        """
        args:
            n_iters: int
                number of training iterations
            batch_size: int
                batch size
            seed: int
                random seed for train-dev split
            log_every: int
                how often to perform evalution on dev set
        """
        # data
        self.seed = seed
        self.batch_size = batch_size

        # training loop
        self.n_iters = n_iters
        self.log_every = log_every

        # model
        self.model = DANN_SVHN()

        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.LEARNING_RATE, momentum=self.MOMENTUM)

        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_loaders(self, train_size=0.8, seed=None):
        # source data
        ds_src = get_svhn_data(self.DATA_DIR, train_size, seed)
        ds_src_train, ds_src_dev, ds_src_test = ds_src
        # target data
        ds_trg = get_mnist_data(self.DATA_DIR, train_size, seed)
        ds_trg_train, ds_trg_dev, ds_trg_test = ds_trg
        # combine datasets
        datasets = [ds_src_train, ds_src_dev, ds_src_test,
                    ds_trg_train, ds_trg_dev, ds_trg_test]
        # dataloaders
        loader_train, loader_src_dev, loader_src_test, loader_trg_dev, loader_trg_test = get_loaders(datasets, self.batch_size)

        return loader_train, loader_src_dev, loader_src_test, loader_trg_dev, loader_trg_test
