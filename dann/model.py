#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
from torchvision import models
from .layers import GradScaleLayer


class DANN(nn.Module):
    def forward(self, x, l=0):
        features = self.feature_extractor(x)
        labels = self.classifier_label(features)
        domain = self.classifier_domain(GradScaleLayer()(features, -l))
        # domain = grad_reverse(self.classifier_domain(features), l)
        return labels, domain


class DANN_MNIST(DANN):
    def __init__(self, n_out=10):
        super().__init__()

        # feature extractor
        # mnist image are padded to 3x32x32
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 5),  # 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x14x14
            nn.Conv2d(32, 48, 5),  # 48x10x10
            nn.ReLU(),
            nn.MaxPool2d(2),  # 48x5x5
            nn.Flatten(),  # 1200
            )

        # label classifier
        self.classifier_label = nn.Sequential(
                nn.Linear(1200, 100),
                nn.ReLU(),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, n_out),
                )

        # domain classifier
        self.classifier_domain = nn.Sequential(
                nn.Linear(1200, 100),
                nn.ReLU(),
                nn.Linear(100, 2),
                )


class DANN_SVHN(DANN):
    def __init__(self, n_out=10):
        super().__init__()

        # feature extractor
        # svhn image is 3x32x32
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 5),  # 64x28x28
            nn.BatchNorm2d(64),
            nn.Dropout2d(1-0.9),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 64x13x13
            nn.ReLU(),
            nn.Conv2d(64, 64, 5),  # 64x9x9
            nn.BatchNorm2d(64),
            nn.Dropout2d(1-0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 64x4x4
            nn.ReLU(),
            nn.Conv2d(64, 128, 4),  # 128x1x1
            nn.BatchNorm2d(128),
            nn.Dropout2d(1-0.75),
            nn.ReLU(),
            nn.Flatten(),  # 128x1
            )

        # label classifier
        self.classifier_label = nn.Sequential(
                nn.Linear(128, 3072),
                nn.BatchNorm1d(3072),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(3072, 2048),
                nn.BatchNorm1d(2048),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(2048, n_out),
                )

        # domain classifier
        self.classifier_domain = nn.Sequential(
                nn.Linear(128, 1024),
                nn.BatchNorm1d(1024),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.BatchNorm1d(1024),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(1024, 2),
                )


class DANN_GTSRB(DANN):
    def __init__(self, n_out=10):
        super().__init__()

        # feature extractor
        # mnist image is 3x28x28
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 5), # 32x24x24
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x12x12
            nn.Conv2d(32, 48, 5),  # 48x8x8
            nn.ReLU(),
            nn.MaxPool2d(2),  # 48x4x4
            nn.Flatten(),  # 768
            )

        # label classifier
        self.classifier_label = nn.Sequential(
                nn.Linear(768, 100),
                nn.ReLU(),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, n_out),
                )

        # domain classifier
        self.classifier_domain = nn.Sequential(
                nn.Linear(768, 100),
                nn.ReLU(),
                nn.Linear(100, 2),
                )


class DANN_Alexnet(DANN):
    def __init__(self, n_out=10):

        # feature extractor
        alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        alexnet.classifier.pop(-1)  # remove last fc
        self.feature_extractor = alexnet

        # label classifier
        self.classifier_label = nn.Sequential(
                nn.Linear(4096, n_out),
                )

        # domain classifier
        self.classifier_domain = nn.Sequential(
                nn.Linear(4096, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2),
                )


class DANN_ResNet(DANN):
    def __init__(self, n_out=10):
        # feature extractor
        resnet = models.resnet18(pretrained=True)
        resnet.fc = nn.Identity()  # remove last fc
        self.feature_extractor = resnet

        # label classifier
        self.classifier_label = nn.Sequential(
                nn.Linear(512, n_out),
                )

        # domain classifier
        self.classifier_domain = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 2),
                )
