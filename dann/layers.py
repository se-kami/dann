#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn

class GradScaleLayer(nn.Module):
    """
    gradient scaling layer
    """
    def forward(self, x, scale):
        """
        pass x to forward
        pass x_scaled to backward
        """
        x_scaled = x * scale
        return torch.detach(x - x_scaled) + x_scaled
