#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class cnnNet(nn.Module):
    def __init__(self, cond_len=48, pred_len=24, inter_m=2, dropout=0.):
        super(cnnNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=cond_len,
            out_channels=inter_m * cond_len,
            kernel_size=9,
            padding='same'
        )
        self.conv2 = nn.Conv2d(
            in_channels=inter_m * cond_len,
            out_channels=inter_m * cond_len,
            kernel_size=5,
            padding='same'
        )
        self.conv3 = nn.Conv2d(
            in_channels=inter_m * cond_len,
            out_channels=inter_m * cond_len,
            kernel_size=5,
            padding='same'
        )
        self.Dropout = nn.Dropout(dropout)
        self.fc1 = nn.LazyLinear(1200)
        self.fc2 = nn.LazyLinear(pred_len)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = self.Dropout(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = self.Dropout(x)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.Dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
