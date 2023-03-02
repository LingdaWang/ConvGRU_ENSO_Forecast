#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Time    :   01/29/2023
@Author  :   lingdaw2
@Mail    :   lingdaw2@illinois.edu
@Description:   utils
"""

from torch import nn
import netCDF4 as nc
from collections import OrderedDict
import numpy as np
import os
import torch
import torch.utils.data as data


def make_layers(block):
    """
    Making layers using parameters from NetParams.py
    :param block: OrderedDict
    :return: layers
    """
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0],
                                                 out_channels=v[1],
                                                 kernel_size=v[2],
                                                 stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0],
                               out_channels=v[1],
                               kernel_size=v[2],
                               stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError

    return nn.Sequential(OrderedDict(layers))


def load_data(root):
    """
    load datasets
    :param root: position of the dataset
    :return: dataset contains X_train, X_test, Y_train, Y_test
    """
    x = dict()
    ds = nc.Dataset(os.path.join(root, 'air.2m.mon.mean.nc'))
    ds = ds["air"][:, 15:-15, 64:].data - 273.15
    x["X_train"] = ds[:1560, :, :]  # 1968
    x["X_test"] = ds[1560:, :, :]
    return x


class SstSeq(data.Dataset):
    def __init__(self, root, is_train, cond_len, pred_len, transform=None):
        super(SstSeq, self).__init__()

        self.SST_dataset = load_data(root)
        self.is_train = is_train
        self.cond_len = cond_len
        self.pred_len = pred_len
        self.transform = transform
        self.train_len = self.SST_dataset['X_train'].shape[0] - self.cond_len - self.pred_len  # 1560
        self.test_len = self.SST_dataset['X_test'].shape[0] - self.cond_len - self.pred_len  # 408
        self.length = self.train_len  # size of each epoch

    def __getitem__(self, idx):
        if self.is_train:
            # random training
            start_point = np.random.randint(0, self.train_len)
            inputs = self.SST_dataset['X_train'][start_point:start_point + self.cond_len, ...]
            outputs = self.SST_dataset['X_train'][
                      start_point + self.cond_len:start_point + self.cond_len + self.pred_len, ...]
        else:
            # sequentially testing
            start_point = idx
            inputs = self.SST_dataset['X_test'][start_point:start_point + self.cond_len, ...]
            outputs = self.SST_dataset['X_test'][
                      start_point + self.cond_len:start_point + self.cond_len + self.pred_len, ...]

        inputs = inputs[:, np.newaxis, :, :]
        outputs = outputs[:, np.newaxis, :, :]
        outputs = torch.from_numpy(outputs).contiguous().float()
        inputs = torch.from_numpy(inputs).contiguous().float()
        out = [idx, outputs, inputs, start_point]

        return out

    def __len__(self):
        return self.length


class RecordHist:
    def __init__(self, verbose=False):
        """
        Args:
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.verbose = verbose
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, epoch, save_path):
        self.save_checkpoint(val_loss, model, epoch, save_path)

    def save_checkpoint(self, val_loss, model, epoch, save_path):
        """
        Saves model.
        """
        if self.verbose:
            print(
                f'Validation loss from ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            )
        torch.save(
            model, save_path + "/" +
            "checkpoint_{}_{:.6f}.pth.tar".format(epoch, val_loss))
        self.val_loss_min = val_loss


