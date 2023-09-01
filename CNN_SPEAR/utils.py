#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import torch
import netCDF4 as nc
import numpy as np
import torch.utils.data as data
from skimage.transform import downscale_local_mean


def load_data(root, if_verbose=True):
    """
    load datasets
    :param root: position of the dataset
    :param if_verbose:
    :return:
    """
    if if_verbose:
        print("Reading Data...")
    tos = dict()
    avgTos = np.zeros((2160, 180, 360))
    for i in range(1, 31):
        print("reading {}th dataset...".format(i))
        fileHist = os.path.join(root, "tos_Omon_GFDL-SPEAR-MED_historical_r{}i1p1f1_gr_192101-201412.nc".format(i))
        fileFut = os.path.join(root, "tos_Omon_GFDL-SPEAR-MED_scenarioSSP5-85_r{}i1p1f1_gr_201501-210012.nc".format(i))
        tosTot = np.ma.vstack((nc.Dataset(fileHist)["tos"][:], nc.Dataset(fileFut)["tos"][:]))
        tos[i] = tosTot.filled(fill_value=0)
        avgTos += tos[i] / 30

    for i in range(1, 31):
        tos[i] -= avgTos

    if if_verbose:
        print("Done!")
        print("Downsampling Data...")

    x, x_train, x_test, y_train, y_test = dict(), dict(), dict(), dict(), dict()
    for i in range(1, 31):
        tmp_x = downscale_local_mean(tos[i][:, 21:149, 150:278], (1, 2, 2))
        tmp_y = np.mean(tmp_x[:, 32:37, 20:45], axis=(1, 2))
        x_train[i], x_test[i] = tmp_x[:1560, ...], tmp_x[1560:, ...]
        y_train[i], y_test[i] = tmp_y[:1560], tmp_y[1560:]
        # print(x_train[i].shape, x_test[i].shape, y_train[i].shape, y_test[i].shape)
    if if_verbose:
        print("Done!")
    x["X_train"], x["X_test"], x["Y_train"], x["Y_test"] = x_train, x_test, y_train, y_test
    return x


class SstSeq(data.Dataset):
    def __init__(self, SST_dataset, is_train, cond_len, pred_len, transform=None):
        super(SstSeq, self).__init__()
        self.SST_dataset = SST_dataset
        self.is_train = is_train
        self.cond_len = cond_len
        self.pred_len = pred_len
        self.transform = transform
        self.train_len = self.SST_dataset['X_train'][1].shape[0] - self.cond_len - self.pred_len   # 1560*30
        self.test_len = self.SST_dataset['X_test'][1].shape[0] - self.cond_len - self.pred_len  # 600*30
        self.length = self.train_len * 10  # size of each epoch

    def __getitem__(self, idx):
        if self.is_train:
            # random training
            data_set_index = np.random.randint(1, 30)
            start_point = np.random.randint(0, self.train_len)
            inputs = self.SST_dataset['X_train'][data_set_index][start_point:start_point + self.cond_len, ...]
            outputs = self.SST_dataset['Y_train'][data_set_index][
                      start_point + self.cond_len:start_point + self.cond_len + self.pred_len]
        else:
            # sequentially testing
            data_set_index = (idx // self.test_len) + 1
            start_point = idx % self.test_len
            inputs = self.SST_dataset['X_test'][data_set_index][start_point:start_point + self.cond_len, ...]
            outputs = self.SST_dataset['Y_test'][data_set_index][
                      start_point + self.cond_len:start_point + self.cond_len + self.pred_len]

        outputs = torch.from_numpy(outputs).contiguous().float()
        inputs = torch.from_numpy(inputs).contiguous().float()

        out = [idx, outputs, inputs, data_set_index, start_point]
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
