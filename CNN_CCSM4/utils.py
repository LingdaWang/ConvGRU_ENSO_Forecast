#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import pickle
import torch
import numpy as np
import torch.utils.data as data


def load_data(root):
    """
    load datasets
    :param root: position of the dataset
    :return: dataset contains X_train, X_test, Y_train, Y_test
    """
    path = os.path.join(root, 'dataset_region_DS.pckl')
    with open(path, 'rb') as f:
        D = pickle.load(f)

    # data_X = np.vstack((D['X_train'], D['X_test']))
    # data_Y = np.vstack((D['Y_train'], D['Y_test']))
    # mean_X = np.mean(data_X, axis=0)
    # mean_Y = np.mean(data_Y, axis=0)
    # D['X_train'] = D['X_train'] - mean_X
    # D['X_test'] = D['X_test'] - mean_X
    # D['Y_train'] = D['Y_train'] - mean_Y
    # D['Y_test'] = D['Y_test'] - mean_Y

    D['Y_train'] = np.mean(D['Y_train'], axis=(1, 2))
    D['Y_test'] = np.mean(D['Y_test'], axis=(1, 2))

    return D


class SstSeq(data.Dataset):
    def __init__(self, root, is_train, cond_len, pred_len, transform=None):
        super(SstSeq, self).__init__()

        self.SST_dataset = load_data(root)
        self.is_train = is_train
        self.cond_len = cond_len
        self.pred_len = pred_len
        self.transform = transform
        self.train_len = self.SST_dataset['X_train'].shape[0] - self.cond_len - self.pred_len  # 13188
        self.test_len = self.SST_dataset['X_test'].shape[0] - self.cond_len - self.pred_len  # 2412
        self.length = self.train_len  # size of each epoch

    def __getitem__(self, idx):
        if self.is_train:
            # random training
            start_point = np.random.randint(0, self.train_len)
            inputs = self.SST_dataset['X_train'][start_point:start_point + self.cond_len, ...]
            outputs = self.SST_dataset['Y_train'][
                      start_point + self.cond_len:start_point + self.cond_len + self.pred_len]
        else:
            # sequentially testing
            start_point = idx
            inputs = self.SST_dataset['X_test'][start_point:start_point + self.cond_len, ...]
            outputs = self.SST_dataset['Y_test'][
                      start_point + self.cond_len:start_point + self.cond_len + self.pred_len]

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
