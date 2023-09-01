#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import random
import pickle
import torch
import numpy as np
from torch import nn, mps
from tqdm import tqdm
from scipy.stats import pearsonr
from utils import SstSeq
from model import cnnNet


random_seed = 123
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
mps.manual_seed(random_seed)


if __name__ == "__main__":
    # make dir
    saved_CNN_model_folder = "./results/saved_CNN_model_3"
    save_dir = "plots"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # average temperature of Nino 3.4 region from CCSM4 dataset
    avg_temp_by_month = np.array([24.33743506, 24.33262092, 24.40992943, 24.72788656, 25.12441604, 25.10031928,
                                  24.64784378, 23.99778991, 23.64409438, 23.82734981, 24.18979863, 24.33067972])

    # import the true Nino 3.4 index
    with open("data/nino_truth.pckl", "rb") as f:
        nino_test = pickle.load(f)

    cond_len, pred_len, test_data_len, batch_size = 3, 24, 2340, 1
    validFolder = SstSeq(root='./data', is_train=False, cond_len=cond_len, pred_len=pred_len)
    validLoader = torch.utils.data.DataLoader(validFolder, batch_size=batch_size, shuffle=False)

    device = torch.device('mps')
    model = cnnNet(pred_len=pred_len, cond_len=cond_len, inter_m=16)
    model.to(device)
    PATH = saved_CNN_model_folder + '/checkpoint_26_0.469964.pth.tar'
    model_info = torch.load(PATH)
    model.load_state_dict(model_info['state_dict'])
    model.eval()
    lossfunction = nn.MSELoss().to(device)
    ori_res, pred_res, start_points = (
        np.zeros((test_data_len, pred_len)),
        np.zeros((test_data_len, pred_len)),
        np.zeros(test_data_len)
    )
    test_losses = []
    with torch.no_grad():
        t = tqdm(validLoader, leave=False, total=len(validLoader))
        for i, (_, targetVar, inputVar, start_point) in enumerate(t):
            inputs = inputVar.to(device)
            label = targetVar.to(device)
            pred = model(inputs)
            loss = lossfunction(pred, label)
            loss_aver = loss.item()
            test_losses.append(loss_aver)
            t.set_postfix({
                'validloss': '{:.6f}'.format(loss_aver)
            })
            ori_res[i, :] = (torch.squeeze(label.to('cpu'))).numpy()
            pred_res[i, :] = (torch.squeeze(pred.to('cpu'))).numpy()
            start_points[i] = start_point
            if i + 1 == test_data_len:
                break
    test_loss = np.average(test_losses)
    print("test loss is {}".format(test_loss))

    # compute the true Nino 3.4 index and predicted Nino 3.4 index
    Nino_ori, Nino_pred = np.zeros((test_data_len, pred_len)), np.zeros((test_data_len, pred_len))
    for i in range(test_data_len):
        for j in range(pred_len):
            Nino_ori[i, j] = nino_test[int(start_points[i]) + cond_len + j]
            Nino_pred[i, j] = (pred_res[i, j] - avg_temp_by_month[(int(start_points[i]) + cond_len + j) % 12]) / 0.9171

        # Compute the Person Correlation
        P_corr_CNN = np.array([pearsonr(Nino_ori[:, i], Nino_pred[:, i])[0] for i in range(pred_len)])
        np.save(save_dir + '/P_corr_CNN_cond_3.npy', P_corr_CNN)

        # Compute the RMSE
        RMSE_CNN = np.sqrt(np.mean((Nino_ori - Nino_pred) ** 2, axis=0))
        np.save(save_dir + '/RMSE_CNN_cond_3.npy', RMSE_CNN)

        wMAPE_CNN = np.sum(np.abs(Nino_ori - Nino_pred), axis=0) / np.sum(np.abs(Nino_ori), axis=0)
        np.save(save_dir + '/wMAPE_CNN_cond_3.npy', wMAPE_CNN)
