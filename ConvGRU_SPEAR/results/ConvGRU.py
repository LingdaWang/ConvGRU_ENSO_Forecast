#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   ConvGRU.py
@Time    :   01/02/2023
@Author  :   lingdaw2
@Mail    :   lingdaw2@illinois.edu
@Description:   analyze the performance on test dataset.
"""


import sys
import os
import random
import torch
import netCDF4 as nc
import numpy as np
from torch import nn
from tqdm import tqdm
from scipy.stats.stats import pearsonr
from skimage.transform import downscale_local_mean
sys.path.append("..")
from model import Encoder, Decoder, ED
from NetParams import convgru_encoder_params, convgru_decoder_params
from utils import SstSeq

random_seed = 123
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.cuda.manual_seed(random_seed)

if __name__ == "__main__":
    # make dir
    saved_ConvGRU_model_folder = "./saved_ConvGRU_model"
    save_dir = "./res_ConvGRU"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params

    print("Reading Data...")
    tosTotal = dict()
    root = '../data'
    avgTos = np.zeros((2160, 180, 360))
    for i in range(1, 31):
        print("reading {}th dataset...".format(i))
        fileHist = os.path.join(root, "tos_Omon_GFDL-SPEAR-MED_historical_r{}i1p1f1_gr_192101-201412.nc".format(i))
        fileFut = os.path.join(root, "tos_Omon_GFDL-SPEAR-MED_scenarioSSP5-85_r{}i1p1f1_gr_201501-210012.nc".format(i))
        tosTot = np.ma.vstack((nc.Dataset(fileHist)["tos"][:], nc.Dataset(fileFut)["tos"][:]))
        tosTotal[i] = tosTot.filled(fill_value=0)
        avgTos += tosTotal[i] / 30
    for i in range(1, 31):
        tosTotal[i] -= avgTos
    print("Done!")

    # down-sample and reformulate the data
    SST_by_month, avg_SST_yearly = np.zeros((30, 2160)), np.zeros((30, 12))
    for i in range(1, 31):
        tos = downscale_local_mean(tosTotal[i][:, 21:149, 150:278], (1, 2, 2))
        SST_by_month[i - 1, :] = tos[:, 32:37, 20:45].mean(axis=(1, 2))
        avg_SST_yearly[i - 1, :] = np.array([np.mean(SST_by_month[j::12]) for j in range(12)])

    cond_len, pred_len, batch_size = 48, 24, 1
    validFolder = SstSeq(root=root, is_train=False, cond_len=cond_len, pred_len=pred_len)
    validLoader = torch.utils.data.DataLoader(validFolder, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(encoder_params[0], encoder_params[1], cond_len=cond_len).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1], pred_len=pred_len).cuda()
    net = ED(encoder, decoder)
    net.to(device)
    # name of saved model
    PATH = saved_ConvGRU_model_folder + '/checkpoint_7_0.352630.pth.ta'
    model_info = torch.load(PATH)
    net.load_state_dict(model_info['state_dict'])

    lossfunction = nn.MSELoss().cuda()
    test_data_size = (600 - pred_len - cond_len) * 30
    ori_res, pred_res, start_points = np.zeros((test_data_size, pred_len, 64, 64)), \
                                      np.zeros((test_data_size, pred_len, 64, 64)), \
                                      np.zeros((test_data_size, 2))
    test_losses = []
    with torch.no_grad():
        net.eval()
        t = tqdm(validLoader, leave=False, total=len(validLoader))
        for i, (_, targetVar, inputVar, data_set_index, start_point) in enumerate(t):
            inputs = inputVar.to(device)
            label = targetVar.to(device)
            pred = net(inputs)
            loss = lossfunction(pred, label)
            loss_aver = loss.item()
            test_losses.append(loss_aver)
            t.set_postfix({
                'validloss': '{:.6f}'.format(loss_aver)
            })
            ori_res[i, ...] = (torch.squeeze(label.to('cpu'))).numpy()
            pred_res[i, ...] = (torch.squeeze(pred.to('cpu'))).numpy()
            start_points[i, 0], start_points[i, 1] = data_set_index, start_point
            if i + 1 == test_data_size:
                break

    test_loss = np.average(test_losses)
    print("test loss is {}".format(test_loss))

    # compute the true Nino 3.4 index and predicted Nino 3.4 index
    Nino_ori, Nino_pred = np.zeros((test_data_size, pred_len)), np.zeros((test_data_size, pred_len))
    for i in range(test_data_size):
        for j in range(pred_len):
            Nino_ori[i, j] = np.mean(ori_res[i, j, 32:37, 20:45]) - \
                            avg_SST_yearly[int(start_points[i, 0] - 1), int(start_points[i, 1] + cond_len + j) % 12]
            Nino_pred[i, j] = np.mean(pred_res[i, j, 32:37, 20:45]) - \
                             avg_SST_yearly[int(start_points[i, 0] - 1), int(start_points[i, 1] + cond_len + j) % 12]

    # Compute the Person Correlation
    P_corr_ConvGRU = np.array([pearsonr(Nino_ori[:, i], Nino_pred[:, i])[0] for i in range(pred_len)])
    np.save(save_dir + '/P_corr_ConvGRU.npy', P_corr_ConvGRU)

    # Compute the RMSE
    RMSE_ConvGRU = np.sqrt(np.mean((Nino_ori - Nino_pred) ** 2, axis=0))
    np.save(save_dir + '/RMSE_ConvGRU.npy', RMSE_ConvGRU)
