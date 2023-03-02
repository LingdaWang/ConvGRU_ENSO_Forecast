#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   ConvGRU.py
@Time    :   01/01/2023
@Author  :   lingdaw2
@Mail    :   lingdaw2@illinois.edu
@Description:   analyze the performance on test dataset.
"""

import sys 
import os
import random
import pickle
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from scipy.stats.stats import pearsonr
sys.path.append("..")
from model import Encoder, Decoder, ED
from NetParams import convgru_encoder_params, convgru_decoder_params
from utils import SstSeq

random_seed = 5
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.cuda.manual_seed(random_seed)

if __name__ == "__main__":
    # make dir
    saved_ConvGRU_model_folder = "./saved_ConvGRU_model"
    save_dir = "./plots"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # average temperature of Nino 3.4 region from CCSM4 dataset
    avg_temp_by_month = np.array([24.33743506, 24.33262092, 24.40992943, 24.72788656, 25.12441604, 25.10031928,
                                  24.64784378, 23.99778991, 23.64409438, 23.82734981, 24.18979863, 24.33067972])
    # import the true Nino 3.4 index
    with open("../data/nino_truth.pckl", "rb") as f:
        nino_test = pickle.load(f)

    cond_len, pred_len, test_data_len, batch_size = 48, 24, 2340, 1
    validFolder = SstSeq(root='../data', is_train=False, cond_len=cond_len, pred_len=pred_len)
    validLoader = torch.utils.data.DataLoader(validFolder, batch_size=batch_size, shuffle=False)
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(encoder_params[0], encoder_params[1], cond_len=cond_len).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1], pred_len=pred_len).cuda()
    net = ED(encoder, decoder)
    net.to(device)
    PATH = saved_ConvGRU_model_folder + '/checkpoint_4_0.484522.pth.tar'
    model_info = torch.load(PATH)
    net.load_state_dict(model_info['state_dict'])

    lossfunction = nn.MSELoss().cuda()
    ori_res, pred_res, start_points = np.zeros((test_data_len, pred_len, 12, 10)), \
                                      np.zeros((test_data_len, pred_len, 12, 10)), \
                                      np.zeros(test_data_len)
    test_losses = []
    with torch.no_grad():
        net.eval()
        t = tqdm(validLoader, leave=False, total=len(validLoader))
        for i, (_, targetVar, inputVar, start_point) in enumerate(t):
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
            Nino_pred[i, j] = (np.mean(pred_res[i, j, :, :]) -
                               avg_temp_by_month[(int(start_points[i]) + cond_len + j) % 12]) / 0.9171

    # Save the data for plot sample truth and forecast
    np.save(save_dir + '/ori_res.npy', ori_res)
    np.save(save_dir + '/pred_res.npy', pred_res)

    # Compute the RMSE per Pixel
    RMSE_ConvGRU_per_pixel = np.sqrt(np.mean((pred_res - ori_res) ** 2, axis=(0, 2, 3)))
    np.save(save_dir + '/RMSE_ConvGRU_per_pixel.npy', RMSE_ConvGRU_per_pixel)

    # Compute the Person Correlation
    P_corr_ConvGRU = np.array([pearsonr(Nino_ori[:, i], Nino_pred[:, i])[0] for i in range(pred_len)])
    np.save(save_dir + '/P_corr_ConvGRU.npy', P_corr_ConvGRU)

    # Compute the RMSE
    RMSE_ConvGRU = np.sqrt(np.mean((Nino_ori - Nino_pred) ** 2, axis=0))
    np.save(save_dir + '/RMSE_ConvGRU.npy', RMSE_ConvGRU)
