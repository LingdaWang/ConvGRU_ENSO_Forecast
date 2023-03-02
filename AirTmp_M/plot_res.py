#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   ConvGRU.py
@Time    :   01/29/2023
@Author  :   lingdaw2
@Mail    :   lingdaw2@illinois.edu
@Description:   analyze the performance on test dataset.
"""

import os
from model import Encoder, Decoder, ED
from NetParams import convgru_encoder_params, convgru_decoder_params
from utils import SstSeq
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import random
from scipy.stats.stats import pearsonr

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

    cond_len, pred_len, batch_size, test_data_len = 24, 12, 1, 372
    validFolder = SstSeq(root='./data', is_train=False, cond_len=cond_len, pred_len=pred_len)
    validLoader = torch.utils.data.DataLoader(validFolder, batch_size=batch_size, shuffle=False)
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(encoder_params[0], encoder_params[1], cond_len=cond_len).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1], pred_len=pred_len).cuda()
    net = ED(encoder, decoder)
    net.to(device)
    PATH = saved_ConvGRU_model_folder + '/checkpoint_21_1.428732.pth.tar'
    model_info = torch.load(PATH)
    net.load_state_dict(model_info['state_dict'])

    lossfunction = nn.MSELoss().cuda()
    ori_res, pred_res, start_points = np.zeros((test_data_len, pred_len, 64, 128)), \
                                      np.zeros((test_data_len, pred_len, 64, 128)), \
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

    # Save the data for plot sample truth and forecast
    np.save(save_dir + '/ori_res.npy', ori_res)
    np.save(save_dir + '/pred_res.npy', pred_res)

    # Compute the RMSE per Pixel
    RMSE = np.sqrt(np.mean((pred_res - ori_res)**2, axis=(0, 2, 3)))
    np.save(save_dir + '/RMSE.npy', RMSE)

    # Compute the Person Correlation
    PC = np.zeros(12)
    for i in range(12):
        PC[i], _ = pearsonr(pred_res[:, i, :, :].reshape(-1), ori_res[:, i, :, :].reshape(-1))
    np.save(save_dir + '/PC.npy', PC)
