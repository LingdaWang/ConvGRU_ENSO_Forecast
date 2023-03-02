#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   LR.py
@Time    :   02/05/2023
@Author  :   lingdaw2
@Mail    :   lingdaw2@illinois.edu
@Description:   Linear Regression
"""

import os
import numpy as np
import netCDF4 as nc
from scipy.stats.stats import pearsonr
from skimage.transform import downscale_local_mean
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    # Create saving dir
    save_dir = "./res_LR"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    print("Reading Data...")
    num_series = 30
    tosTotal = dict()
    root = '../data'
    avgTos = np.zeros((2160, 180, 360))
    for i in range(1, num_series + 1):
        print("reading {}th dataset...".format(i))
        fileHist = os.path.join(root, "tos_Omon_GFDL-SPEAR-MED_historical_r{}i1p1f1_gr_192101-201412.nc".format(i))
        fileFut = os.path.join(root, "tos_Omon_GFDL-SPEAR-MED_scenarioSSP5-85_r{}i1p1f1_gr_201501-210012.nc".format(i))
        tosTot = np.ma.vstack((nc.Dataset(fileHist)["tos"][:], nc.Dataset(fileFut)["tos"][:]))
        tosTotal[i] = tosTot.filled(fill_value=0)
        avgTos += tosTotal[i] / num_series
    for i in range(1, num_series + 1):
        tosTotal[i] -= avgTos
    print("Done!")

    # down-sample and reformulate the data
    SST_by_month, avg_SST_yearly = np.zeros((num_series, 2160)), np.zeros((num_series, 12))
    for i in range(1, num_series + 1):
        tos = downscale_local_mean(tosTotal[i][:, 21:149, 150:278], (1, 2, 2))
        SST_by_month[i - 1, :] = tos[:, 32:37, 20:45].mean(axis=(1, 2))
        avg_SST_yearly[i - 1, :] = np.array([np.mean(SST_by_month[j::12]) for j in range(12)])

    # condition length and prediction length
    cond_len, pred_len, train_len, test_len = 48, 24, 1560, 600
    SST_by_month_pred = np.zeros((num_series, test_len + 1 - cond_len - pred_len, pred_len))
    for idx in range(num_series):
        for i in range(pred_len):
            Y_train = SST_by_month[idx, cond_len + i: train_len + 1 + i - pred_len]
            X_train = np.array([SST_by_month[idx, j:train_len + 1 - pred_len - cond_len + j] for j in range(cond_len)])
            reg = LinearRegression().fit(X_train.T, Y_train)
            print("Coef of Determination R^2 of {idx}th series at {i}-month ahead is {score}".format(
                idx=idx+1, i=i+1, score=reg.score(X_train.T, Y_train))
            )
            X_test = np.array(
                [SST_by_month[idx, train_len + j:train_len + test_len + 1 - pred_len - cond_len + j] for j in
                 range(cond_len)])
            SST_by_month_pred[idx, :, i] = reg.predict(X_test.T)

    # compute the true Nino 3.4 index and predicted Nino 3.4 index
    Nino_ori, Nino_pred = np.zeros((num_series, test_len + 1 - cond_len - pred_len, pred_len)), \
                          np.zeros((num_series, test_len + 1 - cond_len - pred_len, pred_len))
    for i in range(num_series):
        for j in range(test_len + 1 - cond_len - pred_len):
            for k in range(pred_len):
                temp = train_len + cond_len + j + k
                Nino_ori[i, j, k] = SST_by_month[i, temp] - avg_SST_yearly[i, temp % 12]
                Nino_pred[i, j, k] = SST_by_month_pred[i, j, k] - avg_SST_yearly[i, temp % 12]
    Nino_ori, Nino_pred = np.reshape(Nino_ori, (-1, pred_len)), np.reshape(Nino_pred, (-1, pred_len))

    # Compute the Person Correlation
    P_corr_LR = np.array([pearsonr(Nino_ori[:, i], Nino_pred[:, i])[0] for i in range(pred_len)])
    np.save(save_dir + '/P_corr_LR.npy', P_corr_LR)

    # Compute the RMSE
    RMSE_LR = np.sqrt(np.mean((Nino_ori - Nino_pred) ** 2, axis=0))
    np.save(save_dir + '/RMSE_LR.npy', RMSE_LR)
