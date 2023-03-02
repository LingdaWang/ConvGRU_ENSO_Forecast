#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   LIM.py
@Time    :   02/05/2023
@Author  :   lingdaw2
@Mail    :   lingdaw2@illinois.edu
@Description:   implement the linear inverse model
"""

import os
import netCDF4 as nc
import numpy as np
from scipy.linalg import eigh, inv
from skimage.transform import downscale_local_mean
from sklearn.linear_model import LinearRegression
from scipy.stats.stats import pearsonr


if __name__ == "__main__":

    # Create saving dir
    save_dir = "./res_LIM"
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
    SST_by_month, avg_SST_yearly, tos = np.zeros((num_series, 2160)), np.zeros((num_series, 12)), dict()
    for i in range(num_series):
        tos[i] = downscale_local_mean(tosTotal[i + 1][:, 21:149, 150:278], (1, 2, 2))
        SST_by_month[i, :] = tos[i][:, 32:37, 20:45].mean(axis=(1, 2))
        avg_SST_yearly[i, :] = np.array([np.mean(SST_by_month[j::12]) for j in range(12)])
        tos[i] = tos[i].reshape((2160, -1))

    # Training Process
    train_len, test_len, num_eigen, cond_len, pred_len = 1560, 600, 50, 48, 24
    SST_by_month_pred = np.zeros((num_series, test_len + 1 - cond_len - pred_len, pred_len))
    for idx in range(num_series):
        tos_train = tos[idx][:train_len, :]
        SST_by_month_train = SST_by_month[idx, :train_len]
        # Compute the Psi matrix
        Cov_tos = tos_train.T @ tos_train
        eigen_val, eigen_vec = eigh(
            Cov_tos,
            subset_by_index=[tos_train.shape[1] - num_eigen, tos_train.shape[1] - 1]
        )
        Psi = tos_train @ eigen_vec / np.sqrt(eigen_val)
        reg = LinearRegression(fit_intercept=False).fit(Psi, SST_by_month_train)
        z = reg.coef_
        print("Coeff of determination of {}th series is {}.".format(idx, reg.score(Psi, SST_by_month_train)))

        # Testing Process
        # PreComp
        G = np.zeros((num_eigen, num_eigen))
        for i in range(train_len + cond_len - 1):
            tmp = tos[idx][i, :].reshape((1, -1)) @ eigen_vec / np.sqrt(eigen_val)
            G += tmp.T @ tmp

        G_i = dict()
        for i in range(1, pred_len + 1):
            G_i[i] = np.zeros((num_eigen, num_eigen))
            for j in range(train_len + cond_len - 1 - i):
                tmp1 = tos[idx][j, :].reshape((1, -1)) @ eigen_vec / np.sqrt(eigen_val)
                tmp2 = tos[idx][i + j, :].reshape((1, -1)) @ eigen_vec / np.sqrt(eigen_val)
                G_i[i] += tmp2.T @ tmp1

        for i in range(test_len + 1 - cond_len - pred_len):
            tmp = tos[idx][train_len + cond_len + i - 1, :].reshape((1, -1)) @ eigen_vec / np.sqrt(
                eigen_val)
            G += tmp.T @ tmp
            G_normalized = G / (train_len + cond_len + i)
            G_normalized_inv = inv(G_normalized)
            for j in range(1, pred_len + 1):
                tmp2 = tos[idx][train_len + cond_len + i - j - 1, :].reshape((1, -1)) @ eigen_vec / np.sqrt(
                    eigen_val)
                G_i[j] += tmp.T @ tmp2
                G_i_normalized = G_i[j] / (train_len + cond_len + i - j)
                G_tau = G_i_normalized @ G_normalized_inv
                SST_by_month_pred[idx, i, j - 1] = np.sum((G_tau @ tmp.T).reshape(-1) * z) - \
                                     avg_SST_yearly[idx, (train_len + cond_len + i + j - 1) % 12]

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
    P_corr_LIM = np.array([pearsonr(Nino_ori[:, i], Nino_pred[:, i])[0] for i in range(pred_len)])
    np.save(save_dir + '/P_corr_LIM.npy', P_corr_LIM)

    # Compute the RMSE
    RMSE_LIM = np.sqrt(np.mean((Nino_ori - Nino_pred) ** 2, axis=0))
    np.save(save_dir + '/RMSE_LIM.npy', RMSE_LIM)
