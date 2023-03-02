#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   LIM.py
@Time    :   02/05/2023
@Author  :   lingdaw2
@Mail    :   lingdaw2@illinois.edu
@Description:   Linear Inverse Model
"""

import os
import pickle
import numpy as np
from scipy.linalg import eigh, inv
from sklearn.linear_model import LinearRegression
from scipy.stats.stats import pearsonr


def loadData(root):
    path = os.path.join(root, 'dataset_region_DS.pckl')
    with open(path, 'rb') as f:
        x = pickle.load(f)
    avg_SST_by_month = np.hstack((np.mean(x['Y_train'], axis=(1, 2)), np.mean(x['Y_test'], axis=(1, 2))))
    gridded_SST_by_month = np.reshape(np.vstack((x['X_train'], x['X_test'])), (15600, -1))
    return avg_SST_by_month, gridded_SST_by_month


if __name__ == "__main__":
    # make dir
    save_dir = "./plots"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # import the true Nino 3.4 index
    with open("../data/nino_truth.pckl", "rb") as f:
        nino_test = pickle.load(f)

    # Load Yearly Avg SST of Nino 3.4 Region
    avg_SST_yearly = [24.33743506, 24.33262092, 24.40992943, 24.72788656, 25.12441604, 25.10031928,
                      24.64784378, 23.99778991, 23.64409438, 23.82734981, 24.18979863, 24.33067972]

    # Load the gridded data
    avg_SST_by_month, gridded_SST_by_month = loadData('../data')
    avg_SST_yearly = np.array(avg_SST_yearly * 1300)

    # Training Process
    train_len, test_len, num_eigen = 13188, 2412, 50
    tos_train = gridded_SST_by_month[:train_len, :]
    avg_SST_by_month_train = avg_SST_by_month[:train_len]
    # Compute the Psi matrix
    Cov_tos = tos_train.T @ tos_train
    eigen_val, eigen_vec = eigh(
        Cov_tos,
        subset_by_index=[tos_train.shape[1] - num_eigen, tos_train.shape[1] - 1]
    )
    Psi = tos_train @ eigen_vec / np.sqrt(eigen_val)
    reg = LinearRegression(fit_intercept=False).fit(Psi, avg_SST_by_month_train)
    z = reg.coef_
    print("Coeff of determination is {}.".format(reg.score(Psi, avg_SST_by_month_train)))

    # Testing Process
    cond_len, pred_len = 48, 24
    res_pred = np.zeros((test_len - cond_len - pred_len + 1, pred_len))
    res_ori = np.zeros((test_len - cond_len - pred_len + 1, pred_len))
    # PreComp
    G = np.zeros((num_eigen, num_eigen))
    for i in range(train_len + cond_len - 1):
        tmp = gridded_SST_by_month[i, :].reshape((1, -1)) @ eigen_vec / np.sqrt(eigen_val)
        G += tmp.T @ tmp

    G_i = dict()
    for i in range(1, pred_len +1):
        G_i[i] = np.zeros((num_eigen, num_eigen))
        for j in range(train_len + cond_len - 1 - i):
            tmp1 = gridded_SST_by_month[j, :].reshape((1, -1)) @ eigen_vec / np.sqrt(eigen_val)
            tmp2 = gridded_SST_by_month[i + j, :].reshape((1, -1)) @ eigen_vec / np.sqrt(eigen_val)
            G_i[i] += tmp2.T @ tmp1

    for i in range(test_len + 1 - cond_len - pred_len):
        tmp = gridded_SST_by_month[train_len + cond_len + i - 1, :].reshape((1, -1)) @ eigen_vec / np.sqrt(eigen_val)
        G += tmp.T @ tmp
        G_normalized = G / (train_len + cond_len + i)
        G_normalized_inv = inv(G_normalized)
        for j in range(1, pred_len + 1):
            tmp2 = gridded_SST_by_month[train_len + cond_len + i - j - 1, :].reshape((1, -1)) @ eigen_vec / np.sqrt(eigen_val)
            G_i[j] += tmp.T @ tmp2
            G_i_normalized = G_i[j] / (train_len + cond_len + i - j)
            G_tau = G_i_normalized @ G_normalized_inv
            res_pred[i, j - 1] = np.sum((G_tau @ tmp.T).reshape(-1) * z) - \
                                avg_SST_yearly[train_len + cond_len + i + j - 1]
            res_ori[i, j - 1] = nino_test[cond_len + i + j - 1]

    # Compute Pearson Correlation
    P_corr_LIM = np.zeros(pred_len)
    for i in range(pred_len):
        P_corr_LIM[i], _ = pearsonr(res_ori[:, i], res_pred[:, i])
    np.save(save_dir + '/P_corr_LIM.npy', P_corr_LIM)

    # Compute RMSE
    RMSE_LIM = np.sqrt(np.mean((res_ori - res_pred) ** 2, axis=0))
    np.save(save_dir + '/RMSE_LIM.npy', RMSE_LIM)
