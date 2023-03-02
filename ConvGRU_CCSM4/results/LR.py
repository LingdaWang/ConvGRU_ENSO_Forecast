#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   LR.py
@Time    :   02/04/2023
@Author  :   lingdaw2
@Mail    :   lingdaw2@illinois.edu
@Description:   Linear Regression
"""

import os
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats.stats import pearsonr


def computeMonthlySST(root):
    path = os.path.join(root, 'dataset_region_DS.pckl')
    with open(path, 'rb') as f:
        x = pickle.load(f)
    SST_by_month = np.hstack((np.mean(x['Y_train'], axis=(1, 2)), np.mean(x['Y_test'], axis=(1, 2))))
    return SST_by_month


if __name__ == "__main__":

    # Create saving dir
    save_dir = "./plots"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # import the true Nino 3.4 index
    with open("../data/nino_truth.pckl", "rb") as f:
        nino_test = pickle.load(f)
    # Load Monthly Avg SST of Nino 3.4 Region
    avg_SST_by_month = np.array([24.33743506, 24.33262092, 24.40992943, 24.72788656, 25.12441604, 25.10031928,
                                  24.64784378, 23.99778991, 23.64409438, 23.82734981, 24.18979863, 24.33067972])
    # Compute Monthly Temp of Nino 3.4 Region
    SST_by_month = computeMonthlySST(root='../data')

    # condition length and prediction length
    cond_len, pred_len, training_len, test_len = 48, 24, 13188, 2412
    SST_pred = np.zeros((test_len - cond_len - pred_len + 1, pred_len))
    for i in range(pred_len):
        Y_train = SST_by_month[cond_len + i: training_len + 1 + i - pred_len]
        X_train = np.array([SST_by_month[j:training_len + 1 - pred_len - cond_len + j] for j in range(cond_len)])
        reg = LinearRegression().fit(X_train.T, Y_train)
        print("Coef of Determination R^2 at {i}-month ahead is {score}".format(
            i=i + 1, score=reg.score(X_train.T, Y_train))
        )
        X_test = np.array(
            [SST_by_month[training_len + j:training_len + test_len + 1 - pred_len - cond_len + j] for
             j in range(cond_len)]
        )
        SST_pred[:, i] = reg.predict(X_test.T)

    # compute the true Nino 3.4 index and predicted Nino 3.4 index
    Nino_ori, Nino_pred = np.zeros((test_len - cond_len - pred_len + 1, pred_len)), \
                          np.zeros((test_len - cond_len - pred_len + 1, pred_len))
    for i in range(test_len - cond_len - pred_len + 1):
        for j in range(pred_len):
            Nino_ori[i, j] = nino_test[i + cond_len + j]
            Nino_pred[i, j] = SST_pred[i, j] - avg_SST_by_month[(i + cond_len + j) % 12]

    # Compute the Person Correlation
    P_corr_LR = np.array([pearsonr(Nino_ori[:, i], Nino_pred[:, i])[0] for i in range(pred_len)])
    np.save(save_dir + '/P_corr_LR.npy', P_corr_LR)

    # Compute the RMSE
    RMSE_LR = np.sqrt(np.mean((Nino_ori - Nino_pred) ** 2, axis=0))
    np.save(save_dir + '/RMSE_LR.npy', RMSE_LR)
