#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   GRU.py
@Time    :   01/28/2022
@Author  :   lingdaw2
@Mail    :   lingdaw2@illinois.edu
@Description:
"""

import os
import sys
import random
sys.path.append("..")
import numpy as np
import netCDF4 as nc
import pandas as pd
import mxnet as mx
from mxnet import gluon
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.transform import ExpectedNumInstanceSampler
from gluonts.mx import Trainer
from gluonts.evaluation import make_evaluation_predictions
from scipy.stats.stats import pearsonr
from skimage.transform import downscale_local_mean

random_seed = 123
np.random.seed(random_seed)
random.seed(random_seed)
mx.random.seed(random_seed)


if __name__ == "__main__":
    save_dir = "./res_GRU"
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

    # Configure the Dataset Information
    custom_ds_metadata = {
        'num_series': num_series,
        'num_steps': SST_by_month.shape[1],
        'prediction_length': 24,
        'condition_length': 48,
        'freq': 'M',
        'start': pd.Timestamp("01-01-1921")  # time set randomly
    }

    # Define the Training Dataset
    train_len, test_len = 1560, 600
    train_ds = ListDataset(
        [
            {
                FieldName.TARGET: d[:train_len],
                FieldName.START: custom_ds_metadata['start'],
                FieldName.FEAT_STATIC_CAT: [idx]
            }
            for idx, d in enumerate(SST_by_month)
        ],
        freq=custom_ds_metadata['freq']
    )

    # Configure the GRU Network
    estimator = DeepAREstimator(
        freq=custom_ds_metadata['freq'],
        prediction_length=custom_ds_metadata['prediction_length'],
        context_length=custom_ds_metadata['condition_length'],
        num_layers=1,
        num_cells=20,
        cell_type='gru',
        train_sampler=ExpectedNumInstanceSampler(
            num_instances=640,
            min_future=custom_ds_metadata['prediction_length']
        ),
        trainer=Trainer(
            ctx='cpu',
            epochs=50,
            num_batches_per_epoch=600,
            learning_rate=1e-3,
            weight_decay=1e-8
        ),
        dropout_rate=0.1,
        batch_size=32
    )

    # Training
    predictor = estimator.train(train_ds)

    # Testing
    len_tot = custom_ds_metadata['condition_length'] + custom_ds_metadata['prediction_length']
    SST_by_month_pred = np.zeros((num_series, test_len - len_tot, custom_ds_metadata['prediction_length']))
    for i in range(test_len - len_tot):
        if i % 100 == 0:
            print("{}th Testing ...".format(i))
        test_ds = ListDataset(
            [
                {
                    FieldName.TARGET: d[:test_len + len_tot + i],
                    FieldName.START: custom_ds_metadata['start'],
                    FieldName.FEAT_STATIC_CAT: [idx]
                }
                for idx, d in enumerate(SST_by_month)
            ],
            freq=custom_ds_metadata['freq']
        )
        forecast_it, _ = make_evaluation_predictions(
            dataset=test_ds,  # test dataset
            predictor=predictor,  # predictor
            num_samples=500,  # number of sample paths we want for evaluation
        )
        forecasts = list(forecast_it)
        for j in range(num_series):
            SST_by_month_pred[j, i, :] = forecasts[j].median

    # compute the true Nino 3.4 index and predicted Nino 3.4 index
    Nino_ori, Nino_pred = np.zeros((num_series, test_len - len_tot, custom_ds_metadata['prediction_length'])), \
                          np.zeros((num_series, test_len - len_tot, custom_ds_metadata['prediction_length']))
    for i in range(num_series):
        for j in range(test_len - len_tot):
            for k in range(custom_ds_metadata['prediction_length']):
                temp = train_len + custom_ds_metadata['condition_length'] + j + k
                Nino_ori[i, j, k] = SST_by_month[i, temp] - avg_SST_yearly[i, temp % 12]
                Nino_pred[i, j, k] = SST_by_month_pred[i, j, k] - avg_SST_yearly[i, temp % 12]
    Nino_ori, Nino_pred = np.reshape(Nino_ori, (-1, custom_ds_metadata['prediction_length'])), \
                          np.reshape(Nino_pred, (-1, custom_ds_metadata['prediction_length']))

    # compute the Person Correlation
    P_corr_GRU = np.array([pearsonr(Nino_ori[:, i], Nino_pred[:, i])[0] for
                           i in range(custom_ds_metadata['prediction_length'])])
    np.save(save_dir + '/P_corr_GRU.npy', P_corr_GRU)

    # compute the RMSE
    RMSE_GRU = np.sqrt(np.mean((Nino_ori - Nino_pred) ** 2, axis=0))
    np.save(save_dir + '/RMSE_GRU.npy', RMSE_GRU)
