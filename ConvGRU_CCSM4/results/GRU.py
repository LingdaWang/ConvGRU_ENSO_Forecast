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
import pickle
import sys
import random
sys.path.append("..")
import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import gluon
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.transform import ExpectedNumInstanceSampler
from gluonts.mx import Trainer
from gluonts.evaluation import make_evaluation_predictions
from gluonts.evaluation import Evaluator
from scipy.stats.stats import pearsonr

random_seed = 123
np.random.seed(random_seed)
random.seed(random_seed)
mx.random.seed(random_seed)


def computeMonthlySST(root):
    path = os.path.join(root, 'dataset_region_DS.pckl')
    with open(path, 'rb') as f:
        x = pickle.load(f)
    SST_by_month = np.hstack((np.mean(x['Y_train'], axis=(1, 2)), np.mean(x['Y_test'], axis=(1, 2))))
    return SST_by_month


if __name__ == "__main__":
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

    # Configure the Dataset Information
    custom_ds_metadata = {
        'num_series': 1,
        'num_steps': SST_by_month.shape[0],
        'prediction_length': 24,
        'condition_length': 48,
        'freq': 'M',
        'start': pd.Timestamp("01-01-1800")  # time set randomly
    }

    # Define the Training Dataset
    train_len, test_len = 13188, 2412
    train_ds = ListDataset(
        [
            {
                FieldName.TARGET: d[:train_len],
                FieldName.START: custom_ds_metadata['start'],
                FieldName.FEAT_STATIC_CAT: [idx]
            }
            for idx, d in enumerate([SST_by_month])
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
            num_instances=6400.0,
            min_future=custom_ds_metadata['prediction_length']
        ),
        trainer=Trainer(
            ctx='cpu',
            epochs=50,
            num_batches_per_epoch=200,
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
    SST_by_month_pred = np.zeros((test_len - len_tot, custom_ds_metadata['prediction_length']))
    for i in range(test_len - len_tot):
        if i % 100 == 0:
            print("{}th Testing ...".format(i))
        test_ds = ListDataset(
            [
                {
                    FieldName.TARGET: d[:train_len + len_tot + i],
                    FieldName.START: custom_ds_metadata['start'],
                    FieldName.FEAT_STATIC_CAT: [idx]
                }
                for idx, d in enumerate([SST_by_month])
            ],
            freq=custom_ds_metadata['freq']
        )
        forecast_it, _ = make_evaluation_predictions(
            dataset=test_ds,  # test dataset
            predictor=predictor,  # predictor
            num_samples=500,  # number of sample paths we want for evaluation
        )
        forecasts = list(forecast_it)
        SST_by_month_pred[i, :] = forecasts[0].median

    # compute the true Nino 3.4 index and predicted Nino 3.4 index
    Nino_ori, Nino_pred = np.zeros((test_len - len_tot, custom_ds_metadata['prediction_length'])), \
                          np.zeros((test_len - len_tot, custom_ds_metadata['prediction_length']))
    for i in range(test_len - len_tot):
        for j in range(custom_ds_metadata['prediction_length']):
            Nino_ori[i, j] = nino_test[custom_ds_metadata['condition_length'] + i + j]
            Nino_pred[i, j] = SST_by_month_pred[i, j] - \
                              avg_SST_by_month[(train_len + custom_ds_metadata['condition_length'] + i + j) % 12]

    # compute the Pearson Correlation
    P_corr_GRU = np.array([pearsonr(Nino_ori[:, i], Nino_pred[:, i])[0] for
                           i in range(custom_ds_metadata['prediction_length'])])
    np.save(save_dir + '/P_corr_GRU.npy', P_corr_GRU)

    # compute the RMSE
    RMSE_GRU = np.sqrt(np.mean((Nino_ori - Nino_pred) ** 2, axis=0))
    np.save(save_dir + '/RMSE_GRU.npy', RMSE_GRU)
