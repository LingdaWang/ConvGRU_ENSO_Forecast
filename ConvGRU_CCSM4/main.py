#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   main.py
@Time    :   12/27/2022
@Author  :   lingdaw2
@Mail    :   lingdaw2@illinois.edu
@Description:   main function
"""

import os
import torch
import argparse
import random
import numpy as np
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from torch.optim import lr_scheduler
from model import Encoder, Decoder, ED
from NetParams import convgru_encoder_params, convgru_decoder_params
from utils import SstSeq, RecordHist


FOLDERNAME = 'saved_ConvGRU_model'
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',
                    default=4,
                    type=int,
                    help='mini-batch size')
parser.add_argument('-lr', default=1e-4, type=float, help='G learning rate')
parser.add_argument('-cond_len',
                    default=48,
                    type=int,
                    help='condition length')
parser.add_argument('-pred_len',
                    default=24,
                    type=int,
                    help='prediction length')
parser.add_argument('-epochs', default=20, type=int, help='sum of epochs')
args = parser.parse_args()

random_seed = 123
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

save_dir = './results/' + FOLDERNAME
trainFolder = SstSeq(root='./data', is_train=True, cond_len=args.cond_len, pred_len=args.pred_len)
validFolder = SstSeq(root='./data', is_train=False, cond_len=args.cond_len, pred_len=args.pred_len)
trainLoader = torch.utils.data.DataLoader(trainFolder, batch_size=args.batch_size, shuffle=False)
validLoader = torch.utils.data.DataLoader(validFolder, batch_size=args.batch_size, shuffle=False)

encoder_params = convgru_encoder_params
decoder_params = convgru_decoder_params


def train():
    """
    main function to run the training
    """
    encoder = Encoder(encoder_params[0], encoder_params[1], cond_len=args.cond_len).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1], pred_len=args.pred_len).cuda()
    net = ED(encoder, decoder)
    # initialize the record hist object
    record_hist = RecordHist(verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
        # load existing model
        print('==> loading existing model')
        model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
        net.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(net.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cur_epoch = 0
    lossfunction = nn.MSELoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4, verbose=True)

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    for epoch in range(cur_epoch, args.epochs + 1):
        ###################
        # train the model #
        ###################
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        for i, (_, targetVar, inputVar, _) in enumerate(t):
            inputs = inputVar.to(device)  # B,S,C,H,W
            label = targetVar.to(device)  # B,S,C,H,W
            optimizer.zero_grad()
            net.train()
            pred = net(inputs)  # B,S,C,H,W
            loss = lossfunction(pred, label)
            loss_aver = loss.item()
            train_losses.append(loss_aver)
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
            optimizer.step()
            t.set_postfix({
                'trainloss': '{:.6f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })

        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            net.eval()
            t = tqdm(validLoader, leave=False, total=len(validLoader))
            for i, (_, targetVar, inputVar, _) in enumerate(t):
                inputs = inputVar.to(device)
                label = targetVar.to(device)
                pred = net(inputs)
                loss = lossfunction(pred, label)
                loss_aver = loss.item()
                # record validation loss
                valid_losses.append(loss_aver)
                t.set_postfix({
                    'validloss': '{:.6f}'.format(loss_aver),
                    'epoch': '{:02d}'.format(epoch)
                })
                if i + 1 == 2340 // args.batch_size:
                    break

        torch.cuda.empty_cache()
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        epoch_len = len(str(args.epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.6f} ' +
                     f'valid_loss: {valid_loss:.6f}')
        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        pla_lr_scheduler.step(valid_loss)  # lr_scheduler
        model_dict = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        record_hist(valid_loss.item(), model_dict, epoch, save_dir)

    with open("avg_train_losses.txt", 'wt') as f:
        for i in avg_train_losses:
            print(i, file=f)

    with open("avg_valid_losses.txt", 'wt') as f:
        for i in avg_valid_losses:
            print(i, file=f)


if __name__ == "__main__":
    train()
