'''
model trained on only one dataset samples
'''

# libraries
import numpy as np
import pandas as pd 
import torch
from utils_bilstm_sh import trainLSTM, RiboDatasetGWSDepr, GWSDatasetFromPandas # custom dataset and trainer
import random
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional import pearson_corrcoef
from torchmetrics import Metric
import argparse
from sklearn.model_selection import KFold
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

# reproducibility
pl.seed_everything(42)

# training arguments
tot_epochs = 100
batch_size = 1
dropout_val = 0.1
annot_thresh = 0.3
longZerosThresh_val = 20
percNansThresh_val = 0.05
lr = 1e-4

parser = argparse.ArgumentParser()
parser.add_argument('--condition', type=str, default='CTRL', help='condition to train on')
args = parser.parse_args()

condition_training = args.condition

model_name = 'LSTM DS: ' + str(condition_training) + ' [' + str(annot_thresh) + ', ' + str(longZerosThresh_val) + ', ' + str(percNansThresh_val) + ', BS ' + str(batch_size) + ', D ' + str(dropout_val) + ' E ' + str(tot_epochs) + ' LR ' + str(lr) + ']'

# start a new wandb run to track this script
wandb_logger = WandbLogger(log_model="all", project="XLNet-DH", name=model_name)

# model parameters
save_loc = 'saved_models/' + model_name

# load datasets train and test
# GWS dataset
if condition_training == 'CTRL':
    train_dataset, test_dataset = RiboDatasetGWSDepr(threshold = annot_thresh, longZerosThresh = longZerosThresh_val, percNansThresh = percNansThresh_val, cond = 'CTRL')
elif condition_training == 'LEU':
    train_dataset, test_dataset = RiboDatasetGWSDepr(threshold = annot_thresh, longZerosThresh = longZerosThresh_val, percNansThresh = percNansThresh_val, cond = 'LEU')
elif condition_training == 'ILE':
    train_dataset, test_dataset = RiboDatasetGWSDepr(threshold = annot_thresh, longZerosThresh = longZerosThresh_val, percNansThresh = percNansThresh_val, cond = 'ILE')
elif condition_training == 'LEU-ILE':
    train_dataset, test_dataset = RiboDatasetGWSDepr(threshold = annot_thresh, longZerosThresh = longZerosThresh_val, percNansThresh = percNansThresh_val, cond = 'LEU_ILE')
elif condition_training == 'VAL':
    train_dataset, test_dataset = RiboDatasetGWSDepr(threshold = annot_thresh, longZerosThresh = longZerosThresh_val, percNansThresh = percNansThresh_val, cond = 'VAL')
elif condition_training == 'LEU-ILE-VAL':
    train_dataset, test_dataset = RiboDatasetGWSDepr(threshold = annot_thresh, longZerosThresh = longZerosThresh_val, percNansThresh = percNansThresh_val, cond = 'LEU_ILE_VAL')
elif condition_training == 'OnlyLiver':
    train_dataset, test_dataset = RiboDatasetGWSDepr(threshold = annot_thresh, longZerosThresh = longZerosThresh_val, percNansThresh = percNansThresh_val, cond = 'OnlyLiver')

# convert to torch dataset
train_dataset = GWSDatasetFromPandas(train_dataset)
test_dataset = GWSDatasetFromPandas(test_dataset)

print("samples in train dataset: ", len(train_dataset))
print("samples in test dataset: ", len(test_dataset))

# train model
model, result = trainLSTM(tot_epochs, batch_size, lr, save_loc, wandb_logger, train_dataset, test_dataset, dropout_val)

# print(1.0 - result['test'][0]['test_loss'])
