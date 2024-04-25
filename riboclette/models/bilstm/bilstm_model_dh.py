# libraries
import numpy as np
import torch
from utils_bilstm_dh import trainLSTM, RiboDatasetGWSDepr, GWSDatasetFromPandas # custom dataset and trainer
import random
from sklearn.model_selection import KFold
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import argparse

# reproducibility
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='seed value')
args = parser.parse_args()

seed_val = args.seed

pl.seed_everything(seed_val)

# training arguments
tot_epochs = 100
batch_size = 1
dropout_val = 0.1
annot_thresh = 0.3
longZerosThresh_val = 20
percNansThresh_val = 0.05
lr = 1e-4

model_name = 'LSTM DS: DeprNA [' + str(annot_thresh) + ', ' + str(longZerosThresh_val) + ', ' + str(percNansThresh_val) + ', BS ' + str(batch_size) + ', D ' + str(dropout_val) + ' E ' + str(tot_epochs) + ' LR ' + str(lr) + '] ' + 'Seed: ' + str(seed_val)

# start a new wandb run to track this script
wandb_logger = WandbLogger(log_model="all", project="XLNet-DH", name=model_name)

# model parameters
save_loc = 'saved_models/' + model_name

# load datasets train and test
# GWS dataset
train_dataset, test_dataset = RiboDatasetGWSDepr(threshold = annot_thresh, longZerosThresh = longZerosThresh_val, percNansThresh = percNansThresh_val)

# convert to torch dataset
train_dataset = GWSDatasetFromPandas(train_dataset)
test_dataset = GWSDatasetFromPandas(test_dataset)

print("samples in train dataset: ", len(train_dataset))
print("samples in test dataset: ", len(test_dataset))

# train model
model, result = trainLSTM(tot_epochs, batch_size, lr, save_loc, wandb_logger, train_dataset, test_dataset, dropout_val)

# print(1.0 - result['test'][0]['test_loss'])
