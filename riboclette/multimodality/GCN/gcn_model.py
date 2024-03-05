'''
model trained on only one dataset samples
'''

# libraries
import numpy as np
import pandas as pd 
import torch
from utils import trainGCN, RiboDataset # custom dataset and trainer
import random
from torch.nn.utils.rnn import pad_sequence
import torch_geometric
import torch_geometric.transforms as T
from torch import nn
from torchmetrics.functional import pearson_corrcoef
from torchmetrics import Metric
import argparse
from sklearn.model_selection import KFold
from pytorch_lightning.loggers import WandbLogger

# reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# training arguments
feature_folder = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/mm/'
data_folder = '/nfs_home/nallapar/final/riboclette/riboclette/models/xlnet/data/sh/'
tot_epochs = 50
batch_size = 2
dropout_val = 0.4
annot_thresh = 0.3
longZerosThresh_val = 20
percNansThresh_val = 0.05
random_walk_length = 32
alpha = -1
lr = 1e-3
edge_attr = True # can put True only for GAT
loss_fn = 'MAE + PCC'
features = ['cbert_full', 'codon_ss']
model_type = 'DirSeq+' # USeq, USeq+, DirSeq, DirSeq+
algo = 'GAT' # SAGE, GAT
features_str = '_'.join(features)
model_name = model_type + '-' + algo + ' DS: Liver' + '[' + str(annot_thresh) + ', ' + str(longZerosThresh_val) + ', ' + str(percNansThresh_val) + ', BS ' + str(batch_size) + ', D ' + str(dropout_val) + ' E ' + str(tot_epochs) + ' LR ' + str(lr) + '] F: ' + features_str + ' VN RW 32 -1 + GraphNorm ' + loss_fn

input_nums_dict = {'cbert_full': 768, 'codon_ss': 0}
num_inp_ft = sum([input_nums_dict[ft] for ft in features])

# start a new wandb run to track this script
wandb_logger = WandbLogger(log_model="all", project="GCN_MM", name=model_name)

gcn_layers = [256, 128, 128, 64]

# model parameters
save_loc = 'saved_models/' + model_name

# make torch datasets from pandas dataframes
transforms = T.Compose([T.AddRandomWalkPE(walk_length=random_walk_length), T.VirtualNode()])

train_ds = RiboDataset('train', feature_folder, data_folder, model_type, transforms, edge_attr, sampler=False)
test_ds = RiboDataset('test', feature_folder, data_folder, model_type, transforms, edge_attr, sampler=False)

print("samples in train dataset: ", len(train_ds))
print("samples in test dataset: ", len(test_ds))

train_loader = torch_geometric.loader.DataListLoader(train_ds, batch_size=1, shuffle=True)
test_loader = torch_geometric.loader.DataListLoader(test_ds, batch_size=1, shuffle=False)

# train model
model, result = trainGCN(gcn_layers, tot_epochs, batch_size, lr, save_loc, wandb_logger, train_loader, test_loader, dropout_val, num_inp_ft, alpha, model_type, algo)

print(1.0 - result['test'][0]['test_loss'])
