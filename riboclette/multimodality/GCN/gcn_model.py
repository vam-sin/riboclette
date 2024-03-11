'''
model trained on only one dataset samples
'''

# libraries
import numpy as np
import pandas as pd 
import torch
from utils import trainGCN, FileRiboDataset # custom dataset and trainer
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

edge_attr = 'None' # 'Yes' (default, will send the 2 features), 'None' (will make then None), 'Zero' (will convert them to zeros)

loss_fn = 'MAE + PCC'
features = ['cbert_full', 'codon_ss', 'pos_enc']

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='DirSeq+', help='condition to train on') # USeq, USeq+, DirSeq, DirSeq+
args = parser.parse_args()
model_type = args.model_type # USeq, USeq+, DirSeq, DirSeq+

if model_type == 'USeq':
    proc_data_folder = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/mm/USeq/' # USeq, USeqPlus, DirSeq, DirSeqPlus
elif model_type == 'USeq+':
    proc_data_folder = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/mm/USeqPlus/' # USeq, USeqPlus, DirSeq, DirSeqPlus
elif model_type == 'DirSeq':
    proc_data_folder = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/mm/DirSeq/' # USeq, USeqPlus, DirSeq, DirSeqPlus
elif model_type == 'DirSeq+':
    proc_data_folder = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/mm/DirSeqPlus/' # USeq, USeqPlus, DirSeq, DirSeqPlus

algo = 'TF' # SAGE, GAT, GATv2, GINE, TF
features_str = '_'.join(features)
model_name = 'Noisy' + model_type + '-' + algo + ' EA: ' + str(edge_attr) + ' DS: Liver' + '[' + str(annot_thresh) + ', ' + str(longZerosThresh_val) + ', ' + str(percNansThresh_val) + ', BS ' + str(batch_size) + ', D ' + str(dropout_val) + ' E ' + str(tot_epochs) + ' LR ' + str(lr) + '] F: ' + features_str + ' VN RW 32 -1 + GraphNorm ' + loss_fn

input_nums_dict = {'cbert_full': 768, 'codon_ss': 0, 'pos_enc': 32}
num_inp_ft = sum([input_nums_dict[ft] for ft in features])

# start a new wandb run to track this script
wandb_logger = WandbLogger(log_model="all", project="GCN_MM", name=model_name)

gcn_layers = [256, 128, 128, 64]

# model parameters
save_loc = 'saved_models/' + model_name

# make torch datasets from pandas dataframes
transforms = T.Compose([T.AddRandomWalkPE(walk_length=random_walk_length), T.VirtualNode()])

train_ds = FileRiboDataset(proc_data_folder, 'train', edge_attr, shuffle=True)
test_ds = FileRiboDataset(proc_data_folder, 'test', edge_attr, shuffle=False)

print("samples in train dataset: ", len(train_ds))
print("samples in test dataset: ", len(test_ds))

# train model
model, result = trainGCN(gcn_layers, tot_epochs, batch_size, lr, save_loc, wandb_logger, train_ds, test_ds, dropout_val, num_inp_ft, model_type, algo, edge_attr)

print(1.0 - result['test'][0]['test_loss'])
