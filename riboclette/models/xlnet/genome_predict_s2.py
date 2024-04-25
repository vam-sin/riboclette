# libraries
import numpy as np
import pandas as pd 
import torch
from transformers import XLNetConfig, XLNetForTokenClassification, TrainingArguments
import random
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.functional import pearson_corrcoef
from torchmetrics import Metric
from Bio import SeqIO
import os
import seaborn as sns
import shutil
from ipynb.fs.full.utils_dh import RegressionTrainerFive, RiboDatasetGWS, GWSDatasetFromPandas, collate_fn, compute_metrics, compute_metrics_saved  # custom dataset and trainer, CorrCoef, collate_fn, compute_metrics, compute_metrics_saved  # custom dataset and trainer
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import itertools
from tqdm import tqdm

# reproducibility
pl.seed_everything(42)

# conditions
conditions_list = ['CTRL', 'LEU', 'ILE', 'VAL', 'LEU_ILE', 'LEU_ILE_VAL']
condition_values = {'CTRL': 64, 'ILE': 65, 'LEU': 66, 'LEU_ILE': 67, 'LEU_ILE_VAL': 68, 'VAL': 69}
id_to_codon = {idx:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
codon_to_id = {v:k for k,v in id_to_codon.items()}

def coverageMod(a, window_size=30):
    '''
    returns the modified coverage function val in the sequence
    '''
    a = a[1:-1].split(', ')
    a = [float(k) for k in a]
    for i in range(len(a) - window_size):
        if np.all(a[i:i+window_size] == 0.0):
            a[i:i+window_size] = np.nan

    # num non zero, non nan
    num = 0
    den = 0
    for i in a:
        if i != 0.0 and not np.isnan(i):
            num += 1
        if not np.isnan(i):
            den += 1
    
    return num / den

def ntseqtoCodonSeq(seq, condition, add_cond=True):
    """
    Convert nucleotide sequence to codon sequence
    """
    codon_seq = []
    # cut seq to remove last codon if not complete
    for i in range(0, len(seq), 3):
        # check if codon is complete
        if len(seq[i:i+3]) == 3:
            codon_seq.append(seq[i:i+3])

    codon_seq = [codon_to_id[codon] for codon in codon_seq]

    if add_cond:
        # prepend condition token
        codon_seq = [condition_values[condition]] + codon_seq

    return codon_seq

# model parameters
annot_thresh = 0.3
longZerosThresh_val = 20
percNansThresh_val = 0.05
d_model_val = 256
n_layers_val = 3
n_heads_val = 8
dropout_val = 0.1
lr_val = 1e-4
batch_size_val = 1
loss_fun_name = '5L' # 5L

# model name and output folder path
model_name1 = 'saved_models/XLNetDHConds DS: DeprNA [' + str(n_layers_val) + ', ' + str(d_model_val) + ', ' + str(n_heads_val) + '] FT: [PEL] BS: ' + str(batch_size_val) + ' Loss: ' + str(loss_fun_name) + ' Data Conds: [NZ: ' + str(longZerosThresh_val) + ', PNTh: ' + str(percNansThresh_val) + ', AnnotThresh: ' + str(annot_thresh) + '] Seed: 1'
model_name2 = 'saved_models/XLNetDHConds DS: DeprNA [' + str(n_layers_val) + ', ' + str(d_model_val) + ', ' + str(n_heads_val) + '] FT: [PEL] BS: ' + str(batch_size_val) + ' Loss: ' + str(loss_fun_name) + ' Data Conds: [NZ: ' + str(longZerosThresh_val) + ', PNTh: ' + str(percNansThresh_val) + ', AnnotThresh: ' + str(annot_thresh) + '] Seed: 2'
model_name3 = 'saved_models/XLNetDHConds DS: DeprNA [' + str(n_layers_val) + ', ' + str(d_model_val) + ', ' + str(n_heads_val) + '] FT: [PEL] BS: ' + str(batch_size_val) + ' Loss: ' + str(loss_fun_name) + ' Data Conds: [NZ: ' + str(longZerosThresh_val) + ', PNTh: ' + str(percNansThresh_val) + ', AnnotThresh: ' + str(annot_thresh) + '] Seed: 3'
model_name4 = 'saved_models/XLNetDHConds DS: DeprNA [' + str(n_layers_val) + ', ' + str(d_model_val) + ', ' + str(n_heads_val) + '] FT: [PEL] BS: ' + str(batch_size_val) + ' Loss: ' + str(loss_fun_name) + ' Data Conds: [NZ: ' + str(longZerosThresh_val) + ', PNTh: ' + str(percNansThresh_val) + ', AnnotThresh: ' + str(annot_thresh) + '] Seed: 4'
model_name42 = 'saved_models/XLNetDHConds DS: DeprNA [' + str(n_layers_val) + ', ' + str(d_model_val) + ', ' + str(n_heads_val) + '] FT: [PEL] BS: ' + str(batch_size_val) + ' Loss: ' + str(loss_fun_name) + ' Data Conds: [NZ: ' + str(longZerosThresh_val) + ', PNTh: ' + str(percNansThresh_val) + ', AnnotThresh: ' + str(annot_thresh) + '] Seed: 42'

class XLNetDH(XLNetForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = torch.nn.Linear(d_model_val, 2, bias=True)

config = XLNetConfig(vocab_size=71, pad_token_id=70, d_model = d_model_val, n_layer = n_layers_val, n_head = n_heads_val, d_inner = d_model_val, num_labels = 1, dropout=dropout_val) # 64*6 tokens + 1 for padding
model = XLNetDH(config)

# load model best weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# load model from the saved model
model1 = model.from_pretrained(model_name1 + "/best_model")
model2 = model.from_pretrained(model_name2 + "/best_model")
model3 = model.from_pretrained(model_name3 + "/best_model")
model4 = model.from_pretrained(model_name4 + "/best_model")
model42 = model.from_pretrained(model_name42 + "/best_model")

models_list = [model1, model2, model3, model4, model42]

for model_chosen in models_list:
    model_chosen.to(device)
    model_chosen.eval()

print("Loaded all the models")

# load in the set to predict
df_set = pd.read_csv("pseudolabeling/data_preds/df_set2.csv")

print("Loaded In set to predict")

# load in the full dataset
df_full = pd.read_csv("pseudolabeling/data_preds/tmp/df_full.csv")

print("Loaded DF Full with all the Lina Data")

final_mean_preds_list = []
final_stds_preds_list = []
final_sequence_list = []

# load genes file 
sequences_df = list(df_set['sequence'])
genes_df = list(df_set['gene'])
transcripts_df = list(df_set['transcript'])
conditions_df = list(df_set['condition'])

# make predictions on all the sequences, using the five models
for j in tqdm(range(len(sequences_df))):
    # process the sequence
    X_non_cond = ntseqtoCodonSeq(sequences_df[j], conditions_df[j], add_cond=False)
    final_sequence_list.append(X_non_cond)

    # process with condition for the model
    X = sequences_df[j]
    X = ntseqtoCodonSeq(X, conditions_df[j], add_cond=True)
    X = np.asarray(X)
    X = torch.from_numpy(X).long()

    preds_list_sample = []

    with torch.no_grad():
        for model_chosen in models_list:
            y_pred = model_chosen(X.unsqueeze(0).to(device).to(torch.int32))
            y_pred = torch.sum(y_pred["logits"], dim=2)
            y_pred = y_pred.squeeze(0)

            # remove first token which is condition token
            y_pred = y_pred[1:]

            preds_list_sample.append(y_pred.detach().cpu().numpy())

    # add preds_list_sample to genes_file 
    preds_list_sample = np.asarray(preds_list_sample)
    # take mean and std of the predictions over each codon
    mean_preds = np.mean(preds_list_sample, axis=0)
    stds_preds = np.std(preds_list_sample, axis=0)

    # print(mean_preds.shape, stds_preds.shape)

    final_mean_preds_list.append(mean_preds)
    final_stds_preds_list.append(stds_preds)

# create a dataframe with the final predictions
df_final_preds = pd.DataFrame({'gene': genes_df, 'transcript': transcripts_df, 'sequence': final_sequence_list, 'mean_preds': final_mean_preds_list, 'stds_preds': final_stds_preds_list, 'condition': conditions_df})
print(df_final_preds)
# save the dataframe
df_final_preds.to_pickle("pseudolabeling/data_preds/set2_preds.pkl")