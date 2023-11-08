'''
Control Utils
'''
# libraries
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Trainer
from sklearn.model_selection import train_test_split
import itertools

id_to_codon = {idx:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
codon_to_id = {v:k for k,v in id_to_codon.items()}

def longestZeroSeqLength(a):
    '''
    length of the longest sub-sequence of zeros
    '''
    a = a[1:-1].split(', ')
    a = [float(k) for k in a]
    # longest sequence of zeros
    longest = 0
    current = 0
    for i in a:
        if i == 0.0:
            current += 1
        else:
            longest = max(longest, current)
            current = 0
    return longest

def percNans(a):
    '''
    returns the percentage of nans in the sequence
    '''
    a = a[1:-1].split(', ')
    a = [float(k) for k in a]
    a = np.asarray(a)
    perc = np.count_nonzero(np.isnan(a)) / len(a)

    return perc

def RiboDatasetGWSDepr(df_depr_path: str, threshold: float = 0.6, longZerosThresh: int = 20, percNansThresh: float = 0.1, cond: str = 'LEU'):
    '''
    Dataset generation function
    '''
    df_full = pd.read_csv(df_depr_path)

    # drop first column
    df_full = df_full.drop(df_full.columns[0], axis=1)

    df_full.columns = ['gene', 'transcript', 'sequence', 'annotations', 'perc_non_zero_annots']
    # apply annot threshold
    df_full = df_full[df_full['perc_non_zero_annots'] >= threshold]

    # get longest zero sequence length for each sequence in annotations and ctrl_sequence
    annotations_list = list(df_full['annotations'])
    annotation_long_zeros = []
    num_nans_full = []
    for i in range(len(annotations_list)):
        annotation_long_zeros.append(longestZeroSeqLength(annotations_list[i]))
        num_nans_full.append(percNans(annotations_list[i]))

    # add the longest zero sequence length to the df
    df_full['longest_zero_seq_length_annotation'] = annotation_long_zeros

    # add the number of nans to the df
    df_full['perc_nans_annotation'] = num_nans_full

    # apply the threshold for the longest zero sequence length
    df_full = df_full[df_full['longest_zero_seq_length_annotation'] <= longZerosThresh]

    # apply the threshold for the number of nans
    df_full = df_full[df_full['perc_nans_annotation'] <= percNansThresh]

    # gene wise split
    genes = df_full['gene'].unique()
    genes_train, genes_test = train_test_split(genes, test_size=0.2, random_state=42)

    # split the dataframe
    df_train = df_full[df_full['gene'].isin(genes_train)]
    df_test = df_full[df_full['gene'].isin(genes_test)]

    # save the dataframes
    out_train_path = 'data/sh/ribo_train_' + str(cond) + '-NA_sh_' + str(threshold) + '_NZ_' + str(longZerosThresh) + '_PercNan_' + str(percNansThresh) + '.csv'
    out_test_path = 'data/sh/ribo_test_' + str(cond) + '-NA_sh_' + str(threshold) + '_NZ_' + str(longZerosThresh) + '_PercNan_' + str(percNansThresh) + '.csv'

    df_train.to_csv(out_train_path, index=False)
    df_test.to_csv(out_test_path, index=False)

    df_train = pd.read_csv(out_train_path)
    df_test = pd.read_csv(out_test_path)

    return df_train, df_test

class GWSDatasetFromPandas(Dataset):
    '''
    converts dataset from pandas dataframe to pytorch dataset
    '''
    def __init__(self, df):
        self.df = df
        self.counts = list(self.df['annotations'])
        self.sequences = list(self.df['sequence'])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        X = self.df['sequence'].iloc[idx]
        # convert to int
        X = X[1:-1].split(', ')
        X = [int(i) for i in X]

        y = self.df['annotations'].iloc[idx]
        # convert string into list of floats
        y = y[1:-1].split(', ')
        y = [float(i) for i in y]

        # min max scaling (Do not do this, this reduces the performance)
        y = [1+i for i in y]
        y = np.log(y)

        X = np.array(X)
        y = np.array(y)

        X = torch.from_numpy(X).long()
        y = torch.from_numpy(y).float()

        return X, y

class MaskedPearsonLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def __call__(self, y_pred, y_true, mask, eps=1e-6):
        y_pred_mask = torch.masked_select(y_pred, mask)
        y_true_mask = torch.masked_select(y_true, mask)
        cos = nn.CosineSimilarity(dim=0, eps=eps)
        return 1 - cos(
            y_pred_mask - y_pred_mask.mean(),
            y_true_mask - y_true_mask.mean(),
        )

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, y_pred, y_true, mask):
        y_pred_mask = torch.masked_select(y_pred, mask).float()
        y_true_mask = torch.masked_select(y_true, mask).float()

        loss = nn.functional.l1_loss(y_pred_mask, y_true_mask, reduction="none")
        return torch.sqrt(loss.mean())

class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels") # y true full
        outputs = model(**inputs)
        logits = outputs.logits
        logits = torch.squeeze(logits, dim=2)
        lengths = inputs['lengths']

        loss_fnc = MaskedPearsonLoss()
        
        mask = torch.arange(logits.shape[1])[None, :].to(lengths) < lengths[:, None]
        mask = torch.logical_and(mask, torch.logical_not(torch.isnan(labels)))

        loss = loss_fnc(logits, labels, mask)

        return (loss, outputs) if return_outputs else loss 