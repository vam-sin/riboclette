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

# dataset generation functions
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
    longest = max(longest, current)
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

def slidingWindowZeroToNan(a, window_size=30):
    '''
    use a sliding window, if all the values in the window are 0, then replace them with nan
    '''
    a = np.asarray(a)
    for i in range(len(a) - window_size):
        if np.all(a[i:i+window_size] == 0.0):
            a[i:i+window_size] = np.nan

    return a

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

def sequenceLength(a):
    '''
    returns the length of the sequence
    '''
    a = a[1:-1].split(', ')
    a = [float(k) for k in a]
    return len(a)

def uniqueGenes(df):
    # add sequence length column
    df['sequence_length'] = df['annotations'].apply(sequenceLength)
    # keep only longest transcript for each gene
    df = df.sort_values('sequence_length', ascending=False).drop_duplicates('gene').sort_index()
    # drop sequence length column
    df = df.drop(columns=['sequence_length'])

    assert len(df['gene'].unique()) == len(df['gene'])

    return df

def RiboDatasetGWSDepr(threshold: float = 0.6, longZerosThresh: int = 20, percNansThresh: float = 0.1):
    '''
    Dataset generation function
    '''
    # save the dataframes
    out_train_path = '../../data/orig/train_' + str(threshold) + '_NZ_' + str(longZerosThresh) + '_PercNan_' + str(percNansThresh) + '.csv'
    out_test_path = '../../data/orig/test_' + str(threshold) + '_NZ_' + str(longZerosThresh) + '_PercNan_' + str(percNansThresh) + '.csv'
    out_val_path = '../../data/orig/val_' + str(threshold) + '_NZ_' + str(longZerosThresh) + '_PercNan_' + str(percNansThresh) + '.csv'

    # out_train_path = 'data/orig/train_' + str(threshold) + '_NZ_' + str(longZerosThresh) + '_PercNan_' + str(percNansThresh) + '.csv'
    # out_test_path = 'data/orig/test_' + str(threshold) + '_NZ_' + str(longZerosThresh) + '_PercNan_' + str(percNansThresh) + '.csv'
    # out_val_path = 'data/orig/val_' + str(threshold) + '_NZ_' + str(longZerosThresh) + '_PercNan_' + str(percNansThresh) + '.csv'

    df_train = pd.read_csv(out_train_path)
    df_test = pd.read_csv(out_test_path)
    df_val = pd.read_csv(out_val_path)

    return df_train, df_val, df_test

class GWSDatasetFromPandas(Dataset):
    def __init__(self, df):
        self.df = df
        self.counts = list(self.df['annotations'])
        self.sequences = list(self.df['sequence'])
        self.condition_lists = list(self.df['condition'])
        self.condition_values = {'CTRL': 64, 'ILE': 65, 'LEU': 66, 'LEU_ILE': 67, 'LEU_ILE_VAL': 68, 'VAL': 69}

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

        y = slidingWindowZeroToNan(y)

        y = [1+i for i in y]
        y = np.log(y)

        X = np.array(X)
        # multiply X with condition value times 64 + 1
        cond_token = self.condition_values[self.condition_lists[idx]]
        
        # prepend the condition token to X
        X = np.insert(X, 0, cond_token)

        y = np.array(y)

        X = torch.from_numpy(X).long()
        y = torch.from_numpy(y).float()

        gene = self.df['gene'].iloc[idx]
        transcript = self.df['transcript'].iloc[idx]

        return X, y, gene, transcript

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

class MaskedPCCL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = MaskedL1Loss()
        self.pcc_loss = MaskedPearsonLoss()

    def __call__(self, y_pred, y_true, mask):
        '''
        loss is the sum of the l1 loss and the pearson correlation coefficient loss
        '''

        l1 = self.l1_loss(y_pred, y_true, mask)
        pcc = self.pcc_loss(y_pred, y_true, mask)

        return l1 + pcc

class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels") # y true full
        outputs = model(**inputs)
        logits = outputs.logits
        logits = torch.squeeze(logits, dim=2)
        logits = logits[:, 1:]
        lengths = inputs['lengths']

        loss_fnc = MaskedPCCL1Loss()
        
        mask = torch.arange(logits.shape[1])[None, :].to(lengths) < lengths[:, None]
        mask = torch.logical_and(mask, torch.logical_not(torch.isnan(labels)))

        loss = loss_fnc(logits, labels, mask)

        return (loss, outputs) if return_outputs else loss 