# libraries
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Trainer
from torchmetrics import Metric
import itertools
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.functional import pearson_corrcoef

id_to_codon = {idx:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
codon_to_id = {v:k for k,v in id_to_codon.items()}

class CorrCoef(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("corrcoefs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
    def update(self, preds, target, mask):
        preds = preds[:, 1:]
        assert preds.shape == target.shape
        assert preds.shape == mask.shape
        coeffs = []
        for p, t, m in zip(preds, target, mask):
            mp, mt = torch.masked_select(p, m), torch.masked_select(t, m)
            temp_pearson = pearson_corrcoef(mp, mt)
            coeffs.append(temp_pearson)
        coeffs = torch.stack(coeffs)
        self.corrcoefs += torch.sum(coeffs)
        self.total += len(coeffs)
    def compute(self):
        return self.corrcoefs / self.total

def collate_fn(batch):
    x, y, gene, transcript = zip(*batch)

    # sequence lenghts 
    lengths = torch.tensor([len(x) for x in x])
    x = pad_sequence(x, batch_first=True, padding_value=64) 
    y = pad_sequence(y, batch_first=True, padding_value=-1)

    out_batch = {}

    out_batch["input_ids"] = x
    out_batch["labels"] = y
    out_batch["lengths"] = lengths

    return out_batch

def compute_metrics(pred):
    labels = pred.label_ids 
    preds = pred.predictions
    mask = labels != -100.0
    labels = torch.tensor(labels)
    preds = torch.tensor(preds)
    preds = torch.squeeze(preds, dim=2)
    mask = torch.tensor(mask)
    mask = torch.logical_and(mask, torch.logical_not(torch.isnan(labels)))
    corr_coef = CorrCoef()
    corr_coef.update(preds, labels, mask)

    return {"r": corr_coef.compute()}

def slidingWindowZeroToNan(a, window_size=30):
    '''
    use a sliding window, if all the values in the window are 0, then replace them with nan
    '''
    a = np.asarray(a)
    for i in range(len(a) - window_size):
        if np.all(a[i:i+window_size] == 0.0):
            a[i:i+window_size] = np.nan

    return a

def RiboDatasetGWSDepr():
    '''
    Dataset generation function
    '''
    dataset_folder = '../../../data/orig/'

    df_train = pd.read_csv(dataset_folder + 'train.csv')
    df_val = pd.read_csv(dataset_folder + 'val.csv')
    df_test = pd.read_csv(dataset_folder + 'test.csv')

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