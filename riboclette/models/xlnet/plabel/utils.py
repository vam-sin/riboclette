# %%
# libraries
import numpy as np
import pandas as pd 
import torch
import os
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.functional import pearson_corrcoef
from torchmetrics.regression import MeanAbsolutePercentageError, MeanAbsoluteError
from torchmetrics import Metric
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Trainer
import itertools
from tqdm import tqdm
from pyhere import here

# %%
# model functions
class MAECoef(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("mae", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
    def update(self, preds, target, mask):
        preds = torch.sum(preds, dim=2)
        preds = preds[:, 1:]
        assert preds.shape == target.shape
        assert preds.shape == mask.shape
        coeffs = []
        abs_error = MeanAbsoluteError()
        for p, t, m in zip(preds, target, mask):
            mp, mt = torch.masked_select(p, m), torch.masked_select(t, m)
            temp_mae = abs_error(mp, mt)
            coeffs.append(temp_mae)
        coeffs = torch.stack(coeffs)
        self.mae += torch.sum(coeffs)
        self.total += len(coeffs)
    def compute(self):
        return self.mae / self.total

class CorrCoef(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("corrcoefs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
    def update(self, preds, target, mask):
        preds = torch.sum(preds, dim=2)
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

class MAPECoef(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("mapecoefs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
    def update(self, preds, target, mask):
        preds = torch.sum(preds, dim=2)
        preds = preds[:, 1:]
        assert preds.shape == target.shape
        assert preds.shape == mask.shape
        coeffs = []
        perc_error = MeanAbsolutePercentageError()
        for p, t, m in zip(preds, target, mask):
            # remove first token in p
            mp, mt = torch.masked_select(p, m), torch.masked_select(t, m)
            temp_mape = perc_error(mp, mt)
            coeffs.append(temp_mape)
        coeffs = torch.stack(coeffs)
        self.mapecoefs += torch.sum(coeffs)
        self.total += len(coeffs)
    def compute(self):
        return self.mapecoefs / self.total

# collate function
def collate_fn(batch):
    # batch is a list of tuples (x, y)
    x, y, ctrl_y, gene, transcript = zip(*batch)

    # sequence lenghts 
    lengths = torch.tensor([len(x) for x in x])
    
    x = pad_sequence(x, batch_first=True, padding_value=384) 
    y = pad_sequence(y, batch_first=True, padding_value=-1)
    ctrl_y = pad_sequence(ctrl_y, batch_first=True, padding_value=-1)

    out_batch = {}

    out_batch["input_ids"] = x
    out_batch["labels"] = y
    out_batch["lengths"] = lengths
    out_batch["labels_ctrl"] = ctrl_y

    return out_batch

# compute metrics
def compute_metrics(pred):
    labels = pred.label_ids 
    preds = pred.predictions
    inputs = pred.inputs
    mask = labels != -100.0
    labels = torch.tensor(labels)
    preds = torch.tensor(preds)
    preds = torch.squeeze(preds, dim=2)
    
    mask = torch.tensor(mask)
    
    # mask = torch.arange(preds.shape[1])[None, :].to(lengths) < lengths[:, None]
    mask = torch.logical_and(mask, torch.logical_not(torch.isnan(labels)))

    corr_coef = CorrCoef()
    corr_coef.update(preds, labels, mask)

    mae_coef = MAECoef()
    mae_coef.update(preds, labels, mask)

    mape_coef = MAPECoef()
    mape_coef.update(preds, labels, mask)

    return {"r": corr_coef.compute(), "mae": mae_coef.compute(), "mape": mape_coef.compute()}

# compute metrics
def compute_metrics_saved(pred):
    '''
    additional function to just save everything to do analysis later
    '''
    labels = pred.label_ids 
    preds = pred.predictions
    inputs = pred.inputs
    mask = labels != -100.0
    labels = torch.tensor(labels)
    preds = torch.tensor(preds)
    preds = torch.squeeze(preds, dim=2)
    
    mask = torch.tensor(mask)
    
    # mask = torch.arange(preds.shape[1])[None, :].to(lengths) < lengths[:, None]
    mask = torch.logical_and(mask, torch.logical_not(torch.isnan(labels)))

    mae_coef = MAECoef()
    mae_coef.update(preds, labels, mask)

    corr_coef = CorrCoef()
    corr_coef.update(preds, labels, mask)

    mape_coef = MAPECoef()
    mape_coef.update(preds, labels, mask)

    # save predictions
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    # np.save("preds/preds.npy", preds)
    # np.save("preds/labels.npy", labels)
    # np.save("preds/inputs.npy", inputs)

    return {"r": corr_coef.compute(), "mae": mae_coef.compute(), "mape": mape_coef.compute()}

# %%
# global variables
id_to_codon = {idx:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
codon_to_id = {v:k for k,v in id_to_codon.items()}

def slidingWindowZeroToNan(a, window_size=30):
    '''
    use a sliding window, if all the values in the window are 0, then replace them with nan
    '''
    a = [float(k) for k in a]
    a = np.asarray(a)
    for i in range(len(a) - window_size):
        if np.all(a[i:i+window_size] == 0.0):
            a[i:i+window_size] = np.nan

    return a

def RiboDatasetPlabel():
    # load training and testing original sets
    train_path = here('data', 'plabel', 'plabel_train.csv')
    val_path = here('data', 'orig', 'val.csv')
    test_path = here('data', 'orig', 'test.csv')

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

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

        # ctrl sequence 
        ctrl_y = self.df['ctrl_sequence'].iloc[idx]
        # convert string into list of floats
        ctrl_y = ctrl_y[1:-1].split(', ')
        ctrl_y = [float(i) for i in ctrl_y]

        ctrl_y = slidingWindowZeroToNan(ctrl_y)

        # no min max scaling
        ctrl_y = [1+i for i in ctrl_y]
        ctrl_y = np.log(ctrl_y)

        X = np.array(X)
        # multiply X with condition value times 64 + 1
        cond_token = self.condition_values[self.condition_lists[idx]]
        
        # prepend the condition token to X
        X = np.insert(X, 0, cond_token)

        y = np.array(y)

        X = torch.from_numpy(X).long()
        y = torch.from_numpy(y).float()
        ctrl_y = torch.from_numpy(ctrl_y).float()

        gene = self.df['gene'].iloc[idx]
        transcript = self.df['transcript'].iloc[idx]

        return X, y, ctrl_y, gene, transcript

# %%
# loss functions
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

class MaskedCombinedFourDH(nn.Module):
    def __init__(self):
        super().__init__()
        self.pearson = MaskedPearsonLoss()
        self.l1 = MaskedL1Loss()
    
    def __call__(self, y_pred, labels, labels_ctrl, mask_full, mask_ctrl, condition_):
        # remove the first output cause that corresponds to the condition token
        # y_pred_ctrl = y_pred[:, :, 0]
        # relu on ctrl prediction
        y_pred_ctrl = torch.relu(y_pred[:, :, 0])
        
        y_pred_depr_diff = y_pred[:, :, 1]
        y_pred_full = torch.sum(y_pred, dim=2)

        labels_diff = labels - labels_ctrl

        # combine masks to make mask diff 
        mask_diff = mask_full & mask_ctrl

        loss_ctrl = self.pearson(y_pred_ctrl, labels_ctrl, mask_ctrl)
        if condition_ != 64:
            loss_depr_diff = self.pearson(y_pred_depr_diff, labels_diff, mask_diff)
        loss_full = self.pearson(y_pred_full, labels, mask_full) + self.l1(y_pred_full, labels, mask_full)

        if condition_ != 64:
            return loss_ctrl + loss_depr_diff + loss_full
        else:
            return loss_ctrl + loss_full 

# custom Four regression trainer
class RegressionTrainerFour(Trainer):
    def __init__(self, **kwargs,):
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        condition_ = inputs['input_ids'][0][0]
        labels_ctrl = inputs.pop("labels_ctrl")
        outputs = model(**inputs)
        logits = outputs.logits
        logits = torch.squeeze(logits, dim=2)
        # remove the first output cause that corresponds to the condition token
        logits = logits[:, 1:, :]
        lengths = inputs['lengths']

        loss_fnc = MaskedCombinedFourDH()
        
        mask_full = torch.arange(labels.shape[1])[None, :].to(lengths) < lengths[:, None]
        mask_full = torch.logical_and(mask_full, torch.logical_not(torch.isnan(labels)))

        mask_ctrl = torch.arange(labels_ctrl.shape[1])[None, :].to(lengths) < lengths[:, None]
        mask_ctrl = torch.logical_and(mask_ctrl, torch.logical_not(torch.isnan(labels_ctrl)))
        
        loss = loss_fnc(logits, labels, labels_ctrl, mask_full, mask_ctrl, condition_)

        return (loss, outputs) if return_outputs else loss 
