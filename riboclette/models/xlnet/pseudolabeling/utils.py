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

    np.save("preds/preds.npy", preds)
    np.save("preds/labels.npy", labels)
    np.save("preds/inputs.npy", inputs)

    return {"r": corr_coef.compute(), "mae": mae_coef.compute(), "mape": mape_coef.compute()}

# %%
# global variables
id_to_codon = {idx:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
codon_to_id = {v:k for k,v in id_to_codon.items()}

def checkArrayEquality(arr1, arr2):
    '''
    inputs: two arrays
    outputs: True if the arrays are equal, False otherwise
    '''
    if len(arr1) != len(arr2):
        return False
    
    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            return False
    
    return True

# dataset generation functions
def longestZeroSeqLength(a):
    '''
    length of the longest sub-sequence of zeros
    '''
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
    a = [float(k) for k in a]
    a = np.asarray(a)
    perc = np.count_nonzero(np.isnan(a)) / len(a)

    return perc

def coverageMod(a, window_size=30):
    '''
    returns the modified coverage function val in the sequence
    '''
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

    if den == 0:
        return 0
    
    return num / den
    
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

def pseudolabelExp1(ground_truth, mean_preds, stds_preds, threshold, impden):
    # process ground truth
    if ground_truth == 'NA':
        # make a list of np.nans
        ground_truth = [np.nan for j in range(len(mean_preds))]
    else:
        ground_truth = ground_truth[1:-1].split(', ')
        ground_truth = [float(k) for k in ground_truth]

    if np.mean(stds_preds) <= threshold:
        if impden == 'impute': # only imputation
            annot = []
            for j in range(len(mean_preds)):
                if (np.isnan(ground_truth[j]) or ground_truth[j] == 0.0):
                    annot.append(np.abs(mean_preds[j]))
                else:
                    annot.append(ground_truth[j])
            
            return annot
        elif impden == 'impden': # impute and denoise so you can change everything
            annot = []
            for j in range(len(mean_preds)):
                annot.append(np.abs(mean_preds[j]))

            return annot
    else:
        return ground_truth
    
def pseudolabelExp2_Preds(mean_preds, stds_preds, threshold):
    annot = [np.nan for j in range(len(mean_preds))]
    # print(np.mean(stds_preds), threshold)
    if np.mean(stds_preds) <= threshold:
        for j in range(len(mean_preds)):
            annot[j] = np.abs(mean_preds[j])

    return annot

def ntseqtoCodonSeq(seq):
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

    return codon_seq

def RiboDatasetExp1(threshold: float = 0.6, longZerosThresh: int = 20, percNansThresh: float = 0.1, plabel_thresh: float = 0.5, plabel_quartile: float = 0.25, impden: str = 'impute'):
    # load training and testing original sets
    out_train_path = 'data_dh/train_' + str(threshold) + '_NZ_' + str(longZerosThresh) + '_PercNan_' + str(percNansThresh) + '_Exp1PL-' + str(impden) +  '_' + str(plabel_quartile) + '.csv'
    df_test_orig = pd.read_csv('data_dh/orig/test_0.3_NZ_20_PercNan_0.05.csv')
    
    # check if the file exists
    if os.path.exists(out_train_path):
        exp1_preds = pd.read_csv(out_train_path)
        return exp1_preds, df_test_orig
    
    orig_test_genes = list(set(list(df_test_orig['gene'])))
    orig_test_transcripts = list(set(list(df_test_orig['transcript'])))

    # predictions paths
    exp1_preds_path = 'data_preds/set1_preds.pkl'
    # load preds files
    exp1_preds = pd.read_pickle(exp1_preds_path)

    # remove those for test genes and transcripts
    exp1_preds = exp1_preds[~exp1_preds['gene'].isin(orig_test_genes)]
    exp1_preds = exp1_preds[~exp1_preds['transcript'].isin(orig_test_transcripts)]
    
    annots_imputed = []

    # go through each of the samples in the df_full 
    # and impute the predictions
    if impden != 'same':
        for i in tqdm(range(len(exp1_preds))):
            # get the condition
            ground_truth_sample = exp1_preds['annotations'].iloc[i]
            mean_preds_sample = exp1_preds['mean_preds'].iloc[i]
            stds_preds_sample = exp1_preds['stds_preds'].iloc[i]

            pred_sample = pseudolabelExp1(ground_truth_sample, mean_preds_sample, stds_preds_sample, plabel_thresh, impden)

            annots_imputed.append(pred_sample)
        
        exp1_preds['annotations'] = annots_imputed

        # drop those that have NA in the annotations
        exp1_preds = exp1_preds[exp1_preds['annotations'] != 'NA']

        # coverage threshold
        exp1_preds['coverage_mod'] = exp1_preds['annotations'].apply(coverageMod)
        exp1_preds = exp1_preds[exp1_preds['coverage_mod'] >= threshold]

        # add the longest zero sequence length to the df
        exp1_preds['longest_zero_seq_length_annotation'] = exp1_preds['annotations'].apply(longestZeroSeqLength)
        # add the number of nans to the df
        exp1_preds['perc_nans_annotation'] = exp1_preds['annotations'].apply(percNans)

        # apply the threshold for the longest zero sequence length
        exp1_preds = exp1_preds[exp1_preds['longest_zero_seq_length_annotation'] <= longZerosThresh]

        # apply the threshold for the number of nans
        exp1_preds = exp1_preds[exp1_preds['perc_nans_annotation'] <= percNansThresh]

        print("Added Thresholds on all the factors")

        # for all the sequences in a condition that is not CTRL, add their respective CTRL sequence to them
        sequences_ctrl = []
        annotations_list = list(exp1_preds['annotations'])
        condition_df_list = list(exp1_preds['condition'])
        transcripts_list = list(exp1_preds['transcript'])

        for i in tqdm(range(len(condition_df_list))):
            try:
                if condition_df_list[i] != 'CTRL':
                    # find the respective CTRL sequence for the transcript
                    ctrl_sequence = exp1_preds[(exp1_preds['transcript'] == transcripts_list[i]) & (exp1_preds['condition'] == 'CTRL')]['annotations'].iloc[0]
                    sequences_ctrl.append(ctrl_sequence)
                else:
                    sequences_ctrl.append(annotations_list[i])
            except:
                sequences_ctrl.append('NA')

        # add the sequences_ctrl to the df
        print(len(sequences_ctrl), len(annotations_list))
        exp1_preds['ctrl_sequence'] = sequences_ctrl

        # remove those rows where the ctrl_sequence is NA
        exp1_preds = exp1_preds[exp1_preds['ctrl_sequence'] != 'NA']

        # sanity check for the ctrl sequences
        # get the ds with only condition as CTRL
        df_ctrl_full = exp1_preds[exp1_preds['condition'] == 'CTRL']
        ctrl_sequences_san = list(df_ctrl_full['annotations'])
        ctrl_sequences_san2 = list(df_ctrl_full['ctrl_sequence'])

        for i in range(len(ctrl_sequences_san)):
            assert ctrl_sequences_san[i] == ctrl_sequences_san2[i]

        print("Sanity Checked")
        # this part is messing up the values, save this and see

    exp1_preds.to_csv(out_train_path, index=False)

    exp1_preds = pd.read_csv(out_train_path)

    return exp1_preds, df_test_orig

def RiboDatasetExp2(threshold: float = 0.6, longZerosThresh: int = 20, percNansThresh: float = 0.1, plabel_thresh1: float = 0.5, plabel_quartile1: float = 0.25, impden: str = 'impute',  plabel_thresh2: float = 0.5, plabel_quartile2: float = 0.25):
    out_train_path = 'data_dh/train_' + str(threshold) + '_NZ_' + str(longZerosThresh) + '_PercNan_' + str(percNansThresh) + '_Exp2_PLQ1-' + str(impden) +  '_' + str(plabel_quartile1) + '_PLQ2_' + str(plabel_quartile2) + '.csv'
    df_test_orig = pd.read_csv('data_dh/orig/test_0.3_NZ_20_PercNan_0.05.csv')
    df_train_orig = pd.read_csv('data_dh/orig/train_0.3_NZ_20_PercNan_0.05.csv')

    # check if the file exists
    if os.path.exists(out_train_path):
        exp2_preds = pd.read_csv(out_train_path)
        return exp2_preds, df_test_orig

    orig_test_genes = list(set(list(df_test_orig['gene'])))
    orig_test_transcripts = list(set(list(df_test_orig['transcript'])))

    # predictions paths
    exp2_preds_path = 'data_preds/set2_preds.pkl'
    # load preds files
    exp2_preds = pd.read_pickle(exp2_preds_path)

    exp1_preds_path = 'data_preds/set1_preds.pkl'
    exp1_preds = pd.read_pickle(exp1_preds_path)

    # remove those for test genes and transcripts
    exp2_preds = exp2_preds[~exp2_preds['gene'].isin(orig_test_genes)]
    exp2_preds = exp2_preds[~exp2_preds['transcript'].isin(orig_test_transcripts)]

    if impden != 'same':
        # impute set2 preds
        annots_imputed = []
        for i in tqdm(range(len(exp2_preds))):
            # get the condition
            mean_preds_sample = exp2_preds['mean_preds'].iloc[i]
            stds_preds_sample = exp2_preds['stds_preds'].iloc[i]

            pred_sample = pseudolabelExp2_Preds(mean_preds_sample, stds_preds_sample, plabel_thresh2)

            annots_imputed.append(pred_sample)
        
        exp2_preds['annotations'] = annots_imputed

        # drop mean_preds and stds_preds
        exp2_preds = exp2_preds.drop(columns=['mean_preds', 'stds_preds'])

        # coverage threshold
        exp2_preds['coverage_mod'] = exp2_preds['annotations'].apply(coverageMod)

        # add the longest zero sequence length to the df
        exp2_preds['longest_zero_seq_length_annotation'] = exp2_preds['annotations'].apply(longestZeroSeqLength)
        # add the number of nans to the df
        exp2_preds['perc_nans_annotation'] = exp2_preds['annotations'].apply(percNans)

        # apply the threshold for the longest zero sequence length
        exp2_preds = exp2_preds[exp2_preds['longest_zero_seq_length_annotation'] <= longZerosThresh]
        exp2_preds = exp2_preds[exp2_preds['coverage_mod'] >= threshold]
        # apply the threshold for the number of nans
        exp2_preds = exp2_preds[exp2_preds['perc_nans_annotation'] <= percNansThresh]

        print("Added Thresholds on all the factors")

        # remove conditions columns from exp2_preds
        exp2_preds = exp2_preds.drop(columns=['coverage_mod', 'longest_zero_seq_length_annotation', 'perc_nans_annotation'])

        # impute df_train_orig
        annots_imputed = []
        for i in tqdm(range(len(df_train_orig))):
            # get the mean_preds, and stds_preds using condition and transcript from the df_train_orig, use set1_preds
            transcript = df_train_orig['transcript'].iloc[i]
            condition = df_train_orig['condition'].iloc[i]

            mean_preds_sample = exp1_preds[(exp1_preds['transcript'] == transcript) & (exp1_preds['condition'] == condition)]['mean_preds'].iloc[0]
            stds_preds_sample = exp1_preds[(exp1_preds['transcript'] == transcript) & (exp1_preds['condition'] == condition)]['stds_preds'].iloc[0]
            ground_truth = df_train_orig['annotations'].iloc[i]

            pred_sample = pseudolabelExp1(ground_truth, mean_preds_sample, stds_preds_sample, plabel_thresh1, impden)

            annots_imputed.append(pred_sample)

        df_train_orig['annotations'] = annots_imputed

        # merge the two dataframes
        # only keep gene, transcript, sequence, condition, annotations
        df_train_orig = df_train_orig[['gene', 'transcript', 'sequence', 'condition', 'annotations']]

        exp2_preds = pd.concat([exp2_preds, df_train_orig])

        # for all the sequences in a condition that is not CTRL, add their respective CTRL sequence to them
        sequences_ctrl = []
        annotations_list = list(exp2_preds['annotations'])
        condition_df_list = list(exp2_preds['condition'])
        transcripts_list = list(exp2_preds['transcript'])

        for i in tqdm(range(len(condition_df_list))):
            try:
                if condition_df_list[i] != 'CTRL':
                    # find the respective CTRL sequence for the transcript
                    ctrl_sequence = exp2_preds[(exp2_preds['transcript'] == transcripts_list[i]) & (exp2_preds['condition'] == 'CTRL')]['annotations'].iloc[0]
                    sequences_ctrl.append(ctrl_sequence)
                else:
                    sequences_ctrl.append(annotations_list[i])
            except:
                sequences_ctrl.append('NA')

        # add the sequences_ctrl to the df
        print(len(sequences_ctrl), len(annotations_list))
        exp2_preds['ctrl_sequence'] = sequences_ctrl

        # remove those rows where the ctrl_sequence is NA
        exp2_preds = exp2_preds[exp2_preds['ctrl_sequence'] != 'NA']

        # sanity check for the ctrl sequences
        # get the ds with only condition as CTRL
        df_ctrl_full = exp2_preds[exp2_preds['condition'] == 'CTRL']
        ctrl_sequences_san = list(df_ctrl_full['annotations'])
        ctrl_sequences_san2 = list(df_ctrl_full['ctrl_sequence'])

        for i in range(len(ctrl_sequences_san)):
            assert ctrl_sequences_san[i] == ctrl_sequences_san2[i]

        print("Sanity Checked")
        # this part is messing up the values, save this and see
    exp2_preds.to_csv(out_train_path, index=False)

    exp2_preds = pd.read_csv(out_train_path)

    return exp2_preds, df_test_orig

def RiboDatasetExp1_2(threshold: float = 0.6, longZerosThresh: int = 20, percNansThresh: float = 0.1, plabel_thresh1: float = 0.5, plabel_quartile1: float = 0.25, impden: str = 'impute',  plabel_thresh2: float = 0.5, plabel_quartile2: float = 0.25):
    out_train_path = 'data_dh/train_' + str(threshold) + '_NZ_' + str(longZerosThresh) + '_PercNan_' + str(percNansThresh) + '_Exp1-2_PLQ1-' + str(impden) +  '_' + str(plabel_quartile1) + '_PLQ2_' + str(plabel_quartile2) + '.csv'
    df_test_orig = pd.read_csv('data_dh/orig/test_0.3_NZ_20_PercNan_0.05.csv')
    df_train_orig = pd.read_csv('data_dh/orig/train_0.3_NZ_20_PercNan_0.05.csv')

    # check if the file exists
    if os.path.exists(out_train_path):
        exp2_preds = pd.read_csv(out_train_path)
        return exp2_preds, df_test_orig

    orig_test_genes = list(set(list(df_test_orig['gene'])))
    orig_test_transcripts = list(set(list(df_test_orig['transcript'])))

    # predictions paths
    exp2_preds_path = 'data_preds/set2_preds.pkl'
    # load preds files
    exp2_preds = pd.read_pickle(exp2_preds_path)

    # remove those for test genes and transcripts
    exp2_preds = exp2_preds[~exp2_preds['gene'].isin(orig_test_genes)]
    exp2_preds = exp2_preds[~exp2_preds['transcript'].isin(orig_test_transcripts)]

    if impden != 'same':
        # impute set2 preds
        annots_imputed = []
        for i in tqdm(range(len(exp2_preds))):
            # get the condition
            mean_preds_sample = exp2_preds['mean_preds'].iloc[i]
            stds_preds_sample = exp2_preds['stds_preds'].iloc[i]

            pred_sample = pseudolabelExp2_Preds(mean_preds_sample, stds_preds_sample, plabel_thresh2)

            annots_imputed.append(pred_sample)
        
        exp2_preds['annotations'] = annots_imputed

        # drop mean_preds and stds_preds
        exp2_preds = exp2_preds.drop(columns=['mean_preds', 'stds_preds'])

        # coverage threshold
        exp2_preds['coverage_mod'] = exp2_preds['annotations'].apply(coverageMod)

        # add the longest zero sequence length to the df
        exp2_preds['longest_zero_seq_length_annotation'] = exp2_preds['annotations'].apply(longestZeroSeqLength)
        # add the number of nans to the df
        exp2_preds['perc_nans_annotation'] = exp2_preds['annotations'].apply(percNans)

        # apply the threshold for the longest zero sequence length
        exp2_preds = exp2_preds[exp2_preds['longest_zero_seq_length_annotation'] <= longZerosThresh]
        exp2_preds = exp2_preds[exp2_preds['coverage_mod'] >= threshold]
        # apply the threshold for the number of nans
        exp2_preds = exp2_preds[exp2_preds['perc_nans_annotation'] <= percNansThresh]

        print("Added Thresholds on all the factors")

        # remove conditions columns from exp2_preds
        exp2_preds = exp2_preds.drop(columns=['coverage_mod', 'longest_zero_seq_length_annotation', 'perc_nans_annotation'])

        # for all the sequences in a condition that is not CTRL, add their respective CTRL sequence to them
        sequences_ctrl = []
        annotations_list = list(exp2_preds['annotations'])
        condition_df_list = list(exp2_preds['condition'])
        transcripts_list = list(exp2_preds['transcript'])

        for i in tqdm(range(len(condition_df_list))):
            try:
                if condition_df_list[i] != 'CTRL':
                    # find the respective CTRL sequence for the transcript
                    ctrl_sequence = exp2_preds[(exp2_preds['transcript'] == transcripts_list[i]) & (exp2_preds['condition'] == 'CTRL')]['annotations'].iloc[0]
                    sequences_ctrl.append(ctrl_sequence)
                else:
                    sequences_ctrl.append(annotations_list[i])
            except:
                sequences_ctrl.append('NA')

        # add the sequences_ctrl to the df
        print(len(sequences_ctrl), len(annotations_list))
        exp2_preds['ctrl_sequence'] = sequences_ctrl

        # remove those rows where the ctrl_sequence is NA
        exp2_preds = exp2_preds[exp2_preds['ctrl_sequence'] != 'NA']

        # sanity check for the ctrl sequences
        # get the ds with only condition as CTRL
        df_ctrl_full = exp2_preds[exp2_preds['condition'] == 'CTRL']
        ctrl_sequences_san = list(df_ctrl_full['annotations'])
        ctrl_sequences_san2 = list(df_ctrl_full['ctrl_sequence'])

        for i in range(len(ctrl_sequences_san)):
            assert ctrl_sequences_san[i] == ctrl_sequences_san2[i]

        print("Sanity Checked")

    # load in experiment 1 file for this 
    exp_1_train_path = 'data_dh/train_' + str(threshold) + '_NZ_' + str(longZerosThresh) + '_PercNan_' + str(percNansThresh) + '_Exp1PL-' + str(impden) +  '_' + str(plabel_quartile1) + '.csv'

    exp1_train = pd.read_csv(exp_1_train_path)

    exp1_train = exp1_train[['gene', 'transcript', 'sequence', 'condition', 'annotations', 'ctrl_sequence']]
    exp2_preds = exp2_preds[['gene', 'transcript', 'sequence', 'condition', 'annotations', 'ctrl_sequence']]

    exp1_2_train = pd.concat([exp1_train, exp2_preds])

    exp1_2_train.to_csv(out_train_path, index=False)

    exp1_2_train = pd.read_csv(out_train_path)

    return exp1_2_train, df_test_orig

class GWSDatasetFromPandas(Dataset):
    def __init__(self, df, split, noise_flag):
        self.df = df
        self.counts = list(self.df['annotations'])
        self.sequences = list(self.df['sequence'])
        self.condition_lists = list(self.df['condition'])
        self.condition_values = {'CTRL': 64, 'ILE': 65, 'LEU': 66, 'LEU_ILE': 67, 'LEU_ILE_VAL': 68, 'VAL': 69}
        self.split = split
        self.noise_flag = noise_flag

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

        # add noise for the training set
        if self.split == 'train' and self.noise_flag == True:
            noise = torch.normal(0, 0.2, size=y.shape).to(y.device)
            y = y + noise
            ctrl_y = ctrl_y + noise

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

class MaskedNormMAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, y_pred, y_true, mask):
        y_pred_mask = torch.masked_select(y_pred, mask).float()
        y_true_mask = torch.masked_select(y_true, mask).float()

        loss = nn.functional.l1_loss(y_pred_mask, y_true_mask, reduction="none") 
        # get mean y true without nans
        # convert y_true_mask to numpy
        y_true_mask = y_true_mask.cpu().numpy()
        y_true_max = np.nanmax(y_true_mask)

        if y_true_max == 0:
            return torch.sqrt(loss.mean())
        else:
            return torch.sqrt(loss.mean()) / y_true_max

class MaskedCombinedFiveDH(nn.Module):
    def __init__(self):
        super().__init__()
        self.pearson = MaskedPearsonLoss()
        self.l1 = MaskedL1Loss()
    
    def __call__(self, y_pred, labels, labels_ctrl, mask_full, mask_ctrl):
        # remove the first output cause that corresponds to the condition token
        y_pred_ctrl = y_pred[:, :, 0]
        y_pred_depr_diff = y_pred[:, :, 1]
        y_pred_full = torch.sum(y_pred, dim=2)

        labels_diff = labels - labels_ctrl

        # combine masks to make mask diff 
        mask_diff = mask_full & mask_ctrl

        loss_ctrl = self.pearson(y_pred_ctrl, labels_ctrl, mask_ctrl) + self.l1(y_pred_ctrl, labels_ctrl, mask_ctrl)
        loss_depr_diff = self.l1(y_pred_depr_diff, labels_diff, mask_diff)
        loss_full = self.pearson(y_pred_full, labels, mask_full) + self.l1(y_pred_full, labels, mask_full)

        return loss_ctrl + loss_depr_diff + loss_full

# %%
# custom Five regression trainer
class RegressionTrainerFive(Trainer):
    def __init__(self, **kwargs,):
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        labels_ctrl = inputs.pop("labels_ctrl")
        outputs = model(**inputs)
        logits = outputs.logits
        logits = torch.squeeze(logits, dim=2)
        # remove the first output cause that corresponds to the condition token
        logits = logits[:, 1:, :]
        lengths = inputs['lengths']

        loss_fnc = MaskedCombinedFiveDH()
        
        mask_full = torch.arange(labels.shape[1])[None, :].to(lengths) < lengths[:, None]
        mask_full = torch.logical_and(mask_full, torch.logical_not(torch.isnan(labels)))

        mask_ctrl = torch.arange(labels_ctrl.shape[1])[None, :].to(lengths) < lengths[:, None]
        mask_ctrl = torch.logical_and(mask_ctrl, torch.logical_not(torch.isnan(labels_ctrl)))
        
        loss = loss_fnc(logits, labels, labels_ctrl, mask_full, mask_ctrl)

        return (loss, outputs) if return_outputs else loss 


