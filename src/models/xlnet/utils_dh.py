'''
double head model utils
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

def RiboDatasetGWS(data_folder: str, ds: str, threshold: float = 0.6, longZerosThresh: int = 20, percNansThresh: float = 0.1):
    if ds == 'ALL':
        # # # paths 
        # ctrl_path = data_folder + 'CTRL.csv'
        # leu_path = data_folder + 'LEU.csv'
        # arg_path = data_folder + 'ARG.csv'

        # # load the control data
        # df_ctrl = pd.read_csv(ctrl_path)
        # df_ctrl['condition'] = 'CTRL'

        # # other dataset
        # df_leu = pd.read_csv(leu_path)
        # df_leu['condition'] = 'LEU'
        # df_arg = pd.read_csv(arg_path)
        # df_arg['condition'] = 'ARG'

        # df_full = pd.concat([df_ctrl, df_leu, df_arg], axis=0) # ctrl, leu, arg

        # # drop first column
        # df_full = df_full.drop(['Unnamed: 0'], axis=1)

        # df_full.columns = ['gene', 'transcript', 'sequence', 'annotations', 'perc_non_zero_annots', 'condition']

        # # apply annot threshold
        # df_full = df_full[df_full['perc_non_zero_annots'] >= threshold]

        # # for all the sequences in a condition that is not CTRL, add their respective CTRL sequence to them
        # sequences_ctrl = []
        # annotations_list = list(df_full['annotations'])
        # condition_df_list = list(df_full['condition'])
        # transcripts_list = list(df_full['transcript'])

        # for i in range(len(condition_df_list)):
        #     try:
        #         if condition_df_list != 'CTRL':
        #             # find the respective CTRL sequence for the transcript
        #             ctrl_sequence = df_full[(df_full['transcript'] == transcripts_list[i]) & (df_full['condition'] == 'CTRL')]['annotations'].iloc[0]
        #             sequences_ctrl.append(ctrl_sequence)
        #         else:
        #             sequences_ctrl.append(annotations_list[i])
        #     except:
        #         sequences_ctrl.append('NA')

        # # add the sequences_ctrl to the df
        # print(len(sequences_ctrl), len(annotations_list))
        # df_full['ctrl_sequence'] = sequences_ctrl

        # # remove those rows where the ctrl_sequence is NA
        # df_full = df_full[df_full['ctrl_sequence'] != 'NA']

        # # sanity check for the ctrl sequences
        # # get the ds with only condition as CTRL
        # df_ctrl_full = df_full[df_full['condition'] == 'CTRL']
        # ctrl_sequences_san = list(df_ctrl_full['annotations'])
        # ctrl_sequences_san2 = list(df_ctrl_full['ctrl_sequence'])

        # for i in range(len(ctrl_sequences_san)):
        #     assert ctrl_sequences_san[i] == ctrl_sequences_san2[i]

        # print("Sanity Checked")

        # # get longest zero sequence length for each sequence in annotations and ctrl_sequence
        # annotations_list = list(df_full['annotations'])
        # sequences_ctrl = list(df_full['ctrl_sequence'])
        # annotation_long_zeros = []
        # ctrl_sequence_long_zeros = []
        # num_nans_full = []
        # num_nans_ctrl = []
        # for i in range(len(annotations_list)):
        #     annotation_long_zeros.append(longestZeroSeqLength(annotations_list[i]))
        #     ctrl_sequence_long_zeros.append(longestZeroSeqLength(sequences_ctrl[i]))
        #     num_nans_full.append(percNans(annotations_list[i]))
        #     num_nans_ctrl.append(percNans(sequences_ctrl[i]))

        # # add the longest zero sequence length to the df
        # df_full['longest_zero_seq_length_annotation'] = annotation_long_zeros
        # df_full['longest_zero_seq_length_ctrl_sequence'] = ctrl_sequence_long_zeros

        # # add the number of nans to the df
        # df_full['perc_nans_annotation'] = num_nans_full
        # df_full['perc_nans_ctrl_sequence'] = num_nans_ctrl

        # # apply the threshold for the longest zero sequence length
        # df_full = df_full[df_full['longest_zero_seq_length_annotation'] <= longZerosThresh]
        # df_full = df_full[df_full['longest_zero_seq_length_ctrl_sequence'] <= longZerosThresh]

        # # apply the threshold for the number of nans
        # df_full = df_full[df_full['perc_nans_annotation'] <= percNansThresh]
        # df_full = df_full[df_full['perc_nans_ctrl_sequence'] <= percNansThresh]

        # # GWS
        # genes = df_full['gene'].unique()
        # genes_train, genes_test = train_test_split(genes, test_size=0.2, random_state=42)

        # # split the dataframe
        # df_train = df_full[df_full['gene'].isin(genes_train)]
        # df_test = df_full[df_full['gene'].isin(genes_test)]

        out_train_path = 'data/ribo_train_ALL-NA_dh_' + str(threshold) + '_NZ_' + str(longZerosThresh) + '_PercNan_' + str(percNansThresh) + '.csv'
        out_test_path = 'data/ribo_test_ALL-NA_dh_' + str(threshold) + '_NZ_' + str(longZerosThresh) + '_PercNan_' + str(percNansThresh) + '.csv'

        # df_train.to_csv(out_train_path, index=False)
        # df_test.to_csv(out_test_path, index=False)

        df_train = pd.read_csv(out_train_path)
        df_test = pd.read_csv(out_test_path)

        return df_train, df_test

def sequence_to_OH(codon_sequence):
    oh_out = np.zeros((len(codon_sequence), 64))
    for i in range(len(codon_sequence)):
        oh_out[i][codon_sequence[i]] = 1

    return oh_out

class GWSDatasetFromPandas(Dataset):
    def __init__(self, df):
        self.df = df
        self.counts = list(self.df['annotations'])
        self.sequences = list(self.df['sequence'])
        self.condition_lists = list(self.df['condition'])
        self.condition_values = {'CTRL': 0, 'LEU': 1, 'ARG': 2}

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

        y = [1+i for i in y]
        y = np.log(y)

        # ctrl sequence 
        ctrl_y = self.df['ctrl_sequence'].iloc[idx]
        # convert string into list of floats
        ctrl_y = ctrl_y[1:-1].split(', ')
        ctrl_y = [float(i) for i in ctrl_y]

        # no min max scaling
        ctrl_y = [1+i for i in ctrl_y]
        ctrl_y = np.log(ctrl_y)

        X = np.array(X)
        # multiply X with condition value times 64 + 1
        add_factor = (self.condition_values[self.condition_lists[idx]]) * 64
        X += add_factor

        y = np.array(y)
        len_X = len(X)

        X = torch.from_numpy(X).long()
        y = torch.from_numpy(y).float()
        ctrl_y = torch.from_numpy(ctrl_y).float()

        gene = self.df['gene'].iloc[idx]
        transcript = self.df['transcript'].iloc[idx]

        return X, y, ctrl_y, gene, transcript

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

class MaskedPoissonLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, y_pred, y_true, mask):
        y_pred_mask = torch.masked_select(y_pred, mask)
        y_true_mask = torch.masked_select(y_true, mask)
        return nn.functional.poisson_nll_loss(y_pred_mask, y_true_mask, log_input=False)

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, y_pred, y_true, mask):
        y_pred_mask = torch.masked_select(y_pred, mask).float()
        y_true_mask = torch.masked_select(y_true, mask).float()

        loss = nn.functional.l1_loss(y_pred_mask, y_true_mask, reduction="none")
        return torch.sqrt(loss.mean())

class MaskedCombinedPearsonLoss(nn.Module):
    def __init__(self, comb_max_duration: int = 200):
        super().__init__()
        self.pearson = MaskedPearsonLoss()
        self.poisson = MaskedPoissonLoss()
        self.comb_max_duration = comb_max_duration

    def __call__(self, y_pred, y_true, mask, timestamp, eps=1e-6):
        poisson = self.poisson(y_pred, y_true, mask)
        pearson = self.pearson(y_pred, y_true, mask, eps=eps)

        return pearson + max(0, 1 - timestamp / self.comb_max_duration) * poisson

class MaskedCombinedDoubleHeadPearsonLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.pearson = MaskedPearsonLoss()
        self.l1 = MaskedL1Loss()
    
    def __call__(self, y_pred, labels, labels_ctrl, mask_full, mask_ctrl):
        y_pred_ctrl = y_pred[:, :, 0]
        y_pred_depr_diff = y_pred[:, :, 1]
        y_pred_full = torch.sum(y_pred, dim=2)
        labels_diff = labels - labels_ctrl
        # combine masks to make mask diff 
        mask_diff = mask_full & mask_ctrl

        loss_ctrl = self.pearson(y_pred_ctrl, labels_ctrl, mask_ctrl)
        loss_depr_diff = self.l1(y_pred_depr_diff, labels_diff, mask_diff)
        loss_full = self.pearson(y_pred_full, labels, mask_full)

        return loss_ctrl + loss_depr_diff + loss_full

class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # print(inputs)
        labels = inputs.pop("labels")
        labels_ctrl = inputs.pop("labels_ctrl")
        # print("inputs: ", inputs)
        outputs = model(**inputs)
        logits = outputs.logits
        logits = torch.squeeze(logits, dim=2)
        lengths = inputs['lengths']
        # loss_fnc = MaskedCombinedPearsonLoss()

        loss_fnc = MaskedCombinedDoubleHeadPearsonLoss()
        
        mask_full = torch.arange(labels.shape[1])[None, :].to(lengths) < lengths[:, None]
        mask_full = torch.logical_and(mask_full, torch.logical_not(torch.isnan(labels)))

        mask_ctrl = torch.arange(labels_ctrl.shape[1])[None, :].to(lengths) < lengths[:, None]
        mask_ctrl = torch.logical_and(mask_ctrl, torch.logical_not(torch.isnan(labels_ctrl)))

        # print(logits, labels, labels_ctrl, mask)
        # print("logits: ", logits)
        # print("labels: ", labels)
        # print("labels_ctrl: ", labels_ctrl)
        
        loss = loss_fnc(logits, labels, labels_ctrl, mask_full, mask_ctrl)
        # print("loss: ", loss)

        return (loss, outputs) if return_outputs else loss 