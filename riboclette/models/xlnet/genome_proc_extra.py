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

def sequenceLength(a):
    '''
    returns the length of the sequence
    '''
    a = a[1:-1].split(', ')
    a = [float(k) for k in a]
    return len(a)

def mergeAnnotations(annots):
    '''
    merge the annotations for the same gene
    '''
    # get the annotations
    annots = [a[1:-1].split(', ') for a in annots]
    annots = [[float(k) for k in a] for a in annots]

    # merge the annotations
    merged_annots = []
    for i in range(len(annots[0])):
        # get the ith annotation for all the transcripts, only non zero and non nan
        ith_annots = [a[i] for a in annots if a[i] != 0.0 and not np.isnan(a[i])]
        # take the mean of the ith annotation
        ith_mean = np.mean(ith_annots)
        merged_annots.append(ith_mean)

    return merged_annots

def uniqueGenes(df):
    # add sequence length column
    df['sequence_length'] = df['annotations'].apply(sequenceLength)

    unique_genes = list(df['gene'].unique())

    # iterate through each gene, and choose the longest transcript, for the annotation, merge the annotations
    for gene in unique_genes:
        # get the df for the gene
        df_gene = df[df['gene'] == gene]
        if len(df_gene) > 1:
            # get the transcript with the longest sequence
            df_gene = df_gene.sort_values('sequence_length', ascending=False)
            # chosen transcript
            chosen_transcript = df_gene['transcript'].values[0]
            other_transcripts = df_gene['transcript'].values[1:]
            # merge the annotations
            annotations = df_gene['annotations'].values
            merged_annotations = mergeAnnotations(annotations)
            # drop the other transcripts from the df
            df = df[~df['transcript'].isin(other_transcripts)]

            # change the annotations for the chosen transcript
            df.loc[df['transcript'] == chosen_transcript, 'annotations'] = str(merged_annotations)

    # drop sequence length column
    df = df.drop(columns=['sequence_length'])

    assert len(df['gene'].unique()) == len(df['gene'])
    assert len(df['transcript'].unique()) == len(df['transcript'])
    assert len(df['transcript']) == len(df['gene'])

    return df

def seqLenMouse(a):
    '''
    returns the length of the sequence
    '''
    return len(a)

def removeFullGenes(df_mouse, df_full):
    '''
    remove the genes that are already in df_full
    '''
    # gene transcript dict
    tr_unique_full = list(df_full['transcript'].unique())
    transcripts_full_sans_version = [tr.split('.')[0] for tr in tr_unique_full]

    df_mouse_tr_sans_version = [tr.split('.')[0] for tr in df_mouse['transcript']]
    df_mouse_genes = list(df_mouse['gene'])

    mouse_tg_dict = dict(zip(df_mouse_tr_sans_version, df_mouse_genes))

    # for each transcript in df_full, remove the gene from df_mouse
    for tran in transcripts_full_sans_version:
        mouse_gene_for_full_transcript = mouse_tg_dict[tran]
        # remove the gene from df_mouse
        df_mouse = df_mouse[df_mouse['gene'] != mouse_gene_for_full_transcript]

    # get one transcript per gene, choose the longest one
    df_mouse['sequence_length'] = df_mouse['sequence'].apply(seqLenMouse)
    df_mouse = df_mouse.sort_values('sequence_length', ascending=False).drop_duplicates('gene')
    df_mouse = df_mouse.drop(columns=['sequence_length'])

    return df_mouse

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

# # model name and output folder path
# model_name1 = 'saved_models/XLNetDHConds DS: DeprNA [' + str(n_layers_val) + ', ' + str(d_model_val) + ', ' + str(n_heads_val) + '] FT: [PEL] BS: ' + str(batch_size_val) + ' Loss: ' + str(loss_fun_name) + ' Data Conds: [NZ: ' + str(longZerosThresh_val) + ', PNTh: ' + str(percNansThresh_val) + ', AnnotThresh: ' + str(annot_thresh) + '] Seed: 1'
# model_name2 = 'saved_models/XLNetDHConds DS: DeprNA [' + str(n_layers_val) + ', ' + str(d_model_val) + ', ' + str(n_heads_val) + '] FT: [PEL] BS: ' + str(batch_size_val) + ' Loss: ' + str(loss_fun_name) + ' Data Conds: [NZ: ' + str(longZerosThresh_val) + ', PNTh: ' + str(percNansThresh_val) + ', AnnotThresh: ' + str(annot_thresh) + '] Seed: 2'
# model_name3 = 'saved_models/XLNetDHConds DS: DeprNA [' + str(n_layers_val) + ', ' + str(d_model_val) + ', ' + str(n_heads_val) + '] FT: [PEL] BS: ' + str(batch_size_val) + ' Loss: ' + str(loss_fun_name) + ' Data Conds: [NZ: ' + str(longZerosThresh_val) + ', PNTh: ' + str(percNansThresh_val) + ', AnnotThresh: ' + str(annot_thresh) + '] Seed: 3'
# model_name4 = 'saved_models/XLNetDHConds DS: DeprNA [' + str(n_layers_val) + ', ' + str(d_model_val) + ', ' + str(n_heads_val) + '] FT: [PEL] BS: ' + str(batch_size_val) + ' Loss: ' + str(loss_fun_name) + ' Data Conds: [NZ: ' + str(longZerosThresh_val) + ', PNTh: ' + str(percNansThresh_val) + ', AnnotThresh: ' + str(annot_thresh) + '] Seed: 4'
# model_name42 = 'saved_models/XLNetDHConds DS: DeprNA [' + str(n_layers_val) + ', ' + str(d_model_val) + ', ' + str(n_heads_val) + '] FT: [PEL] BS: ' + str(batch_size_val) + ' Loss: ' + str(loss_fun_name) + ' Data Conds: [NZ: ' + str(longZerosThresh_val) + ', PNTh: ' + str(percNansThresh_val) + ', AnnotThresh: ' + str(annot_thresh) + '] Seed: 42'

# class XLNetDH(XLNetForTokenClassification):
#     def __init__(self, config):
#         super().__init__(config)
#         self.classifier = torch.nn.Linear(d_model_val, 2, bias=True)

# config = XLNetConfig(vocab_size=71, pad_token_id=70, d_model = d_model_val, n_layer = n_layers_val, n_head = n_heads_val, d_inner = d_model_val, num_labels = 1, dropout=dropout_val) # 64*6 tokens + 1 for padding
# model = XLNetDH(config)

# # load model best weights
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# # load model from the saved model
# model1 = model.from_pretrained(model_name1 + "/best_model")
# model2 = model.from_pretrained(model_name2 + "/best_model")
# model3 = model.from_pretrained(model_name3 + "/best_model")
# model4 = model.from_pretrained(model_name4 + "/best_model")
# model42 = model.from_pretrained(model_name42 + "/best_model")

# models_list = [model1, model2, model3, model4, model42]

# for model_chosen in models_list:
#     model_chosen.to(device)
#     model_chosen.eval()

# print("Loaded all the models")

# depr_folder = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/' # depr data folder

# ctrl_depr_path = depr_folder + 'CTRL_AA.csv'
# ile_path = depr_folder + 'ILE_AA.csv'
# leu_path = depr_folder + 'LEU_AA.csv'
# val_path = depr_folder + 'VAL_AA.csv'
# leu_ile_path = depr_folder + 'LEU-ILE_AA_remBadRep.csv'
# leu_ile_val_path = depr_folder + 'LEU-ILE-VAL_AA.csv'
# liver_path = depr_folder + 'LIVER.csv'

# # load the control data
# df_liver = pd.read_csv(liver_path)
# df_liver['condition'] = 'CTRL'

# # load ctrl_aa data
# df_ctrl_depr = pd.read_csv(ctrl_depr_path)
# df_ctrl_depr['condition'] = 'CTRL'

# # add to the liver data the genes from ctrl depr which are not in liver
# tr_liver = df_liver['transcript'].unique()
# tr_ctrl_depr = df_ctrl_depr['transcript'].unique()
# tr_to_add = [g for g in tr_liver if g not in tr_ctrl_depr]

# df_liver = df_liver[df_liver['transcript'].isin(tr_to_add)]

# # df ctrldepr without liver intersection
# df_ctrldepr_liver = pd.concat([df_liver, df_ctrl_depr], axis=0)

# # unique genes
# df_ctrldepr_liver = uniqueGenes(df_ctrldepr_liver)

# # get ctrl gene, transcript tuple pairs from the df_ctrldepr_liver
# ctrl_genes_transcripts = list(zip(df_ctrldepr_liver['gene'], df_ctrldepr_liver['transcript']))
# # make a list of lists
# ctrl_genes_transcripts = [[gene, transcript] for gene, transcript in ctrl_genes_transcripts]

# print("CTRL Done")

# # other conditions
# df_ile = pd.read_csv(ile_path)
# df_ile['condition'] = 'ILE'
# # unique genes
# df_ile = uniqueGenes(df_ile)
# # only choose those genes+transcripts that are in ctrl_depr_liver
# # iterate through the df_ile and choose those genes that are in ctrl_genes_transcripts
# for index, row in df_ile.iterrows():
#     if [row['gene'], row['transcript']] not in ctrl_genes_transcripts:
#         df_ile.drop(index, inplace=True) 

# print("ILE Done")

# df_leu = pd.read_csv(leu_path)
# df_leu['condition'] = 'LEU'
# # unique genes
# df_leu = uniqueGenes(df_leu)
# # choose those transcripts that are in ctrl_depr_liver
# for index, row in df_leu.iterrows():
#     if [row['gene'], row['transcript']] not in ctrl_genes_transcripts:
#         df_leu.drop(index, inplace=True)

# print("LEU Done")

# df_val = pd.read_csv(val_path)
# df_val['condition'] = 'VAL'
# # unique genes
# df_val = uniqueGenes(df_val)
# # choose those transcripts that are in ctrl_depr_liver
# for index, row in df_val.iterrows():
#     if [row['gene'], row['transcript']] not in ctrl_genes_transcripts:
#         df_val.drop(index, inplace=True)

# print("VAL Done")

# df_leu_ile = pd.read_csv(leu_ile_path)
# df_leu_ile['condition'] = 'LEU_ILE'
# # unique genes
# df_leu_ile = uniqueGenes(df_leu_ile)
# # choose those transcripts that are in ctrl_depr_liver
# for index, row in df_leu_ile.iterrows():
#     if [row['gene'], row['transcript']] not in ctrl_genes_transcripts:
#         df_leu_ile.drop(index, inplace=True)

# print("LEU_ILE Done")

# df_leu_ile_val = pd.read_csv(leu_ile_val_path)
# df_leu_ile_val['condition'] = 'LEU_ILE_VAL'
# # unique genes
# df_leu_ile_val = uniqueGenes(df_leu_ile_val)
# # choose those transcripts that are in ctrl_depr_liver
# for index, row in df_leu_ile_val.iterrows():
#     if [row['gene'], row['transcript']] not in ctrl_genes_transcripts:
#         df_leu_ile_val.drop(index, inplace=True)

# print("LEU_ILE_VAL Done")

# df_full = pd.concat([df_ctrldepr_liver, df_ile, df_leu, df_val, df_leu_ile, df_leu_ile_val], axis=0) # liver + ctrl depr + ile + leu + val + leu ile + leu ile val

# df_full.columns = ['index_val', 'gene', 'transcript', 'sequence', 'annotations', 'perc_non_zero_annots', 'condition']

# # drop index_val column
# df_full = df_full.drop(columns=['index_val'])

# df_full.to_csv("pseudolabeling/data_preds/tmp/df_full.csv", index=False)

df_full = pd.read_csv("pseudolabeling/data_preds/tmp/df_full.csv")

print("Processed All Ribo-Seq Lina Data")

print("Number of Genes in Full Data: ", len(df_full['gene'].unique()))
print("Number of Transcripts in Full Data: ", len(df_full['transcript'].unique()))

# # get the gene, transcript, sequence data 
# df_set1 = df_full[['gene', 'transcript', 'sequence']]

# # drop duplicates
# df_set1 = df_set1.drop_duplicates()

# len_sf_set1 = len(df_set1)

# # replicate this 6 times, and add condition column
# df_set1 = pd.concat([df_set1]*6, ignore_index=True)

# # add condition column
# cond_col = []
# for i in range(6):
#     for j in range(len_sf_set1):
#         cond_col.append(conditions_list[i])

# df_set1['condition'] = cond_col

# print(df_set1)

# # save this dataframe
# df_set1.to_csv("pseudolabeling/data_preds/df_set1.csv", index=False)


# # # ensembl cds file process for mouse genome, and remove those transcripts that are already in train and test
# # read fasta file
fasta_file = "/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/reference/ensembl.cds.fa"
max_codon_len = 2000

gene_id_mouse  = []
transcript_id_mouse = []
sequences_mouse = []

for record in SeqIO.parse(fasta_file, "fasta"):
    rec = record.description.split(' ')[3]
    gene_id = rec.split(':')[1]
    transcript_id = record.description.split(' ')[0]

    if len(str(record.seq)) <= max_codon_len*3 and len(str(record.seq)) >= 120: # 2000 codons max length for the genes and min 40 codons 
        gene_id_mouse.append(gene_id)
        sequences_mouse.append(str(record.seq))
        transcript_id_mouse.append(transcript_id)

# create dataframe
df_mouse = pd.DataFrame({'gene': gene_id_mouse, 'transcript': transcript_id_mouse, 'sequence': sequences_mouse})

# remove those genes that had a 'N' in the sequence
df_mouse = df_mouse[~df_mouse['sequence'].str.contains('N')]

print("Number of Genes in Mouse Data: ", len(df_mouse['gene'].unique()))

# # remove the genes that are already in df_full
df_mouse = removeFullGenes(df_mouse, df_full)

print("Number of Genes in Mouse Data After Removing those from DF Full: ", len(df_mouse['gene'].unique()))

len_sf_set2 = len(df_mouse)

# replicate this 6 times, and add condition column
df_set2 = pd.concat([df_mouse]*6, ignore_index=True)

# add condition column
cond_col = []
for i in range(6):
    for j in range(len_sf_set2):
        cond_col.append(conditions_list[i])

df_set2['condition'] = cond_col

print(df_set2)

# save this dataframe
df_set2.to_csv("pseudolabeling/data_preds/df_set2.csv", index=False)


        

            
