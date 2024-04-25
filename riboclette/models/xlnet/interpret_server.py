# libraries
import numpy as np
import seaborn as sns
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import math
import os
from scipy import stats
from scipy.stats import pearsonr, spearmanr 
from torchmetrics.functional import pearson_corrcoef
import itertools
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.interpolate import make_interp_spline 
from captum.attr import IntegratedGradients, LayerIntegratedGradients, LayerGradientXActivation
from transformers import XLNetConfig, XLNetForTokenClassification, TrainingArguments
from ipynb.fs.full.utils_dh import RegressionTrainerFive, RiboDatasetGWS, GWSDatasetFromPandas, collate_fn, compute_metrics, compute_metrics_saved  # custom dataset and trainer, CorrCoef, collate_fn, compute_metrics, compute_metrics_saved  # custom dataset and trainer
import pytorch_lightning as pl
import h5py
from tqdm import tqdm
# suppress warnings
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class model_finalexpLIG(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model 
    
    def forward(self, x, index_val):
        # input dict
        out_batch = {}

        out_batch["input_ids"] = x.unsqueeze(0)
        for k, v in out_batch.items():
            out_batch[k] = v.to(device)

        out_batch["input_ids"] = torch.tensor(out_batch["input_ids"]).to(device).to(torch.int32)

        pred = self.model(out_batch["input_ids"])

        pred_fin = torch.sum(pred["logits"], dim=2)

        pred_fin = pred_fin.squeeze(0)

        out = pred_fin[index_val].unsqueeze(0)

        return out 
    
def integratedgrad_output(model, x, y):
    model_fin = model_finalexpLIG(model)
        
    lig = LayerIntegratedGradients(model_fin, model_fin.model.transformer.word_embedding)

    # set torch graph to allow unused tensors
    with torch.autograd.set_detect_anomaly(True):

        out_batch = {}

        out_batch["input_ids"] = x
        
        out_batch["input_ids"] = torch.tensor(out_batch["input_ids"]).to(device).to(torch.int32)

        baseline_inp = torch.ones(out_batch["input_ids"].shape) * 70 # 70 is the padding token
        baseline_inp = baseline_inp.to(device).to(torch.int32)
    
        # get indices of the top 10 values 
        # add first token to y, which is -epsilon
        y = torch.cat((torch.tensor([-1e-6]), y))
        indices = torch.topk(y, 10).indices

        len_sample = len(x)
        attributions_sample = np.zeros((10, len_sample))

        for j in range(len(indices)):
            index_val = indices[j]

            index_val = torch.tensor(index_val).to(device)

            attributions, approximation_error = lig.attribute((out_batch["input_ids"]), baselines=baseline_inp, 
                                                    method = 'gausslegendre', return_convergence_delta = True, additional_forward_args=index_val, n_steps=20, internal_batch_size=2048)

            
            attributions = attributions.squeeze(1)
            attributions = torch.sum(attributions, dim=1)
            attributions = attributions / torch.norm(attributions)
            attributions = attributions.detach().cpu().numpy()
            attributions_sample[j] = attributions
        
        attributions_sample = np.array(attributions_sample)

        # remove first column which is padding token
        attributions_sample = attributions_sample[:, 1:]

        # flatten the attributions
        attributions_sample = attributions_sample.flatten()

    return attributions_sample, indices

# reproducibility
seed_val = 1
pl.seed_everything(seed_val)

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

# dataset paths 
data_folder = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/'

# model name and output folder path
model_name = 'XLNetDHConds DS: DeprNA [' + str(n_layers_val) + ', ' + str(d_model_val) + ', ' + str(n_heads_val) + '] FT: [PEL] BS: ' + str(batch_size_val) + ' Loss: ' + str(loss_fun_name) + ' Data Conds: [NZ: ' + str(longZerosThresh_val) + ', PNTh: ' + str(percNansThresh_val) + ', AnnotThresh: ' + str(annot_thresh) + '] Seed: ' + str(seed_val)
output_loc = "saved_models/" + model_name

class XLNetDH(XLNetForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = torch.nn.Linear(d_model_val, 2, bias=True)

config = XLNetConfig(vocab_size=71, pad_token_id=70, d_model = d_model_val, n_layer = n_layers_val, n_head = n_heads_val, d_inner = d_model_val, num_labels = 1, dropout=dropout_val) # 64*6 tokens + 1 for padding
model = XLNetDH(config)

# generate dataset
ds = 'ALL' # uses all the three conditions
train_dataset, test_dataset = RiboDatasetGWS(data_folder, ds, threshold = annot_thresh, longZerosThresh = longZerosThresh_val, percNansThresh = percNansThresh_val)

# convert pandas dataframes into torch datasets
train_dataset = GWSDatasetFromPandas(train_dataset)
test_dataset = GWSDatasetFromPandas(test_dataset)
print("samples in train dataset: ", len(train_dataset))
print("samples in test dataset: ", len(test_dataset))

# load model from the saved model
model = model.from_pretrained(output_loc + "/best_model")
model.to(device)
# set model to evaluation mode
model.eval()

# h5py dataset
out_ds = h5py.File('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/xlnet_interpret/XLNet_DH_S1.h5', 'w')

# make datasets in out_ds
x_input_ds = out_ds.create_dataset(
    'x_input',
    (len(test_dataset),),
    dtype=h5py.special_dtype(vlen=np.dtype('float64'))
)

y_true_full_ds = out_ds.create_dataset(
    'y_true_full',
    (len(test_dataset),),
    dtype=h5py.special_dtype(vlen=np.dtype('float64'))
)

y_pred_ds_full = out_ds.create_dataset(
    'y_pred',
    (len(test_dataset),),
    dtype=h5py.special_dtype(vlen=np.dtype('float64'))
)

y_true_ctrl_ds = out_ds.create_dataset(
    'y_true_ctrl',
    (len(test_dataset),),
    dtype=h5py.special_dtype(vlen=np.dtype('float64'))
)

# gene ds with string datatype
gene_ds = out_ds.create_dataset(
    'gene',
    (len(test_dataset),),
    dtype=h5py.special_dtype(vlen=str)
)

transcript_ds = out_ds.create_dataset(
    'transcript',
    (len(test_dataset),),
    dtype=h5py.special_dtype(vlen=str)
)

indices_ds = out_ds.create_dataset(
    'indices',
    (len(test_dataset),),
    dtype=h5py.special_dtype(vlen=np.dtype('float64'))
)

atrr_ds = out_ds.create_dataset(
    'attributions',
    (len(test_dataset),),
    dtype=h5py.special_dtype(vlen=np.dtype('float64'))
)

y_pred_ctrl_ds = out_ds.create_dataset(
    'y_pred_ctrl',
    (len(test_dataset),),
    dtype=h5py.special_dtype(vlen=np.dtype('float64'))
)

y_pred_depr_diff_ds = out_ds.create_dataset(
    'y_pred_depr_diff',
    (len(test_dataset),),
    dtype=h5py.special_dtype(vlen=np.dtype('float64'))
)

with torch.autograd.set_detect_anomaly(True):
    for i, (x, y, ctrl_y, gene, transcript) in tqdm(enumerate(test_dataset)):
        x = torch.tensor(x)
        y = torch.tensor(y)

        attributions_sample, indices_sample = integratedgrad_output(model, x, y)

        # make predictions for sample
        y_pred_full = model(x.unsqueeze(0).to(device).to(torch.int32))
        y_pred = torch.sum(y_pred_full["logits"], dim=2)
        y_pred = y_pred.squeeze(0)

        # remove first token which is condition token
        y_pred = y_pred[1:]

        # add to h5py datasets
        x_input_ds[i] = x
        y_true_full_ds[i] = y
        y_pred_ds_full[i] = y_pred.detach().cpu().numpy()
        y_true_ctrl_ds[i] = ctrl_y
        gene_ds[i] = gene
        transcript_ds[i] = transcript
        indices_ds[i] = indices_sample.detach().cpu().numpy()
        atrr_ds[i] = attributions_sample

        # index 0 for ctrl pred from y_pred
        y_pred_ = y_pred_full['logits'].detach().cpu().numpy()
        y_pred_ctrl = y_pred_[:, :, 0]
        y_pred_ctrl = y_pred_ctrl.squeeze(0)
        y_pred_ctrl_ds[i] = y_pred_ctrl

        # index 1 for depr pred from y_pred
        y_pred_depr = y_pred_[:, :, 1]
        y_pred_depr = y_pred_depr.squeeze(0)
        y_pred_depr_diff_ds[i] = y_pred_depr
        