'''
model trained on only one condition samples
'''

# libraries
import numpy as np
import pandas as pd 
import torch
from transformers import XLNetConfig, XLNetForTokenClassification, EarlyStoppingCallback
from utils_sh import RegressionTrainer, RiboDatasetGWSDepr, GWSDatasetFromPandas  # custom dataset and trainer
from transformers import TrainingArguments
import random
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch import nn
from torchmetrics.functional import pearson_corrcoef
from torchmetrics import Metric
import argparse

# reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--condition', type=str, default='CTRL', help='condition to train on')
args = parser.parse_args()

# training arguments
condition_training = args.condition
data_folder = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/'
annot_thresh = 0.3
n_layers_val = 3
d_model_val = 256
n_heads_val = 8
batch_size_val = 1
loss_fun_name = 'MAE_PCC'
longZerosThresh_val = 20
percNansThresh_val = 0.05
model_name = 'XLNetSHConds DS: ' + str(condition_training) + ' [' + str(n_layers_val) + ', ' + str(d_model_val) + ', ' + str(n_heads_val) + '] FT: [PEL] BS: ' + str(batch_size_val) + ' Loss: ' + str(loss_fun_name) + ' Data Conds: [NZ: ' + str(longZerosThresh_val) + ', PNTh: ' + str(percNansThresh_val) + ', AnnotThresh: ' + str(annot_thresh) + ']'
output_loc = "saved_models/" + model_name

liver_path = data_folder + 'LIVER.csv'

# dataset paths
if condition_training == 'CTRL':
    ctrl_path = data_folder + 'CTRL_AA.csv'
elif condition_training == 'LEU':
    leu_path = data_folder + 'LEU_AA.csv'
elif condition_training == 'ILE':
    ile_path = data_folder + 'ILE_AA.csv'
elif condition_training == 'LEU-ILE':
    leu_ile_path = data_folder + 'LEU-ILE_AA_remBadRep.csv'
elif condition_training == 'VAL':
    val_path = data_folder + 'VAL_AA.csv'
elif condition_training == 'LEU-ILE-VAL':
    leu_ile_val_path = data_folder + 'LEU-ILE-VAL_AA.csv'

# GWS dataset
if condition_training == 'CTRL':
    train_dataset, test_dataset = RiboDatasetGWSDepr(ctrl_path, threshold = annot_thresh, longZerosThresh = longZerosThresh_val, percNansThresh = percNansThresh_val, cond = 'CTRL', liver_path = liver_path)
elif condition_training == 'LEU':
    train_dataset, test_dataset = RiboDatasetGWSDepr(leu_path, threshold = annot_thresh, longZerosThresh = longZerosThresh_val, percNansThresh = percNansThresh_val, cond = 'LEU', liver_path = liver_path)
elif condition_training == 'ILE':
    train_dataset, test_dataset = RiboDatasetGWSDepr(ile_path, threshold = annot_thresh, longZerosThresh = longZerosThresh_val, percNansThresh = percNansThresh_val, cond = 'ILE', liver_path = liver_path)
elif condition_training == 'LEU-ILE':
    train_dataset, test_dataset = RiboDatasetGWSDepr(leu_ile_path, threshold = annot_thresh, longZerosThresh = longZerosThresh_val, percNansThresh = percNansThresh_val, cond = 'LEU_ILE', liver_path = liver_path)
elif condition_training == 'VAL':
    train_dataset, test_dataset = RiboDatasetGWSDepr(val_path, threshold = annot_thresh, longZerosThresh = longZerosThresh_val, percNansThresh = percNansThresh_val, cond = 'VAL', liver_path = liver_path)
elif condition_training == 'LEU-ILE-VAL':
    train_dataset, test_dataset = RiboDatasetGWSDepr(leu_ile_val_path, threshold = annot_thresh, longZerosThresh = longZerosThresh_val, percNansThresh = percNansThresh_val, cond = 'LEU_ILE_VAL', liver_path = liver_path)
elif condition_training == 'OnlyLiver':
    train_dataset, test_dataset = RiboDatasetGWSDepr(liver_path, threshold = annot_thresh, longZerosThresh = longZerosThresh_val, percNansThresh = percNansThresh_val, cond = 'OnlyLiver', liver_path = liver_path)

# convert to torch dataset
train_dataset = GWSDatasetFromPandas(train_dataset)
test_dataset = GWSDatasetFromPandas(test_dataset)

print("samples in train dataset: ", len(train_dataset))
print("samples in test dataset: ", len(test_dataset))

# load xlnet to train from scratch
# GWS
config = XLNetConfig(vocab_size=65, pad_token_id=64, d_model = d_model_val, n_layer = n_layers_val, n_head = n_heads_val, d_inner = d_model_val, num_labels = 1) # 4^3 + 1 for padding
model = XLNetForTokenClassification(config)

model.classifier = torch.nn.Linear(d_model_val, 1, bias=True)

class CorrCoef(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("corrcoefs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
    def update(self, preds, target, mask):
        # # sum preds in dim 2
        # preds = torch.sum(preds, dim=2)
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

# collate function
def collate_fn(batch):
    # batch is a list of tuples (x, y)
    x, y = zip(*batch)

    # sequence lenghts 
    lengths = torch.tensor([len(x) for x in x])
    x = pad_sequence(x, batch_first=True, padding_value=64) 
    y = pad_sequence(y, batch_first=True, padding_value=-1)

    out_batch = {}

    out_batch["input_ids"] = x
    out_batch["labels"] = y
    out_batch["lengths"] = lengths

    return out_batch

# compute metrics
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

output_loc = "saved_models/" + model_name

# train xlnet
training_args = TrainingArguments(
    output_dir=output_loc,
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=4,
    num_train_epochs=100,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    dataloader_pin_memory=True,
    dataloader_num_workers=4
)

trainer = RegressionTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=20)]
)

trainer.train()

# save best model
trainer.save_model(output_loc + "/best_model")

# evaluate model
trainer.evaluate()

# # # # # load model best weights
# model.load_state_dict(torch.load(output_loc + "/best_model/pytorch_model.bin"))

# # evaluate model
# trainer.evaluate()
