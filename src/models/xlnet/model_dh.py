'''
double head models
'''
# libraries
import numpy as np
import pandas as pd 
import torch
from transformers import XLNetConfig, XLNetForTokenClassification
from utils_dh import RegressionTrainer, RiboDatasetGWS, GWSDatasetFromPandas  # custom dataset and trainer
from transformers import TrainingArguments
import random
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.functional import pearson_corrcoef
from torchmetrics import Metric

# reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# dataset paths 
data_folder = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Darnell_Full/Darnell/data_conds_split/processed/'

# current params
annot_thresh = 0.5
longZerosThresh_val = 20
percNansThresh_val = 0.05
d_model_val = 256
n_layers_val = 3
n_heads_val = 8
dropout_val = 0.3
lr_val = 1e-4
batch_size_val = 1

model_name = 'XLNetDepr-' + str(n_layers_val) + '_' + str(d_model_val) + '_' + str(n_heads_val) + '-ALL_NA-PEL-BS' + str(batch_size_val) + '-GWS_PCC_IT192_DH_3L_NZ' + str(longZerosThresh_val) + '_PNTh' + str(percNansThresh_val) + '_AnnotThresh' + str(annot_thresh)

# GWS dataset
ds = 'ALL' # this uses both liver and deprivation datasets all the conditions
train_dataset, test_dataset = RiboDatasetGWS(data_folder, ds, threshold = annot_thresh, longZerosThresh = longZerosThresh_val, percNansThresh = percNansThresh_val)

# convert to torch dataset
train_dataset = GWSDatasetFromPandas(train_dataset)
test_dataset = GWSDatasetFromPandas(test_dataset)
print("samples in train dataset: ", len(train_dataset))
print("samples in test dataset: ", len(test_dataset))

# load xlnet to train from scratch
# GWS
config = XLNetConfig(vocab_size=385, pad_token_id=384, d_model = d_model_val, n_layer = n_layers_val, n_head = n_heads_val, d_inner = d_model_val, num_labels = 1, dropout=dropout_val) # 5^3 + 1 for padding
model = XLNetForTokenClassification(config)
# modify the input layer to take 384 to 256
model.classifier = torch.nn.Linear(d_model_val, 2, bias=True)

class CorrCoef(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("corrcoefs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
    def update(self, preds, target, mask):
        preds = torch.sum(preds, dim=2)
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

    return {"r": corr_coef.compute()}

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
    corr_coef = CorrCoef()
    corr_coef.update(preds, labels, mask)

    # save predictions
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    np.save("preds/" + model_name + "/preds.npy", preds)
    np.save("preds/" + model_name + "/labels.npy", labels)
    np.save("preds/" + model_name + "/inputs.npy", inputs)

    return {"r": corr_coef.compute()}

output_loc = "saved_models/" + model_name

# train xlnet
# save max 5 checkpoints
training_args = TrainingArguments(
    output_dir=output_loc,
    learning_rate=lr_val,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=batch_size_val,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=4,
    num_train_epochs=200,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    dataloader_pin_memory=True,
    save_total_limit=5,
    dataloader_num_workers=4,
    include_inputs_for_metrics=True
)

trainer = RegressionTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

trainer.train()

# save best model
trainer.save_model(output_loc + "/best_model")

# # evaluate model
trainer.evaluate()

# trainer = RegressionTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
#     data_collator=collate_fn,
#     compute_metrics=compute_metrics_saved,
# )

# # load model best weights
# model.load_state_dict(torch.load(output_loc + "/best_model/pytorch_model.bin"))

# # trainer.evaluate()

# # analyse preds
# preds = np.load("preds/" + model_name + "/preds.npy")
# labels = np.load("preds/" + model_name + "/labels.npy")
# inputs = np.load("preds/" + model_name + "/inputs.npy")

# # quantile_metric(preds, labels, inputs, "preds/" + model_name + "/analysis_dh/quantile_metric_plots/")

# # analyse_dh_outputs(preds, labels, inputs, "preds/" + model_name + "/analysis_dh")

# captum_interpretability(model, inputs, labels)
