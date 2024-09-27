# %%
# libraries
import torch
from transformers import XLNetConfig, XLNetForTokenClassification, TrainingArguments, EarlyStoppingCallback
from xlnet_plabel_utils import RegressionTrainerFour, RiboDatasetExp1, RiboDatasetExp2, GWSDatasetFromPandas, collate_fn, compute_metrics  # custom dataset and trainer
import pytorch_lightning as pl
import wandb
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

# %%
# model parameters
annot_thresh = 0.3
longZerosThresh_val = 20
percNansThresh_val = 0.05
loss_fun_name = '4L' # 4L
seed_val = 1
experiment_type = 'exp2' # exp1, exp2

# %%
# generate dataset
if experiment_type == 'exp1': # impute all train genes
    train_dataset, val_dataset, test_dataset = RiboDatasetExp1(threshold = annot_thresh, longZerosThresh = longZerosThresh_val, percNansThresh = percNansThresh_val)
elif experiment_type == 'exp2': # impute train genes + extra mouse genome genes
    train_dataset, val_dataset, test_dataset = RiboDatasetExp2(threshold = annot_thresh, longZerosThresh = longZerosThresh_val, percNansThresh = percNansThresh_val)

# convert pandas dataframes into torch datasets
train_dataset = GWSDatasetFromPandas(train_dataset)
val_dataset = GWSDatasetFromPandas(val_dataset)
test_dataset = GWSDatasetFromPandas(test_dataset)

print("samples in train dataset: ", len(train_dataset))
print("samples in val dataset: ", len(val_dataset))
print("samples in test dataset: ", len(test_dataset))

# %%
# reproducibility
pl.seed_everything(seed_val)

# dataset paths 
data_folder = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/'


# %%
def objective(trials):
    # hyperparameters
    tot_epochs = 10
    n_layers_val = trials.suggest_categorical('num_layers', [2, 3, 4, 5, 6])
    batch_size_val = trials.suggest_categorical('batch_size', [1, 2, 4])
    dropout_val = trials.suggest_categorical('dropout_val', [0.0, 0.1])
    lr_val = trials.suggest_categorical('lr', [1e-3, 1e-4, 1e-5])
    d_model_val = trials.suggest_categorical('num_nodes', [64, 128, 256, 512])
    n_heads_val = trials.suggest_categorical('n_heads_val', [4, 8, 16])

    # model name and output folder path
    model_name = 'XLNet-PLabelDHOptuna ' + ' Exp: ' + experiment_type + ' [NL: ' + str(n_layers_val) + ', NH: ' + str(n_heads_val) + ', D: ' + str(d_model_val) + ', LR: ' + str(lr_val) + ', BS: ' + str(batch_size_val) + ', LF: ' + loss_fun_name + ', Dr: ' + str(dropout_val) + ', S: ' + str(seed_val) + ']'
    output_loc = "saved_models/" + model_name

    # set wandb name to model_name
    wandb.init(project="PLabel-Optuna2", name=model_name)

    # load xlnet to train from scratch
    config = XLNetConfig(vocab_size=71, pad_token_id=70, d_model = d_model_val, n_layer = n_layers_val, n_head = n_heads_val, d_inner = d_model_val, num_labels = 1, dropout=dropout_val) # 6 conds, 64 codons, 1 for padding
    model = XLNetForTokenClassification(config)

    # modify the output layer
    # model.classifier is a linear layer followed by a softmax layer
    model.classifier = torch.nn.Linear(d_model_val, 2, bias=True)

    # xlnet training arguments
    training_args = TrainingArguments(
        output_dir = output_loc,
        learning_rate = lr_val,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = batch_size_val, # training batch size = per_device_train_batch_size * gradient_accumulation_steps
        per_device_eval_batch_size = 1,
        eval_accumulation_steps = 4, 
        num_train_epochs = tot_epochs,
        weight_decay = 0.01,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        load_best_model_at_end = True,
        push_to_hub = False,
        dataloader_pin_memory = True,
        save_total_limit = 5,
        dataloader_num_workers = 4,
        include_inputs_for_metrics = True
    )

    # initialize trainer
    trainer = RegressionTrainerFour(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # train model
    trainer.train()

    # save best model
    trainer.save_model(output_loc + "/best_model")

    # evaluate model
    res = trainer.evaluate(eval_dataset=test_dataset)

    mini_perf = 1 - res['eval/r']

    return mini_perf

# %%
study = optuna.load_study(
    study_name='xlnet_dh_plabel_exp2', 
    storage='mysql://optuna_vamsi:RJ-zf8JJFLxwK3ey1WbrRLbHmmz-W_EznNkdDefMsNyLHTehNgXa4TtX92qa1kmI@lts2srv1.epfl.ch/optuna_vamsi'
    )
study.optimize(
    objective,
    callbacks=[MaxTrialsCallback(50, states=(TrialState.COMPLETE,))],
)

# %%
# best results
print("Best trial:")
print(study.best_trial)


