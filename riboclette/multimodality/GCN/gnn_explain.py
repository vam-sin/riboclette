from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer, CaptumExplainer
import torch 
import captum
from utils import GCN

# load a sample from the test folder
feature_folder = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/mm/DirSeqPlus/train/'
sample_number = 1
file_name = feature_folder + 'sample_' + str(sample_number) + '.pt'

# load the sample
data = torch.load(file_name)
data.edge_attr = None

# load the model
tot_epochs = 50
batch_size = 2
dropout_val = 0.4
annot_thresh = 0.3
longZerosThresh_val = 20
percNansThresh_val = 0.05
random_walk_length = 32
alpha = -1
lr = 1e-3
model_type = 'DirSeq+'
algo = 'TF'
edge_attr = 'None'
features = ['cbert_full', 'codon_ss', 'pos_enc']
features_str = '_'.join(features)
loss_fn = 'MAE + PCC'
gcn_layers = [256, 128, 128, 64]
input_nums_dict = {'cbert_full': 768, 'codon_ss': 0, 'pos_enc': 32}
num_inp_ft = sum([input_nums_dict[ft] for ft in features])

model_name = 'Noisy' + model_type + '-' + algo + ' EA: ' + str(edge_attr) + ' DS: Liver' + '[' + str(annot_thresh) + ', ' + str(longZerosThresh_val) + ', ' + str(percNansThresh_val) + ', BS ' + str(batch_size) + ', D ' + str(dropout_val) + ' E ' + str(tot_epochs) + ' LR ' + str(lr) + '] F: ' + features_str + ' VN RW 32 -1 + GraphNorm ' + loss_fn
save_loc = 'saved_models/' + model_name
l_model = GCN.load_from_checkpoint(save_loc + '/epoch=14-step=37020.ckpt', gcn_layers=gcn_layers, dropout_val=dropout_val, num_epochs=tot_epochs, bs=batch_size, lr=lr, num_inp_ft=num_inp_ft, alpha=alpha, model_type=model_type, algo=algo, edge_attr=edge_attr)

explainer = Explainer(
    model=l_model, # get torch module from lightning module
    algorithm=CaptumExplainer(attribution_method=captum.attr.InputXGradient),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='regression',
        task_level='node',
        return_type='raw',  # Model returns log probabilities.
    ),
)

# Generate explanation for the node at index `10`:
explanation = explainer(data.x, data.edge_index, index=10)
print(explanation.edge_mask)
print(explanation.node_mask)