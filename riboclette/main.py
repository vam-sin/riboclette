import numpy as np
import torch
from transformers import XLNetConfig, XLNetForTokenClassification
import itertools
from tqdm import tqdm
from captum.attr import LayerIntegratedGradients
import pkg_resources

def hello():
    print("Hello, from Riboclette!")

def sequence2codonids(seq):
    '''
    converts nt sequence into one-hot codon ids
    '''
    id_to_codon = {idx:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
    codon_to_id = {v:k for k,v in id_to_codon.items()}
    codon_ids = []
    for i in range(0, len(seq), 3):
        codon = seq[i:i+3]
        if len(codon) == 3:
            codon_ids.append(codon_to_id[codon])

    return codon_ids

class XLNetDH(XLNetForTokenClassification):
        def __init__(self, config):
            super().__init__(config)
            self.classifier = torch.nn.Linear(512, 2, bias=True)

class model_CTRL(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, x, index_val):
        # input dict
        out_batch = {}

        out_batch["input_ids"] = x.unsqueeze(0)
        for k, v in out_batch.items():
            out_batch[k] = v.to(self.device)

        out_batch["input_ids"] = torch.tensor(out_batch["input_ids"]).to(self.device).to(torch.int32)
        pred = self.model(out_batch["input_ids"])

        # get dim 0
        pred_fin = torch.relu(pred["logits"][:, :, 0])

        pred_fin = pred_fin.squeeze(0)

        out = pred_fin[index_val].unsqueeze(0)

        return out 
    
class model_DD(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, x, index_val):
        # input dict
        out_batch = {}

        out_batch["input_ids"] = x.unsqueeze(0)
        for k, v in out_batch.items():
            out_batch[k] = v.to(self.device)

        out_batch["input_ids"] = torch.tensor(out_batch["input_ids"]).to(self.device).to(torch.int32)

        pred = self.model(out_batch["input_ids"])

        # get dim 1
        pred_fin = pred["logits"][:, :, 1]

        pred_fin = pred_fin.squeeze(0)

        out = pred_fin[index_val].unsqueeze(0)

        return out 

def lig_output(model, x, mode='ctrl', ibs=32):
    if mode == 'ctrl':
        model_fin = model_CTRL(model)
    elif mode == 'dd':
        model_fin = model_DD(model)
        
    lig = LayerIntegratedGradients(model_fin, model_fin.model.transformer.word_embedding)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set torch graph to allow unused tensors
    with torch.autograd.set_detect_anomaly(True):

        out_batch = {}

        out_batch["input_ids"] = x
        
        out_batch["input_ids"] = torch.tensor(out_batch["input_ids"]).to(device).to(torch.int32)

        baseline_inp = torch.ones(out_batch["input_ids"].shape) * 70 # 70 is the padding token
        baseline_inp = baseline_inp.to(device).to(torch.int32)
    
        # get all indices

        len_sample = len(x)
        attributions_sample = np.zeros((len_sample, len_sample))

        for j in tqdm(range(len_sample)):
            index_val = j

            index_val = torch.tensor(index_val).to(device)

            attributions = lig.attribute((out_batch["input_ids"]), baselines=baseline_inp, 
                                                    method = 'gausslegendre', return_convergence_delta = False, additional_forward_args=index_val, n_steps=10, internal_batch_size=ibs)

            
            attributions = attributions.squeeze(1)
            attributions = torch.sum(attributions, dim=1)
            attributions = attributions / torch.norm(attributions)
            attributions = attributions.detach().cpu().numpy()
            attributions_sample[j] = attributions
        
        attributions_sample = np.array(attributions_sample)

        # remove first column which is padding token
        attributions_sample = attributions_sample[1:, 1:]

    return attributions_sample


def predictRFP(sequence, condition='CTRL'):
    """
    Predict the RFP (Ribosome Footprint) for a given sequence.
    
    Args:
        sequence (str): The nucleotide sequence to predict RFP for.
        condition (str): The condition under which the prediction is made.
            Options: 'CTRL', 'ILE', 'LEU', 'LEU_ILE', 'LEU_ILE_VAL', 'VAL'.
            Default is 'CTRL'.
    Raises:
        AssertionError: If the condition is not one of the specified options.
    Note:
        The sequence should be a string of nucleotide bases (A, T, C, G).
        The function uses a pre-trained model to predict the RFP.
        The model is loaded from the specified location in the package.
    Example:
        >>> sequence = "ATGCGTACGTAGCTAGCTAGC"
        >>> condition = "ILE"
        >>> rfp = predictRFP(sequence, condition)
        >>> print(rfp)
        (ctrl_rfp, dd_rfp, dc_rfp)
        
    Returns:
        arr: The predicted RFPs for the given sequence and condition. includes three values: 
        ctrl_rfp, dd_rfp, and dc_rfp.
        ctrl_rfp: Control RFP value.
        dd_rfp: DD RFP value.
        dc_rfp: Deprivation Condition RFP value.
    """
    condition_values = {'CTRL': 64, 'ILE': 65, 'LEU': 66, 'LEU_ILE': 67, 'LEU_ILE_VAL': 68, 'VAL': 69}

    assert condition in condition_values.keys(), f"Condition {condition} not in {condition_values.keys()}"
    
    # model name and output folder path
    model_loc = pkg_resources.resource_filename('riboclette', 'checkpoints/XLNet-PLabelDH_S2/best_model/')

    # model parameters
    d_model_val = 512
    n_layers_val = 6
    n_heads_val = 4
    dropout_val = 0.1

    config = XLNetConfig(vocab_size=71, pad_token_id=70, d_model = d_model_val, n_layer = n_layers_val, n_head = n_heads_val, d_inner = d_model_val, num_labels = 1, dropout=dropout_val) # 64*6 tokens + 1 for padding
    model = XLNetDH(config)

    # load model from the saved model
    model = model.from_pretrained(model_loc)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # set model to evaluation mode
    model.eval()

    X = sequence2codonids(sequence)
    X = [int(k) for k in X]

    # prepend condition token
    X = [condition_values[condition]] + X

    X = np.asarray(X)
    X = torch.from_numpy(X).long()

    with torch.no_grad():
        y_pred = model(X.unsqueeze(0).to(device).to(torch.int32))
        y_pred = y_pred["logits"].squeeze(0)
        y_pred = y_pred[1:]

    # convert to numpy
    y_pred = y_pred.cpu().numpy()

    ctrl_rfp = y_pred[:, 0]
    dd_rfp = y_pred[:, 1]
    dc_rfp = ctrl_rfp + dd_rfp

    # return ctrl, dd, and dc
    return ctrl_rfp, dd_rfp, dc_rfp

def getAttr(sequence, condition = 'CTRL', head='CTRL'):
    """
    Get the attributes of the sequence for the specified head.
    
    Args:
        sequence (str): The nucleotide sequence to get attributes for.
        condition (str): The condition under which the attributes are calculated.
            Options: 'CTRL', 'ILE', 'LEU', 'LEU_ILE', 'LEU_ILE_VAL', 'VAL'.
            Default is 'CTRL'.
        head (str): The head for which to get attributes.
            Options: 'CTRL', 'DD'.
            Default is 'CTRL'.
    Raises:
        AssertionError: If the head is not one of the specified options.
    Note:
        The sequence should be a string of nucleotide bases (A, T, C, G).
        The function uses a pre-trained model to get the attributes.
    Example:
        >>> sequence = "ATGCGTACGTAGCTAGCTAGC"
        >>> condition = "ILE"
        >>> head = "DD"
        >>> attr = getAttr(sequence, condition, head)
        >>> print(attr)
        
    Returns:
        arr: The attributes for the given sequence and head.
    """
    # Check if the head is valid
    assert head in ['CTRL', 'DD'], f"Head {head} not in ['CTRL', 'DD']"

    # Check if the condition is valid
    condition_values = {'CTRL': 64, 'ILE': 65, 'LEU': 66, 'LEU_ILE': 67, 'LEU_ILE_VAL': 68, 'VAL': 69}
    assert condition in condition_values.keys(), f"Condition {condition} not in {condition_values.keys()}"
    
    # model name and output folder path
    model_loc = pkg_resources.resource_filename('riboclette', 'checkpoints/XLNet-PLabelDH_S2/best_model/')

    # model parameters
    d_model_val = 512
    n_layers_val = 6
    n_heads_val = 4
    dropout_val = 0.1

    config = XLNetConfig(vocab_size=71, pad_token_id=70, d_model = d_model_val, n_layer = n_layers_val, n_head = n_heads_val, d_inner = d_model_val, num_labels = 1, dropout=dropout_val) # 64*6 tokens + 1 for padding
    model = XLNetDH(config)

    # load model from the saved model
    model = model.from_pretrained(model_loc)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # set model to evaluation mode
    model.eval()

    X = sequence2codonids(sequence)
    X = [int(k) for k in X]

    # prepend condition token
    X = [condition_values[condition]] + X

    X = np.asarray(X)
    X = torch.from_numpy(X).long()

    if head == 'CTRL':
        lig_sample_ctrl = lig_output(model, X, mode='ctrl')
        return lig_sample_ctrl
    elif head == 'DD':
        lig_sample_dd = lig_output(model, X, mode='dd')
        return lig_sample_dd
        