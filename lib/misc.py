import math
import hashlib
import sys
from collections import OrderedDict
from numbers import Number
import operator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        if isinstance(self.underlying_dataset.data, np.ndarray):
            self.data = self.underlying_dataset.data[self.keys]
            self.shape=self.data.shape
        elif isinstance(self.underlying_dataset.data, list):
            self.data = [self.underlying_dataset.data[i] for i in self.keys]
            self.shape=(len(self.data),3,224,224)
        else:
            raise TypeError("Unsupported data type for splitting")
        self.targets = self.underlying_dataset.targets[self.keys]
    def __getitem__(self, key):
        idx, img, target = self.underlying_dataset[self.keys[key]]
        return key, img, target
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, n, seed=0): 
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)
'''

def accuracy(network, loader, device):
    correct = 0
    total = 0
    network.eval() 
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)
            batch_weights = torch.ones(len(x))
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total
'''
def compute_metrics(network, loader, device):
    """
    Compute Accuracy, Precision, Recall, F1, AUC.
    Args:
        network: The model.
        loader: DataLoader (inputs x and labels y).
        device: Device (e.g. 'cuda' or 'cpu').
    Returns:
        Dict with Accuracy, Precision, Recall, F1, AUC.
    """
    network.eval()
    all_preds = []
    all_labels = []
    all_scores = []  # Positive-class scores for AUC

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)
            
            # Predicted class (argmax) and positive-class score
            preds = p.argmax(1).cpu().numpy()  # Predicted class (0 or 1)
            scores = p[:, 1].cpu().numpy()     # Positive-class score for AUC
            labels = y.cpu().numpy()           # Ground-truth labels
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_scores.extend(scores)

    network.train()

    # Compute metrics
    try:
        metrics = {
            'Accuracy': accuracy_score(all_labels, all_preds),
            'Precision': precision_score(all_labels, all_preds, zero_division=0),
            'Recall': recall_score(all_labels, all_preds),
            'F1': f1_score(all_labels, all_preds),
            'AUC': roc_auc_score(all_labels, all_scores)
        }
    except:
        metrics = {
            'Accuracy': accuracy_score(all_labels, all_preds),
            'Precision': precision_score(all_labels, all_preds, zero_division=0),
            'Recall': recall_score(all_labels, all_preds),
            'F1': f1_score(all_labels, all_preds),
            'AUC': 0.5
        }
    return metrics
    
class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

def val_oracle_accuracy(network, p_loader, u_loader, device, u_type):
    correct_p = 0
    correct_u = 0
    total_p = 0
    total_u = 0

    network.eval()
    with torch.no_grad():
        for _, (inputs_p_w, inputs_p_s), targets_p in p_loader:
            inputs_p_w = inputs_p_w.to(device)
            targets_p = targets_p.to(device)
            p = network.predict(inputs_p_w)
            if p.size(1) == 1:
                correct_p += p.gt(0.5).eq(targets_p).float().sum().item()
            else:
                correct_p += p.argmax(1).eq(targets_p).float().sum().item()
            total_p += len(inputs_p_w)
        for _, (inputs_u_w, inputs_u_s), (targets_u, targets_u_true) in u_loader:
            inputs_u_w = inputs_u_w.to(device)
            targets_u_true = targets_u_true.to(device)
            p = network.predict(inputs_u_w)
            if p.size(1) == 1:
                correct_u += p.gt(0.5).eq(targets_u_true).float().sum().item()
            else:
                correct_u += p.argmax(1).eq(targets_u_true).float().sum().item()
            total_u += len(inputs_u_w)
    network.train()

    #return correct_p / total_p, correct_u / total_u, (correct_p + correct_u) / (total_p + total_u)
    if u_type == 'one_sample':
        return (correct_p + correct_u) / (total_p + total_u)
    else:
        return correct_u / total_u

def val_proxy_accuracy(network, p_loader, u_loader, device, pi, u_type):
    correct_p = 0
    correct_u = 0
    total_p = 0
    total_u = 0

    network.eval()
    with torch.no_grad():
        for _, (inputs_p_w, inputs_p_s), targets_p in p_loader:
            inputs_p_w = inputs_p_w.to(device)
            targets_p = targets_p.to(device)
            p = network.predict(inputs_p_w)
            if p.size(1) == 1:
                correct_p += p.gt(0.5).eq(targets_p).float().sum().item()
            else:
                correct_p += p.argmax(1).eq(targets_p).float().sum().item()
            total_p += len(inputs_p_w)
        for _, (inputs_u_w, inputs_u_s), (targets_u, _) in u_loader:
            inputs_u_w = inputs_u_w.to(device)
            targets_u = targets_u.to(device)
            p = network.predict(inputs_u_w)
            if p.size(1) == 1:
                correct_u += p.gt(0.5).eq(targets_u).float().sum().item()
            else:
                correct_u += p.argmax(1).eq(targets_u).float().sum().item()
            total_u += len(inputs_u_w)
    network.train()

    if u_type == 'one_sample':
        return 2 * pi * correct_p / total_p + (correct_p + correct_u) / (total_p + total_u)
    else:
        return 2 * pi * correct_p / total_p + correct_u / total_u

def val_proxy_auc(network, p_loader, u_loader, device):
    network.eval()
    scores, labels = [], []

    with torch.no_grad():
        # Positive samples (label=1): use second column of predict (positive-class score)
        for _, (inputs_p_w, _), _ in p_loader:
            inputs_p_w = inputs_p_w.to(device)
            p = network.predict(inputs_p_w)[:, 1].cpu().numpy()  # Positive-class score
            scores.extend(p)
            labels.extend(np.ones_like(p))  # Label 1

        # Negative samples (label=0): same, use positive-class score
        for _, (inputs_u_w, _), _ in u_loader:
            inputs_u_w = inputs_u_w.to(device)
            u = network.predict(inputs_u_w)[:, 1].cpu().numpy()  # Positive-class score
            scores.extend(u)
            labels.extend(np.zeros_like(u))  # Label 0

    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = 0.5 
    network.train()
    return auc
# def val_accuracy(network, loader, device):
#     correct = 0
#     total = 0
#     network.eval()
#     with torch.no_grad():
#         for x, x_weak, x_strong, x_distill, partial_y, y, _ in loader:
#             x = x.to(device)
#             partial_y = partial_y.to(device)
#             y = y.to(device)
#             p = network.predict(x)
#             if p.size(1) == 1:
#                 correct += p.gt(0).eq(y).float().sum().item()
#             else:
#                 correct += p.argmax(1).eq(y).float().sum().item()
#             total += len(x)
#     network.train()
#
#     return correct / total

'''
def val_covering_rate(network, loader, device):
    correct = 0
    total = 0
    network.eval()
    with torch.no_grad():
        for x, x_weak, x_strong, x_distill, partial_y, _, _ in loader:
            x = x.to(device)
            partial_y = partial_y.to(device)
            p = network.predict(x)
            predicted_label = p.argmax(1)
            covering_per_example = partial_y[torch.arange(len(x)), predicted_label]
            correct += covering_per_example.sum().item()
            total += len(x)
    network.train()

    return correct / total
'''
def gen_configs(args):
    if args.dataset == 'CIFAR10':
        args.num_classes_original = 10
        args.pi = 0.5
        set_mode, positive_label_mode = args.setting.split('_')[0], args.setting.split('_')[1]
        if positive_label_mode == '1':
            args.positive_label_list = [0, 1, 2, 8, 9]
        elif positive_label_mode == '2':
            args.positive_label_list = [2, 3, 5, 7, 9]
        else:
            raise ValueError("Unknown setting for CIFAR10 in set1: {}".format(args.setting))

        if set_mode == 'set1':
            args.num_p = 2000
            args.num_p_and_u = 20000
            args.u_type = 'one_sample'
        elif set_mode == 'set2':    
            args.num_p = 4000
            args.num_p_and_u = 20000
            args.u_type = 'one_sample'
        elif set_mode == 'set3':
            args.num_p = 6000
            args.num_p_and_u = 20000
            args.u_type = 'one_sample'
        elif set_mode == 'set4':
            args.num_p = 8000
            args.num_p_and_u = 20000
            args.u_type = 'one_sample'
        elif set_mode == 'set5':
            args.num_p = 10000
            args.num_p_and_u = 20000
            args.u_type = 'one_sample'
        elif set_mode == 'set6':
            args.num_p = 2000
            args.num_u = 18000
            args.u_type = 'two_sample'
        elif set_mode == 'set7':
            args.num_p = 4000
            args.num_u = 16000
            args.u_type = 'two_sample'
        elif set_mode == 'set8':
            args.num_p = 6000
            args.num_u = 14000
            args.u_type = 'two_sample'
        elif set_mode == 'set9':
            args.num_p = 8000
            args.num_u = 12000
            args.u_type = 'two_sample'
        elif set_mode == 'set10':
            args.num_p = 10000
            args.num_u = 10000
            args.u_type = 'two_sample'
    elif args.dataset == "IMAGENETTE":
        args.num_classes_original = 10
        args.pi = 0.5
        set_mode, positive_label_mode = args.setting.split('_')[0], args.setting.split('_')[1]
        if positive_label_mode == '1':
            args.positive_label_list = [0, 1, 2, 8, 9]
        elif positive_label_mode == '2':
            args.positive_label_list = [2, 3, 5, 7, 9]
        else:
            raise ValueError("Unknown setting for CIFAR10 in set1: {}".format(args.setting))

        if set_mode == 'set1':
            args.num_p = 600
            args.num_p_and_u = 6000
            args.u_type = 'one_sample'
        elif set_mode == 'set2':    
            args.num_p = 1200
            args.num_p_and_u = 6000
            args.u_type = 'one_sample'
        elif set_mode == 'set3':
            args.num_p = 1800
            args.num_p_and_u = 6000
            args.u_type = 'one_sample'
        elif set_mode == 'set4':
            args.num_p = 2400
            args.num_p_and_u = 6000
            args.u_type = 'one_sample'
        elif set_mode == 'set5':
            args.num_p = 3000
            args.num_p_and_u = 6000
            args.u_type = 'one_sample'
        elif set_mode == 'set6':
            args.num_p = 600
            args.num_u = 5400
            args.u_type = 'two_sample'
        elif set_mode == 'set7':
            args.num_p = 1200
            args.num_u = 4800
            args.u_type = 'two_sample'
        elif set_mode == 'set8':
            args.num_p = 1800
            args.num_u = 4200
            args.u_type = 'two_sample'
        elif set_mode == 'set9':
            args.num_p = 2400
            args.num_u = 3600
            args.u_type = 'two_sample'
        elif set_mode == 'set10':
            args.num_p = 3000
            args.num_u = 3000
            args.u_type = 'two_sample'
    elif args.dataset == 'USPS':
        args.num_classes_original = 10
        args.pi = 0.5
        set_mode, positive_label_mode = args.setting.split('_')[0], args.setting.split('_')[1]
        if positive_label_mode == '1':
            args.positive_label_list = [4, 7, 9, 5, 8]  # 3039 / 7291
        elif positive_label_mode == '2':
            args.positive_label_list = [1, 6, 4, 9, 8]  # 3783/ 7291
        else:
            raise ValueError("Unknown setting for CIFAR10 in set1: {}".format(args.setting))

        if set_mode == 'set1':
            args.num_p = 400
            args.num_p_and_u = 4000
            args.u_type = 'one_sample'
        elif set_mode == 'set2':    
            args.num_p = 800
            args.num_p_and_u = 4000
            args.u_type = 'one_sample'
        elif set_mode == 'set3':
            args.num_p = 1200
            args.num_p_and_u = 4000
            args.u_type = 'one_sample'
        elif set_mode == 'set4':
            args.num_p = 1600
            args.num_p_and_u = 4000
            args.u_type = 'one_sample'
        elif set_mode == 'set5':
            args.num_p = 2000
            args.num_p_and_u = 4000
            args.u_type = 'one_sample'
        elif set_mode == 'set6':
            args.num_p = 400
            args.num_u = 3600
            args.u_type = 'two_sample'
        elif set_mode == 'set7':
            args.num_p = 800
            args.num_u = 3200
            args.u_type = 'two_sample'
        elif set_mode == 'set8':
            args.num_p = 1200
            args.num_u = 2800
            args.u_type = 'two_sample'
        elif set_mode == 'set9':
            args.num_p = 1600
            args.num_u = 2400
            args.u_type = 'two_sample'
        elif set_mode == 'set10':
            args.num_p = 2000
            args.num_u = 2000
            args.u_type = 'two_sample'
    elif args.dataset == 'Letter':
        args.num_classes_original = 26
        args.pi = 0.5
        set_mode, positive_label_mode = args.setting.split('_')[0], args.setting.split('_')[1]
        if positive_label_mode == '1':
            args.positive_label_list = ['B', 'V', 'L', 'R', 'I', 'O', 'W', 'S', 'J', 'K', 'C', 'H', 'Z']  # 9747/ 20000
        elif positive_label_mode == '2':
            args.positive_label_list = ['D', 'T', 'A', 'Y', 'Q', 'G', 'B', 'L', 'I', 'W', 'J', 'C', 'Z']  # 9983/ 20000
        else:
            raise ValueError("Unknown setting for CIFAR10 in set1: {}".format(args.setting))

        if set_mode == 'set1':
            args.num_p = 1300
            args.num_p_and_u = 13000
            args.u_type = 'one_sample'
        elif set_mode == 'set2':    
            args.num_p = 2600
            args.num_p_and_u = 13000
            args.u_type = 'one_sample'
        elif set_mode == 'set3':
            args.num_p = 3900
            args.num_p_and_u = 13000
            args.u_type = 'one_sample'
        elif set_mode == 'set4':
            args.num_p = 5200
            args.num_p_and_u = 13000
            args.u_type = 'one_sample'
        elif set_mode == 'set5':
            args.num_p = 6500
            args.num_p_and_u = 13000
            args.u_type = 'one_sample'
        elif set_mode == 'set6':
            args.num_p = 1300
            args.num_u = 11700
            args.u_type = 'two_sample'
        elif set_mode == 'set7':
            args.num_p = 2600
            args.num_u = 10400
            args.u_type = 'two_sample'
        elif set_mode == 'set8':
            args.num_p = 3900
            args.num_u = 9100
            args.u_type = 'two_sample'
        elif set_mode == 'set9':
            args.num_p = 5200
            args.num_u = 7800
            args.u_type = 'two_sample'
        elif set_mode == 'set10':
            args.num_p = 6500
            args.num_u = 6500
            args.u_type = 'two_sample'

    elif args.dataset == 'Creditcard':
        args.num_classes_original = 2
        args.pi = 0.5
        set_mode, positive_label_mode = args.setting.split('_')[0], args.setting.split('_')[1]
        if positive_label_mode == '1':
            args.positive_label_list = [1]  # 284315 / 284807
        elif positive_label_mode == '2':
            args.positive_label_list = [0]  # 492/ 284807
        else:
            raise ValueError("Unknown setting for CIFAR10 in set1: {}".format(args.setting))

        if set_mode == 'set1':
            args.num_p = 60
            args.num_p_and_u = 600
            args.u_type = 'one_sample'
        elif set_mode == 'set2':    
            args.num_p = 120
            args.num_p_and_u = 600
            args.u_type = 'one_sample'
        elif set_mode == 'set3':
            args.num_p = 180
            args.num_p_and_u = 600
            args.u_type = 'one_sample'
        elif set_mode == 'set4':
            args.num_p = 240
            args.num_p_and_u = 600
            args.u_type = 'one_sample'
        elif set_mode == 'set5':
            args.num_p = 300
            args.num_p_and_u = 600
            args.u_type = 'one_sample'
        elif set_mode == 'set6':
            args.num_p = 60
            args.num_u = 540
            args.u_type = 'two_sample'
        elif set_mode == 'set7':
            args.num_p = 120
            args.num_u = 480
            args.u_type = 'two_sample'
        elif set_mode == 'set8':
            args.num_p = 180
            args.num_u = 420
            args.u_type = 'two_sample'
        elif set_mode == 'set9':
            args.num_p = 240
            args.num_u = 360
            args.u_type = 'two_sample'
        elif set_mode == 'set10':
            args.num_p = 300
            args.num_u = 300
            args.u_type = 'two_sample'

    return args




