import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
import random
import scipy.sparse as sp
import os
import jenkspy
import pandas as pd
# from mindspore import EarlyStopping
import transformers
from sklearn.metrics import euclidean_distances
# nfa is adopted from GWLS
from copy import deepcopy
from .nfa import create_proportion_graph
from . import networks
from torch.optim.lr_scheduler import MultiStepLR

ALGORITHMS = [
    'uPU',
    'nnPU',
    'nnPU_GA',
    'PUbN',
    'VPU',
    'PAN',
    'CVIR',
    'Dist_PU',
    'P3MIX_E',
    'P3MIX_C',
    'Count_Loss',
    'Robust_PU',
    'HolisticPU',
    'GLWS',
    'PUSB',
    'LBE',
    'PUe',
    'PULDA'
]

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# @torch.no_grad()
# def mixup_two_target(x, y, alpha=1.0, is_bias=False):
#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1
#     if is_bias:
#         lam = max(lam, 1 - lam)
#
#     index = torch.randperm(x.size(0)).to(x.device)
#
#     mixed_x = lam * x + (1 - lam) * x[index]
#     y_a, y_b = y, y[index]
#     return mixed_x, y_a, y_b, lam


@torch.no_grad()
def mixup_one_target(data_a, data_b, target_a, target_b, alpha=1.0, is_bias=False):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    if is_bias:
        lam = max(lam, 1 - lam)

    mixed_data = lam * data_a + (1 - lam) * data_b
    mixed_target = lam * target_a + (1 - lam) * target_b
    return mixed_data, mixed_target, lam


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

## WDD - top ##
class Algorithm(nn.Module):
    """
    A subclass of Algorithm implements a partial-label learning algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, p_input_shape, u_input_shape, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams
        self.num_classes = 2
        self.pi = hparams["pi"]

        self.p_input_shape = p_input_shape
        self.u_input_shape = u_input_shape

        self.featurizer = networks.Featurizer(p_input_shape, self.hparams)
        if self.hparams['dataset']== 'IMAGENETTE':
            self.classifier = networks.Classifier(256, self.num_classes)
        elif self.hparams['dataset'] in ['CIFAR10','USPS','Letter','Creditcard']:
            self.classifier = networks.Classifier(self.featurizer.n_outputs, self.num_classes)
        else:
            pass
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.hparams["lr"], momentum=self.hparams["momentum"])
        self.base_loss = nn.CrossEntropyLoss()

    def interleave(self, x, size):
        s = list(x.shape)
        return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])
    def de_interleave(self, x, size):
        s = list(x.shape)
        return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])
    def output_after_interleave(self,x,batch_size):
        inputs=self.interleave(x,2)
        outputs=self.network(inputs)
        outputs=self.de_interleave(outputs,2)
        outputs_p_w, outputs_u_w=outputs[:batch_size], outputs[batch_size:]
        return outputs_p_w, outputs_u_w

    def update(self, inputs_p_w, inputs_p_s, targets_p, inputs_u_w, inputs_u_s, targets_u):
        """
        Perform one update step
        """
        raise NotImplementedError

    def predict(self, x):
        return self.network(x)
"""
class Algorithm(torch.nn.Module):
    def __init__(self, p_input_shape, u_input_shape, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams
        #self.num_p = p_input_shape[0]
        #self.num_u = u_input_shape[0]
        self.num_classes = 2
        self.pi = hparams["pi"]

    def interleave(self, x, size):
        s = list(x.shape)
        return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


    def de_interleave(self, x, size):
        s = list(x.shape)
        return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

    def update(self, inputs_p_w, inputs_p_s, targets_p, inputs_u_w, inputs_u_s, targets_u):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
"""
## WDD - bottom ##

class uPU(Algorithm):
    """
    uPU
    Reference: Convex Formulation for Learning from Positive and Unlabeled Data, ICML 2015.
    """

    def __init__(self, p_input_shape, u_input_shape, hparams):
        super(uPU, self).__init__(p_input_shape, u_input_shape, hparams)
        ## WDD - top ##
        """
        self.featurizer = networks.Featurizer(p_input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            self.num_classes)

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.hparams["lr"],
            momentum=self.hparams["momentum"]
        )
        self.base_loss = nn.CrossEntropyLoss()
        """
        ## WDD - bottom ##

    def update(self, inputs_p_w, targets_p, inputs_u_w, targets_u):
        ## WDD - top ##
        outputs_p_w, outputs_u_w=self.output_after_interleave(torch.cat((inputs_p_w, inputs_u_w)), inputs_p_w.shape[0])
        """
        batch_size = inputs_p_w.shape[0]
        inputs_all = self.interleave(torch.cat((inputs_p_w, inputs_u_w)), 2)
        outputs_all = self.predict(inputs_all)
        outputs_all = self.de_interleave(outputs_all, 2)
        outputs_p_w = outputs_all[:batch_size]
        outputs_u_w = outputs_all[batch_size:]
        """
        ## WDD - bottom ##
        loss = self.upu_loss(outputs_p_w, outputs_u_w, targets_p, targets_u)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    def upu_loss(self, outputs_p_w, outputs_u_w, targets_p, targets_u):
        loss_pos_pos = self.pi * self.base_loss(outputs_p_w, targets_p)
        loss_unlabel_neg = self.base_loss(outputs_u_w, targets_u)
        targets_p_reverse = (1 - targets_p).to(targets_p.dtype)
        loss_pos_neg = - self.pi * self.base_loss(outputs_p_w, targets_p_reverse)
        average_loss = loss_pos_pos + loss_unlabel_neg + loss_pos_neg
        return average_loss

class nnPU(Algorithm):
    """
    nnPU
    Reference: Positive-Unlabeled Learning with Non-Negative Risk Estimator, NIPS 2017.
    """

    def __init__(self, p_input_shape, u_input_shape, hparams):
        super(nnPU, self).__init__(p_input_shape, u_input_shape, hparams)
        ## WDD - top ##
        """
        self.featurizer = networks.Featurizer(p_input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            self.num_classes)

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.hparams["lr"],
            momentum=self.hparams["momentum"]
        )
        self.base_loss = nn.CrossEntropyLoss()
        """
        ## WDD - bottom ##

    def update(self, inputs_p_w, targets_p, inputs_u_w, targets_u):
        ## WDD - top ##
        outputs_p_w, outputs_u_w = self.output_after_interleave(torch.cat((inputs_p_w, inputs_u_w)), inputs_p_w.shape[0])
        """
        batch_size = inputs_p_w.shape[0]
        inputs_all = self.interleave(torch.cat((inputs_p_w, inputs_u_w)), 2)
        outputs_all = self.predict(inputs_all)
        outputs_all = self.de_interleave(outputs_all, 2)
        outputs_p_w = outputs_all[:batch_size]
        outputs_u_w = outputs_all[batch_size:]
        """
        ## WDD - bottom ##
        loss = self.nnpu_loss(outputs_p_w, outputs_u_w, targets_p, targets_u)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    def nnpu_loss(self, outputs_p_w, outputs_u_w, targets_p, targets_u):
        loss_pos_pos = self.pi * self.base_loss(outputs_p_w, targets_p)
        loss_unlabel_neg = self.base_loss(outputs_u_w, targets_u)
        targets_p_reverse = (1 - targets_p).to(targets_p.dtype)
        loss_pos_neg = - self.pi * self.base_loss(outputs_p_w, targets_p_reverse)
        self.lda = torch.tensor([0.0]).to(outputs_p_w.device)
        average_loss = loss_pos_pos + torch.max(self.lda, loss_unlabel_neg + loss_pos_neg)
        return average_loss

class nnPU_GA(Algorithm):
    """
    nnPU
    Reference: Positive-Unlabeled Learning with Non-Negative Risk Estimator, NIPS 2017.
    """

    def __init__(self, p_input_shape, u_input_shape, hparams):
        super(nnPU_GA, self).__init__(p_input_shape, u_input_shape, hparams)
        ## WDD - top ##
        """
        self.featurizer = networks.Featurizer(p_input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            self.num_classes)

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.hparams["lr"],
            momentum=self.hparams["momentum"]
        )
        self.base_loss = nn.CrossEntropyLoss()
        """
        ## WDD - bottom ##

    def update(self, inputs_p_w, targets_p, inputs_u_w, targets_u):
        ## WDD - top ##
        outputs_p_w, outputs_u_w = self.output_after_interleave(torch.cat((inputs_p_w, inputs_u_w)),inputs_p_w.shape[0])
        """
        batch_size = inputs_p_w.shape[0]
        inputs_all = self.interleave(torch.cat((inputs_p_w, inputs_u_w)), 2)
        outputs_all = self.predict(inputs_all)
        outputs_all = self.de_interleave(outputs_all, 2)
        outputs_p_w = outputs_all[:batch_size]
        outputs_u_w = outputs_all[batch_size:]
        """
        ## WDD - bottom ##
        loss = self.upu_loss(outputs_p_w, outputs_u_w, targets_p, targets_u)
        self.optimizer.zero_grad()
        loss.backward()
        if loss.item() < 0:
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    p.grad = -1*p.grad
        self.optimizer.step()
        return {'loss': loss.item()}

    def upu_loss(self, outputs_p_w, outputs_u_w, targets_p, targets_u):
        # R_p^+
        loss_pos_pos = self.pi * self.base_loss(outputs_p_w, targets_p)
        # R_u^_
        loss_unlabel_neg = self.base_loss(outputs_u_w, targets_u)
        # R_p^_
        targets_p_reverse = (1 - targets_p).to(targets_p.dtype)
        loss_pos_neg = - self.pi * self.base_loss(outputs_p_w, targets_p_reverse) 
        average_loss = loss_pos_pos + loss_unlabel_neg + loss_pos_neg
        return average_loss   


class PUbN(Algorithm):
    """
    PUbN
    Reference: Classification from Positive, Unlabeled and Biased Negative Data
    """
    
    def __init__(self, p_input_shape, u_input_shape, hparams):
        super(PUbN,self).__init__(p_input_shape, u_input_shape, hparams)
        self.rho=0.2
        self.tau=hparams['tau']
        self.u_numbers=u_input_shape[0]
        self.pi=0.5
        self.base_loss = nn.CrossEntropyLoss(reduction="none")


    def update(self, inputs_p_w, targets_p, inputs_u_w, targets_u):
        outputs_p_w, outputs_u_w = self.output_after_interleave(torch.cat((inputs_p_w, inputs_u_w)), inputs_p_w.shape[0])
        loss = self.pubn_loss(outputs_p_w, outputs_u_w, targets_p, targets_u)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    def get_eta(self,sigma):
        with torch.no_grad():
            k = (self.tau * (1 - self.pi - self.rho) * self.u_numbers)
            q = k / sigma.numel()   # Convert to quantile
            q = min(max(q, 0.0), 1.0)       # Clamp to [0, 1]
            eta = torch.quantile(sigma, q).item()
        return eta

    def pubn_loss(self, outputs_p_w, outputs_u_w, targets_p, targets_u):
        loss_pos_pos = self.pi * self.base_loss(outputs_p_w, targets_p)
        loss_pos_pos = loss_pos_pos.mean()

        sigma_unlabel=torch.softmax(outputs_u_w,dim=1)[:,1].detach()
        eta=self.get_eta(sigma_unlabel)
        mask=sigma_unlabel <= eta
        if mask.any():
            loss_unlabel_neg = self.base_loss(outputs_u_w, targets_u)
            loss_unlabel_neg = loss_unlabel_neg[mask]*(1-sigma_unlabel[mask])
            loss_unlabel_neg = loss_unlabel_neg.mean()
        else:
            loss_unlabel_neg = torch.tensor(0.0).cuda()

        sigma_pos=torch.softmax(outputs_p_w, dim=1)[:,1].detach()
        reverse_mask = sigma_pos > eta
        if reverse_mask.any():
            targets_p_reverse = (1 - targets_p).to(targets_p.dtype)
            loss_pos_neg = - self.pi * self.base_loss(outputs_p_w, targets_p_reverse)[reverse_mask]*(1-sigma_pos[reverse_mask])/(sigma_pos[reverse_mask])
            loss_pos_neg=loss_pos_neg.mean()
        else:
            loss_pos_neg = 0
        
        average_loss = loss_pos_pos + loss_unlabel_neg + loss_pos_neg
        return average_loss


class GLWS(Algorithm):
    """
    nnPU
    Reference: A General Framework for Learning from Weak Supervision, ICML 2024.
    """

    def __init__(self, p_input_shape, u_input_shape, hparams):
        super(GLWS, self).__init__(p_input_shape, u_input_shape, hparams)
        ## WDD - top ##
        """
        self.featurizer = networks.Featurizer(p_input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            self.num_classes)

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.hparams["lr"],
            momentum=self.hparams["momentum"]
        )
        self.base_loss = nn.CrossEntropyLoss()
        """
        ## WDD - bottom ##

    ## WDD - top ##
    def update(self, inputs_p_w, targets_p, inputs_u_w, targets_u):
        outputs_p_w, outputs_u_w = self.output_after_interleave(torch.cat((inputs_p_w, inputs_u_w)), inputs_p_w.shape[0])
        loss = self.glws_loss(outputs_p_w, outputs_u_w, targets_p)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}
    """
    def update(self, inputs_p_w, inputs_p_s, targets_p, inputs_u_w, inputs_u_s, targets_u):
            batch_size = inputs_p_w.shape[0]
            inputs_all = self.interleave(torch.cat((inputs_p_w, inputs_u_w)), 2)
            outputs_all = self.predict(inputs_all)
            outputs_all = self.de_interleave(outputs_all, 2)
            outputs_p_w = outputs_all[:batch_size]
            outputs_u_w = outputs_all[batch_size:]
    
            inputs_all_s = self.interleave(torch.cat((inputs_p_s, inputs_u_s)), 2)
            outputs_all_s = self.predict(inputs_all_s)
            outputs_all_s = self.de_interleave(outputs_all_s, 2)
            outputs_p_s = outputs_all_s[:batch_size]
            outputs_u_s = outputs_all_s[batch_size:]
    
    
            loss = self.glws_loss(outputs_p_w, outputs_u_w, outputs_u_s, targets_p, targets_u)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return {'loss': loss.item()}
    """
    ## WDD - bottom ##

    def glws_loss(self, outputs_p_w, outputs_u_w, targets_p):
        # calculate labeled loss
        lb_sup_loss = self.base_loss(outputs_p_w, targets_p)
        num_ulb = outputs_u_w.shape[0]

        # forward-backward algorithm
        with torch.no_grad():
            probs_x_ulb_w = outputs_u_w.softmax(dim=1)
            pseudo_probs_x, count_probs_x = create_proportion_graph(torch.log(probs_x_ulb_w), int(self.pi * num_ulb))

        # calculate unsup loss
        ## WDD - top ##
        unsup_loss = self.base_loss(outputs_u_w, pseudo_probs_x)
        """
        unsup_loss = self.base_loss(outputs_u_s, pseudo_probs_x)
        """
        ## WDD- bottom ##
        total_loss = lb_sup_loss + unsup_loss
        return total_loss


class VPU(Algorithm):
    """
    VPU
    Reference: TBA
    """

    def __init__(self, p_input_shape, u_input_shape, hparams):
        super(VPU, self).__init__(p_input_shape, u_input_shape, hparams)
        self.loss_weight_mixup = 0.03
        self.mixup_alpha= 0.3
        ## WDD - top ##
        """
        self.featurizer = networks.Featurizer(p_input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            self.num_classes)

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.hparams["lr"],
            momentum=self.hparams["momentum"]
        )
        self.base_loss = nn.CrossEntropyLoss()
        """
        ## WDD - bottom ##

    def update(self, inputs_p_w, targets_p, inputs_u_w, targets_u):
        ## WDD - top ##
        outputs_p_w, outputs_u_w = self.output_after_interleave(torch.cat((inputs_p_w, inputs_u_w)),inputs_p_w.shape[0])
        """
        batch_size = inputs_p_w.shape[0]
        inputs_all = self.interleave(torch.cat((inputs_p_w, inputs_u_w)), 2)
        outputs_all = self.predict(inputs_all)
        outputs_all = self.de_interleave(outputs_all, 2)
        outputs_p_w = outputs_all[:batch_size]
        outputs_u_w = outputs_all[batch_size:]
        """
        ## WDD - bottom ##
        loss = self.vpu_loss(outputs_p_w, outputs_u_w, targets_p, targets_u,inputs_p_w,inputs_u_w)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    def vpu_loss(self, outputs_p_w, outputs_u_w, targets_p, targets_u,x_lb,x_ulb):
        # get probability
        log_probs_x_lb = outputs_p_w.log_softmax(dim=1)[:, 1]
        log_probs_x_ulb = outputs_u_w.log_softmax(dim=1)[:, 1]

        # compute varitional loss
        var_loss = torch.logsumexp(log_probs_x_ulb, dim=0).mean() - math.log(len(log_probs_x_ulb)) - torch.mean(
            log_probs_x_lb)

        # mixup regularization
        with torch.no_grad():
            target_x_ulb = outputs_u_w.softmax(dim=1)[:, 1]
            target_x_lb = targets_p
            rand_perm = torch.randperm(x_lb.size(0))
            x_lb_perm, target_x_lb_perm = x_lb[rand_perm], target_x_lb[rand_perm]
            m = torch.distributions.beta.Beta(self.mixup_alpha, self.mixup_alpha)
            lam = m.sample()
            ## WDD - top ##
            x_ulb, target_x_ulb = x_ulb[:x_lb.size(0)], target_x_ulb[:x_lb.size(0)]
            ## WDD - bottom ##
            mixed_x = lam * x_ulb + (1 - lam) * x_lb_perm
            mixed_y = lam * target_x_ulb + (1 - lam) * target_x_lb_perm
        mixed_logits = self.predict(mixed_x)
        reg_mix_loss = ((torch.log(mixed_y) - mixed_logits[:, 1]) ** 2).mean()

        # calculate total loss
        total_loss = var_loss + self.loss_weight_mixup * reg_mix_loss
        return total_loss

class Count_Loss(Algorithm):
    """
    Count_Loss
    Reference: TBA.
    """

    def __init__(self, p_input_shape, u_input_shape, hparams):
        super(Count_Loss, self).__init__(p_input_shape, u_input_shape, hparams)
        ## WDD - top ##
        """
        self.featurizer = networks.Featurizer(p_input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            self.num_classes)

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.hparams["lr"],
            momentum=self.hparams["momentum"]
        )
        self.base_loss = nn.CrossEntropyLoss()
        """
        ## WDD - bottom ##

    def dp_count_proportion(self,log_probs, targets):
        batch_size, _ = log_probs.shape

        # dynamic programming initialization
        # where the column is the number of positive instances and row is the idx of samples
        log_alpha = torch.full((batch_size + 1, batch_size + 1), -float("Inf"), device=log_probs.device)
        log_alpha[0, 0] = 0

        for i in range(1, batch_size + 1):
            for j in range(batch_size + 1):
                alpha_plus_zero = log_alpha[i - 1, j] + log_probs[i - 1, 0]
                alpha_plus_one = log_alpha[i - 1, j - 1] + log_probs[i - 1, 1]

                # Mask to check if alpha_plus_zero and alpha_plus_one are -inf
                alpha_plus_zero[alpha_plus_zero == -float("Inf")] = -1e10
                alpha_plus_one[alpha_plus_one == -float("Inf")] = -1e10

                log_alpha[i, j] = torch.logsumexp(torch.stack([alpha_plus_zero, alpha_plus_one], dim=-1), dim=-1)

        m = torch.logsumexp(log_alpha[-1, :], dim=0)
        alpha = torch.exp(log_alpha[-1, :] - m)
        alpha = alpha / (alpha.sum() + 1e-6)
        count_prob = alpha[targets].view(1, -1)
        return count_prob
    def update(self, inputs_p_w, targets_p, inputs_u_w, targets_u):
        ## WDD - top ##
        outputs_p_w, outputs_u_w = self.output_after_interleave(torch.cat((inputs_p_w, inputs_u_w)),inputs_p_w.shape[0])
        """
        batch_size = inputs_p_w.shape[0]
        inputs_all = self.interleave(torch.cat((inputs_p_w, inputs_u_w)), 2)
        outputs_all = self.predict(inputs_all)
        outputs_all = self.de_interleave(outputs_all, 2)
        outputs_p_w = outputs_all[:batch_size]
        outputs_u_w = outputs_all[batch_size:]
        """
        ## WDD - bottom ##
        loss = self.count_loss(outputs_p_w, outputs_u_w, targets_p, targets_u)
        #t = torch.randint(1,5,(1,))
        #if t==2:
        #    loss = torch.tensor(float('nan'))*loss
        self.optimizer.zero_grad()
        if torch.isnan(loss):
            print("NaN loss detected, skipping backward pass.")
            #loss.backward()
            #self.optimizer.step()
        else:
            loss.backward()
            # gradient clip
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=10.0)
            self.optimizer.step()
        return {'loss': loss.item()}

    def count_loss(self, outputs_p_w, outputs_u_w, targets_p, targets_u):
        num_lb = outputs_p_w.shape[0]
        num_ulb = outputs_u_w.shape[0]

        logits_x_lb = outputs_p_w
        logits_x_ulb = outputs_u_w

        # calculate labeled loss
        sup_loss = self.base_loss(logits_x_lb, targets_p)

        # get probs for unlabeled
        probs_x_ulb = logits_x_ulb.softmax(dim=1)
        # get count probability

        count_prob = self.dp_count_proportion(torch.log(probs_x_ulb), int(num_ulb * self.pi))
        # get unsup loss
        unsup_loss = - torch.log(count_prob).mean()
        total_loss = sup_loss + unsup_loss
        # print(unsup_loss,sup_loss)

        return total_loss



class CVIR(Algorithm):
    """
    CVIR
    Reference: TBA.
    """

    def __init__(self, p_input_shape, u_input_shape, hparams):
        super(CVIR, self).__init__(p_input_shape, u_input_shape, hparams)
        ## WDD - top ##
        """
        self.featurizer = networks.Featurizer(p_input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            self.num_classes)

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.hparams["lr"],
            momentum=self.hparams["momentum"]
        )
        self.base_loss = nn.CrossEntropyLoss()
        """
        ## WDD - bottom ##

    def update(self, inputs_p_w, targets_p, inputs_u_w, targets_u, idx_ulb):
        ## WDD - top ##
        outputs_p_w, outputs_u_w = self.output_after_interleave(torch.cat((inputs_p_w, inputs_u_w)), inputs_p_w.shape[0])
        """
        batch_size = inputs_p_w.shape[0]
        inputs_all = self.interleave(torch.cat((inputs_p_w, inputs_u_w)), 2)
        outputs_all = self.predict(inputs_all)
        outputs_all = self.de_interleave(outputs_all, 2)
        outputs_p_w = outputs_all[:batch_size]
        outputs_u_w = outputs_all[batch_size:]
        """
        ## WDD - bottom ##
        loss = self.cvir_loss(outputs_p_w, outputs_u_w, targets_p, targets_u, idx_ulb)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    def cvir_loss(self, outputs_p_w, outputs_u_w, targets_p, targets_u, idx_ulb):
        num_lb = targets_p.shape[0]
        idx_new = []
        try:
            for i in idx_ulb:
                idx_new.append(torch.where(self.all_idxs == i)[0][0])
            idx_new = torch.tensor(idx_new)
        except:
            print(torch.where(self.all_idxs == i))
        keep_ulb = torch.nonzero(self.keep_samples[idx_new] == 1).squeeze()

        try:
            l = len(keep_ulb)
        except:
            l = 0

        if l == 0:
            return self.base_loss(outputs_p_w, targets_p)
        else:
            outputs_u_w = outputs_u_w[keep_ulb]
            # compute loss
            loss_pos = self.base_loss(outputs_p_w, targets_p)
            loss_neg = self.base_loss(outputs_u_w, torch.zeros((outputs_u_w.shape[0],), device=outputs_p_w.device, dtype=torch.long))
            # total loss
            #total_loss = 0.5 * (loss_pos + loss_neg)
            total_loss = self.pi * loss_pos + (1 - self.pi) * loss_neg
            return total_loss

    @torch.no_grad()
    def before_train_epoch(self, train_eval_u_loader, device, num_ulb_data):
        self.network.eval()
        with torch.no_grad():
            all_probs = []
            all_idxs = []
            l = 0
            for data_batch in train_eval_u_loader:
                # if l % 1000 == 0:
                #    print(l)
                l = l + 1
                ## WDD - top ##
                x_idx, (images_u_w, images_u_s), (targets_u, targets_u_true) = data_batch
                """
                x_idx, (images_u_w, images_u_s), (targets_u, targets_u_true) = data_batch
                """
                ## WDD - bottom ##
                # x_idx = data_batch['idx_ulb']
                images_u_w = images_u_w.to(device)

                logits = self.network(images_u_w)
                probs = logits.softmax(dim=-1)
                all_probs.append(probs.detach())
                all_idxs.append(x_idx.detach())
            self.network.train()

            all_probs = torch.cat(all_probs, dim=0)[:, 0]
            all_idxs = torch.cat(all_idxs, dim=0)
            sorted_idx = torch.argsort(all_probs, descending=True)
            keep_samples = torch.ones_like(all_probs)
            keep_samples[sorted_idx[num_ulb_data - int(self.pi * num_ulb_data):]] = 0

            self.keep_samples = keep_samples
            self.all_idxs = all_idxs

class Dist_PU(Algorithm):
    """
    Dist_PU
    Reference: TBA.
    """

    def __init__(self, p_input_shape, u_input_shape, hparams):
        super(Dist_PU, self).__init__(p_input_shape, u_input_shape, hparams)
        ## WDD - top ##
        self.steps_per_epoch_u = (self.u_input_shape[0] // self.hparams['batch_size'])
        """
        self.featurizer = networks.Featurizer(p_input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            self.num_classes)

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.hparams["lr"],
            momentum=self.hparams["momentum"]
        )
        self.base_loss = nn.CrossEntropyLoss()
        """
        ## WDD - bottom ##

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.label_dist_loss = self.LabelDistributionLoss(prior=self.pi, device=torch.device(self.device))


        mixup_alpha = 6.0
        loss_weight_ent = 0.004
        loss_weight_mixup = 5.0
        loss_weight_mixup_ent = 0.04
        warmup_epoch = 7
        self.mixup_alpha = mixup_alpha
        self.loss_weight_ent = loss_weight_ent
        self.loss_weight_mixup_ent = loss_weight_mixup_ent
        self.loss_weight_mixup = loss_weight_mixup
        self.warmup_epochs = warmup_epoch

    def update(self, inputs_p_w, targets_p, inputs_u_w, targets_u, step, inputs_p_s, inputs_u_s, epoch, epochs):
        ## WDD - top ##
        outputs_p_w, outputs_u_w = self.output_after_interleave(torch.cat((inputs_p_w, inputs_u_w)), inputs_p_w.shape[0])
        """
        batch_size = inputs_p_w.shape[0]
        inputs_all = self.interleave(torch.cat((inputs_p_w, inputs_u_w)), 2)
        outputs_all = self.predict(inputs_all)
        outputs_all = self.de_interleave(outputs_all, 2)
        outputs_p_w = outputs_all[:batch_size]
        outputs_u_w = outputs_all[batch_size:]
        """
        ## WDD - bottom ##
        loss = self.dist_loss(outputs_p_w, outputs_u_w, targets_p, targets_u,inputs_p_w,inputs_u_w, epoch, epochs)
        #t = torch.randint(1,5,(1,))
        #if t==2:
        #    loss = torch.tensor(float('nan'))*loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        ## WDD - top ##
        #if step % self.steps_per_epoch_u == 0:
        #    self.after_train_epoch((step + 1) / self.steps_per_epoch_u)
        ## WDD - bottom ##

        return {'loss': loss.item()}

    @torch.no_grad()
    def mixup_two_target(self, x, y, alpha=1.0, is_bias=False):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        if is_bias:
            lam = max(lam, 1 - lam)

        index = torch.randperm(x.size(0)).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    @torch.no_grad()
    def mixup_one_target(self, x, y, alpha=1.0, is_bias=False):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        if is_bias:
            lam = max(lam, 1 - lam)

        index = torch.randperm(x.size(0)).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        return mixed_x, mixed_y, lam

    # adapted from https://github.com/valerystrizh/pytorch-histogram-loss
    class LabelDistributionLoss(nn.Module):
        def __init__(self, prior, device, num_bins=1, proxy='polar', dist='L1'):
            super().__init__()
            self.prior = prior
            self.frac_prior = 1.0 / (2 * prior)

            self.step = 1 / num_bins  # bin width. predicted scores in [0, 1].
            self.device = device
            self.t = torch.arange(0, 1 + self.step, self.step).view(1, -1).requires_grad_(False)  # [0, 1+bin width)
            self.t_size = num_bins + 1

            self.dist = None
            if dist == 'L1':
                self.dist = F.l1_loss
            else:
                raise NotImplementedError("The distance: {} is not defined!".format(dist))

            # proxy
            proxy_p, proxy_n = None, None
            if proxy == 'polar':
                proxy_p = np.zeros(self.t_size, dtype=float)
                proxy_n = np.zeros_like(proxy_p)
                proxy_p[-1] = 1
                proxy_n[0] = 1
            else:
                raise NotImplementedError("The proxy: {} is not defined!".format(proxy))

            proxy_mix = prior * proxy_p + (1 - prior) * proxy_n
            print('#### Label Distribution Loss ####')
            print('ground truth P:')
            print(proxy_p)
            print('ground truth U:')
            print(proxy_mix)

            # to torch tensor
            self.proxy_p = torch.from_numpy(proxy_p).requires_grad_(False).float()
            self.proxy_mix = torch.from_numpy(proxy_mix).requires_grad_(False).float()

            # to device
            self.t = self.t.to(self.device)
            self.proxy_p = self.proxy_p.to(self.device)
            self.proxy_mix = self.proxy_mix.to(self.device)

        def histogram(self, scores):
            scores_rep = scores.repeat(1, self.t_size)

            hist = torch.abs(scores_rep - self.t)

            inds = (hist > self.step)
            hist = self.step - hist  # switch values
            hist[inds] = 0

            return hist.sum(dim=0) / (len(scores) * self.step)

        def forward(self, scores_p, scores_n):
            # scores=torch.sigmoid(outputs)
            # labels=labels.view(-1,1)

            s_p = scores_p.view(-1, 1)
            s_u = scores_n.view(-1, 1)

            l_p = 0
            l_u = 0
            if s_p.numel() > 0:
                hist_p = self.histogram(s_p)
                l_p = self.dist(hist_p, self.proxy_p, reduction='mean')
            if s_u.numel() > 0:
                hist_u = self.histogram(s_u)
                l_u = self.dist(hist_u, self.proxy_mix, reduction='mean')

            return l_p + self.frac_prior * l_u
    def dist_loss(self, outputs_p_w, outputs_u_w, targets_p, targets_u,x_lb,x_ulb,epoch,epochs):
        num_ulb = outputs_u_w.shape[0]


        logits_x_lb = outputs_p_w
        logits_x_ulb = outputs_u_w

        # get probs
        probs_x_lb = logits_x_lb.softmax(dim=1)
        probs_x_ulb = logits_x_ulb.softmax(dim=1)

        # calculate label dist loss
        label_dist_loss = self.label_dist_loss(probs_x_lb[:, 1], probs_x_ulb[:, 0])
        total_loss = label_dist_loss

        # calculate entropy minimization loss
        if epoch >= self.warmup_epochs:
            entropy_loss = - (probs_x_ulb * torch.log(probs_x_ulb)).sum(dim=1).mean()

            # get pseudo labels for mixup
            with torch.no_grad():
                pseudo_labels = probs_x_ulb.argmax(dim=1)
                x, y = torch.cat((x_lb, x_ulb), dim=0), torch.cat((targets_p, pseudo_labels), dim=0)
                mixed_x, mixed_y, lam = self.mixup_one_target(x, y, self.mixup_alpha)

            mixed_logits = self.network(mixed_x)
            mixup_loss = self.base_loss(mixed_logits, mixed_y.to(torch.long))

            # calculate mixup entropy loss
            mixed_probs = mixed_logits.softmax(dim=1)
            mixup_entropy_loss = - (mixed_probs * torch.log(mixed_probs)).sum(dim=1).mean()

            loss_weight_ent = (1 - math.cos(
                (float(epoch - self.warmup_epochs) / (epochs - self.warmup_epochs)) * (
                        math.pi / 2))) * self.loss_weight_ent
            total_loss += self.loss_weight_mixup * mixup_loss + loss_weight_ent * entropy_loss + self.loss_weight_mixup_ent * mixup_entropy_loss

        return total_loss
    '''
    def before_run(self):
        # set warmup optimizer, scheduler (if needed)

        self.optimizer = torch.optim.SGD(
                self.network.parameters(),
                lr=self.warmup_lr,
                momentum=self.hparams["momentum"]
        )
        # add revised scheduler if needed
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=int(
        #        iter_per_epoch * self.warmup_epochs), eta_min=1e-6)
    '''
    '''
    def after_train_epoch(self, epoch):
        if epoch >= self.warmup_epochs:
            self.optimizer = torch.optim.SGD(
                self.network.parameters(),
                lr=self.hparams["lr"],
                momentum=self.hparams["momentum"]
            )
            # add revised scheduler if needed
            #algorithm.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=int(
            #    iter_per_epoch * (self.epochs - self.warmup_epochs)), eta_min=1e-6)
        print('warmup end!')
    '''
class PUSB(Algorithm):
    """
    PUSB
    Reference: Learning from Positive and Unlabeled Data with a Selection Bias, ICLR 2019.
    """

    def __init__(self, p_input_shape, u_input_shape, hparams):
        super(PUSB, self).__init__(p_input_shape, u_input_shape, hparams)
        self.threshold=0.5


    def update(self, inputs_p_w, targets_p, inputs_u_w, targets_u, cur_step, train_p_loader,train_u_loader):
        outputs_p_w, outputs_u_w = self.output_after_interleave(torch.cat((inputs_p_w, inputs_u_w)), inputs_p_w.shape[0])
        loss = self.pusb_loss(outputs_p_w, outputs_u_w, targets_p, targets_u)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if cur_step==self.hparams["max_steps"]:
            self.network.eval()
            self.set_threshold(train_p_loader,train_u_loader)
        return {'loss': loss.item()}

    def pusb_loss(self,outputs_p_w, outputs_u_w, targets_p, targets_u):
        loss_pos_pos = self.pi * self.base_loss(outputs_p_w, targets_p)
        loss_unlabel_neg = self.base_loss(outputs_u_w, targets_u)
        targets_p_reverse = (1 - targets_p).to(targets_p.dtype)
        loss_pos_neg = - self.pi * self.base_loss(outputs_p_w, targets_p_reverse)
        average_loss = loss_pos_pos + torch.clamp(loss_unlabel_neg + loss_pos_neg, min=0)
        return average_loss

    def set_threshold(self,train_p_loader, train_u_loader):
        Z=torch.tensor([], dtype=torch.float32).cuda()
        with torch.no_grad():
            for p_idx, (inputs_p_w1, inputs_p_s), targets_p in train_p_loader:
                inputs_p_w1=inputs_p_w1.cuda()
                outputs_p_w1=self.network(inputs_p_w1)
                Z = torch.cat((Z, torch.softmax(outputs_p_w1, 1)[:, 1]))
            for u_idx, (inputs_u_w1, inputs_u_s), (targets_u, targets_u_true) in train_u_loader:
                inputs_u_w1=inputs_u_w1.cuda()
                outputs_u_w1=self.network(inputs_u_w1)
                Z = torch.cat((Z, torch.softmax(outputs_u_w1, 1)[:, 1]))
            index_threshold = (1 - self.pi) * len(Z)
            sorted_Z, _ = torch.sort(Z, dim=0)
            self.threshold = sorted_Z[int(index_threshold)]

    def predict(self,x):
        output=self.network(x)
        output=torch.softmax(output,dim=1)
        output=output[:,1]-self.threshold
        output[output>=0]=1
        output[output<0]=-1
        output=output.unsqueeze(1)
        return torch.cat(((1-output)/2,(1+output)/2),1)

class PUe(Algorithm):
    """
    PUe
    Reference: Biased Positive-Unlabeled Learning Enhancement by Causal Inference
    """

    def __init__(self, p_input_shape, u_input_shape, hparams):
        super(PUe, self).__init__(p_input_shape, u_input_shape, hparams)
        self.e_network=deepcopy(self.network)
        self.e_optimizer=torch.optim.SGD(self.e_network.parameters(), lr=self.hparams["lr"], momentum=self.hparams["momentum"])
        self.base_loss_none=nn.CrossEntropyLoss(reduction='none')
        self.base_loss_sum=nn.CrossEntropyLoss(reduction='sum')

    def update(self, inputs_p_w, targets_p, inputs_u_w, targets_u, cur_step, train_p_loader):
        if cur_step<=self.hparams['warmup_steps']:
            e_outputs_p_w, e_outputs_u_w = self.e_output_after_interleave(torch.cat((inputs_p_w, inputs_u_w)),inputs_p_w.shape[0])
            loss = self.pue_e_loss(e_outputs_p_w, e_outputs_u_w, targets_p, targets_u)

            self.e_optimizer.zero_grad()
            loss.backward()
            self.e_optimizer.step()

        else:
            outputs_p_w, outputs_u_w = self.output_after_interleave(torch.cat((inputs_p_w, inputs_u_w)),inputs_p_w.shape[0])
            e_outputs_p_w, e_outputs_u_w = self.e_output_after_interleave(torch.cat((inputs_p_w, inputs_u_w)), inputs_p_w.shape[0])
            e_outputs_p_w, e_outputs_u_w = e_outputs_p_w.detach(), e_outputs_u_w.detach()
            with torch.no_grad():
                prop_scores=torch.softmax(e_outputs_p_w,dim=1)[:,1]
                ### calculate normalizing_factor
                normalizing_factor=torch.sum(1/prop_scores)
                normalized_prop_scores=prop_scores*normalizing_factor
                ###
            loss = self.pue_clf_loss(outputs_p_w, outputs_u_w, targets_p, targets_u, normalized_prop_scores)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {'loss': loss.item()}

    def pue_e_loss(self, outputs_p_w, outputs_u_w, targets_p, targets_u):
        loss1 = 1 / (outputs_p_w.shape[0] + outputs_u_w.shape[0]) * self.base_loss_sum(outputs_p_w, targets_p)
        loss2 = 1 / (outputs_p_w.shape[0] + outputs_u_w.shape[0]) * self.base_loss_sum(outputs_u_w, targets_u)
        regularisation_former=torch.sum(torch.softmax(outputs_p_w,1)[:,1])+torch.sum(torch.softmax(outputs_u_w,1)[:,1])
        regularisation_latter=outputs_p_w.shape[0]
        if regularisation_former-regularisation_latter>=0:
            regularisation=regularisation_former-regularisation_latter # the sum of postive sample in this batch
        else:
            regularisation=regularisation_latter-regularisation_former
        average_loss = loss1 + loss2 + self.hparams['alpha'] * regularisation
        return average_loss

    def pue_clf_loss(self,outputs_p_w, outputs_u_w, targets_p, targets_u, prop_score_p):
        # prop_score_p=outputs_p_w.shape[0]
        loss_pos_pos = self.pi*torch.sum(self.base_loss_none(outputs_p_w, targets_p)/prop_score_p)
        loss_unlabel_neg = self.base_loss(outputs_u_w, targets_u)
        targets_p_reverse = (1 - targets_p).to(targets_p.dtype)
        loss_pos_neg = - self.pi * torch.sum(self.base_loss_none(outputs_p_w, targets_p_reverse)/ prop_score_p)
        average_loss = loss_pos_pos + loss_unlabel_neg + loss_pos_neg
        return average_loss

    def e_output_after_interleave(self,x,batch_size):
        inputs=self.interleave(x,2)
        outputs=self.e_network(inputs)
        outputs=self.de_interleave(outputs,2)
        outputs_p_w, outputs_u_w=outputs[:batch_size], outputs[batch_size:]
        return outputs_p_w, outputs_u_w

class LBE(Algorithm):
    """
    LBE
    Reference: Instance-Dependent Positive and Unlabeled Learning With Labeling Bias Estimation
    """

    def __init__(self, p_input_shape, u_input_shape, hparams):
        super(LBE, self).__init__(p_input_shape, u_input_shape, hparams)
        self.steps_per_epoch_u = (self.u_input_shape[0] // self.hparams['batch_size'])
        self.epochs = (self.hparams['max_steps'] // self.steps_per_epoch_u)
        self.warmup_epoch = (self.hparams['warmup_steps'] // self.steps_per_epoch_u)

        self.eta_network=deepcopy(self.network)
        self.network_frozen=deepcopy(self.network)
        self.eta_network_frozen=deepcopy(self.eta_network)

        self.whole_optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams["lr"], momentum=self.hparams["momentum"])
        self.base_loss_none = nn.CrossEntropyLoss(reduction='none')


    def update(self, inputs_p_w, targets_p, inputs_u_w, targets_u, cur_step, test_loader):
        if cur_step <= self.hparams['warmup_steps']:
            outputs_p_w, outputs_u_w = self.output_after_interleave(torch.cat((inputs_p_w, inputs_u_w)), inputs_p_w.shape[0])
            loss = self.base_loss(outputs_p_w, targets_p) + self.base_loss(outputs_u_w, targets_u)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if cur_step / self.steps_per_epoch_u == self.warmup_epoch: # at the end of warmup
                self.network_frozen=deepcopy(self.network)
        else:
            P_y_hat_p, P_y_hat_u=self.E_step(inputs_p_w, inputs_u_w)
            loss = self.lbe_loss(inputs_p_w, inputs_u_w,targets_p, targets_u, P_y_hat_p, P_y_hat_u)

            self.whole_optimizer.zero_grad()
            loss.backward()
            self.whole_optimizer.step()

            if cur_step % self.steps_per_epoch_u == 0 and cur_step > 0: # for each epoch
                self.network_frozen=deepcopy(self.network)
                self.eta_network_frozen=deepcopy(self.eta_network)

        return {'loss': loss.item()}



    def lbe_loss(self, inputs_p_w, inputs_u_w, targets_p, targets_u, P_y_hat_p, P_y_hat_u):
        outputs_p_w, outputs_u_w = self.output_after_interleave(torch.cat((inputs_p_w, inputs_u_w)), inputs_p_w.shape[0])
        eta_outputs_p_w, eta_outputs_u_w = self.eta_output_after_interleave(torch.cat((inputs_p_w, inputs_u_w)), inputs_p_w.shape[0])

        targets_p_reverse = (1 - targets_p).to(targets_p.dtype)
        targets_u_reverse = (1 - targets_u).to(targets_u.dtype)

        loss1_p = self.base_loss_none(eta_outputs_p_w, targets_p)*P_y_hat_p[:,1] + self.base_loss_none(eta_outputs_p_w, targets_p)*P_y_hat_p[:,0]
        loss1_n = self.base_loss_none(eta_outputs_u_w, targets_u)*P_y_hat_u[:,1] + self.base_loss_none(eta_outputs_u_w, targets_u)*P_y_hat_u[:,0]

        loss2_p = self.base_loss_none(outputs_p_w, targets_p)*P_y_hat_p[:,1] + self.base_loss_none(outputs_p_w, targets_p_reverse)*P_y_hat_p[:,0]
        loss2_n = self.base_loss_none(outputs_u_w, targets_u_reverse)*P_y_hat_u[:,1] + self.base_loss_none(outputs_u_w, targets_u)*P_y_hat_u[:,0]

        loss = torch.sum(loss1_p + loss1_n + loss2_p + loss2_n)

        return loss
    def E_step(self, inputs_p_w, inputs_u_w):
        with torch.no_grad():
            frozen_outputs_p_w, frozen_outputs_u_w = self.frozen_output_after_interleave(torch.cat((inputs_p_w, inputs_u_w)), inputs_p_w.shape[0])
            frozen_outputs_p_w, frozen_outputs_u_w =torch.softmax(frozen_outputs_p_w, dim=1)[:,1], torch.softmax(frozen_outputs_u_w, dim=1)[:,1]
            frozen_eta_outputs_p_w, frozen_eta_outputs_u_w = self.frozen_eta_output_after_interleave(torch.cat((inputs_p_w, inputs_u_w)), inputs_p_w.shape[0])
            frozen_eta_outputs_p_w, frozen_eta_outputs_u_w = torch.softmax(frozen_eta_outputs_p_w, dim=1)[:,1],torch.softmax(frozen_eta_outputs_u_w, dim=1)[:,1]

            P_y_hat_1_p = frozen_eta_outputs_p_w * frozen_outputs_p_w
            P_y_hat_1_u = (1-frozen_eta_outputs_u_w) * frozen_outputs_u_w
            P_y_hat_0_p = 0 * (1-frozen_outputs_p_w)
            P_y_hat_0_u = 1 * (1-frozen_outputs_u_w)

            P_y_hat_p = torch.cat([P_y_hat_0_p.reshape(-1, 1), P_y_hat_1_p.reshape(-1, 1)], axis=1)
            P_y_hat_u = torch.cat([P_y_hat_0_u.reshape(-1, 1), P_y_hat_1_u.reshape(-1, 1)], axis=1)
            P_y_hat_p = P_y_hat_p/(P_y_hat_p.sum(axis=1).reshape(-1, 1)+1e-10)
            P_y_hat_u = P_y_hat_u/(P_y_hat_u.sum(axis=1).reshape(-1, 1)+1e-10)
            return P_y_hat_p, P_y_hat_u

    def eta_output_after_interleave(self, x, batch_size):
        inputs = self.interleave(x, 2)
        outputs = self.eta_network(inputs)
        outputs = self.de_interleave(outputs, 2)
        outputs_p_w, outputs_u_w = outputs[:batch_size], outputs[batch_size:]
        return outputs_p_w, outputs_u_w
    def frozen_eta_output_after_interleave(self,x,batch_size):
        inputs=self.interleave(x,2)
        outputs=self.eta_network_frozen(inputs)
        outputs=self.de_interleave(outputs,2)
        outputs_p_w, outputs_u_w=outputs[:batch_size], outputs[batch_size:]
        return outputs_p_w, outputs_u_w

    def frozen_output_after_interleave(self,x,batch_size):
        inputs=self.interleave(x,2)
        outputs=self.network_frozen(inputs)
        outputs=self.de_interleave(outputs,2)
        outputs_p_w, outputs_u_w=outputs[:batch_size], outputs[batch_size:]
        return outputs_p_w, outputs_u_w



class HolisticPU(Algorithm):
    """
    HolisticPU
    Reference: Beyond Myopia: Learning from Positive and Unlabeled Data through Holistic Predictive Trends
    """

    def __init__(self, p_input_shape, u_input_shape, hparams):
        super(HolisticPU, self).__init__(p_input_shape, u_input_shape, hparams)
        self.steps_per_epoch_u = (self.u_input_shape[0] // self.hparams['batch_size'])
        self.epochs = (self.hparams['max_steps'] // self.steps_per_epoch_u)
        self.warmup_epoch = (self.hparams['warmup_steps'] // self.steps_per_epoch_u)

        grouped_parameters = [{'params': [p for n, p in self.network.named_parameters() if not any(nd in n for nd in ['bias', 'bn'])], 'weight_decay': 5e-4},
            {'params': [p for n, p in self.network.named_parameters() if any(nd in n for nd in ['bias', 'bn'])], 'weight_decay': 0.0}]
        self.optimizer=torch.optim.SGD(grouped_parameters, lr=self.hparams["lr"], momentum=self.hparams["momentum"],nesterov=True)
        self.scheduler = self.get_cosine_schedule_with_warmup(self.optimizer, self.hparams["warmup_steps"],self.hparams["max_steps"])
        self.ema_network = self.ModelEMA(self.network, decay=0.999)
        self.base_loss_ls=nn.CrossEntropyLoss(label_smoothing=self.hparams["rho"])

        self.model1=deepcopy(self.network)
        grouped_parameters1 = [{'params': [p for n, p in self.model1.named_parameters() if not any(nd in n for nd in ['bias', 'bn'])], 'weight_decay': 5e-4},
            {'params': [p for n, p in self.model1.named_parameters() if any(nd in n for nd in ['bias', 'bn'])], 'weight_decay': 0.0}]
        self.optimizer1 = torch.optim.SGD(grouped_parameters1,lr=self.hparams["lr"], momentum=self.hparams["momentum"],nesterov=True)
        self.scheduler1 = self.get_cosine_schedule_with_warmup(self.optimizer1, 0,self.hparams["max_steps"])

    def update(self, inputs_p_w, targets_p, inputs_u_w, targets_u, cur_step, inputs_p_s, inputs_u_s, u_idx, train_u_loader):
        if cur_step <= self.hparams['warmup_steps']:
            outputs_p_w, outputs_u_w = self.output_after_interleave(torch.cat((inputs_p_w, inputs_u_w)),inputs_p_w.shape[0])
            outputs_p_s, outputs_u_s = self.output_after_interleave(torch.cat((inputs_p_s, inputs_u_s)),inputs_p_w.shape[0])
            Lx=(self.base_loss_ls(outputs_p_w, targets_p)+ self.base_loss_ls(outputs_p_s, targets_p))/2
            Ln=self.base_loss(outputs_u_w, targets_u)
            loss=Lx+Ln

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if self.hparams["use_ema"]:
                self.ema_network.update(self.network)

            if cur_step % self.steps_per_epoch_u == 0 and cur_step > 0: # for each epoch
                preds, targets_true = self.record(train_u_loader)
                preds, targets_true = preds, targets_true
                if cur_step / self.steps_per_epoch_u == 1:
                    self.preds_sequence=preds
                else:
                    self.preds_sequence = torch.cat((self.preds_sequence, preds), dim=1)

            if cur_step / self.steps_per_epoch_u == self.warmup_epoch: # at the end of warmup
                self.preds_sequence = self.preds_sequence.cpu().numpy()
                trends = np.zeros(len(self.preds_sequence))
                for i, sequence in enumerate(self.preds_sequence):
                    sequence = pd.Series(sequence)
                    diff_1 = sequence.diff(periods=1)
                    diff_1 = np.array(diff_1)
                    diff_1 = diff_1[1:]
                    diff_1 = np.log(1 + diff_1 + 0.5 * diff_1 ** 2)
                    trends[i] = diff_1.mean()
                intervals = jenkspy.jenks_breaks(trends, n_classes=2)
                break_point = intervals[1]
                if break_point > 0:
                    trends_std = self.three_sigma(trends)
                    intervals = jenkspy.jenks_breaks(trends_std, n_classes=2)
                    break_point = intervals[1]
                self.pseudo_targets = np.where(trends > break_point, 1, 0)

                self.unlabeled_num = len(self.pseudo_targets)
        else:
            outputs_p_w, outputs_u_w = self.model1_output_after_interleave(torch.cat((inputs_p_w, inputs_u_w)),inputs_p_w.shape[0])
            outputs_p_s, outputs_u_s = self.model1_output_after_interleave(torch.cat((inputs_p_s, inputs_u_s)),inputs_p_w.shape[0])
            targets_pseudo = self.pseudo_targets[u_idx]
            targets_pseudo = torch.tensor(targets_pseudo).to(torch.long).cuda()

            Lx1 = self.base_loss(outputs_p_w, targets_p)
            Lu1 = self.loss_ft(outputs_u_w, outputs_u_s, targets_u, targets_pseudo, cur_step)
            loss = Lx1 + Lu1

            self.optimizer1.zero_grad()
            loss.backward()
            self.optimizer1.step()
            self.scheduler1.step()

            if self.hparams["use_ema"]:
                self.ema_network.update(self.model1)

        return {'loss': loss.item()}

    def predict(self, x):
        if self.hparams["use_ema"]:
            return self.ema_network.ema(x)
        else:
            return self.network(x)

    class ModelEMA(object):
        def __init__(self,  model, decay):
            self.ema = deepcopy(model)
            self.ema.cuda()
            self.ema.eval()
            self.decay = decay
            self.ema_has_module = hasattr(self.ema, 'module')
            # Fix EMA. https://github.com/valencebond/FixMatch_pytorch thank you!
            self.param_keys = [k for k, _ in self.ema.named_parameters()]
            self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
            for p in self.ema.parameters():
                p.requires_grad_(False)

        def update(self, model):
            needs_module = hasattr(model, 'module') and not self.ema_has_module
            with torch.no_grad():
                msd = model.state_dict()
                esd = self.ema.state_dict()
                for k in self.param_keys:
                    if needs_module:
                        j = 'module.' + k
                    else:
                        j = k
                    model_v = msd[j].detach()
                    ema_v = esd[k]
                    esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

                for k in self.buffer_keys:
                    if needs_module:
                        j = 'module.' + k
                    else:
                        j = k
                    esd[k].copy_(msd[j])

    def get_cosine_schedule_with_warmup(self,optimizer, num_warmup_steps, num_training_steps, num_cycles=7. / 16., last_epoch=-1):
        from torch.optim.lr_scheduler import LambdaLR
        def _lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            no_progress = float(current_step - num_warmup_steps) / \
                          float(max(1, num_training_steps - num_warmup_steps))
            return max(0., math.cos(math.pi * num_cycles * no_progress))
        return LambdaLR(optimizer, _lr_lambda, last_epoch)
    def loss_ft(self, logits1_u, logits1_u_s, targets_u, targets_p, cur_step):
        label_u = F.one_hot(targets_u, 2).float().cuda()
        label_p = F.one_hot(targets_p, 2).float().cuda()
        lamda = ((cur_step-self.hparams['warmup_steps']) / (self.hparams['max_steps']-self.hparams['warmup_steps'])) ** 0.8
        label = lamda * label_p + (1 - lamda) * label_u
        loss = F.cross_entropy(logits1_u, label, reduction='mean')

        pseudo_label = torch.softmax(logits1_u.detach() / self.hparams['T'], dim=-1)
        max_probs, pseudo_targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(0.9).float()
        loss2 = (F.cross_entropy(logits1_u_s, pseudo_targets_u,reduction='none') * mask).mean()
        return loss + loss2

    def model1_output_after_interleave(self, x, batch_size):
        inputs = self.interleave(x, 2)
        outputs = self.model1(inputs)
        outputs = self.de_interleave(outputs, 2)
        outputs_p_w, outputs_u_w = outputs[:batch_size], outputs[batch_size:]
        return outputs_p_w, outputs_u_w

    def three_sigma(self,x):
        idx = np.where(x < 0.2 / 9)
        return x[idx]
    def record(self, unlabeled_trainloader):
        with torch.no_grad():
            preds=torch.ones((self.u_input_shape[0],1)).cuda()
            targets=torch.ones((self.u_input_shape[0],1)).long().cuda()
            for batch_idx, (u_idx, (inputs_u_w1, inputs_u_s), (targets_u, targets_u_true)) in enumerate(unlabeled_trainloader):
                self.ema_network.ema.eval()
                self.network.eval()
                inputs_u_w1, targets_u_true = inputs_u_w1.cuda(), targets_u_true.cuda()
                outputs= self.predict(inputs_u_w1)
                pred=torch.softmax(outputs, dim=1)[:,1]
                preds[u_idx]=pred.unsqueeze(1)
                targets[u_idx]=targets_u_true.unsqueeze(1)

            self.ema_network.ema.train()
            self.network.train()
        return preds, targets


class PULDA(Algorithm):
    """
    PULDA
    Reference: Positive-Unlabeled Learning With Label Distribution Alignment
    """

    def __init__(self, p_input_shape, u_input_shape, hparams):
        super(PULDA, self).__init__(p_input_shape, u_input_shape, hparams)

        self.steps_per_epoch_u = (self.u_input_shape[0] // self.hparams['batch_size'])

        self.epochs=(self.hparams['max_steps']//self.steps_per_epoch_u)
        self.warmup_epoch=(self.hparams['warmup_steps']//self.steps_per_epoch_u)

        self.loss_fn = self.create_loss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs)

    def update(self,inputs_p_w, targets_p, inputs_u_w, targets_u, cur_step,p_idx,u_idx,train_p_loader, train_u_loader):
        if cur_step<=self.hparams["warmup_steps"]:
            outputs_p_w, outputs_u_w = self.output_after_interleave(torch.cat((inputs_p_w, inputs_u_w)),inputs_p_w.shape[0])
            one_outputs_p_w=outputs_p_w[:,1]-outputs_p_w[:,0]
            one_outputs_u_w=outputs_u_w[:,1]-outputs_u_w[:,0]
            loss=self.loss_fn(one_outputs_p_w, targets_p.float())+self.loss_fn(one_outputs_u_w, targets_u.float())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if cur_step % self.steps_per_epoch_u == 0  and cur_step > 0: # for each epoch
                self.scheduler.step()

            if cur_step / self.steps_per_epoch_u == self.warmup_epoch: # at the end of warmup
                self.network.eval()
                self.p_psudo_labels = torch.ones(len(train_p_loader.dataset)).squeeze().cuda()
                self.u_psudo_labels = torch.zeros(len(train_u_loader.dataset)).squeeze().cuda()
                with torch.no_grad():
                    for u_idx, (inputs_u_w1, inputs_u_s), (targets_u, targets_u_true) in train_u_loader:
                        inputs_u_w1=inputs_u_w1.cuda()
                        outputs=self.network(inputs_u_w1)
                        outputs=torch.softmax(outputs, dim=1)[:,1]
                        self.u_psudo_labels[u_idx]=outputs
                    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs, 0.7*self.hparams['lr'])
                self.network.train()
        else:
            u_psudos=self.u_psudo_labels[u_idx]
            p_psudos=torch.ones_like(self.p_psudo_labels[p_idx]).cuda()
            inputs_w=torch.cat((inputs_p_w, inputs_u_w))
            targets  = torch.cat((targets_p, targets_u))
            psudos=torch.cat((p_psudos,u_psudos))
            mixed_x, y_a, y_b, lam = self.mixup_two_targets(inputs_w, psudos, self.hparams["alpha"])
            outputs = self.network(mixed_x)
            outputs = torch.clamp(outputs[:,1]-outputs[:,0],min=-10, max=10)
            scores = torch.sigmoid(outputs)
            outputs_ = self.network(inputs_w)
            outputs_ = torch.clamp(outputs_[:,1]-outputs_[:,0],min=-10, max=10)
            scores_ = torch.sigmoid(outputs_)
            loss = self.loss_fn(outputs_,targets.float())+self.hparams["co_mixup"]*self.mixup_bce(scores,y_a,y_b,lam)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.u_psudo_labels[u_idx]=scores_[inputs_p_w.shape[0]:].detach()

            if cur_step % self.steps_per_epoch_u == 0:
                self.scheduler.step()

        return {'loss': loss.item()}


    def mixup_bce(self, scores, targets_a, targets_b, lam):
        mixup_loss_a = F.binary_cross_entropy(scores, targets_a)
        mixup_loss_b = F.binary_cross_entropy(scores, targets_b)
        mixup_loss = lam * mixup_loss_a + (1 - lam) * mixup_loss_b
        return mixup_loss
    def mixup_two_targets(self, x, y, alpha=1.0, device='cuda', is_bias=False):
        """
            Returns mixed inputs, pairs of targets, and lambda
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        if is_bias: lam = max(lam, 1 - lam)

        index = torch.randperm(x.size(0)).to(device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    class LabelDistributionLoss(nn.Module):
        def __init__(self, prior, device, dist='softplus', tmpr=1):
            super().__init__()
            self.prior = prior
            self.two_times_prior = 2 * prior
            self.exp_y_P = torch.tensor(1, dtype=torch.float, requires_grad=False).to(device)
            self.exp_y_U = torch.tensor(prior, dtype=torch.float, requires_grad=False).to(device)

            self.dist = None
            if dist == 'softplus':
                def softplus_loss(x1, x2, reduction='mean'):
                    x_diff = x1 - x2
                    return torch.mean(F.softplus(x_diff, tmpr) + F.softplus(-x_diff, tmpr))

                self.dist = softplus_loss
            else:
                raise NotImplementedError("The distance function: {} is not defined!".format(dist))

            print('#### Label Distribution Alignment ####')
            print('Expectation of Labels from Labeled Positive data: ')
            print(self.exp_y_P)
            print('Expectation of Labels from Unlabeled data: ')
            print(self.exp_y_U)
            print('Distance Measure Function: ')
            print(dist)

        def forward(self, outputs, labels):
            scores = torch.sigmoid(outputs)
            labels = labels.view(-1, 1)

            scores = scores.view_as(labels)
            scores_P = scores[labels == 1].view(-1, 1)
            scores_U = scores[labels == 0].view(-1, 1)

            R_L = 0
            R_U = 0
            if scores_P.numel() > 0:
                exp_y_hat_P = scores_P.mean()
                R_L = torch.mean(self.exp_y_P - exp_y_hat_P)
            if scores_U.numel() > 0:
                exp_y_hat_U = scores_U.mean()
                R_U = self.dist(exp_y_hat_U, self.exp_y_U, reduction='mean')

            return self.two_times_prior * R_L + R_U

    class LabelDistributionLossWithEMA(LabelDistributionLoss):
        def __init__(self, prior, device, dist='softplus', tmpr=1, alpha_U=0.9):
            super().__init__(prior, device, dist, tmpr)
            self.alpha_U = alpha_U
            self.one_minus_alpha_U = 1 - alpha_U
            self.exp_y_hat_U_ema = None

        def forward(self, outputs, labels):
            scores = torch.sigmoid(outputs)
            labels = labels.view(-1, 1)
            scores = scores.view_as(labels)
            scores_P = scores[labels == 1].view(-1, 1)
            scores_U = scores[labels == 0].view(-1, 1)

            R_L = 0
            R_U = 0
            if scores_P.numel() > 0:
                exp_y_hat_P = scores_P.mean()
                R_L = torch.mean(self.exp_y_P - exp_y_hat_P)
            if scores_U.numel() > 0:
                exp_y_hat_U = scores_U.mean()
                R_U = self.dist(exp_y_hat_U, self.exp_y_U, reduction='mean')
                self.exp_y_hat_U_ema = exp_y_hat_U.detach()

            if scores_U.numel() > 0:
                self.forward = self.second_forward

            return self.two_times_prior * R_L + R_U

        def second_forward(self, outputs, labels):
            scores = torch.sigmoid(outputs)
            labels = labels.view(-1, 1)

            scores = scores.view_as(labels)
            scores_P = scores[labels == 1].view(-1, 1)
            scores_U = scores[labels == 0].view(-1, 1)

            R_L = 0
            R_U = 0
            if scores_P.numel() > 0:
                exp_y_hat_P = scores_P.mean()
                R_L = torch.mean(self.exp_y_P - exp_y_hat_P)
            if scores_U.numel() > 0:
                exp_y_hat_U = self.alpha_U * self.exp_y_hat_U_ema + self.one_minus_alpha_U * scores_U.mean()
                R_U = self.dist(exp_y_hat_U, self.exp_y_U, reduction='mean')
                self.exp_y_hat_U_ema = exp_y_hat_U.detach()

            return self.two_times_prior * R_L + R_U / self.one_minus_alpha_U

    class TwoWaySigmoidLoss(nn.Module):
        def __init__(self, prior, margin=1.0, dist='softplus', tmpr=1):
            super().__init__()
            self.prior = prior
            self.margin = margin

            tmpr = 1.0
            print('tmpr of 2-way sigmoid: ', tmpr)
            self.slope_P_left = tmpr  # 1.0
            self.slope_P_right = tmpr  # 1.0
            self.slope_N_left = tmpr  # 1.0
            self.slope_N_right = tmpr  # 1.0

            self.dist = None
            if dist == 'softplus':
                def softplus_loss(x1, x2, reduction='mean'):
                    x_diff = x1 - x2
                    return torch.mean(F.softplus(x_diff, tmpr) + F.softplus(-x_diff, tmpr))

                self.dist = softplus_loss
            else:
                raise NotImplementedError("The distance function: {} is not defined!".format(dist))

        def twoWaySigmoidLossForPositive(self, outputs):
            return torch.sigmoid(self.slope_P_left * outputs) * torch.sigmoid(
                -self.slope_P_right * (outputs - self.margin))

        def twoWaySigmoidLossForNegative(self, outputs):
            return torch.sigmoid(self.slope_N_left * (outputs + self.margin)) * torch.sigmoid(
                -self.slope_N_right * outputs)

        def forward(self, outputs, labels):
            labels = labels.view(-1, 1)
            outputs = outputs.view_as(labels)

            outputs_P = outputs[labels == 1].view(-1, 1)
            outputs_U = outputs[labels != 1].view(-1, 1)

            if outputs_P.numel() > 0:
                C_P_plus = self.twoWaySigmoidLossForPositive(outputs_P).mean()
                C_P_minus = self.twoWaySigmoidLossForNegative(outputs_P).mean()
            else:
                C_P_plus = torch.tensor(0., device=labels.device)
                C_P_minus = torch.tensor(0., device=labels.device)
            if outputs_U.numel() > 0:
                C_U_minus = self.twoWaySigmoidLossForNegative(outputs_U).mean()
            else:
                C_U_minus = torch.tensor(0., device=labels.device)

            return self.prior * C_P_plus + self.dist(C_U_minus, self.prior * C_P_minus, reduction='mean')

    class TwoWaySigmoidLossWithEMA(TwoWaySigmoidLoss):
        def __init__(self, prior, margin=1.0, dist='softplus', tmpr=1, alpha_=0.9):
            super().__init__(prior, margin, dist, tmpr)

            self.alpha_ = alpha_
            self.one_minus_alpha_ = 1 - alpha_

            self.C_P_minus_ema = None
            self.C_U_minus_ema = None

        def forward(self, outputs, labels):
            labels = labels.view(-1, 1)
            outputs = outputs.view_as(labels)

            outputs_P = outputs[labels == 1].view(-1, 1)
            outputs_U = outputs[labels != 1].view(-1, 1)

            if outputs_P.numel() > 0:
                C_P_plus = self.twoWaySigmoidLossForPositive(outputs_P).mean()
                C_P_minus = self.twoWaySigmoidLossForNegative(outputs_P).mean()
                self.C_P_minus_ema = C_P_minus.detach()
            else:
                C_P_plus = torch.tensor(0., device=labels.device)
                C_P_minus = torch.tensor(0., device=labels.device)
            if outputs_U.numel() > 0:
                C_U_minus = self.twoWaySigmoidLossForNegative(outputs_U).mean()
                self.C_U_minus_ema = C_U_minus.detach()
            else:
                C_U_minus = torch.tensor(0., device=labels.device)

            if self.C_P_minus_ema != None and self.C_U_minus_ema != None:
                self.forward = self.second_forward

            return self.prior * C_P_plus + self.dist(C_U_minus, self.prior * C_P_minus, reduction='mean')

        def second_forward(self, outputs, labels):
            labels = labels.view(-1, 1)
            outputs = outputs.view_as(labels)

            outputs_P = outputs[labels == 1].view(-1, 1)
            outputs_U = outputs[labels != 1].view(-1, 1)

            if outputs_P.numel() > 0:
                C_P_plus = self.twoWaySigmoidLossForPositive(outputs_P).mean()
                C_P_minus = self.alpha_ * self.C_P_minus_ema + self.one_minus_alpha_ * self.twoWaySigmoidLossForNegative(
                    outputs_P).mean()
                self.C_P_minus_ema = C_P_minus.detach()
            else:
                C_P_plus = torch.tensor(0., device=labels.device)
                C_P_minus = self.C_P_minus_ema
            if outputs_U.numel() > 0:
                C_U_minus = self.alpha_ * self.C_U_minus_ema + self.one_minus_alpha_ * self.twoWaySigmoidLossForNegative(
                    outputs_U).mean()
                self.C_U_minus_ema = C_U_minus.detach()
            else:
                C_U_minus = self.C_U_minus_ema

            return self.prior * C_P_plus + self.dist(C_U_minus, self.prior * C_P_minus,
                                                     reduction='mean') / self.one_minus_alpha_

    def create_loss(self):
        if self.hparams['use_ema']:
            base_loss = self.LabelDistributionLossWithEMA(prior=self.pi, device='cuda', dist='softplus',
                                                     tmpr=self.hparams['tmpr'], alpha_U=self.hparams['alpha_U'])
        else:
            base_loss = self.LabelDistributionLoss(prior=self.pi, device='cuda', dist='softplus', tmpr=self.hparams['tmpr'])

        if self.hparams['two_way'] == 1:
            if self.hparams['use_ema']:
                two_loss = self.TwoWaySigmoidLossWithEMA(prior=self.pi, margin=self.hparams['margin'], dist='softplus',
                                                    tmpr=self.hparams['tmpr'], alpha_=self.hparams['alpha_CN'])
            else:
                two_loss = self.TwoWaySigmoidLoss(prior=self.pi, margin=self.hparams['margin'], dist='softplus',
                                             tmpr=self.hparams['tmpr'])

            def loss_fn_2way(outputs, labels):
                return base_loss(outputs, labels) + two_loss(outputs, labels)

            return loss_fn_2way
        return base_loss

class PAN(Algorithm):
    """
    PAN
    Reference: Predictive Adversarial Learning from Positive and Unlabeled Data, AAAI 2021.
    """
    def __init__(self, p_input_shape, u_input_shape, hparams):
        super(PAN, self).__init__(p_input_shape, u_input_shape, hparams)
        # self.featurizer_D = networks.Featurizer(p_input_shape, self.hparams)
        self.featurizer_C = networks.Featurizer(p_input_shape, self.hparams)
        # self.classifier_D = networks.Classifier(
        #     self.featurizer_D.n_outputs, self.num_classes)
        if self.hparams['dataset']== 'IMAGENETTE':
            self.classifier_C = networks.Classifier(256, self.num_classes)
        elif self.hparams['dataset'] in ['CIFAR10','USPS','Letter','Creditcard']:
            self.classifier_C = networks.Classifier(self.featurizer.n_outputs, self.num_classes)
        else:
            pass
        # Discriminator
        # self.network_D = nn.Sequential(self.featurizer_D, self.classifier_D)
        # Classifier
        self.network_C = nn.Sequential(self.featurizer_C, self.classifier_C)
        # self.optimizer_D = torch.optim.SGD(
        #     self.network_D.parameters(),
        #     lr=self.hparams["lr"],
        #     momentum=self.hparams["momentum"]
        # )
        self.optimizer_C = torch.optim.SGD(
            self.network_C.parameters(),
            lr=self.hparams["lr"],
            momentum=self.hparams["momentum"]
        )
        # self.base_loss = nn.CrossEntropyLoss()

    def update(self, inputs_p_w, targets_p, inputs_u_w, targets_u, epoch):

        # update Discriminator

        batch_size = inputs_p_w.shape[0]
        outputs_C_u_w = self.predict(inputs_u_w)
        outputs_C_u_w = torch.softmax(outputs_C_u_w, dim=1)[:, 1]

        # inputs_all = self.interleave(torch.cat((inputs_p_w, inputs_u_w)), 2)
        # outputs_D_all = self.network_D(inputs_all)
        # outputs_D_all = self.de_interleave(outputs_D_all, 2)
        # outputs_D_all = torch.softmax(outputs_D_all, dim=1)[:, 1]
        # outputs_D_p_w = outputs_D_all[:batch_size]
        # outputs_D_u_w = outputs_D_all[batch_size:]
        outputs_D_p_w, outputs_D_u_w = self.output_after_interleave(torch.cat((inputs_p_w, inputs_u_w)), inputs_p_w.shape[0])
        outputs_D_p_w = torch.softmax(outputs_D_p_w, dim=1)[:, 1]
        outputs_D_u_w = torch.softmax(outputs_D_u_w, dim=1)[:, 1]
        loss_old_D = self.pan_loss(outputs_D_p_w, outputs_D_u_w, outputs_C_u_w)
        loss_D = - loss_old_D
        self.optimizer.zero_grad()
        loss_D.backward(retain_graph=True)
        self.optimizer.step()

        # update Classifier

        outputs_C_u_w_new = self.predict(inputs_u_w)
        outputs_C_u_w_new = torch.softmax(outputs_C_u_w_new, dim=1)[:, 1]
        # outputs_D_all_new = self.network_D(inputs_all)
        # outputs_D_all_new = self.de_interleave(outputs_D_all_new, 2)
        # outputs_D_all_new = torch.softmax(outputs_D_all_new, dim=1)[:, 1]
        # outputs_D_p_w_new = outputs_D_all_new[:batch_size]
        # outputs_D_u_w_new = outputs_D_all_new[batch_size:]
        outputs_D_p_w_new, outputs_D_u_w_new = self.output_after_interleave(torch.cat((inputs_p_w, inputs_u_w)),
                                                                    inputs_p_w.shape[0])
        outputs_D_p_w_new = torch.softmax(outputs_D_p_w_new, dim=1)[:, 1]
        outputs_D_u_w_new = torch.softmax(outputs_D_u_w_new, dim=1)[:, 1]
        loss_new_D = self.pan_loss(outputs_D_p_w_new, outputs_D_u_w_new, outputs_C_u_w_new)
        loss_C = loss_new_D
        self.optimizer_C.zero_grad()
        loss_C.backward()
        self.optimizer_C.step()
        return {'loss': loss_D.item()}

    def pan_loss(self, outputs_D_p_w, outputs_D_u_w, outputs_C_u_w):
        l = self.hparams["l"]
        eps = 1e-2
        d_rate_u = outputs_D_u_w.clone().detach()
        loss1 = torch.sum(torch.log(outputs_D_p_w + eps)) + 1.0 * torch.sum(torch.max(torch.log(1.0 - outputs_D_u_w + eps) - torch.log(1.0 - torch.mean(d_rate_u)), torch.zeros(d_rate_u.shape).cuda()))
        loss2 = l * torch.sum((1 * torch.log(1 - outputs_C_u_w + eps + 0.0) - torch.log(outputs_C_u_w + eps + 0.0)) * (2 * outputs_D_u_w - 1.0 ))
        average_loss = loss1 + loss2
        return average_loss

    def predict(self, x):
        return self.network_C(x)

class P3MIX_C(Algorithm):
    """
    P3MIX_C
    Reference: Who Is Your Right Mixup Partner in Positive and Unlabeled Learning, ICLR 2022.
    """

    def __init__(self, p_input_shape, u_input_shape, hparams):
        super(P3MIX_C, self).__init__(p_input_shape, u_input_shape, hparams)
        # self.featurizer = networks.Featurizer(p_input_shape, self.hparams)
        # self.classifier = networks.Classifier(
        #     self.featurizer.n_outputs,
        #     self.num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        # self.optimizer = torch.optim.SGD(
        #     self.network.parameters(),
        #     lr=self.hparams["lr"],
        #     momentum=self.hparams["momentum"]
        # )
        # self.base_loss = nn.CrossEntropyLoss()
        self.scheduler = MultiStepLR(self.optimizer,
                                     milestones=self.hparams["milestones"],
                                     gamma=self.hparams["scheduler_gamma"])
        self.h_inputs_x = torch.Tensor([]).to(device)
        self.h_preds_x = torch.Tensor([]).to(device)

    def update(self, inputs_p_w, targets_p, inputs_u_w, targets_u, epoch, train_p_dataset):
        start_hmix = self.hparams["start_hmix"]
        gamma = self.hparams["gamma"]
        alpha = self.hparams["alpha"]
        top_k = self.hparams["top_k"]

        batch_size = inputs_p_w.shape[0]
        # inputs_all = self.interleave(data, 2)
        # outputs_all = self.predict(inputs_all)
        # outputs_all = self.de_interleave(outputs_all, 2)
        # outputs_p_w = outputs_all[:batch_size]
        # outputs_u_w = outputs_all[batch_size:]
        outputs_p_w, outputs_u_w = self.output_after_interleave(torch.cat((inputs_p_w, inputs_u_w)), inputs_p_w.shape[0])
        data = torch.cat((inputs_p_w, inputs_u_w), dim=0)

        outputs_u_w = torch.softmax(outputs_u_w, dim=1)[:, 1]
        idx_p = slice(0, batch_size)
        idx_u = slice(batch_size, len(data))
        target_p = targets_p[:, None]
        target_u = targets_u[:, None]
        target_p_ = torch.cat((1. - target_p, target_p), dim=1)
        target_u_ = torch.cat((1. - target_u, target_u), dim=1)
        targets_ = torch.cat((target_p_, target_u_), dim=0)

        # update cnd each epoch
        if epoch.is_integer():
            # print("epoch= ", epoch)
            train_p_loader = torch.utils.data.DataLoader(dataset=train_p_dataset, batch_size=len(train_p_dataset), shuffle=False)
            self.update_hinput(top_k, train_p_loader)

        with torch.no_grad():
            if epoch > start_hmix:
                outputs_u_w = outputs_u_w[:, None]
                p_indicator_u = outputs_u_w
                p_indicator_p = torch.ones(batch_size,
                                           dtype=torch.float32,
                                           device=device)
                p_indicator_p = p_indicator_p[:, None]
                p_indicator = torch.cat((p_indicator_p, p_indicator_u),
                                        dim=0)
                p_indicator = p_indicator.view(p_indicator.size(0))

                # correct unlabeled
                targets_u_fix = targets_u
                targets_u_fix[p_indicator[idx_u] >= gamma] = 1.
                targets_u_fix = targets_u_fix[:, None]
                targets_u_fix = torch.cat((1. - targets_u_fix, targets_u_fix), dim=1)
                targets_ = torch.cat((target_p_, targets_u_fix), dim=0)

        if epoch > start_hmix:
            h_input_b_idx = self.get_hinput(top_k, batch_size)  # for all unlabeled data
            h_target_b = torch.ones(len(h_input_b_idx),
                                    dtype=torch.float32,
                                    device=device)
            h_target_b = h_target_b[:, None]
            h_target_b_ = torch.cat([1. - h_target_b, h_target_b],
                                    dim=1)  # [0,1]

            idx1 = torch.tensor(
                np.random.randint(data.size(0), size=batch_size))  # for positive data, randomly select a partner
            data_b1 = torch.cat(
                [data[idx1], self.h_inputs_x[h_input_b_idx]], dim=0)
            targets_b1 = torch.cat([targets_[idx1], h_target_b_],
                                   dim=0)
            idx2 = torch.tensor(
                np.random.randint(data.size(0), size=batch_size))  # for unlabeled data, randomly select a partner

            idx = torch.cat([idx1, idx2])
            data_b, targets_b = data[idx], targets_[idx]

            p_indicator[p_indicator >= gamma] = 1.
            p_indicator[p_indicator <= 1-gamma] = 1.
            p_indicator[idx_p] = 1.
            p_indicator[p_indicator != 1.] = 0.

            data_b = (p_indicator * data_b.swapdims(0, -1) +
                    (1. - p_indicator) *
                      data_b1.swapdims(0, -1)).swapdims(0, -1)
            targets_b = (p_indicator * targets_b.swapdims(0, -1) +
                         (1. - p_indicator) *
                         targets_b1.swapdims(0, -1)).swapdims(0, -1)
        else:
            idx = torch.randperm(data.size(0))
            data_b, targets_b = data[idx], targets_[idx]
        data_a, targets_a = data, targets_

        l = np.random.beta(alpha, alpha)
        l = max(l, 1. - l)

        data_a = self.interleave(data_a, 2)
        data_b = self.interleave(data_b, 2)
        data_mix, mix_targets, _ = mixup_one_target(data_a, data_b, targets_a, targets_b, alpha=1.0, is_bias=True)
        # data_mix = l * data_a + (1. - l) * data_b
        outputs = self.predict(data_mix)
        # outputs = self.predict_mix(data_a, data_b, l)
        outputs = self.de_interleave(outputs, 2)
        logits_ = torch.softmax(outputs, dim=1)
        # mix_targets = l * targets_a + (1. - l) * targets_b

        logits_ = torch.clamp(logits_, 1e-4, 1. - 1e-4)

        outputs_p_w_mix = logits_[idx_p]
        outputs_u_w_mix = logits_[idx_u]
        targets_p_mix = mix_targets[idx_p]
        targets_u_mix = mix_targets[idx_u]

        loss = self.p3mix_c_loss(outputs_p_w_mix, outputs_u_w_mix, targets_p_mix, targets_u_mix, logits_)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if epoch.is_integer():
            self.scheduler.step()
        return {'loss': loss.item()}

    def get_hinput(self, top_k, batch_size):
        h_input_b_idx = np.random.randint(top_k, size=batch_size)    # select cnd for mpn
        return h_input_b_idx

    # update cnd
    def update_hinput(self, top_k, train_p_loader):
        with torch.no_grad():
            h_inputs_x_ = []
            h_entropys_x_ = []
            h_preds_x_ = []
            for _, data, _ in train_p_loader:
                data = data[0].to(device, non_blocking=True)
                outputs_x = self.predict(data)
                preds_x = torch.softmax(outputs_x, dim=1)[:, 1]
                entropys_x = -(preds_x * torch.log(preds_x) +
                               (1. - preds_x) * torch.log(1. - preds_x))
                preds_x = preds_x.view(data.size(0))
                entropys_x = entropys_x.view(data.size(0))

                h_inputs_x_.extend(np.array(data.cpu()))
                h_entropys_x_.extend(np.array(entropys_x.cpu()))
                h_preds_x_.extend(np.array(preds_x.cpu()))

        h_inputs_x_ = np.array(h_inputs_x_)
        h_entropys_x_ = np.array(h_entropys_x_)
        h_preds_x_ = np.array(h_preds_x_)

        h_group_x = list(
            zip(h_inputs_x_, h_entropys_x_, h_preds_x_))
        h_group_x.sort(key=lambda x: x[1], reverse=True)

        sort_h_inputs_x_c = [x[0] for x in h_group_x[:top_k]]
        sort_h_entropys_x_c = [x[1] for x in h_group_x[:top_k]]
        sort_h_preds_x_c = [x[2] for x in h_group_x[:top_k]]

        self.h_inputs_x = torch.tensor(sort_h_inputs_x_c, device=device)
        self.h_entropys_x = torch.tensor(sort_h_entropys_x_c, device=device)
        self.h_preds_x = torch.tensor(sort_h_preds_x_c, device=device)
        print("===== update cnd ======")


    def p3mix_c_loss(self, outputs_p_w_mix, outputs_u_w_mix, targets_p_mix, targets_u_mix, logits_):
        # coefficient for loss
        positive_weight = self.hparams["positive_weight"]
        unlabeled_weight = self.hparams["unlabeled_weight"]
        entropy_weight = self.hparams["entropy_weight"]
        loss_p = -(targets_p_mix *
                   (outputs_p_w_mix).log()).sum(1).mean()
        loss_u = -(targets_u_mix *
                   (outputs_u_w_mix).log()).sum(1).mean()
        loss_ent = -(logits_ * logits_.log()).sum(1).mean()

        loss = positive_weight * loss_p + unlabeled_weight * loss_u + entropy_weight * loss_ent
        return loss

    def predict(self, x):
        return self.network(x)

    # def predict_mix(self, data_a, data_b, l):
    #     feature_a = self.featurizer(data_a)
    #     feature_b = self.featurizer(data_b)
    #     features = l * feature_a + (1. - l) * feature_b
    #     outputs = self.classifier(features)
    #     return outputs



class P3MIX_E(Algorithm):
    """
    P3MIX_E
    Reference: Who Is Your Right Mixup Partner in Positive and Unlabeled Learning, ICLR 2022.
    """

    def __init__(self, p_input_shape, u_input_shape, hparams):
        super(P3MIX_E, self).__init__(p_input_shape, u_input_shape, hparams)
        # self.featurizer = networks.Featurizer(p_input_shape, self.hparams)
        # self.classifier = networks.Classifier(
        #     self.featurizer.n_outputs,
        #     self.num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.ema_network = copy.deepcopy(self.network)
        self.ema_network.to(device)
        for param in self.ema_network.parameters():
            param.detach_()
        # self.optimizer = torch.optim.SGD(
        #     self.network.parameters(),
        #     lr=self.hparams["lr"],
        #     momentum=self.hparams["momentum"]
        # )
        # self.base_loss = nn.CrossEntropyLoss()
        self.scheduler = MultiStepLR(self.optimizer,
                                     milestones=self.hparams["milestones"],
                                     gamma=self.hparams["scheduler_gamma"])
        self.h_inputs_x = torch.Tensor([]).to(device)
        self.h_preds_x = torch.Tensor([]).to(device)

    def update(self, inputs_p_w, targets_p, inputs_u_w, targets_u, epoch, train_p_dataset):
        self.ema_network.train()
        start_hmix = self.hparams["start_hmix"]
        gamma = self.hparams["gamma"]
        top_k = self.hparams["top_k"]
        ema_start = self.hparams["ema_start"]


        batch_size = inputs_p_w.shape[0]
        # data = torch.cat((inputs_p_w, inputs_u_w), dim=0)
        # inputs_all = self.interleave(data, 2)
        # outputs_all = self.predict(inputs_all)
        # outputs_all = self.de_interleave(outputs_all, 2)
        # outputs_all = torch.softmax(outputs_all, dim=1)[:, 1]
        # outputs_p_w = outputs_all[:batch_size]
        # outputs_u_w = outputs_all[batch_size:]
        # outputs_u_w = outputs_u_w[:, None]
        outputs_p_w, outputs_u_w = self.output_after_interleave(torch.cat((inputs_p_w, inputs_u_w)), inputs_p_w.shape[0])
        data = torch.cat((inputs_p_w, inputs_u_w), dim=0)
        outputs_u_w = torch.softmax(outputs_u_w, dim=1)[:, 1]
        outputs_u_w = outputs_u_w[:, None]
        idx_p = slice(0, batch_size)
        idx_u = slice(batch_size, len(data))
        target_p = targets_p[:, None]
        target_u = targets_u[:, None]
        target_p_ = torch.cat((1. - target_p, target_p), dim=1) # p [0,1]
        target_u_ = torch.cat((1. - target_u, target_u), dim=1) # u [1,0]
        targets_ = torch.cat((target_p_, target_u_), dim=0)

        # update cnd each epoch
        if epoch.is_integer():
            print("===== update cnd ======")
            print("epoch= ", epoch)
            train_p_loader = torch.utils.data.DataLoader(dataset=train_p_dataset, batch_size=len(train_p_dataset), shuffle=False)
            self.update_hinput(top_k, train_p_loader)

        with torch.no_grad():
            outputs_ema = self.ema_predict(data)
            targets_elr = torch.softmax(outputs_ema, dim=1)

            if epoch > start_hmix:
                p_indicator_u = outputs_u_w
                p_indicator_p = torch.ones(batch_size,
                                           dtype=torch.float32,
                                           device=device)
                p_indicator_p = p_indicator_p[:, None]
                p_indicator = torch.cat((p_indicator_p, p_indicator_u),
                                        dim=0)
                p_indicator = p_indicator.view(p_indicator.size(0))

                h_p = torch.ones(len(self.h_inputs_x),
                                 dtype=torch.float32,
                                 device=device)
                h_p = h_p[:, None]
                h_targets_elr = torch.cat([1. - h_p, h_p], dim=1)

        if epoch > start_hmix:
            h_input_b_idx = self.get_hinput(top_k, batch_size)  # for all unlabeled data
            h_target_b = torch.ones(len(h_input_b_idx),
                                    dtype=torch.float32,
                                    device=device)
            h_target_b = h_target_b[:, None]
            h_target_b_ = torch.cat([1. - h_target_b, h_target_b],
                                    dim=1)  # [0,1]

            idx1 = torch.tensor(
                np.random.randint(data.size(0), size=batch_size))  # for positive data, randomly select a partner
            data_b1 = torch.cat(
                [data[idx1], self.h_inputs_x[h_input_b_idx]], dim=0)
            targets_b1 = torch.cat([targets_[idx1], h_target_b_],
                                   dim=0)
            targets_elr_b1 = torch.cat(
                [targets_elr[idx1], h_targets_elr[h_input_b_idx]])

            idx2 = torch.tensor(
                np.random.randint(data.size(0), size=batch_size))  # for unlabeled data, randomly select a partner

            idx = torch.cat([idx1, idx2])
            data_b, targets_b, targets_elr_b = data[idx], targets_[idx], targets_elr[idx]

            p_indicator[p_indicator >= gamma] = 1.
            p_indicator[p_indicator <= 1-gamma] = 1.
            p_indicator[idx_p] = 1.
            p_indicator[p_indicator != 1.] = 0.

            data_b = (p_indicator * data_b.swapdims(0, -1) +
                    (1. - p_indicator) *
                      data_b1.swapdims(0, -1)).swapdims(0, -1)
            targets_b = (p_indicator * targets_b.swapdims(0, -1) +
                         (1. - p_indicator) *
                         targets_b1.swapdims(0, -1)).swapdims(0, -1)
            targets_elr_b = (
                    p_indicator * targets_elr_b.swapdims(0, -1) +
                    (1. - p_indicator) *
                    targets_elr_b1.swapdims(0, -1)).swapdims(0, -1)
        else:
            idx = torch.randperm(data.size(0))
            data_b, targets_b, targets_elr_b = data[idx], targets_[idx], targets_elr[idx]
        data_a, targets_a, targets_elr_a = data, targets_, targets_elr

        data_a = self.interleave(data_a, 2)
        data_b = self.interleave(data_b, 2)
        # data_mix = l * data_a + (1. - l) * data_b
        data_mix, mix_targets, l = mixup_one_target(data_a, data_b, targets_a, targets_b, alpha=1.0, is_bias=True)
        outputs = self.predict(data_mix)
        # outputs = self.predict_mix(data_a, data_b, l)
        outputs = self.de_interleave(outputs, 2)
        logits_ = torch.softmax(outputs, dim=1)
        # mix_targets = l * targets_a + (1. - l) * targets_b
        mix_targets_elr = l * targets_elr_a + (1. - l) * targets_elr_b

        # logits_ = torch.cat([1. - logits, logits], dim=1)
        logits_ = torch.clamp(logits_, 1e-4, 1. - 1e-4)

        outputs_p_w_mix = logits_[idx_p]
        outputs_u_w_mix = logits_[idx_u]
        targets_p_mix = mix_targets[idx_p]
        targets_u_mix = mix_targets[idx_u]

        loss = self.p3mix_e_loss(outputs_p_w_mix, outputs_u_w_mix, targets_p_mix, targets_u_mix, mix_targets_elr, logits_)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if epoch.is_integer() and self.check_mean_teacher(epoch, ema_start):
            ema_decay = self.hparams["ema_decay"]
            # update parameters of ema_net
            self.update_ema_variables(self.network, self.ema_network, ema_decay, epoch, ema_start)
        if epoch.is_integer():
            self.scheduler.step()
        return {'loss': loss.item()}


    def get_hinput(self, top_k, batch_size):
        h_input_b_idx = np.random.randint(top_k, size=batch_size)    # select cnd for mpn
        return h_input_b_idx

    # update cnd
    def update_hinput(self, top_k, train_p_loader):
        with torch.no_grad():
            h_inputs_x_ = []
            h_entropys_x_ = []
            h_preds_x_ = []
            for _, data, _ in train_p_loader:
                data = data[0].to(device, non_blocking=True)
                outputs_x = self.predict(data)
                preds_x = torch.softmax(outputs_x, dim=1)[:, 1]
                entropys_x = -(preds_x * torch.log(preds_x) +
                               (1. - preds_x) * torch.log(1. - preds_x))
                preds_x = preds_x.view(data.size(0))
                entropys_x = entropys_x.view(data.size(0))

                h_inputs_x_.extend(np.array(data.cpu()))
                h_entropys_x_.extend(np.array(entropys_x.cpu()))
                h_preds_x_.extend(np.array(preds_x.cpu()))

        h_inputs_x_ = np.array(h_inputs_x_)
        h_entropys_x_ = np.array(h_entropys_x_)
        h_preds_x_ = np.array(h_preds_x_)

        h_group_x = list(
            zip(h_inputs_x_, h_entropys_x_, h_preds_x_))
        h_group_x.sort(key=lambda x: x[1], reverse=True)

        sort_h_inputs_x_c = [x[0] for x in h_group_x[:top_k]]
        sort_h_entropys_x_c = [x[1] for x in h_group_x[:top_k]]
        sort_h_preds_x_c = [x[2] for x in h_group_x[:top_k]]

        self.h_inputs_x = torch.tensor(sort_h_inputs_x_c, device=device)
        self.h_entropys_x = torch.tensor(sort_h_entropys_x_c, device=device)
        self.h_preds_x = torch.tensor(sort_h_preds_x_c, device=device)


    def p3mix_e_loss(self, outputs_p_w_mix, outputs_u_w_mix, targets_p_mix, targets_u_mix, mix_targets_elr, logits_):
        # coefficient for loss
        positive_weight = self.hparams["positive_weight"]
        unlabeled_weight = self.hparams["unlabeled_weight"]
        entropy_weight = self.hparams["entropy_weight"]
        elr_weight = self.hparams["elr_weight"]

        loss_p = -(targets_p_mix *
                   (outputs_p_w_mix).log()).sum(1).mean()
        loss_u = -(targets_u_mix *
                   (outputs_u_w_mix).log()).sum(1).mean()
        loss_ent = -(logits_ * logits_.log()).sum(1).mean()
        loss_elr = ((1. - (mix_targets_elr * logits_).sum(dim=1)).log()).mean()

        loss = positive_weight * loss_p + unlabeled_weight * loss_u + entropy_weight * loss_ent + elr_weight * loss_elr
        return loss

    def update_ema_variables(self, model, ema_model, alpha, epoch, ema_start):
        # ema_update = False

        alpha = min(1 - 1 / (epoch - ema_start + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_((param), alpha=1. - alpha)
        print("update ema network. current epoch: ", epoch)

    def check_mean_teacher(self, epoch, ema_start):
        mean_teacher = True
        if not mean_teacher:
            return False
        elif epoch < ema_start:
            return False
        else:
            return True

    def predict(self, x):
        return self.network(x)

    def ema_predict(self, x):
        return self.ema_network(x)


class Robust_PU(Algorithm):
    """
    Robust_PU
    Reference: Robust Positive-Unlabeled Learning via Noise Negative Sample Self-correction, KDD 2023.
    """

    def __init__(self, p_input_shape, u_input_shape, hparams):
        super(Robust_PU, self).__init__(p_input_shape, u_input_shape, hparams)
        self.featurizer = networks.Featurizer(p_input_shape, self.hparams)
        if self.hparams['dataset']== 'IMAGENETTE':
            self.classifier = networks.Classifier(256, self.num_classes)
        elif self.hparams['dataset'] in ['CIFAR10','USPS','Letter']:
            self.classifier = networks.Classifier(self.featurizer.n_outputs, self.num_classes)
        else:
            pass

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer_pretrain = torch.optim.SGD(
            self.network.parameters(),
            lr=self.hparams["lr_pretrain"],
            momentum=self.hparams["momentum"]
        )
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.hparams["lr"],
            momentum= self.hparams["momentum"]
        )
        self.scheduler_pretrain = transformers.get_constant_schedule_with_warmup(self.optimizer_pretrain, num_warmup_steps=0)
        self.scheduler = transformers.get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=0)

        self.base_loss = nn.CrossEntropyLoss(reduction='none')
        self.weight_p = torch.ones(60000, device=device, dtype=torch.float32)
        self.weight_u = torch.ones(60000, device=device, dtype=torch.float32)
        # self.best_val_acc = 0
        # self.val_best_epoch = 0

    def update(self, inputs_p_w, targets_p, inputs_u_w, targets_u, epoch, p_idx, u_idx):
        pretrain_epoch = self.hparams["pretrain_epoch"]
        # patience = self.hparams["patience"]

        batch_size = inputs_p_w.shape[0]
        inputs_all = self.interleave(torch.cat((inputs_p_w, inputs_u_w)), 2)
        outputs_all = self.predict(inputs_all)
        outputs_all = self.de_interleave(outputs_all, 2)
        outputs_p_w = outputs_all[:batch_size]
        outputs_u_w = outputs_all[batch_size:]
        # pretrain
        if epoch < pretrain_epoch:
            loss = self.nnpu_loss(outputs_p_w, outputs_u_w, targets_p, targets_u)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if epoch.is_integer():
                self.scheduler_pretrain.step()
        else:
            # if self.check_update_weight(epoch):
            #     val_acc = self.cal_val_acc(val_p_loader, val_u_loader)
            #     if val_acc > self.best_val_acc:
            #         self.best_val_acc = val_acc
            #         self.val_best_epoch = epoch
            #         # print("epoch update val best: ", epoch)
            #     else:
            #         if (epoch - self.val_best_epoch) >= patience:
            #             print(f'=== Break at epoch {self.val_best_epoch + 1} ===')
            # Stage 3: Weighted Supervised Training
            weights_p = self.weight_p[p_idx]
            print("mean of weights_p: ", torch.mean(weights_p))
            weights_u = self.weight_u[u_idx]
            print("mean of weights_u: ", torch.mean(weights_u))
            loss = self.robust_pu_loss(outputs_p_w, outputs_u_w, targets_p, targets_u, weights_p, weights_u)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if epoch.is_integer():
                self.scheduler.step()

        return {'loss': loss.item()}

    # def cal_val_acc(self, val_p_loader, val_u_loader):
    #     correct = 0
    #     total = 0
    #     self.network.eval()
    #     with torch.no_grad():
    #         for _, (inputs_p_w, inputs_p_s), targets_p in val_p_loader:
    #             inputs_p_w = inputs_p_w.to(device)
    #             targets_p = targets_p.to(device)
    #             p = self.network.predict(inputs_p_w)
    #             if p.size(1) == 1:
    #                 correct += p.gt(0.5).eq(targets_p).float().sum().item()
    #             else:
    #                 correct += p.argmax(1).eq(targets_p).float().sum().item()
    #             total += len(inputs_p_w)
    #         for _, (inputs_u_w, inputs_u_s), (targets_u, targets_u_true) in val_u_loader:
    #             inputs_u_w = inputs_u_w.to(device)
    #             targets_u_true = targets_u_true.to(device)
    #             p = self.network.predict(inputs_u_w)
    #             if p.size(1) == 1:
    #                 correct += p.gt(0.5).eq(targets_u_true).float().sum().item()
    #             else:
    #                 correct += p.argmax(1).eq(targets_u_true).float().sum().item()
    #             total += len(inputs_u_w)
    #     self.network.train()
    #
    #     return correct / total


    def robust_pu_loss(self, outputs_p_w, outputs_u_w, targets_p, targets_u, weights_p, weights_u):
        # p
        loss_p = self.base_loss(outputs_p_w, targets_p)
        # print("loss_p: ", loss_p)
        # print(loss_p.shape)
        # print("weights_p: ", weights_p)
        # print(weights_p.shape)
        loss_p = torch.mean(loss_p * weights_p.to(outputs_p_w.device))
        # u
        loss_u = self.base_loss(outputs_u_w, targets_u)
        loss_u = torch.mean(loss_u * weights_u.to(outputs_u_w.device))
        total_loss = loss_p + loss_u
        return total_loss

    def nnpu_loss(self, outputs_p_w, outputs_u_w, targets_p, targets_u):
        # R_p^+
        loss_pos_pos = self.pi * self.base_loss(outputs_p_w, targets_p)
        loss_pos_pos = torch.mean(loss_pos_pos)
        # R_u^_
        loss_unlabel_neg = self.base_loss(outputs_u_w, targets_u)
        loss_unlabel_neg = torch.mean(loss_unlabel_neg)
        # R_p^_
        targets_p_reverse = (1 - targets_p).to(targets_p.dtype)
        loss_pos_neg = - self.pi * self.base_loss(outputs_p_w, targets_p_reverse)
        loss_pos_neg = torch.mean(loss_pos_neg)
        self.lda = torch.tensor([0.0]).to(outputs_p_w.device)
        pretrain_loss = loss_pos_pos + torch.max(self.lda, loss_unlabel_neg + loss_pos_neg)
        return pretrain_loss

    def logistic_loss(self, z, labels):
        return F.softplus(-z * labels)

    def calculate_spl_weights(self, x, thresh, eps=1e-1):
        assert thresh > 0., 'spl threshold must be positive'
        weights = torch.exp(-x / (thresh * thresh))
        assert weights.min() >= 0. - eps and weights.max() <= 1. + eps, f'weight [{weights.min()}, {weights.max()}] must in range [0., 1.]'
        return weights

    def get_next_ratio(self, type, init_ratio, max_thresh, grow_steps, step):
        lam = 0.5
        cal_lib = torch if isinstance(init_ratio, torch.Tensor) else math

        if type == 'const':
            ratio = init_ratio
        elif type == 'linear':
            ratio = init_ratio + (max_thresh - init_ratio) / grow_steps * step
        elif type == 'convex':  # from fast to slow
            ratio = init_ratio + (max_thresh - init_ratio) * cal_lib.sin(step / grow_steps * np.pi * 0.5)
        elif type == 'concave':  # from slow to fast
            if step > grow_steps:
                ratio = max_thresh
            else:
                ratio = init_ratio + (max_thresh - init_ratio) * (1. - cal_lib.cos(step / grow_steps * np.pi * 0.5))
        elif type == 'exp':
            assert 0 <= lam <= 1
            ratio = init_ratio + (max_thresh - init_ratio) * (1. - lam ** step)
        else:
            raise NotImplementedError(f'Invalid Training Scheduler type {type}')

        if init_ratio < max_thresh:
            ratio = min(ratio, max_thresh)
        else:
            ratio = max(ratio, max_thresh)
        # step += 1
        return ratio

    def check_update_weight(self, epoch):
        pretrain_epoch = self.hparams["pretrain_epoch"]
        freq = self.hparams["freq"]
        # print((int(epoch) - pretrain_epoch) % freq)
        if epoch >= pretrain_epoch and (epoch - pretrain_epoch) % freq == 0:
            print("update the weight")
            return True
        else:
            return False

    def update_weight(self, train_p_dataset, train_u_dataset, epoch):
        temper_p = self.hparams["temper_p"]
        temper_n = self.hparams["temper_n"]

        freq = self.hparams["freq"]
        scheduler_type_p = self.hparams["scheduler_type_p"]
        scheduler_type_n = self.hparams["scheduler_type_n"]

        alpha_p = self.hparams["alpha_p"]
        alpha_n = self.hparams["alpha_n"]

        max_thresh_p = self.hparams["max_thresh_p"]
        max_thresh_n = self.hparams["max_thresh_n"]

        grow_steps_p = self.hparams["grow_steps_p"]
        grow_steps_n = self.hparams["grow_steps_n"]

        pretrain_epoch = self.hparams["pretrain_epoch"]

        train_p_loader = torch.utils.data.DataLoader(dataset=train_p_dataset, batch_size=64,
                                                     shuffle=False)
        train_u_loader = torch.utils.data.DataLoader(dataset=train_u_dataset, batch_size=64,
                                                     shuffle=False)

        rel_epoch = (epoch - pretrain_epoch)
        step = int(rel_epoch / freq)

        print("step= ", step)

        self.network.eval()
        with torch.no_grad():
            for data_batch in train_p_loader:
                idx, (inputs_w, _), targets = data_batch
                inputs_w = inputs_w.to(device)
                logits = self.predict(inputs_w)
                weight_p = self.logistic_loss(logits[:, 1] / temper_p, 1)
                thresh_p = self.get_next_ratio(scheduler_type_p, alpha_p, max_thresh_p, grow_steps_p, step)
                weight_p = self.calculate_spl_weights(weight_p.detach(), thresh_p)
                self.weight_p[idx] = weight_p

            for data_batch in train_u_loader:
                idx, (inputs_w, _), targets = data_batch
                inputs_w = inputs_w.to(device)
                logits = self.predict(inputs_w)
                weight_u = self.logistic_loss(logits[:, 1] / temper_n, 1)
                thresh_n = self.get_next_ratio(scheduler_type_n, alpha_n, max_thresh_n, grow_steps_n, step)
                weight_u = self.calculate_spl_weights(weight_u.detach(), thresh_n)
                self.weight_u[idx] = weight_u
        self.network.train()

    def predict(self, x):
        return self.network(x)













