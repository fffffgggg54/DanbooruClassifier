# https://paperswithcode.com/paper/multi-label-classification-with-partial
# carry over functions for multi-label classification

# TODO implement cutouts in dset transforms

# https://github.com/Alibaba-MIIL/PartialLabelingCSL
# https://github.com/Alibaba-MIIL/ASL

# TODO make loss that multiplies loss or logit by tanh(-x) or sigmoid centered around threshold (intuition to not punish unlabeled positives)

# not sure what this ModelEma function does
# seems to be making a duplicate of the model with the weights set to running average of weights from the actual model
# adds a decay to the weights(?) in order to empasize the current model weights over older ones
import torch
from torch import nn as nn, Tensor
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
from copy import deepcopy
import math

import torch.distributed
import scipy.stats





'''


class LeakyLossV1(nn.Module):
    def __init__(self, device):
        super(LeakyLossV1, self).__init__()
        
        self.device = device

        self.targets_weights = None


    def forward(self, logits, targets, priorValues = None):

        
'''


# https://github.com/xinyu1205/robust-loss-mlml

class Hill(nn.Module):
    r""" Hill as described in the paper "Robust Loss Design for Multi-Label Learning with Missing Labels "
    .. math::
        Loss = y \times (1-p_{m})^\gamma\log(p_{m}) + (1-y) \times -(\lambda-p){p}^2 
    where : math:`\lambda-p` is the weighting term to down-weight the loss for possibly false negatives,
          : math:`m` is a margin parameter, 
          : math:`\gamma` is a commonly used value same as Focal loss.
    .. note::
        Sigmoid will be done in loss. 
    Args:
        lambda (float): Specifies the down-weight term. Default: 1.5. (We did not change the value of lambda in our experiment.)
        margin (float): Margin value. Default: 1 . (Margin value is recommended in [0.5,1.0], and different margins have little effect on the result.)
        gamma (float): Commonly used value same as Focal loss. Default: 2
    """

    def __init__(self, lamb: float = 1.5, margin: float = 1.0, gamma: float = 2.0,  reduction: str = 'sum') -> None:
        super(Hill, self).__init__()
        self.lamb = lamb
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        call function as forward
        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
        Returns:
            torch.Tensor: loss
        """

        # Calculating predicted probability
        logits_margin = logits - self.margin
        pred_pos = torch.sigmoid(logits_margin)
        pred_neg = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred_pos) * targets + (1 - targets)
        focal_weight = pt ** self.gamma

        # Hill loss calculation
        los_pos = targets * torch.log(pred_pos)
        los_neg = (1-targets) * -(self.lamb - pred_neg) * pred_neg ** 2

        loss = -(los_pos + los_neg)
        loss *= focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SPLC(nn.Module):
    r""" SPLC loss as described in the paper "Simple Loss Design for Multi-Label Learning with Missing Labels "
    .. math::
        &L_{SPLC}^+ = loss^+(p)
        &L_{SPLC}^- = \mathbb{I}(p\leq \tau)loss^-(p) + (1-\mathbb{I}(p\leq \tau))loss^+(p)
    where :math:'\tau' is a threshold to identify missing label 
          :math:`$\mathbb{I}(\cdot)\in\{0,1\}$` is the indicator function, 
          :math: $loss^+(\cdot), loss^-(\cdot)$ refer to loss functions for positives and negatives, respectively.
    .. note::
        SPLC can be combinded with various multi-label loss functions. 
        SPLC performs best combined with Focal margin loss in our paper. Code of SPLC with Focal margin loss is released here.
        Since the first epoch can recall few missing labels with high precision, SPLC can be used ater the first epoch.
        Sigmoid will be done in loss. 
    Args:
        tau (float): threshold value. Default: 0.6
        change_epoch (int): which epoch to combine SPLC. Default: 1
        margin (float): Margin value. Default: 1
        gamma (float): Hard mining value. Default: 2
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'sum'``
        """

    def __init__(self,
                 tau: float = 0.6,
                 change_epoch: int = 1,
                 margin: float = 1.0,
                 gamma: float = 2.0,
                 reduction: str = 'sum') -> None:
        super(SPLC, self).__init__()
        self.tau = tau
        self.change_epoch = change_epoch
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.LongTensor,
                epoch) -> torch.Tensor:
        """
        call function as forward
        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
            epoch : The epoch of current training.
        Returns:
            torch.Tensor: loss
        """
        # Subtract margin for positive logits
        logits = torch.where(targets == 1, logits-self.margin, logits)
        
        # SPLC missing label correction
        if epoch >= self.change_epoch:
            targets = torch.where(
                torch.sigmoid(logits) > self.tau,
                torch.tensor(1).cuda(), targets)
        
        pred = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred) * targets + pred * (1 - targets)
        focal_weight = pt**self.gamma

        los_pos = targets * F.logsigmoid(logits)
        los_neg = (1 - targets) * F.logsigmoid(-logits)

        loss = -(los_pos + los_neg)
        loss *= focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class SPLCModified(nn.Module):

    def __init__(self,
                 tau: float = 0.6,
                 change_epoch: int = 1,
                 margin: float = 1.0,
                 gamma: float = 2.0,
                 alpha: float = 1e-4,
                 reduction: str = 'sum',
                 loss_fn: nn.Module = Hill()) -> None:
        super().__init__()
        self.tau = tau
        self.tau_per_class = None
        self.change_epoch = change_epoch
        self.margin = margin
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_fn = loss_fn
        if hasattr(self.loss_fn, 'reduction'):
            self.loss_fn.reduction = self.reduction


    def forward(self, logits: torch.Tensor, targets: torch.LongTensor,
                epoch) -> torch.Tensor:
        
        
        if self.tau_per_class == None:
            classCount = logits.size(dim=1)
            currDevice = logits.device
            self.tau_per_class = torch.ones(classCount, device=currDevice) * self.tau
        
        
        
        # Subtract margin for positive logits
        logits = torch.where(targets == 1, logits-self.margin, logits)
        
        pred = torch.sigmoid(logits)
        with torch.no_grad():
            alpha = self.alpha if logits.requires_grad else 0
            self.tau_per_class = self.tau_per_class * (1 - alpha * targets.sum(dim=0)) + alpha * (pred * targets).sum(dim=0)
        
        # SPLC missing label correction
        '''
        if epoch >= self.change_epoch:
            targets = torch.where(
                pred > self.tau_per_class,
                torch.tensor(1).cuda(), targets)
        '''
        
        if epoch >= self.change_epoch:
            targets = torch.where(
                (targets*pred+(1-targets)*(1-pred)) > self.tau_per_class,
                targets, 1-targets)
        
        
        '''

        # Focal margin for postive loss
        pt = (1 - pred) * targets + pred * (1 - targets)
        focal_weight = pt**self.gamma

        los_pos = targets * F.logsigmoid(logits)
        los_neg = (1 - targets) * F.logsigmoid(-logits)

        loss = -(los_pos + los_neg)
        loss *= focal_weight
        '''
        
        loss = self.loss_fn(logits, targets)
        
        '''
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        '''
        
        return loss

# A Modified Logistic Regression for Positive and Unlabeled Learning
# Jaskie et. al., 2019
# http://dx.doi.org/10.1109/IEEECONF44664.2019.9048765
class ModifiedLogisticRegression(nn.Module):
    def __init__(self, num_classes = 1588, initial_weight = 1.0, initial_beta = 1.0, eps = 1e-8):
        super().__init__()
        self.num_classes = num_classes
        self.weight_per_class = nn.Parameter(data=initial_weight * torch.ones(num_classes))
        self.beta_per_class = nn.Parameter(data=initial_beta * torch.ones(num_classes))
        self.eps = eps
        # store intermediate results as attributes to avoid memory realloc as per ASLOptimized
        self.NtC_out = None
        self.c_hat = None
        self.pred = None
        
        
    def forward(self, x):
        # P(s = 1 | x_bar) as per equation #4 and section 4.1 from paper
        self.NtC_out = 1/(1 + (self.beta_per_class ** 2) + torch.exp(-self.weight_per_class * x))
        # c_hat = 1 / (1 + b^2)
        # step isolated since we don't want to optimize beta here, conly compute c_hat
        with torch.no_grad():
            self.c_hat = 1 / (1 + self.beta_per_class ** 2)
        # P(y = 1 | x) as per section 4.2 from paper
        self.pred = self.NtC_out / (self.c_hat + self.eps)
        return self.pred

# rationale is that since this is going after the cls_head of an image backbone, the cls_head, where the linear layer already has a weight and bias term
# is redundant with the weight term of the MLR algorithm, hence an implementation where it is removed from the actual algorithm and handled in the image backbone
class ModifiedLogisticRegression_NoWeight(nn.Module):
    def __init__(self, num_classes = 1588, initial_beta = 1.0, eps = 1e-8):
        super().__init__()
        self.num_classes = num_classes
        self.beta_per_class = nn.Parameter(data=initial_beta * torch.ones(num_classes, dtype=torch.float64))
        self.eps = eps
        
        # store intermediate results as attributes to avoid memory realloc as per ASLOptimized
        self.NtC_out = None
        self.c_hat = None
        self.pred = None
        
    def forward(self, x):
        #return (1 + self.beta_per_class.detach() ** 2) / (1 + (self.beta_per_class ** 2) + torch.exp(-x) + self.eps)
        '''
        # P(s = 1 | x_bar) as per equation #4 and section 4.1 from paper
        # weight term handled in image backbone
        self.NtC_out = 1/(1 + (self.beta_per_class ** 2) + torch.exp(-x))
        # step isolated since we don't want to optimize beta here, only compute c_hat
        with torch.no_grad():
            self.c_hat = 1 / (1 + self.beta_per_class.detach() ** 2)
        # P(y = 1 | x) as per section 4.2 from paper
        self.pred = self.NtC_out / (self.c_hat + self.eps)
        return self.pred
        '''
        #with torch.no_grad():
        #c_hat = 1 / (1 + self.beta_per_class ** 2)
        
        with torch.no_grad():
            c_hat = 1 / (1 + self.beta_per_class.detach() ** 2)
        return c_hat / (1 + (self.beta_per_class ** 2) + torch.exp(-x) + self.eps)

class ModifiedLogisticRegression_Head(nn.Module):
    def __init__(self, num_features, num_classes = 1588, bias = True, initial_beta = 1.0, eps = 1e-8):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.fc = nn.Linear(num_features, num_classes, bias=bias)
        self.beta_per_class = nn.Parameter(data=initial_beta * torch.ones(num_classes, dtype=torch.float64))
        self.eps = eps
        
        # store intermediate results as attributes to avoid memory realloc as per ASLOptimized
        self.NtC_out = None
        self.c_hat = None
        self.pred = None
        
    def forward(self, x):
        
        if self.training:
            c_hat_inv = 1
        else:
            with torch.no_grad():
                c_hat_inv = 1 + self.beta_per_class.detach() ** 2 + self.eps
        
        '''
        with torch.no_grad():
            c_hat_inv = 1 + self.beta_per_class.detach() ** 2 + self.eps
        '''
        return c_hat_inv / (1 + (self.beta_per_class ** 2) + torch.exp(-self.fc(x)) + self.eps)

class DualLogisticRegression_Head(nn.Module):
    def __init__(self, num_features, num_classes = 1588, fc_type='linear', estimator_type='linear', bias_fc = True, bias_estimator = False, eps = 1e-8):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        
        if(fc_type == "linear"):
            self.fc = nn.Linear(num_features, num_classes, bias = bias_fc, dtype=torch.float64)
        elif(fc_type == "mlp"):
            self.fc = nn.Sequential(
                    nn.Linear(num_features, int(num_features * 4), bias = bias_fc, dtype=torch.float64),
                    nn.GELU(),
                    nn.Linear(int(num_features * 4), num_classes, bias = bias_fc, dtype=torch.float64))
                
        if(estimator_type == "linear"):
            self.estimator = nn.Linear(num_features, num_classes, bias = bias_estimator, dtype=torch.float64)
        elif(estimator_type == "mlp"):
            self.estimator = nn.Sequential(
                    nn.Linear(num_features, int(num_features * 4), bias = bias_estimator, dtype=torch.float64),
                    nn.GELU(),
                    nn.Linear(int(num_features * 4), num_classes, bias = bias_estimator, dtype=torch.float64))
            
        self.eps = eps
        
        
    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=False):
            x = x.to(torch.float64)
            
            if self.training:
                propensity_inv = 1
            else:
                with torch.no_grad():
                    propensity_inv = 1 + self.estimator(x.detach()) ** 2 + self.eps
                    #propensity_inv = 1 + torch.exp(-self.estimator(x.detach())) + self.eps
            
            '''
            with torch.no_grad():
                #propensity_inv = 1 + torch.exp(-self.estimator(x.detach())) + self.eps
                propensity_inv = 1 + self.estimator(x.detach()) ** 2 + self.eps
            '''
            return propensity_inv / (1 + self.estimator(x.detach()) ** 2 + torch.exp(-self.fc(x)) + self.eps)
            #return propensity_inv / (1 + torch.exp(-self.estimator(x.detach())) + torch.exp(-self.fc(x)) + self.eps)


class DualLogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes, eps = 1e-8):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(num_features, num_classes, dtype=torch.float64)
        self.estimator = nn.Linear(num_features, num_classes, dtype=torch.float64)
        self.eps = eps

    def forward(self, x):
        '''
        with torch.no_grad():
            propensity = 1/ (1+(self.estimator(x.detach())**2) + self.eps)
        '''
        
        with torch.set_grad_enabled(False):
            propensity = 1 / (1+(x.to(torch.float64) @ self.estimator.weight.transpose(0, 1) + self.estimator.bias)**2)
            #propensity = (x.detach().to(torch.float64) @ self.estimator.weight.transpose(0, 1) + self.estimator.bias).sigmoid()
            #print(propensity)
        
        with torch.set_grad_enabled(True):
            #propensity = (x.detach() @ self.estimator.weight.transpose(0, 1) + self.estimator.bias).sigmoid()
        
            x = torch.special.logit(propensity / (1+(self.estimator(x.detach().to(torch.float64))**2) + torch.exp(-self.fc(x.to(torch.float64))) + self.eps))
            #x = torch.special.logit(propensity / (1+torch.exp(-self.estimator(x.to(torch.float64)).detach()) + torch.exp(-self.fc(x.to(torch.float64))) + self.eps))
            #x = torch.special.logit(propensity / (1+ torch.exp(-self.fc(x)-self.estimator(x)) + self.eps))
            
            #x = torch.special.logit(self.fc(x).sigmoid() * self.estimator(x.detach()).sigmoid())
        return x

class MatryoshkaClassificationHead(nn.Module):
    def __init__(self, num_features, num_classes, k=6):
        super().__init__()
        if isinstance(k, list):
            self.num_subsets = len(k)
            self.feature_subsets = K
        else:
            self.num_subsets = k
            self.feature_subsets = [int(num_features // (2 ** i)) for i in range(k)]

        self.feature_subsets_ = torch.Tensor(self.feature_subsets).reshape(k, 1, 1)
        self.mask = torch.arange(num_features).unsqueeze(0).repeat(self.num_subsets, 1, 1) < self.feature_subsets_
        self.head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        self.mask = self.mask.to(x)
        x = x.repeat(self.num_subsets, 1, 1)
        x = x * self.mask
        x = self.head(x)
        return x

'''
class ClassEmbedClassifierHead(nn.Module):
    def __init__(
        self,
        num_features, 
        num_classes, 
        class_embed, 
        embed_drop=0.1, 
        embed_norm=True,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.embed_dim = class_embed.shape[1]
        
        self.embed_proj = nn.Linear(self.embed_dim, num_features + 1)
        assert len(class_embed) == num_classes, 'ClassEmbedClassifierHead got class_embed where dim 0 != num_classes'
        class_embed = class_embed.clone().detach() # copy instead of reference, detach gradient flow
        self.register_buffer("class_embed", class_embed)

        self.embed_drop = nn.Dropout(embed_drop)
        self.embed_norm = norm_layer(num_features + 1) if embed_norm else nn.Identity()

    def forward(self, x, q=None):
        proj = self.embed_drop(self.embed_norm(self.embed_proj(q or self.class_embed))).transpose(0,1)
        x = x @ proj[:-1] + proj[-1]
        return x

'''

from timm.layers import Mlp, GluMlp

class ClassEmbedClassifierHead(nn.Module):
    def __init__(
        self,
        num_features, 
        num_classes, 
        class_embed, 
        embed_drop=0.1, 
        embed_norm=True,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.embed_dim = class_embed.shape[1]

        self.concat_feature_size = self.embed_dim + self.num_features
        
        self.ffn = GluMlp(
            self.concat_feature_size, 
            hidden_features = self.concat_feature_size * 4,
            out_features = 1,
            norm_layer = norm_layer,
        )
        assert len(class_embed) == num_classes, 'ClassEmbedClassifierHead got class_embed where dim 0 != num_classes'
        class_embed = class_embed.clone().detach() # copy instead of reference, detach gradient flow
        self.register_buffer("class_embed", class_embed)

        self.embed_drop = nn.Dropout(embed_drop)
        self.embed_norm = norm_layer(self.embed_dim) if embed_norm else nn.Identity()

    def forward(self, x, q=None): # [B, C], [K, D]
        q = self.embed_drop(self.embed_norm(q or self.class_embed)).unsqueeze(0) # [1, K, D]
        q = q.expand(x.shape[0], -1, -1) # [B, K, D]
        x = torch.cat([x.unsqueeze(1).expand(-1, q.shape[1], -1), q], dim=-1) # [B, K, C+D]
        x = self.ffn(x).squeeze(-1) # [B, K]

        return x

def stepAtThreshold(x, threshold, k=5, base=10):
    return 1 / (1 + torch.pow(base, (0 - k) * (x - threshold)))

def zero_grad(p, set_to_none=False):
    if p.grad is not None:
        if set_to_none:
            p.grad = None
        else:
            if p.grad.grad_fn is not None:
                p.grad.detach_()
            else:
                p.grad.requires_grad_(False)
            p.grad.zero_()
    return p



# gradient based boundary calculation (not used rn xd)
class getDecisionBoundary(nn.Module):
    def __init__(self, initial_threshold = 0.5, lr = 1e-3, threshold_min = 0.2, threshold_max = 0.8, num_classes = 1588):
        super().__init__()
        self.thresholdPerClass = (torch.ones(num_classes).to(torch.float64) * initial_threshold)
        self.opt = None
        self.lr = lr
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.check_device = True
        
    def forward(self, preds, targs):
        if self.check_device:
            self.thresholdPerClass = self.thresholdPerClass.to(preds).requires_grad_(True)
            self.opt = torch.optim.SGD([self.thresholdPerClass], lr=self.lr, maximize=True)
            self.check_device = False
            
        # update only when training
        if self.training:
            # stepping fn, currently steep version of logistic fn
            predsModified = stepAtThreshold(preds.detach(), self.thresholdPerClass)
            numToMax = getSingleMetric(predsModified, targs, F1).sum()
            numToMax.backward()
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self.thresholdPerClass.data = self.thresholdPerClass.clamp(min=self.threshold_min, max=self.threshold_max)
        
        ''' old code that uses manual optimization calls instead of an optimizer
        # need fp64
        self.thresholdPerClass.retain_grad()
        self.thresholdPerClass = self.thresholdPerClass.to(torch.float64)
        if preds.requires_grad:
            preds = preds.detach()
            
            predsModified = stepAtThreshold(preds, self.thresholdPerClass)
            metrics = getAccuracy(predsModified, targs)

            numToMax = metrics[:,9].sum()

            # TODO clean up this optimization phase
            numToMax.backward()
            with torch.no_grad():
                new_threshold = self.lr * self.thresholdPerClass.grad
                self.thresholdPerClass.add_(new_threshold)
                self.thresholdPerClass = self.thresholdPerClass.clamp(min=self.threshold_min, max=self.threshold_max)
            
            self.thresholdPerClass = zero_grad(self.thresholdPerClass)
            self.thresholdPerClass = self.thresholdPerClass.detach()
            self.thresholdPerClass.requires_grad=True
        '''
        return self.thresholdPerClass.detach()

# gradient based boundary calculation
class getDecisionBoundaryWorking(nn.Module):
    def __init__(self, initial_threshold = 0.5, lr = 1e-3, threshold_min = 0.2, threshold_max = 0.8):
        super().__init__()
        self.initial_threshold = initial_threshold
        self.thresholdPerClass = None
        self.opt = None
        self.needs_init = True
        self.lr = lr
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        
    def forward(self, preds, targs, update=True, step_opt=True):
        # parameter initial_threshold
        # TODO clean this up and make it work consistently, use proper lazy init
        with torch.no_grad():
            if self.needs_init:
                classCount = preds.size(dim=1)
                currDevice = preds.device
                if self.thresholdPerClass == None:
                    self.thresholdPerClass = nn.Parameter(torch.ones(classCount, device=currDevice, requires_grad=True).to(torch.float64) * self.initial_threshold)
                else:
                    self.thresholdPerClass = nn.Parameter(torch.ones(classCount, device=currDevice, requires_grad=True).to(torch.float64) * self.thresholdPerClass)
                self.needs_init = False
                self.opt = torch.optim.SGD(self.parameters(), lr=self.lr, maximize=True)
                #self.opt = torch.optim.SGD(self.parameters(), lr=self.lr, maximize=False)
                #self.criterion = AsymmetricLossOptimized(gamma_neg=0, gamma_pos=0, clip=0.0, eps=1e-8, disable_torch_grad_focal_loss=False)
                
        # update only when training
        if update:
            # ignore what happened before, only need values
            # stepping fn, currently steep version of logistic fn
            predsModified = stepAtThreshold(preds.detach(), self.thresholdPerClass)
            numToMax = getSingleMetric(predsModified, targs, PU_F_Metric).sum()
            #numToMax = AUL(predsModified, targs).sum()
            numToMax.backward()
            #loss = self.criterion(torch.special.logit(predsModified), targs)
            #loss.backward()
            if step_opt:
                with torch.no_grad():
                    self.opt.step()
                    self.opt.zero_grad(set_to_none=True)
                    self.thresholdPerClass.data = self.thresholdPerClass.detach().clamp(min=self.threshold_min, max=self.threshold_max)
        
        ''' old code that uses manual optimization calls instead of an optimizer
        # need fp64
        self.thresholdPerClass.retain_grad()
        self.thresholdPerClass = self.thresholdPerClass.to(torch.float64)
        if preds.requires_grad:
            preds = preds.detach()
            
            predsModified = stepAtThreshold(preds, self.thresholdPerClass)
            metrics = getAccuracy(predsModified, targs)

            numToMax = metrics[:,9].sum()

            # TODO clean up this optimization phase
            numToMax.backward()
            with torch.no_grad():
                new_threshold = self.lr * self.thresholdPerClass.grad
                self.thresholdPerClass.add_(new_threshold)
                self.thresholdPerClass = self.thresholdPerClass.clamp(min=self.threshold_min, max=self.threshold_max)
            
            self.thresholdPerClass = zero_grad(self.thresholdPerClass)
            self.thresholdPerClass = self.thresholdPerClass.detach()
            self.thresholdPerClass.requires_grad=True
        '''
        return self.thresholdPerClass.detach()


class thresholdPenalty(nn.Module):
    def __init__(self, threshold_multiplier, initial_threshold = 0.5, lr = 1e-3, threshold_min = 0.2, threshold_max = 0.8, num_classes = 1588):
        super().__init__()
        self.thresholdCalculator = getDecisionBoundaryWorking(
            initial_threshold = initial_threshold,
            lr = lr, 
            threshold_min = threshold_min, 
            threshold_max = threshold_max,
            #num_classes = num_classes
        )
        self.threshold_multiplier = threshold_multiplier
        # external call changes order, probably insignificant
        self.updateThreshold = self.thresholdCalculator.forward
        self.shift = None
        
    # forward step in model
    def forward(self, logits):
        # detached call should prevent model optim from affecting threshold parameters
        with torch.no_grad():
            self.shift = self.threshold_multiplier * 0 if self.thresholdCalculator.needs_init else torch.special.logit(self.thresholdCalculator.thresholdPerClass.detach().to(logits).detach())

        return logits + self.shift
    
    

# derived from SW-CV-ModelZoo/tools/analyze_metrics.py

class getDecisionBoundaryOld(nn.Module):
    def __init__(self, initial_threshold = 0.5, alpha = 1e-4, threshold_min = 0.01, threshold_max = 0.95):
        super().__init__()
        self.initial_threshold = initial_threshold
        self.thresholdPerClass = None
        self.alpha = alpha
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        
    def forward(self, preds, targs):
        if self.thresholdPerClass == None:
            classCount = preds.size(dim=1)
            currDevice = preds.device
            self.thresholdPerClass = torch.ones(classCount, device=currDevice) * self.initial_threshold
        
        
        with torch.no_grad():
            # TODO update with logic to include current thresholds in calculation of per-batch threshold
            threshold_min = torch.ones(len(self.thresholdPerClass), device=self.thresholdPerClass.device) * self.threshold_min
            threshold_max = torch.ones(len(self.thresholdPerClass), device=self.thresholdPerClass.device) * self.threshold_max
            threshold = (threshold_max + threshold_min) / 2
            recall = torch.ones(len(self.thresholdPerClass), device=self.thresholdPerClass.device) * 0.0
            precision = torch.ones(len(self.thresholdPerClass), device=self.thresholdPerClass.device) * 1.0
            
            adjustmentStopMask = torch.isclose(recall, precision).float()
            lastAdjustmentStopMask = adjustmentStopMask
            lastChange = 0
            
            if preds.requires_grad:
                
                while (1 - adjustmentStopMask).sum() > 0:
                    #print((1 - adjustmentStopMask).sum())
                    predsModified = (preds > threshold).float()
                    metrics = getAccuracy(predsModified, targs)

                    
                    # per-class stopping criterion
                    adjustmentStopMask = torch.isclose(metrics[:,4], metrics[:,6]).float() # recall_P, precision_P
                    
                    
                    # overall exit criterion
                    
                    # if there is any change, reset the no-change counter and update the change reference to new mask
                    if not torch.equal(adjustmentStopMask, lastAdjustmentStopMask):
                        lastAdjustmentStopMask = adjustmentStopMask
                        lastChange = 0
                    # increment the no-change counter
                    lastChange += 1
                    # exit if there hasn't been changes to the stopping mask for a while
                    if lastChange > 2:
                        break
                    
                    threshold = adjustmentStopMask * threshold + (1 - adjustmentStopMask) * (threshold_max + threshold_min) / 2
                    mask = (precision > recall).float()
                    threshold_max = mask * threshold + (1 - mask) * threshold_max
                    threshold_min = (1 - mask) * threshold + mask * threshold_min
        
        
            alpha = self.alpha if preds.requires_grad else 0
            #self.thresholdPerClass = self.thresholdPerClass * (1 - alpha * targs.sum(dim=0)) + alpha * (preds * targs).sum(dim=0)
            # weighting mask of each threshold shift
            deltaPerClass = alpha * targs.sum(dim=0) * adjustmentStopMask
            # EMA of per-class thresholds
            toUpdate = deltaPerClass * adjustmentStopMask
            self.thresholdPerClass = self.thresholdPerClass * (1 - toUpdate)  + threshold * toUpdate
            
        return self.thresholdPerClass

def z_score_to_p_value(x):
    return 0.5 * (1 + torch.erf(x/math.sqrt(2)))



def adjust_labels(logits, labels, dist_tracker, clip_dist = 0.95, clip_logit = 0.95, eps=1e-8):
    # use a z test
    class_z_scores = (dist_tracker.pos_mean - dist_tracker.neg_mean) / ((dist_tracker.pos_var + dist_tracker.neg_var) ** 0.5 + eps)
    class_p_values = z_score_to_p_value(class_z_scores)

    logit_z_scores = (logits - dist_tracker.pos_mean) / (dist_tracker.pos_std + eps)
    logit_p_values = z_score_to_p_value(logit_z_scores)
    
    # use t test
    #class_p_values = torch.Tensor(scipy.stats.ttest_ind_from_stats(dist_tracker.pos_mean.cpu().numpy(), dist_tracker.pos_std.cpu().numpy(), dist_tracker.pos_count.cpu().numpy(), dist_tracker.neg_mean.cpu().numpy(), dist_tracker.neg_std.cpu().numpy(), dist_tracker.neg_count.cpu().numpy(), equal_var=False, alternative="greater").pvalue).to(logits.device)
    
    labels_new = torch.ones_like(labels)
    #labels_new = 1-((1-logit_p_values) * (1-class_p_values))
    if(torch.distributed.get_rank() == 0):
        print(((logit_p_values > clip_logit).float() * (class_p_values > clip_dist).float()).sum())
    labels_new = labels_new.where(logit_p_values > clip_logit, 0).where(class_p_values > clip_dist, 0).where(labels == 0, 1)
    return labels_new

def generate_loss_weights(logits, labels, dist_tracker, clip_dist=0.95, eps=1e-8):
    # use a z test
    class_z_scores = (dist_tracker.pos_mean - dist_tracker.neg_mean) / ((dist_tracker.pos_var + dist_tracker.neg_var) ** 0.5 + eps)
    class_p_values = z_score_to_p_value(class_z_scores)
    
    logit_z_scores = (logits - dist_tracker.pos_mean) / (dist_tracker.pos_std + eps)
    logit_p_values = z_score_to_p_value(logit_z_scores)
    
    # [B, K]
    loss_weights = torch.exp(-logit_z_scores)
    #loss_weights = torch.ones_like(labels)
    #loss_weights = loss_weights.where(class_p_values > clip_dist, 1).where(labels == 1, 1)
    #loss_weights *= (dist_tracker.neg_count / (dist_tracker.pos_count + eps)).where(labels == 1, 1)
    return loss_weights

class DistributionTracker(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self._pos_mean = torch.Tensor([1])
        self._pos_count = torch.Tensor([1])
        #self._pos_M2 = 0
        self._pos_var = torch.Tensor([1])
        self._neg_mean = torch.Tensor([1])
        self._neg_count = torch.Tensor([1])
        #self._neg_M2 = 0
        self._neg_var = torch.Tensor([1])
        self.eps = 1e-8
    
    @property
    def pos_mean(self): return self._pos_mean
    
    @property
    def pos_count(self): return self._pos_count
    
    @property
    def pos_var(self): return self._neg_var #return self._pos_M2/(self._pos_count - 1)
    
    @property
    def pos_std(self): return self.pos_var ** 0.5
    
    @property
    def neg_mean(self): return self._neg_mean
    
    @property
    def neg_count(self): return self._neg_count
    
    @property
    def neg_var(self): return self._neg_var #return self._neg_M2/(self._neg_count - 1)
    
    @property
    def neg_std(self): return self.neg_var ** 0.5

    @property
    def log_odds(self): return torch.special.logit((self._pos_count + self.eps) / (self._pos_count + self._neg_count + self.eps))
    
    def dump(self):
        #return torch.stack([self._pos_mean, self._pos_count, self._pos_M2, self._neg_mean, self._neg_count, self._neg_M2])
    
        return torch.stack([self._pos_mean, self._pos_count, self._pos_var, self._neg_mean, self._neg_count, self._neg_var])
    
    def _zero_grad(self):
        self._pos_mean = zero_grad(self._pos_mean)
        self._pos_count = zero_grad(self._pos_count)
        self._pos_var = zero_grad(self._pos_var)
        self._neg_mean = zero_grad(self._neg_mean)
        self._neg_count = zero_grad(self._neg_count)
        self._neg_var = zero_grad(self._neg_var)
    
    def set_device(self, device):
        self._pos_mean = self._pos_mean.to(device)
        self._pos_count = self._pos_count.to(device)
        self._pos_var = self._pos_var.to(device)
        self._neg_mean = self._neg_mean.to(device)
        self._neg_count = self._neg_count.to(device)
        self._neg_var = self._neg_var.to(device)
    
    def forward(self, logits, labels):
        
        # ([B, K], [B, K])
        
        # [K]
        classSizePos = labels.sum(dim=0)
        #print(classSizePos)
        classSizeNeg = (1-labels).sum(dim=0)
        
        # [K]
        batchMeanPos = logits.where(labels == 1, 0).sum(dim=0) / (classSizePos + self.eps)
        #print(batchMeanPos)
        batchMeanNeg = logits.where(labels == 0, 0).sum(dim=0) / (classSizeNeg + self.eps)
        
        # [K]
        deltaPos = batchMeanPos - self._pos_mean.detach()
        #print(deltaPos)
        deltaNeg = batchMeanNeg - self._neg_mean.detach()
        
        # [K]
        self._pos_mean = self._pos_mean.detach() + classSizePos / ((classSizePos + self._pos_count.detach()) + self.eps) * deltaPos 
        #print(self._pos_mean)
        self._neg_mean = self._neg_mean.detach() + classSizeNeg / ((classSizeNeg + self._neg_count.detach()) + self.eps) * deltaNeg
        
        # [K]
        '''
        self._pos_M2 = self._pos_M2 + ((logits.where(labels == 1, 0) - batchMeanPos) ** 2).sum(0)
        self._pos_M2 = self._pos_M2 + (self._pos_count * classSizePos) / (self._pos_count + classSizePos+ self.eps) * deltaPos ** 2 
        self._neg_M2 = self._neg_M2 + ((logits.where(labels == 0, 0) - batchMeanNeg) ** 2).sum(0)
        self._neg_M2 = self._neg_M2 + (self._neg_count * classSizeNeg) / (self._neg_count + classSizeNeg + self.eps) * deltaNeg ** 2
        '''
        
        self._pos_var = self._pos_var.detach() * ((self._pos_count.detach() - 1) / (self._pos_count.detach() + classSizePos - 1 + self.eps)) + \
            ((logits.where(labels == 1, 0) - batchMeanPos) ** 2).sum(0) / (self._pos_count.detach() + classSizePos - 1 + self.eps) + \
            (self._pos_count.detach() * classSizePos) / ((self._pos_count.detach() + classSizePos - 1 + self.eps) * (self._pos_count.detach() + classSizePos + self.eps)) * deltaPos ** 2 
        #print(self._pos_var)
        self._neg_var = self._neg_var.detach() * ((self._neg_count.detach() - 1) / (self._neg_count.detach() + classSizeNeg - 1 + self.eps)) + \
            ((logits.where(labels == 0, 0) - batchMeanNeg) ** 2).sum(0) / (self._neg_count.detach() + classSizeNeg - 1 + self.eps) + \
            (self._neg_count.detach() * classSizeNeg) / ((self._neg_count.detach() + classSizeNeg - 1 + self.eps) * (self._neg_count.detach() + classSizeNeg + self.eps)) * deltaNeg ** 2
        
        
        # [K]
        self._pos_count = self._pos_count.detach() + classSizePos
        self._neg_count = self._neg_count.detach() + classSizeNeg
        
        return self.pos_mean, self.pos_std, self.neg_mean, self.neg_std
        

class DistributionTrackerEMA(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self._pos_mean = 0
        self._pos_count = 0
        self._pos_M2 = 0
        #self._pos_var = 0
        self._neg_mean = 0
        self._neg_count = 0
        self._neg_M2 = 0
        #self._neg_var = 0
        self.eps = 1e-8
    
    @property
    def pos_mean(self): return self._pos_mean
    
    @property
    def pos_count(self): return self._pos_count
    
    @property
    def pos_var(self): return self._pos_M2/(self._pos_count - 1)
    
    @property
    def pos_std(self): return self.pos_var ** 0.5
    
    @property
    def neg_mean(self): return self._neg_mean
    
    @property
    def neg_count(self): return self._neg_count
    
    @property
    def neg_var(self): return self._neg_M2/(self._neg_count - 1)
    
    @property
    def neg_std(self): return self.neg_var ** 0.5
    
    def dump(self):
        #return torch.stack([self._pos_mean, self._pos_count, self._pos_M2, self._neg_mean, self._neg_count, self._neg_M2])
    
        return torch.stack([self._pos_mean, self._pos_count, self._pos_var, self._neg_mean, self._neg_count, self._neg_var])
    
    def forward(self, logits, labels, alpha=0.998):
        # ([B, K], [B, K])
        
        # [K]
        classSizePos = labels.sum(dim=0)
        #print(classSizePos)
        classSizeNeg = (1-labels).sum(dim=0)
        
        # [K]
        batchMeanPos = logits.where(labels == 1, 0).sum(dim=0) / (classSizePos + self.eps)
        #print(batchMeanPos)
        batchMeanNeg = logits.where(labels == 0, 0).sum(dim=0) / (classSizeNeg + self.eps)
        
        # [K]
        deltaPos = batchMeanPos - self._pos_mean
        #print(deltaPos)
        deltaNeg = batchMeanNeg - self._neg_mean
        
        # [K]
        self._pos_mean = self._pos_mean + classSizePos / ((classSizePos + self._pos_count) + self.eps) * deltaPos 
        #print(self._pos_mean)
        self._neg_mean = self._neg_mean + classSizeNeg / ((classSizeNeg + self._neg_count) + self.eps) * deltaNeg
        
        # [K]
        
        self._pos_M2 = self._pos_M2 * alpha + ((logits.where(labels == 1, 0) - batchMeanPos) ** 2).sum(0) * (1 - alpha)
        self._pos_M2 = self._pos_M2 + (self._pos_count * classSizePos) / (self._pos_count + classSizePos + self.eps) * deltaPos ** 2 * (1 - alpha)
        self._neg_M2 = self._neg_M2 * alpha + ((logits.where(labels == 0, 0) - batchMeanNeg) ** 2).sum(0) * (1 - alpha)
        self._neg_M2 = self._neg_M2 + (self._neg_count * classSizeNeg) / (self._neg_count + classSizeNeg + self.eps) * deltaNeg ** 2 * (1 - alpha)
        
        '''
        self._pos_var = self._pos_var * ((self._pos_count - 1) / (self._pos_count + classSizePos - 1 + self.eps)) + \
            ((logits.where(labels == 1, 0) - batchMeanPos) ** 2).sum(0) / (self._pos_count + classSizePos - 1 + self.eps) + \
            (self._pos_count * classSizePos) / ((self._pos_count + classSizePos - 1 + self.eps) * (self._pos_count + classSizePos + self.eps)) * deltaPos ** 2 
        #print(self._pos_var)
        self._neg_var = self._neg_var * ((self._neg_count - 1) / (self._neg_count + classSizeNeg - 1 + self.eps)) + \
            ((logits.where(labels == 0, 0) - batchMeanNeg) ** 2).sum(0) / (self._neg_count + classSizeNeg - 1 + self.eps) + \
            (self._neg_count * classSizeNeg) / ((self._neg_count + classSizeNeg - 1 + self.eps) * (self._neg_count + classSizeNeg + self.eps)) * deltaNeg ** 2
        '''
        
        # [K]
        self._pos_count = self._pos_count * alpha + classSizePos * (1 - alpha)
        self._neg_count = self._neg_count * alpha + classSizeNeg * (1 - alpha)
        
        return self.pos_mean, self.pos_std, self.neg_mean, self.neg_std

class AdaptiveWeightedLoss(nn.Module):
    def __init__(self, initial_weight = 0.0, lr = 1e-3, weight_limit = 10.0, eps = 1e-8):
        super().__init__()
        self.initial_weight = initial_weight
        self.weight_per_class = None
        self.opt = None
        self.needs_init = True
        self.lr = lr
        assert weight_limit >= 1, "weight_limit must be greater or equal to 1, my borpa code isn't equipped to handle otherwise lol"
        self.weight_limit_upper = weight_limit
        self.weight_limit_lower = 1 / weight_limit
        self.eps = eps
        
    def forward(self, x, y):
    
        
            
        # process logits, ASLOptimized style
        self.targets = y
        self.anti_targets = 1 - y
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos
        
        # basic loss calculation, ASL/ASLOptimized style
        self.loss_pos = self.targets * torch.log(self.xs_pos)
        self.loss_neg = self.anti_targets * torch.log(self.xs_neg)
        
        
        return -(self.loss_neg + self.loss_pos * self.weight_per_class.detach()).sum()
    
    def update(self, x, y, update=True, step_opt=True):
        if not update: return None
        with torch.no_grad():
            # parameter initialization
            # TODO clean this up and make it work consistently, use proper lazy init
            if self.needs_init:
                classCount = x.size(dim=1)
                currDevice = x.device
                if self.weight_per_class == None:
                    self.weight_per_class = nn.Parameter(torch.ones(classCount, device=currDevice, requires_grad=True) * self.initial_weight)
                else:
                    self.weight_per_class = nn.Parameter(torch.ones(classCount, device=currDevice, requires_grad=True) * self.weight_per_class)
                self.needs_init = False
                #self.opt = torch.optim.SGD(self.parameters(), lr=self.lr)
                self.opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
                # TODO maybe another optimizer will work better?
                # TODO maybe a plain EMA?
                
            # process logits, ASLOptimized style
            self.targets = y
            self.anti_targets = 1 - y
            self.xs_pos = torch.sigmoid(x)
            self.xs_neg = 1.0 - self.xs_pos
        
            #self.weight_this_batch = self.anti_targets.sum(dim=1) / (self.targets.sum(dim=1) + self.eps) # via labels
        
            self.weight_this_batch = (self.xs_neg * self.anti_targets).sum(dim=0) / ((self.xs_pos * self.targets).sum(dim=0) + self.eps) # via preds

            #self.weight_this_batch = self.weight_this_batch.detach() # isolate the weight optimization
        
        # optimization
        numToMin = (self.weight_this_batch - self.weight_per_class) ** 2
        numToMin.mean().backward()
        
        if step_opt:
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            
            # EMA
            # TODO get this to work, currently collapsing to high false positive count (87ish %)
            #self.weight_per_class.data = (1-self.lr) * self.weight_per_class.data + (self.lr) * self.weight_this_batch
            with torch.no_grad():
            
                self.weight_per_class.data = self.weight_per_class.clamp(min=self.weight_limit_lower, max=self.weight_limit_upper)

            
        


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y, weight = 1):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
        
        loss *= weight
        return -loss.sum()


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()

class AsymmetricLossAdaptive(nn.Module):
    def __init__(self, gamma_neg=1, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True, adaptive = True, gap_target = 0.1, gamma_step = 0.1):
        super(AsymmetricLossAdaptive, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.adaptive = adaptive
        self.gap_target = gap_target
        self.gamma_step = gamma_step
        self.gamma_neg_per_class = None
        self.gamma_pos_per_class = None
        
        
    def forward(self, x, y, updateAdaptive = True, printAdaptive = False):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        
        output = ""
        x = x.to(torch.float64)
        
        with torch.no_grad():
            if self.gamma_neg_per_class == None or self.gamma_pos_per_class == None:
                print("initializing loss values")
                classCount = y.size(dim=1)
                currDevice = y.device
                self.gamma_neg_per_class = torch.ones(classCount, device=currDevice, dtype=torch.float64) * self.gamma_neg
                self.gamma_pos_per_class = torch.ones(classCount, device=currDevice, dtype=torch.float64) * self.gamma_pos

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        #los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps)) * (self.gamma_neg_per_class ** 0.5)
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            if(self.adaptive == True):
            
                with torch.no_grad():
                    gap = pt0.sum(dim=0) / (y.sum(dim=0) + self.eps) - pt1.sum(dim=0) / ((1 - y).sum(dim=0) + self.eps)

                    if updateAdaptive == True:
                        self.gamma_neg_per_class = self.gamma_neg_per_class - (self.gamma_step) * (gap - self.gap_target)
                        
                        
                        #self.gamma_neg_per_class = self.gamma_neg_per_class - (self.gamma_step * y.mean(dim=0)) * (gap - self.gap_target)
                        #self.gamma_neg_per_class = self.gamma_neg_per_class - (self.gamma_step * y.mean(dim=0).sqrt()) * (gap - self.gap_target)
                        
                        #self.gamma_neg_per_class = self.gamma_neg_per_class - (self.gamma_step * (1-y).mean(dim=0)) * (gap - self.gap_target)
                        #self.gamma_neg_per_class = self.gamma_neg_per_class - (self.gamma_step * (1-y).mean(dim=0).sqrt()) * (gap - self.gap_target)
                        
                        
                        
                        #self.gamma_pos_per_class = self.gamma_pos_per_class - (self.gamma_step * (1-y).mean(dim=0)) * (gap - self.gap_target)
                        #self.gamma_pos_per_class = self.gamma_pos_per_class - (self.gamma_step * (1-y).mean(dim=0).sqrt()) * (gap - self.gap_target)
                        
                        #self.gamma_pos_per_class = self.gamma_pos_per_class - (self.gamma_step * y.mean(dim=0)) * (gap - self.gap_target)
                        #self.gamma_pos_per_class = self.gamma_pos_per_class - (self.gamma_step * y.mean(dim=0).sqrt()) * (gap - self.gap_target)
                        
                        
                        
                        self.gamma_neg_per_class = torch.clamp(self.gamma_neg_per_class, min=0)
                        
                    
                    if printAdaptive == True:
                        #output = str(f'\tpos: {pt0.sum() / (y.sum() + self.eps):.4f},\tneg: {pt1.sum() / ((1 - y).sum() + self.eps):.4f}')
                        output = str(f'pos: {pt0.sum() / (y.sum() + self.eps):.4f},\tneg: {pt1.sum() / ((1 - y).sum() + self.eps):.4f},\tGN: [{self.gamma_neg_per_class.min():.4f}, {self.gamma_neg_per_class.max():.4f}],\tGP: [{self.gamma_pos_per_class.min():.4f}, {self.gamma_pos_per_class.max():.4f}]')
                
            one_sided_gamma = self.gamma_pos_per_class * y + self.gamma_neg_per_class * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum(), output

class AsymmetricLossAdaptiveWorking(nn.Module):
    def __init__(self, gamma_neg=1, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True, adaptive = True, gap_target = 0.1, gamma_step = 0.1):
        super(AsymmetricLossAdaptiveWorking, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.adaptive = adaptive
        self.gap_target = gap_target
        self.gamma_step = gamma_step
        
        
    def forward(self, x, y, updateAdaptive = True, printAdaptive = False):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        
        output=""

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            if(self.adaptive == True):
            
                gap = pt0.sum() / (y.sum() + self.eps) - pt1.sum() / ((1 - y).sum() + self.eps)
                
                
                if updateAdaptive == True:
                    self.gamma_neg = self.gamma_neg - self.gamma_step * (gap - self.gap_target)
                    #self.gamma_neg = self.gamma_neg + self.gamma_step * (gap - self.gap_target)
                    
                
                if printAdaptive == True:
                    output = str(f'\tpos: {pt0.sum() / (y.sum() + self.eps):.4f},\tneg: {pt1.sum() / ((1 - y).sum() + self.eps):.4f},\tgap: {gap:.4f},\tchange: {self.gamma_step * (gap - self.gap_target):.6f},\tgamma neg: {self.gamma_neg:.6f}')
                
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum() + ((x_sigmoid-y) ** 2).sum(), output


class GapWeightLoss(nn.Module):
    def __init__(self, initial_weight = 0., gap_target = 0.1, weight_step = 1e-3):
        super().__init__()
        self.initial_weight = initial_weight
        self.eps = 1e-8
        self.weight_per_class = None
        self.gap_target = gap_target
        self.weight_step = weight_step
    
    def forward(self, x, y, updateAdaptive = True, printAdaptive = False):
        output = ""
        if self.weight_per_class == None:
            classCount = y.size(dim=1)
            currDevice = y.device
            self.weight_per_class = torch.ones(classCount, device=currDevice, dtype=torch.float64) * self.initial_weight
            
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        with torch.amp.autocast('cuda', enabled=False):

            los_pos = y * torch.log(xs_pos.clamp(min=self.eps)) * (10 ** self.weight_per_class.detach())
            los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
            loss = los_pos + los_neg

        with torch.no_grad():
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            gap = pt0.sum(dim=0) / (y.sum(dim=0) + self.eps) - pt1.sum(dim=0) / ((1 - y).sum(dim=0) + self.eps)

            if updateAdaptive:
                self.weight_per_class = self.weight_per_class - (self.weight_step) * (gap - self.gap_target)
                self.weight_per_class = self.weight_per_class.clamp(min=-10, max=10)
            if printAdaptive == True:
                output = str(f'pos: {pt0.sum() / (y.sum() + self.eps):.4f},\tneg: {pt1.sum() / ((1 - y).sum() + self.eps):.4f},\tWPC: [{self.weight_per_class.min():.4f}, {self.weight_per_class.max():.4f}]')
    
        return -loss.sum(), output
            
        


class ASLSingleLabel(nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
        
class ComputePrior:
    def __init__(self, classes, device):
        self.classes = classes
        n_classes = len(self.classes)
        self.sum_pred_train = torch.zeros(n_classes).to(device)
        self.sum_pred_val = torch.zeros(n_classes).to(device)
        self.cnt_samples_train,  self.cnt_samples_val = .0, .0
        self.avg_pred_train, self.avg_pred_val = None, None
        self.path_dest = "./outputs"
        self.path_local = "/class_prior/"

    def update(self, logits, training=True):
        with torch.no_grad():
            preds = torch.sigmoid(logits).detach()
            if training:
                self.sum_pred_train += torch.sum(preds, axis=0)
                self.cnt_samples_train += preds.shape[0]
                self.avg_pred_train = self.sum_pred_train / self.cnt_samples_train

            else:
                self.sum_pred_val += torch.sum(preds, axis=0)
                self.cnt_samples_val += preds.shape[0]
                self.avg_pred_val = self.sum_pred_val / self.cnt_samples_val

    def save_prior(self):

        print('Prior (train), first 5 classes: {}'.format(self.avg_pred_train[:5]))

        # Save data frames as csv files
        if not os.path.exists(self.path_dest):
            os.makedirs(self.path_dest)

        df_train = pd.DataFrame({"Classes": list(self.classes.values()),
                                 "avg_pred": self.avg_pred_train.cpu()})
        df_train.to_csv(path_or_buf=os.path.join(self.path_dest, "train_avg_preds.csv"),
                        sep=',', header=True, index=False, encoding='utf-8')

        if self.avg_pred_val is not None:
            df_val = pd.DataFrame({"Classes": list(self.classes.values()),
                                   "avg_pred": self.avg_pred_val.cpu()})
            df_val.to_csv(path_or_buf=os.path.join(self.path_dest, "val_avg_preds.csv"),
                          sep=',', header=True, index=False, encoding='utf-8')

    def get_top_freq_classes(self):
        n_top = 10
        top_idx = torch.argsort(-self.avg_pred_train.cpu())[:n_top]
        top_classes = np.array(list(self.classes.values()))[top_idx]
        print('Prior (train), first {} classes: {}'.format(n_top, top_classes))

#gamma: focal loss focusing parameter, 0 for BCE

class PartialSelectiveLoss(nn.Module):

    def __init__(self, device, prior_path=None, clip=0, gamma_pos=0.5, gamma_neg=10, gamma_unann=10, alpha_pos=1, alpha_neg=1, alpha_unann=1):
        super(PartialSelectiveLoss, self).__init__()
        self.clip = clip
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.gamma_unann = gamma_unann
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.alpha_unann = alpha_unann
        self.device = device

        self.targets_weights = None

        if prior_path is not None:
            print("Prior file was found in given path.")
            df = pd.read_csv(prior_path)
            self.prior_classes = dict(zip(df.values[:, 0], df.values[:, 1]))
            print("Prior file was loaded successfully. ")

    def forward(self, logits, targets, priorValues = None):

        # Positive, Negative and Un-annotated indexes
        targets_pos = (targets == 1).float()
        targets_neg = (targets == 0).float()
        targets_unann = (targets == -1).float()

        # Activation
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg.add_(self.clip).clamp_(max=1)

        prior_classes = None
        if priorValues is not None:
            prior_classes = torch.tensor(list(priorValues)).to(self.device)
        elif hasattr(self, "prior_classes"):
            prior_classes = torch.tensor(list(self.prior_classes.values())).to(self.device)


        targets_weights = self.targets_weights
        targets_weights, xs_neg = edit_targets_parital_labels(targets, targets_weights, xs_neg, self.device,
                                                              prior_classes=prior_classes)

        # Loss calculation
        BCE_pos = self.alpha_pos * targets_pos * torch.log(torch.clamp(xs_pos, min=1e-8))
        BCE_neg = self.alpha_neg * targets_neg * torch.log(torch.clamp(xs_neg, min=1e-8))
        BCE_unann = self.alpha_unann * targets_unann * torch.log(torch.clamp(xs_neg, min=1e-8))

        BCE_loss = BCE_pos + BCE_neg + BCE_unann

        # Adding asymmetric gamma weights
        with torch.no_grad():
            asymmetric_w = torch.pow(1 - xs_pos * targets_pos - xs_neg * (targets_neg + targets_unann),
                                     self.gamma_pos * targets_pos + self.gamma_neg * targets_neg +
                                     self.gamma_unann * targets_unann)
        BCE_loss *= asymmetric_w

        # partial labels weights
        BCE_loss *= targets_weights

        return -BCE_loss.sum()

# likelihood_topk: K un-annotated labels are allowed to be assumed positive


def edit_targets_parital_labels(targets, targets_weights, xs_neg, device, partial_loss_mode = 'ignore_normalize_classes', prior_classes=None, likelihood_topk = 100, prior_threshold=0.05):
    # targets_weights is and internal state of AsymmetricLoss class. we don't want to re-allocate it every batch
    if partial_loss_mode is None:
        targets_weights = 1.0

    elif partial_loss_mode == 'negative':
        # set all unsure targets as negative
        targets_weights = 1.0

    elif partial_loss_mode == 'ignore':
        # remove all unsure targets (targets_weights=0)
        targets_weights = torch.ones(targets.shape, device=device)
        targets_weights[targets == -1] = 0

    elif partial_loss_mode == 'ignore_normalize_classes':
        # remove all unsure targets and normalize by Durand et al. https://arxiv.org/pdf/1902.09720.pdfs
        alpha_norm, beta_norm = 1, 1
        targets_weights = torch.ones(targets.shape, device=device)
        n_annotated = 1 + torch.sum(targets != -1, axis=1)    # Add 1 to avoid dividing by zero

        g_norm = alpha_norm * (1 / n_annotated) + beta_norm
        n_classes = targets_weights.shape[1]
        targets_weights *= g_norm.repeat([n_classes, 1]).T
        targets_weights[targets == -1] = 0

    elif partial_loss_mode == 'selective':
        #targets[targets == 0] = -1
        if targets_weights is None or targets_weights.shape != targets.shape:
            targets_weights = torch.ones(targets.shape, device=device)
        else:
            targets_weights[:] = 1.0
        num_top_k = likelihood_topk * targets_weights.shape[0]

        xs_neg_prob = xs_neg
        if prior_classes is not None:
            if prior_threshold:
                idx_ignore = torch.where(prior_classes > prior_threshold)[0]
                targets_weights[:, idx_ignore] = 0
                targets_weights += (targets != -1).float()
                targets_weights = targets_weights.bool()

        negative_backprop_fun_jit(targets, xs_neg_prob, targets_weights, num_top_k)

        # set all unsure targets as negative
        # targets[targets == -1] = 0

    return targets_weights, xs_neg

# @torch.jit.script
def negative_backprop_fun_jit(targets: Tensor, xs_neg_prob: Tensor, targets_weights: Tensor, num_top_k: int):
    with torch.no_grad():
        targets_flatten = targets.flatten()
        cond_flatten = torch.where(targets_flatten == -1)[0]
        targets_weights_flatten = targets_weights.flatten()
        xs_neg_prob_flatten = xs_neg_prob.flatten()
        ind_class_sort = torch.argsort(xs_neg_prob_flatten[cond_flatten])
        targets_weights_flatten[
            cond_flatten[ind_class_sort[:num_top_k]]] = 0

def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i

def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()

def AP_partial(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    cnt_class_with_no_neg = 0
    cnt_class_with_no_pos = 0
    cnt_class_with_no_labels = 0

    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]

        # Filter out samples without label
        idx = (targets != -1)
        scores = scores[idx]
        targets = targets[idx]
        if len(targets) == 0:
            cnt_class_with_no_labels += 1
            ap[k] = -1
            continue
        elif sum(targets) == 0:
            cnt_class_with_no_pos += 1
            ap[k] = -1
            continue
        if sum(targets == 0) == 0:
            cnt_class_with_no_neg += 1
            ap[k] = -1
            continue

        # compute average precision
        ap[k] = average_precision(scores, targets)

    idx_valid_classes = np.where(ap != -1)[0]
    ap_valid = ap[idx_valid_classes]
    map = 100 * np.mean(ap_valid)

    # Compute macro-map
    targs_macro_valid = targs[:, idx_valid_classes].copy()
    targs_macro_valid[targs_macro_valid <= 0] = 0  # set partial labels as negative
    n_per_class = targs_macro_valid.sum(0)  # get number of targets for each class
    n_total = np.sum(n_per_class)
    map_macro = 100 * np.sum(ap_valid * n_per_class / n_total)

    return ap, map, map_macro, cnt_class_with_no_labels, cnt_class_with_no_neg, cnt_class_with_no_pos

def mAP_partial(targs, preds):
    """ mean Average precision for partial annotated validatiion set"""

    if np.size(preds) == 0:
        return 0
    results = AP_partial(targs, preds)
    mAP = results[1]
    return mAP

# getAccuracy calculates a confusion matrix and some related values.
def getAccuracy(preds, targs):
    epsilon = 1e-12

    #preds = torch.sigmoid(preds)

    targs_inv = 1 - targs
    batchSize = targs.size(dim=0)
    P = targs * preds
    N = targs_inv * preds
    
    
    TP = P.sum(dim=0) / batchSize
    FN = (targs - P).sum(dim=0) / batchSize
    FP = N.sum(dim=0) / batchSize
    TN = (targs_inv - N).sum(dim=0) / batchSize
    
    Precall = TP / (TP + FN + epsilon)
    Nrecall = TN / (TN + FP + epsilon)
    Pprecision = TP / (TP + FP + epsilon)
    Nprecision = TN / (TN + FN + epsilon)
    

    P4 = (4 * TP * TN) / ((4 * TN * TP) + (TN + TP) * (FP + FN) + epsilon)
    F1 = (2 * TP) / (2 * TP + FP + FN + epsilon)
    
    return torch.column_stack([TP, FN, FP, TN, Precall, Nrecall, Pprecision, Nprecision, P4, F1])
    
def getSingleMetric(preds, targs, metric):
    epsilon = 1e-12

    #preds = torch.sigmoid(preds)

    targs_inv = 1 - targs
    batchSize = targs.size(dim=0)
    P = targs * preds
    N = targs_inv * preds
    
    # [K]
    TP = P.sum(dim=0) / batchSize
    FN = (targs - P).sum(dim=0) / batchSize
    FP = N.sum(dim=0) / batchSize
    TN = (targs_inv - N).sum(dim=0) / batchSize
    
    return metric(TP, FN, FP, TN, epsilon)
    
# recall
def Precall(TP, FN, FP, TN, epsilon):
    zero_grad(FP)
    zero_grad(TN)
    return TP / (TP + FN + epsilon)
    
# specificity
def Nrecall(TP, FN, FP, TN, epsilon):
    zero_grad(FN)
    return TN / (TN + FP + epsilon)

# precision
def Pprecision(TP, FN, FP, TN, epsilon):
    return TP / (TP + FP + epsilon)

# negative predictive value (NPV)
def Nprecision(TP, FN, FP, TN, epsilon):
    return TN / (TN + FN + epsilon)

# P4 metric
def P4(TP, FN, FP, TN, epsilon):
    return (4 * TP * TN) / ((4 * TN * TP) + (TN + TP) * (FP + FN) + epsilon)
    
# F1 metric
def F1(TP, FN, FP, TN, epsilon):
    return (2 * TP) / (2 * TP + FP + FN + epsilon)


# TODO test as boundary opt metric
# https://www.cs.uic.edu/~liub/publications/icml-03.pdf
# metric proposed in 
# Lee, W. S., & Liu, B. (2003).
# Learning with positive and unlabeled examples using weighted logistic regression.
# In Proceedings of the twentieth international conference on machine learning (pp. 448–455).
def PU_F_Metric(TP, FN, FP, TN, epsilon):
    return (Precall(TP, FN, FP, TN, epsilon) ** 2) / (FP + TP + epsilon)

# AUL and AUROC helper, implements shared portion of eqs 1 and 2 in paper
# https://openreview.net/forum?id=2NU7a9AHo-6
# AUL is a better optimization metric in PU learning
# by Shangchuan Huang, Songtao Wang, Dan Li, Liwei Jiang
# originally eqs 4.5 and 4.6 from
# https://doi.org/10.51936/noqf3710
# ROC Curve, Lift Chart and Calibration Plot
# by Miha Vuk and Tomaz Curk (2006)
def chart_inner(preds):

    # [K, B] <- [B, K]
    preds = preds.permute(1,0)
    # [K, B, 1]
    sample1 = preds.unsqueeze(2)
    # [K, 1, B]
    sample2 = preds.unsqueeze(1)
    # [K, B, B]
    #result = (sample1 == sample2).int() * 0.5 + (sample1 > sample2).int()
    result = (sample1 - sample2).sigmoid()
    #print(result)
    return result
    
def AUROC(preds, targs, epsilon = 1e-8):
    # [K] <- [B, K]
    num_pos = targs.sum(dim=0)
    # [K]
    num_neg = targs.size(dim=0) - num_pos
    # [K]
    return (num_pos > 0).int() * torch.tril(chart_inner(preds)).sum(dim=(1,2)) / (num_pos * num_neg + epsilon)
    
def AUL(preds, targs, epsilon = 1e-8):
    # [K] <- [B, K]
    num_pos = targs.sum(dim=0)
    numel = targs.size(dim=0) # = K
    # [K]
    result = (num_pos > 0).int() * torch.tril(chart_inner(preds)).sum(dim=(1,2)) / (num_pos * numel + epsilon)

    return result

# tracking for performance metrics that can be computed from confusion matrix
class MetricTracker():
    def __init__(self):
        self.running_confusion_matrix = None
        self.epsilon = 1e-12
        self.sampleCount = 0
        self.metrics = [Precall, Nrecall, Pprecision, Nprecision, P4, F1, PU_F_Metric]
        
    def get_full_metrics(self):
        with torch.no_grad():
            TP, FN, FP, TN = self.running_confusion_matrix / (self.sampleCount + self.epsilon)
            
            #Precall = TP / (TP + FN + self.epsilon)
            #Nrecall = TN / (TN + FP + self.epsilon)
            #Pprecision = TP / (TP + FP + self.epsilon)
            #Nprecision = TN / (TN + FN + self.epsilon)
            
            #P4 = (4 * TP * TN) / ((4 * TN * TP) + (TN + TP) * (FP + FN) + self.epsilon)
            
            metrics = [metric(TP, FN, FP, TN, self.epsilon) for metric in self.metrics]
        
            return torch.column_stack([TP, FN, FP, TN, *metrics])
        
    def get_aggregate_metrics(self):
        '''
        with torch.no_grad():
        
            TP, FN, FP, TN = (self.running_confusion_matrix / self.sampleCount).mean(dim=1)
            
            Precall = TP / (TP + FN + self.epsilon)
            Nrecall = TN / (TN + FP + self.epsilon)
            Pprecision = TP / (TP + FP + self.epsilon)
            Nprecision = TN / (TN + FN + self.epsilon)
            
            P4 = (4 * TP * TN) / ((4 * TN * TP) + (TN + TP) * (FP + FN) + self.epsilon)
            return torch.stack([TP, FN, FP, TN, Precall, Nrecall, Pprecision, Nprecision, P4])
        '''
        return self.get_full_metrics().mean(dim=0)
    
    def update(self, preds, targs):
        with torch.no_grad():
            self.sampleCount += targs.size(dim=0)
            
            targs_inv = 1 - targs
            P = targs * preds
            N = targs_inv * preds
            
            
            TP = P.sum(dim=0)
            FN = (targs - P).sum(dim=0)
            FP = N.sum(dim=0)
            TN = (targs_inv - N).sum(dim=0)
            
            output = torch.stack([TP, FN, FP, TN])
            if self.running_confusion_matrix is None:
                self.running_confusion_matrix = output
            
            else:
                self.running_confusion_matrix += output
                
            return self.get_aggregate_metrics()
        

class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = self.ema * 0.99 + self.val * 0.01
