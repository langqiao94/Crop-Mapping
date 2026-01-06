import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        return F.cross_entropy(
            inputs, 
            targets,
            weight=self.weight,
            reduction=self.reduction
        )

def compute_fusion_loss(pred, target):
    focal = FocalLoss()(pred, target)
    ce = WeightedCrossEntropyLoss()(pred, target)
    return focal + ce

def compute_swin_loss(pred, target):
    focal = FocalLoss()(pred, target)
    ce = WeightedCrossEntropyLoss()(pred, target)
    return focal + ce

def compute_gru_loss(pred, target):
    focal = FocalLoss()(pred, target)
    ce = WeightedCrossEntropyLoss()(pred, target)
    return focal + ce
