import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification
    
    Args:
        alpha (float): Balance factor for handling class imbalance, default 0.25
        gamma (float): Focusing parameter to reduce weight of easy samples, default 2.0
        reduction (str): Loss calculation method, options 'mean' or 'sum'
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model predicted logits, shape (B, C, H, W)
            targets (torch.Tensor): Target labels, shape (B, H, W), value range [0, C-1]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with class weights
    
    Args:
        weight (torch.Tensor, optional): Weights for each class, shape (C,)
        reduction (str): Loss calculation method, options 'mean' or 'sum'
    """
    def __init__(self, weight=None, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model predicted logits, shape (B, C, H, W)
            targets (torch.Tensor): Target labels, shape (B, H, W), value range [0, C-1]
        """
        return F.cross_entropy(
            inputs, 
            targets,
            weight=self.weight,
            reduction=self.reduction
        )


def compute_fusion_loss(pred, target):
    """Calculate fusion branch loss
    
    Args:
        pred (torch.Tensor): Fusion branch predictions, shape (B, C, H, W)
        target (torch.Tensor): Target labels, shape (B, H, W), value range [0, C-1]
    
    Returns:
        torch.Tensor: Fusion branch loss value
    """
    focal = FocalLoss()(pred, target)
    ce = WeightedCrossEntropyLoss()(pred, target)
    return focal + ce


def compute_swin_loss(pred, target):
    """Calculate swin branch loss
    
    Args:
        pred (torch.Tensor): Swin branch predictions, shape (B, C, H, W)
        target (torch.Tensor): Target labels, shape (B, H, W), value range [0, C-1]
    
    Returns:
        torch.Tensor: Swin branch loss value
    """
    focal = FocalLoss()(pred, target)
    ce = WeightedCrossEntropyLoss()(pred, target)
    return focal + ce


def compute_gru_loss(pred, target):
    """Calculate gru branch loss
    
    Args:
        pred (torch.Tensor): GRU branch predictions, shape (B, C, H, W)
        target (torch.Tensor): Target labels, shape (B, H, W), value range [0, C-1]
    
    Returns:
        torch.Tensor: GRU branch loss value
    """
    focal = FocalLoss()(pred, target)
    ce = WeightedCrossEntropyLoss()(pred, target)
    return focal + ce
