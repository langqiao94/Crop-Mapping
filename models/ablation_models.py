import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .swin_branch import SwinBranch
from .gru_branch import GRUBranch
from .decode_head import UPerHead, LightweightFCNHead
from losses.losses import compute_fusion_loss, compute_swin_loss, compute_gru_loss

class SwinOnlyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.swin_expert = SwinBranch(config)

        uper_head_dropout = getattr(config.decode_head, 'dropout', 0.1)
        uper_head_norm_cfg = getattr(config.decode_head, 'norm_cfg', {'type': 'BN'})
        uper_head_align_corners = getattr(config.decode_head, 'align_corners', False)
        self.swin_decode_head = UPerHead(
            in_channels=config.decode_head.in_channels,
            channels=config.decode_head.channels,
            num_classes=config.dataset.data_info.num_classes,
            dropout_ratio=uper_head_dropout,
            norm_cfg=uper_head_norm_cfg,
            align_corners=uper_head_align_corners
        )

        self.loss_weight = getattr(config, 'swin_loss_weight', 1.0)

    def forward(self, batch):
        if 'hls' not in batch:
            raise KeyError("SwinOnlyModel requires 'hls' in batch")

        hls_data = batch['hls']
        input_size = hls_data.shape[-2:]

        swin_features = self.swin_expert(hls_data)
        swin_logits_raw = self.swin_decode_head(swin_features)
        swin_logits = F.interpolate(swin_logits_raw, size=input_size, mode='bilinear', align_corners=self.swin_decode_head.align_corners)

        return {
            'output': swin_logits,
            'swin_output': swin_logits,
            'gru_output': None,
            'expert_preferences': None
        }

    def forward_loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        if targets.ndim > 3:
            targets = targets.squeeze(1)
        targets = targets.long()

        swin_loss = compute_swin_loss(outputs['swin_output'], targets)
        total_loss = self.loss_weight * swin_loss

        loss_dict = {
            'total_loss': total_loss.item(),
            'fusion_loss': 0.0,
            'swin_loss': swin_loss.item(),
            'gru_loss': 0.0
        }
        return total_loss, loss_dict

class GRUOnlyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gru_expert = GRUBranch(config)

        gru_in_channels = self.gru_expert.output_channels
        num_classes = config.dataset.data_info.num_classes
        gru_decoder_channels = getattr(config.decode_head, 'gru_channels', 128)
        gru_decoder_dropout = getattr(config.decode_head, 'gru_dropout', 0.1)
        gru_decoder_num_convs = getattr(config.decode_head, 'gru_num_convs', 2)
        gru_decoder_scale_factor = getattr(config.decode_head, 'gru_scale_factor', 2)
        gru_decoder_align_corners = getattr(config.decode_head, 'gru_align_corners', False)
        self.gru_decode_head = LightweightFCNHead(
            in_channels=gru_in_channels,
            channels=gru_decoder_channels,
            num_classes=num_classes,
            dropout_ratio=gru_decoder_dropout,
            num_convs=gru_decoder_num_convs,
            scale_factor=gru_decoder_scale_factor,
            align_corners=gru_decoder_align_corners
        )

        self.loss_weight = getattr(config, 'gru_loss_weight', 1.0)

    def forward(self, batch):
        if 'history' not in batch:
            raise KeyError("GRUOnlyModel requires 'history' in batch")
        if 'hls' not in batch:
             raise KeyError("GRUOnlyModel requires 'hls' in batch to determine target size")

        cdl_data = batch['history']
        input_size = batch['hls'].shape[-2:]

        gru_features = self.gru_expert(cdl_data)
        gru_logits_raw = self.gru_decode_head(gru_features)
        gru_logits = F.interpolate(gru_logits_raw, size=input_size, mode='bilinear', align_corners=self.gru_decode_head.align_corners)

        return {
            'output': gru_logits,
            'swin_output': None,
            'gru_output': gru_logits,
            'expert_preferences': None
        }

    def forward_loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        pred_device = outputs['gru_output'].device
        if targets.device != pred_device:
            targets = targets.to(pred_device)
        
        if targets.ndim > 3:
            targets = targets.squeeze(1)
        targets = targets.long()
        
        gru_loss = compute_gru_loss(outputs['gru_output'], targets)
        total_loss = self.loss_weight * gru_loss

        loss_dict = {
            'total_loss': total_loss.item(),
            'fusion_loss': 0.0,
            'swin_loss': 0.0,
            'gru_loss': gru_loss.item()
        }
        return total_loss, loss_dict

class SimpleFusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.swin_expert = SwinBranch(config)
        self.gru_expert = GRUBranch(config)

        uper_head_dropout = getattr(config.decode_head, 'dropout', 0.1)
        uper_head_norm_cfg = getattr(config.decode_head, 'norm_cfg', {'type': 'BN'})
        uper_head_align_corners = getattr(config.decode_head, 'align_corners', False)
        self.swin_decode_head = UPerHead(
            in_channels=config.decode_head.in_channels,
            channels=config.decode_head.channels,
            num_classes=config.dataset.data_info.num_classes,
            dropout_ratio=uper_head_dropout,
            norm_cfg=uper_head_norm_cfg,
            align_corners=uper_head_align_corners
        )

        gru_in_channels = self.gru_expert.output_channels
        num_classes = config.dataset.data_info.num_classes
        gru_decoder_channels = getattr(config.decode_head, 'gru_channels', 128)
        gru_decoder_dropout = getattr(config.decode_head, 'gru_dropout', 0.1)
        gru_decoder_num_convs = getattr(config.decode_head, 'gru_num_convs', 2)
        gru_decoder_scale_factor = getattr(config.decode_head, 'gru_scale_factor', 2)
        gru_decoder_align_corners = getattr(config.decode_head, 'gru_align_corners', False)
        self.gru_decode_head = LightweightFCNHead(
            in_channels=gru_in_channels,
            channels=gru_decoder_channels,
            num_classes=num_classes,
            dropout_ratio=gru_decoder_dropout,
            num_convs=gru_decoder_num_convs,
            scale_factor=gru_decoder_scale_factor,
            align_corners=gru_decoder_align_corners
        )

        self.loss_weights = {
            'fusion': getattr(config, 'fusion_loss_weight', 1.0),
            'swin': getattr(config, 'swin_loss_weight', 0.5),
            'gru': getattr(config, 'gru_loss_weight', 0.1)
        }

    def forward(self, batch):
        if 'hls' not in batch or 'history' not in batch:
            raise KeyError("SimpleFusionModel requires 'hls' and 'history' in batch")

        hls_data = batch['hls']
        cdl_data = batch['history']
        input_size = hls_data.shape[-2:]

        swin_features = self.swin_expert(hls_data)
        swin_logits_raw = self.swin_decode_head(swin_features)
        swin_logits = F.interpolate(swin_logits_raw, size=input_size, mode='bilinear', align_corners=self.swin_decode_head.align_corners)

        gru_features = self.gru_expert(cdl_data)
        gru_logits_raw = self.gru_decode_head(gru_features)
        gru_logits = F.interpolate(gru_logits_raw, size=input_size, mode='bilinear', align_corners=self.gru_decode_head.align_corners)

        fused_logits = 0.5 * swin_logits + 0.5 * gru_logits

        return {
            'output': fused_logits,
            'swin_output': swin_logits,
            'gru_output': gru_logits,
            'expert_preferences': None
        }

    def forward_loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        if targets.ndim > 3:
            targets = targets.squeeze(1)
        targets = targets.long()

        fusion_loss = compute_fusion_loss(outputs['output'], targets)
        swin_loss = compute_swin_loss(outputs['swin_output'], targets)
        gru_loss = compute_gru_loss(outputs['gru_output'], targets)

        total_loss = (
            fusion_loss
        )

        loss_dict = {
            'total_loss': total_loss.item(),
            'fusion_loss': fusion_loss.item(),
            'swin_loss': swin_loss.item(),
            'gru_loss': gru_loss.item()
        }
        return total_loss, loss_dict
