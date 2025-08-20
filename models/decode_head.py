import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

class PPM(nn.Module):
    """Pyramid Pooling Module."""
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 pool_scales: tuple = (1, 2, 3, 6),
                 align_corners: bool = False):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.pools = nn.ModuleList()
        
        for scale in pool_scales:
            self.pools.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, channels, 1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        ppm_outs = []
        for pool in self.pools:
            ppm_out = pool(x)
            upsampled_ppm_out = F.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs

class UPerHead(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding."""

    def __init__(self,
                 in_channels: List[int],
                 channels: int,
                 pool_scales: tuple = (1, 2, 3, 6),
                 dropout_ratio: float = 0.1,
                 num_classes: int = 6,
                 align_corners: bool = False,
                 norm_cfg: Optional[dict] = None):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.align_corners = align_corners

        # PSP Module
        self.psp_modules = PPM(
            in_channels[-1],
            channels,
            pool_scales,
            align_corners)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels[-1] + len(pool_scales) * channels,
                channels,
                3,
                padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channel in in_channels[:-1]:  # skip the top layer
            l_conv = nn.Conv2d(in_channel, channels, 1)
            fpn_conv = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(
                len(in_channels) * channels,
                channels,
                3,
                padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.dropout = nn.Dropout2d(dropout_ratio)
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def psp_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        psp_outs = [inputs]
        psp_outs.extend(self.psp_modules(inputs))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(inputs[i]))

        # FPN
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        fpn_outs = []
        for i in range(used_backbone_levels):
            fpn_outs.append(self.fpn_convs[i](laterals[i]))

        # Append psp feature
        psp_out = self.psp_forward(inputs[-1])
        fpn_outs.append(psp_out)

        # Resize to the same size
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.dropout(output)
        output = self.conv_seg(output)
        return output

class FCNHead(nn.Module):
    """Fully Convolution Networks for Semantic Segmentation."""

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_classes: int,
                 dropout_ratio: float = 0.1,
                 align_corners: bool = False):
        super().__init__()
        self.align_corners = align_corners
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.conv_2 = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.dropout(x)
        x = self.conv_2(x)
        return x

class SimpleFCNHead(nn.Module):
    """
    A simple FCN head for single-scale feature map input.
    """
    def __init__(self, in_channels, channels, num_classes, dropout_ratio=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels # Intermediate channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio

        # Define layers (if BatchNorm follows immediately, Conv2d's bias=False)
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = nn.Identity() # If dropout_ratio is 0, do not use dropout

        # Final classification layer
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, x):
        # This head expects a single feature map
        # Handle input possibly being a list/tuple
        if isinstance(x, (list, tuple)):
             # Assume related feature map is the only element or the last element
             x = x[-1]
        elif not isinstance(x, torch.Tensor):
             raise TypeError(f"SimpleFCNHead expected Tensor input, but got {type(x)}")

        # Check input dimensions
        if x.ndim != 4:
            raise ValueError(f"SimpleFCNHead expected 4D input (B, C, H, W), but got {x.ndim}D shape: {x.shape}")
        if x.shape[1] != self.in_channels:
             raise ValueError(f"SimpleFCNHead expected {self.in_channels} input channels, but got {x.shape[1]}")

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv_seg(x)
        # Output is original logits, resolution same as input feature map
        return x 

class LightweightFCNHead(nn.Module):
    """A lightweight FCN head suitable for single-scale feature input from ConvLSTM.

    Uses a couple of conv layers with intermediate upsampling.
    """
    def __init__(self, 
                 in_channels: int,
                 channels: int, # Intermediate channels
                 num_classes: int, 
                 dropout_ratio: float = 0.1,
                 num_convs: int = 2, # Number of conv blocks
                 scale_factor: int = 2, # Upsampling factor per block
                 align_corners: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.num_convs = num_convs
        self.scale_factor = scale_factor
        self.align_corners = align_corners

        conv_layers = []
        current_channels = in_channels
        for i in range(num_convs):
            conv_block = nn.Sequential(
                nn.Conv2d(current_channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
            conv_layers.append(conv_block)
            current_channels = channels
            
        self.convs = nn.ModuleList(conv_layers)

        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = nn.Identity()

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: (B, C_in, H', W') from GRUBranch (ConvLSTM)
        if isinstance(x, (list, tuple)):
            # Should not happen with ConvLSTM output, but good practice
            x = x[-1]
        elif not isinstance(x, torch.Tensor):
             raise TypeError(f"LightweightFCNHead expected Tensor input, but got {type(x)}")
        if x.ndim != 4:
             raise ValueError(f"LightweightFCNHead expected 4D input, but got {x.ndim}D shape: {x.shape}")
        if x.shape[1] != self.in_channels:
             raise ValueError(f"LightweightFCNHead expected {self.in_channels} input channels, but got {x.shape[1]}")

        output = x
        for i in range(self.num_convs):
            output = self.convs[i](output)
            # Upsample after every block
            if i < self.num_convs: # Upsample after every block
                output = F.interpolate(output, 
                                       scale_factor=self.scale_factor, 
                                       mode='bilinear', 
                                       align_corners=self.align_corners)
        
        output = self.dropout(output)
        output = self.conv_seg(output)
        # Output shape (B, C, H_final, W_final) - depends on scale_factor and num_convs
        return output 