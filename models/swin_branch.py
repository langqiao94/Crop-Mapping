import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional, List, Tuple

"""Image to Patch Embedding."""
# Use same padding, keep output size as input size divided by stride
# Return feature map and shape information
class PatchEmbed(nn.Module):
    """Image to Patch embedding."""
    def __init__(self, 
                 in_channels: int,
                 embed_dims: int,
                 conv_type: str = 'Conv2d',
                 kernel_size: int = 4,
                 stride: int = 4,
                 padding: str = 'same',
                 norm_cfg: Optional[dict] = None):
        super().__init__()
        self.embed_dims = embed_dims
        
        if padding == 'corner':
            self.proj = nn.Conv2d(
                in_channels, embed_dims,
                kernel_size=kernel_size,
                stride=stride,
                padding=0)
        elif padding == 'same':
            # Use same padding to maintain output size as 1/stride of input size
            pad_size = (kernel_size - 1) // 2
            self.proj = nn.Conv2d(
                in_channels, embed_dims,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad_size)
        else:
            self.proj = nn.Conv2d(
                in_channels, embed_dims,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size // 2, kernel_size // 2))
        
        if norm_cfg is not None:
            self.norm = nn.LayerNorm(embed_dims)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.proj.stride[0] != 0:
            pad_w = self.proj.stride[0] - W % self.proj.stride[0]
            x = F.pad(x, (0, pad_w))
        if H % self.proj.stride[0] != 0:
            pad_h = self.proj.stride[0] - H % self.proj.stride[0]
            x = F.pad(x, (0, 0, 0, pad_h))

        x = self.proj(x)  # B C Wh Ww
        Wh, Ww = x.size(2), x.size(3)
        
        if self.norm is not None:
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dims, Wh, Ww)

        # Return feature map and shape information
        return x, (Wh, Ww)

"""Patch Merging Layer.
    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
        norm_layer (nn.Module): Normalization layer
    """
# Input feature has wrong size
# Ensure x size (H*W) are even.
class PatchMerging(nn.Module):
    """Patch Merging Layer.
    
    Args:
        in_channels (int): Input channel count
        out_channels (int): Output channel count
        norm_layer (nn.Module): Normalization layer
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = norm_layer(4 * in_channels)
        self.reduction = nn.Linear(4 * in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, C)
            H, W: Height and width of input feature map
        """
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        assert H % 2 == 0 and W % 2 == 0, f'x size ({H}*{W}) are not even.'

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

"""Partition feature map into windows.
    Args:
        x (torch.Tensor): Input feature map, shape (B, H, W, C)
        window_size (int): Window size
    Returns:
        windows: Windowed features, shape (-1, window_size, window_size, C)
    """
# Calculate required padding
# Pad feature map
# Update height and width
# Partition windows
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Divide feature map into windows.
    
    Args:
        x (torch.Tensor): Input feature map, shape (B, H, W, C)
        window_size (int): Window size
    Returns:
        windows (torch.Tensor): Windowed features with shape (-1, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    
    # Calculate required padding
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    
    if pad_h > 0 or pad_w > 0:
        # Apply padding to feature map
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    
    # Update height and width
    H_pad = H + pad_h
    W_pad = W + pad_w
    
    # Divide into windows
    x = x.view(B, H_pad // window_size, window_size, W_pad // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse windows to feature map.
    
    Args:
        windows: Input windowed features, shape (-1, window_size, window_size, C)
        window_size (int): Window size
        H (int): Original feature map height
        W (int): Original feature map width
    
    Returns:
        x: Feature map, shape (B, H, W, C)
    """
    # Calculate height and width after padding
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    H_pad = H + pad_h
    W_pad = W + pad_w
    
    B = int(windows.shape[0] / (H_pad * W_pad / window_size / window_size))
    x = windows.view(B, H_pad // window_size, W_pad // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_pad, W_pad, -1)
    
    if pad_h > 0 or pad_w > 0:
        # Remove padding
        x = x[:, :H, :W, :]
    
    return x

"""Window based multi-head self attention module."""
# Relative position bias table
# Get relative position index for each pair inside window
class WindowAttention(nn.Module):
    """Window based multi-head self attention module."""
    
    def __init__(self, 
                 dim: int,
                 window_size: int,
                 num_heads: int,
                 qkv_bias: bool = True,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position encoding table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        
        # Get relative position index for each pair inside window
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

"""Swin Transformer Block."""
# Apply LayerNorm, ensure normalization on the last dimension
# Reshape feature to spatial format
# cyclic shift
# partition windows
# W-MSA/SW-MSA
# merge windows
# reverse cyclic shift
# Reshape back to sequence format
class SwinBlock(nn.Module):
    """Swin Transformer Block."""

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 window_size: int = 7,
                 shift_size: int = 0,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN'),
                 with_cp: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp

        assert 0 <= self.shift_size < self.window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x: torch.Tensor, mask_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        def _inner_forward(x: torch.Tensor) -> torch.Tensor:
            identity = x
            B, L, C = x.shape
            H, W = int(np.sqrt(L)), int(np.sqrt(L))
            
            # Apply LayerNorm, ensure normalization on the last dimension
            x = self.norm1(x)  # [B, L, C]
            
            # Reshape features to spatial form
            x = x.reshape(B, H, W, C)

            # cyclic shift
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x

            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # [num_windows*B, window_size, window_size, C]
            x_windows = x_windows.reshape(-1, self.window_size * self.window_size, C)  # [num_windows*B, window_size*window_size, C]

            # W-MSA/SW-MSA
            attn_windows = self.attn(x_windows, mask=mask_matrix)  # [num_windows*B, window_size*window_size, C]

            # merge windows
            attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, C)  # [num_windows*B, window_size, window_size, C]
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # [B, H, W, C]

            # reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = shifted_x

            # Reshape back to sequence form
            x = x.reshape(B, H * W, C)  # [B, L, C]

            # FFN
            x = identity + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

            return x

        if self.with_cp and x.requires_grad:
            x = checkpoint.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x

"""Implements one stage in Swin Transformer."""
# build blocks
# calculate attention mask for SW-MSA
class SwinBlockSequence(nn.Module):
    """Implements one stage in Swin Transformer."""

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 feedforward_channels: int,
                 depth: int,
                 window_size: int = 7,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 downsample: Optional[nn.Module] = None,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN'),
                 with_cp: bool = False):
        super().__init__()

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinBlock(
                dim=embed_dims,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=4.,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(
                    drop_path_rate, list) else drop_path_rate,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp) for i in range(depth)
        ])

        self.downsample = downsample

    def forward(self, x: torch.Tensor, hw_shape: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        H, W = hw_shape
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                   slice(-self.window_size, -self.shift_size),
                   slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                   slice(-self.window_size, -self.shift_size),
                   slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            x = blk(x, attn_mask)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x_down, (Wh, Ww)
        else:
            return x, (H, W)

"""Swin Transformer backbone."""
# Access swin_expert configuration
# Use hls_channels from dataset configuration
# Use other parameters from swin_expert configuration
# Fixed value
# Use current channel count to create LayerNorm
# Update channel count for next stage
class SwinBranch(nn.Module):
    """Swin Transformer backbone."""

    def __init__(self, config):
        super().__init__()
        
        # Access swin_expert configuration
        swin_config = config.swin_expert 
        
        # Use hls_channels from dataset configuration
        self.in_chans = config.dataset.data_info.hls_channels 
        
        # Use other parameters from swin_expert configuration
        self.pretrain_img_size = 224  # Fixed value
        self.patch_size = swin_config.patch_size
        self.embed_dims = swin_config.embed_dims
        self.window_size = swin_config.window_size
        self.mlp_ratio = swin_config.mlp_ratio
        self.depths = swin_config.depths
        self.num_heads = swin_config.num_heads
        self.out_indices = swin_config.out_indices
        self.qkv_bias = swin_config.qkv_bias
        self.patch_norm = swin_config.patch_norm
        self.drop_rate = swin_config.drop_rate
        self.attn_drop_rate = swin_config.attn_drop_rate
        self.drop_path_rate = swin_config.drop_path_rate
        self.use_abs_pos_embed = swin_config.use_abs_pos_embed
        self.act_cfg = swin_config.act_cfg
        self.norm_cfg = swin_config.norm_cfg
        
        self.num_layers = len(self.depths)
        self.out_channels = [self.embed_dims * 2**i for i in range(self.num_layers)]

        # patch embedding
        self.patch_embed = PatchEmbed(
            in_channels=self.in_chans, # Use channel count from dataset
            embed_dims=self.embed_dims,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding='corner',
            norm_cfg=self.norm_cfg if self.patch_norm else None)

        if self.use_abs_pos_embed:
            patch_row = self.pretrain_img_size // self.patch_size
            patch_col = self.pretrain_img_size // self.patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, self.embed_dims)))
            self.drop_after_pos = nn.Dropout(p=self.drop_rate)

        # stochastic depth
        total_depth = sum(self.depths)
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, total_depth)]

        # build stages
        self.stages = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        cur_block_idx = 0
        
        # Track current channel count
        current_dim = self.embed_dims
        
        for i in range(self.num_layers):
            if i in self.out_indices:
                # Use current channel count to create LayerNorm
                norm_layer = nn.LayerNorm(current_dim)
                self.norm_layers.append(norm_layer)

            if i < self.num_layers - 1:
                downsample = PatchMerging(
                    in_channels=current_dim,
                    out_channels=current_dim * 2)
            else:
                downsample = None

            stage = SwinBlockSequence(
                embed_dims=current_dim,
                num_heads=self.num_heads[i],
                feedforward_channels=int(self.mlp_ratio * current_dim),
                depth=self.depths[i],
                window_size=self.window_size,
                qkv_bias=True,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                drop_path_rate=dpr[cur_block_idx:cur_block_idx + self.depths[i]],
                downsample=downsample,
                act_cfg=self.act_cfg,
                norm_cfg=self.norm_cfg,
                with_cp=False)
            self.stages.append(stage)
            cur_block_idx += self.depths[i]
            
            # Update channel count for next stage
            if downsample:
                current_dim *= 2

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward function."""
        outs = []
        
        x, hw_shape = self.patch_embed(x)
        Wh, Ww = hw_shape
        
        if self.use_abs_pos_embed:
            x = x.flatten(2).transpose(1, 2)
            x = x + self.absolute_pos_embed
            x = self.drop_after_pos(x)
        else:
            x = x.flatten(2).transpose(1, 2)
        
        norm_index = 0
        
        for i, stage in enumerate(self.stages):
            # Apply LayerNorm before stage processing
            if i in self.out_indices:
                out = self.norm_layers[norm_index](x)
                out = out.view(-1, *hw_shape, out.size(-1)).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
                norm_index += 1
            
            # Process current stage
            x, hw_shape = stage(x, (Wh, Ww))
            Wh, Ww = hw_shape

        return outs 