import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ConvLSTMCell(nn.Module):
    """Basic ConvLSTM cell."""
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim, # input, forget, cell, output gates
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    """ConvLSTM implementation."""
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super().__init__()

        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]

        # layer_output is list of [(b, t, c, h, w)]
        # last_state is list of [[(b, c, h, w), (b, c, h, w)]]
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ChannelAttention(nn.Module):
    """Channel attention module"""
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Average pooling branch
        avg_out = self.avg_pool(x).view(x.size(0), -1)
        avg_out = self.mlp(avg_out)
        
        # Max pooling branch
        max_out = self.max_pool(x).view(x.size(0), -1)
        max_out = self.mlp(max_out)
        
        # Merge and generate attention weights
        channel_att = self.sigmoid(avg_out + max_out).view(x.size(0), x.size(1), 1, 1)
        
        return x * channel_att

class SpatialAttention(nn.Module):
    """Spatial attention module"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate average and max along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate features
        spatial_features = torch.cat([avg_out, max_out], dim=1)
        
        # Generate spatial attention map
        spatial_att = self.sigmoid(self.conv(spatial_features))
        
        return x * spatial_att

class MultiScaleFeatureEnhancement(nn.Module):
    """Multi-scale feature enhancement module"""
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        
        reduced_channels = in_channels // reduction
        
        # Multi-scale convolution branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(reduced_channels * 3, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual connection 1x1 convolution
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        # Multi-scale feature extraction
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        
        # Feature concatenation and fusion
        out = torch.cat([feat1, feat2, feat3], dim=1)
        out = self.fusion(out)
        
        # Residual connection
        out = self.relu(out + identity)
        
        return out

class GRUBranch(nn.Module):
    """Enhanced GRU branch"""
    def __init__(self, config):
        super().__init__()
        
        expert_config = config.gru_expert
        
        # Embedding layer configuration
        num_classes = getattr(config.dataset.data_info, 'num_classes', 6)
        self.embedding_dim = getattr(expert_config, 'embedding_dim', 32)
        self.embedding = nn.Embedding(
            num_embeddings=num_classes,
            embedding_dim=self.embedding_dim,
            padding_idx=None
        )
        
        # Embedding Dropout
        self.embedding_dropout = nn.Dropout(p=0.1)
        
        # ConvLSTM parameters
        self.convlstm_hidden_dim = expert_config.convlstm_hidden_dim
        self.convlstm_num_layers = expert_config.num_layers
        convlstm_kernel_size = tuple(expert_config.convlstm_kernel_size)
        convlstm_input_dim = self.embedding_dim
        
        # ConvLSTM layers
        self.conv_lstm = ConvLSTM(
            input_dim=convlstm_input_dim,
            hidden_dim=self.convlstm_hidden_dim,
            kernel_size=convlstm_kernel_size,
            num_layers=self.convlstm_num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )
        
        # Set output channels
        if isinstance(self.convlstm_hidden_dim, list):
            self.output_channels = self.convlstm_hidden_dim[-1]
        else:
            self.output_channels = self.convlstm_hidden_dim
            
        # Attention module
        self.channel_attention = ChannelAttention(
            in_channels=self.output_channels,
            reduction_ratio=16
        )
        self.spatial_attention = SpatialAttention(
            kernel_size=7
        )
        
        # Multi-scale feature enhancement
        self.feature_enhancement = MultiScaleFeatureEnhancement(
            in_channels=self.output_channels,
            reduction=4
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass
        """
        if x.ndim != 4 or x.shape[1] != 5:
            raise ValueError(f"Expected input shape (B, 5, H, W), but got {x.shape}")
        if not torch.is_tensor(x) or x.dtype != torch.long:
            raise TypeError(f"Expected LongTensor input for embedding, but got {x.dtype}")
        """
        
        B, T, H, W = x.shape
        
        # 1. Embedding
        embedded_x = self.embedding(x)
        embedded_x = self.embedding_dropout(embedded_x)
        
        # 2. Adjust dimensions for ConvLSTM input format
        embedded_x = embedded_x.permute(0, 1, 4, 2, 3)
        
        # 3. ConvLSTM processing
        layer_output_list, last_state_list = self.conv_lstm(embedded_x)
        
        # 4. Get last layer hidden state
        last_hidden_state = last_state_list[-1][0]
        
        # 5. Apply channel attention
        feat_channel = self.channel_attention(last_hidden_state)
        
        # 6. Apply spatial attention
        feat_spatial = self.spatial_attention(feat_channel)
        
        # 7. Multi-scale feature enhancement
        enhanced_features = self.feature_enhancement(feat_spatial)
        
        return enhanced_features 