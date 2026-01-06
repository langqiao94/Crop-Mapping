import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttentionWithRelativePositionBias(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_length=200, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.max_seq_length = max_seq_length
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        self.dense = nn.Linear(d_model, d_model)
        
        self.relative_position_bias = nn.Parameter(
            torch.randn(2 * max_seq_length - 1, num_heads)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        seq_length = q.size(1)
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        
        dk = torch.tensor(self.depth, dtype=torch.float32)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)
        
        pos_i = torch.arange(seq_length, device=q.device).unsqueeze(1)
        pos_j = torch.arange(seq_length, device=q.device).unsqueeze(0)
        pos_indices = pos_i - pos_j + self.max_seq_length - 1
        pos_indices = torch.clamp(pos_indices, 0, 2 * self.max_seq_length - 2)
        
        pos_bias = self.relative_position_bias[pos_indices]
        pos_bias = pos_bias.permute(2, 0, 1).unsqueeze(0)
        
        scaled_attention_logits = scaled_attention_logits + pos_bias
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, v)
        
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        
        output = self.dense(output)
        
        return output, attention_weights


class PointWiseFeedForward(nn.Module):
    def __init__(self, d_model, dff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.linear2 = nn.Linear(dff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, max_seq_length=200, dropout=0.1):
        super().__init__()
        
        self.mha = MultiHeadAttentionWithRelativePositionBias(
            d_model, num_heads, max_seq_length, dropout
        )
        self.ffn = PointWiseFeedForward(d_model, dff, dropout)
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, 
                 max_seq_length=200, dropout=0.1):
        super().__init__()
        
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dff, max_seq_length, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class SpatialToSequence(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=4, img_size=128):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.proj(x)
        
        x = x.flatten(2)
        x = x.transpose(1, 2)
        
        x = x + self.pos_embed
        
        return x


class CRITTransformer(nn.Module):
    def __init__(
        self,
        in_channels=42,
        num_classes=6,
        img_size=128,
        patch_size=4,
        d_model=256,
        num_heads=8,
        num_layers=4,
        dff=1024,
        dropout=0.1,
        max_seq_length=1024
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_patches_per_side = img_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2
        
        self.spatial_embed = SpatialToSequence(
            in_channels, d_model, patch_size, img_size
        )
        
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        self.patch_classifier = nn.Linear(d_model, num_classes * (patch_size ** 2))
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def create_padding_mask(self, x):
        return None
    
    def forward(self, x, return_features=False):
        batch_size = x.size(0)
        H, W = x.size(2), x.size(3)
        
        patch_embeddings = self.spatial_embed(x)
        
        mask = self.create_padding_mask(patch_embeddings)
        
        encoded = self.encoder(patch_embeddings, mask)
        
        encoded = self.layer_norm(encoded)
        
        patch_logits = self.patch_classifier(encoded)
        
        patch_logits = patch_logits.view(
            batch_size, 
            self.num_patches, 
            self.num_classes, 
            self.patch_size, 
            self.patch_size
        )
        
        patch_logits = patch_logits.view(
            batch_size,
            self.num_patches_per_side,
            self.num_patches_per_side,
            self.num_classes,
            self.patch_size,
            self.patch_size
        )
        
        patch_logits = patch_logits.permute(0, 3, 1, 4, 2, 5).contiguous()
        
        logits = patch_logits.view(batch_size, self.num_classes, H, W)
        
        if return_features:
            return logits, encoded
        return logits


def build_crit_transformer(config):
    model = CRITTransformer(
        in_channels=config.get('in_channels', 42),
        num_classes=config.get('num_classes', 6),
        img_size=config.get('img_size', 128),
        patch_size=config.get('patch_size', 4),
        d_model=config.get('d_model', 256),
        num_heads=config.get('num_heads', 8),
        num_layers=config.get('num_layers', 4),
        dff=config.get('dff', 1024),
        dropout=config.get('dropout', 0.1),
        max_seq_length=config.get('max_seq_length', 1024)
    )
    return model


if __name__ == "__main__":
    print("Testing CRIT Transformer model (PIXEL-LEVEL)...")
    
    batch_size = 2
    in_channels = 77
    img_size = 128
    num_classes = 6
    
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    
    config = {
        'in_channels': in_channels,
        'num_classes': num_classes,
        'img_size': img_size,
        'patch_size': 4,
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 4,
        'dff': 1024,
        'dropout': 0.1,
        'max_seq_length': 1024
    }
    
    model = build_crit_transformer(config)
    
    logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected output shape: (batch={batch_size}, num_classes={num_classes}, H={img_size}, W={img_size})")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    labels = torch.randint(0, num_classes, (batch_size, img_size, img_size))
    print(f"\nLabel shape: {labels.shape}")
    
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    print(f"Loss (pixel-level): {loss.item():.4f}")
    
    print("\nModel test passed!")
