import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetEncoder(nn.Module):
    def __init__(self, in_channels=77, base_channels=64):
        super().__init__()
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )
        
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        bottleneck = self.bottleneck(enc4)
        
        skip_connections = [enc1, enc2, enc3, enc4]
        
        return bottleneck, skip_connections


class BiLSTMFeatureEnhancer(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256, num_layers=1):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, feature_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        seq = x.view(B, C, -1).permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(seq)
        
        attn_scores = self.attention(lstm_out)
        attn_weights = F.softmax(attn_scores, dim=1)
        weighted_out = (lstm_out * attn_weights).sum(dim=1)
        
        enhanced_flat = self.projection(weighted_out)
        
        enhanced = enhanced_flat.view(B, C, 1, 1)
        enhanced = F.interpolate(enhanced, size=(H, W), mode='bilinear', align_corners=False)
        
        return enhanced


class UNetDecoder(nn.Module):
    def __init__(self, base_channels=64, num_classes=6):
        super().__init__()
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 8, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 16, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Conv2d(base_channels, num_classes, kernel_size=1)
    
    def forward(self, bottleneck, skip_connections):
        enc1, enc2, enc3, enc4 = skip_connections
        
        x = self.up1(bottleneck)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec1(x)
        
        x = self.up2(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec2(x)
        
        x = self.up3(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec3(x)
        
        x = self.up4(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec4(x)
        
        x = self.classifier(x)
        
        return x


class BiLSTMModel(nn.Module):
    def __init__(
        self,
        in_channels=77,
        num_classes=6,
        img_size=128,
        spatial_channels=[128, 256, 256],
        pool_size=8,
        lstm_hidden=512,
        lstm_layers=2,
        mlp_dims=[512, 256],
        bilstm_enhance_weight=0.2,
        **kwargs
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.bilstm_enhance_weight = bilstm_enhance_weight
        
        base_channels = 64
        
        self.encoder = UNetEncoder(
            in_channels=in_channels,
            base_channels=base_channels
        )
        
        bottleneck_dim = base_channels * 8
        bilstm_hidden = min(lstm_hidden // 2, 256)
        bilstm_layers = 1
        
        self.bilstm_enhancer = BiLSTMFeatureEnhancer(
            feature_dim=bottleneck_dim,
            hidden_dim=bilstm_hidden,
            num_layers=bilstm_layers
        )
        
        self.decoder = UNetDecoder(
            base_channels=base_channels,
            num_classes=num_classes
        )
        
        print(f"\nBiLSTM Model (U-Net + BiLSTM Hybrid):")
        print(f"  Input: ({in_channels}, {img_size}, {img_size})")
        print(f"  Architecture: U-Net Encoder-Decoder + BiLSTM Enhancement")
        print(f"  U-Net base channels: {base_channels}")
        print(f"  BiLSTM hidden dim: {bilstm_hidden} (reduced from {lstm_hidden})")
        print(f"  BiLSTM layers: {bilstm_layers} (reduced from {lstm_layers})")
        print(f"  BiLSTM enhancement weight: {bilstm_enhance_weight} ({bilstm_enhance_weight*100:.0f}% participation)")
        print(f"  Output: ({num_classes}, {img_size}, {img_size}) - Pixel-level")
    
    def forward(self, x):
        bottleneck, skip_connections = self.encoder(x)
        
        bilstm_enhanced = self.bilstm_enhancer(bottleneck)
        
        enhanced_bottleneck = (1 - self.bilstm_enhance_weight) * bottleneck + \
                              self.bilstm_enhance_weight * bilstm_enhanced
        
        output = self.decoder(enhanced_bottleneck, skip_connections)
        
        return output


def build_bilstm(config):
    model = BiLSTMModel(**config)
    return model


if __name__ == "__main__":
    print("Testing BiLSTM model (U-Net + BiLSTM Hybrid, PIXEL-LEVEL)...")
    
    config = {
        'in_channels': 77,
        'num_classes': 6,
        'img_size': 128,
        'spatial_channels': [128, 256, 256],
        'pool_size': 8,
        'lstm_hidden': 512,
        'lstm_layers': 2,
        'mlp_dims': [512, 256],
        'bilstm_enhance_weight': 0.2
    }
    
    model = build_bilstm(config)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {num_params:,}")
    
    bilstm_params = sum(p.numel() for p in model.bilstm_enhancer.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    bilstm_ratio = bilstm_params / total_params * 100
    print(f"BiLSTM parameters: {bilstm_params:,} ({bilstm_ratio:.1f}% of total)")
    print(f"U-Net parameters: {total_params - bilstm_params:,} ({(1-bilstm_ratio)*100:.1f}% of total)")
    
    x = torch.randn(2, 77, 128, 128)
    output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: (2, 6, 128, 128)")
    print(f"Is pixel-level: {len(output.shape) == 4}")
    
    labels = torch.randint(0, 6, (2, 128, 128))
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, labels)
    print(f"\nLoss (pixel-level): {loss.item():.4f}")
    
    preds = torch.argmax(output, dim=1)
    acc = (preds == labels).float().mean().item() * 100
    print(f"Pixel-level accuracy: {acc:.2f}%")
    
    if output.shape == (2, 6, 128, 128):
        print("\nSUCCESS: Model test passed! (U-Net + BiLSTM Hybrid, PIXEL-LEVEL)")
    else:
        print("\nERROR: Model test failed!")
