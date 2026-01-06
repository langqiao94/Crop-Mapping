import torch
import torch.nn as nn
import torch.nn.functional as F

class CategoryAwareGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.num_classes = config.num_classes
        hidden_dim = config.hidden_dim
        dropout = config.dropout
        
        self.gate_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
                nn.Softmax(dim=1)
            ) for _ in range(self.num_classes)
        ])
        
        preferences_data = []
        for i in range(self.num_classes):
            weight_hls = 0.3 + 0.4 * (i / max(1, self.num_classes - 1))
            weight_cdl = 1.0 - weight_hls
            preferences_data.append([weight_hls, weight_cdl])
        
        self.expert_preferences = nn.Parameter(
            torch.tensor(preferences_data, dtype=torch.float32)
        )
        
        self.epsilon = 1e-8

    def forward(self, swin_logits, gru_logits):
        assert swin_logits.shape == gru_logits.shape, \
            f"Logits shape mismatch: Swin {swin_logits.shape}, GRU {gru_logits.shape}"
        
        B, C, H, W = swin_logits.shape
        assert C == self.num_classes, \
            f"Number of classes mismatch: Expected {self.num_classes}, got {C}"

        swin_probs = F.softmax(swin_logits, dim=1)
        gru_probs = F.softmax(gru_logits, dim=1)

        swin_probs_flat = swin_probs.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        gru_probs_flat = gru_probs.permute(0, 2, 3, 1).reshape(-1, self.num_classes)

        dynamic_weights_list = []
        for i in range(self.num_classes):
            class_swin_probs = swin_probs_flat[:, i:i+1]
            class_gru_probs = gru_probs_flat[:, i:i+1]
            
            gate_input = torch.cat([class_swin_probs, class_gru_probs], dim=1)
            
            class_weights = self.gate_networks[i](gate_input)
            dynamic_weights_list.append(class_weights.unsqueeze(1))

        dynamic_weights = torch.cat(dynamic_weights_list, dim=1)

        current_preferences = F.softmax(self.expert_preferences, dim=1).to(dynamic_weights.device)
        
        expanded_preferences = current_preferences.unsqueeze(0).expand(dynamic_weights.size(0), -1, -1)
        
        final_weights = dynamic_weights

        swin_logits_flat = swin_logits.permute(0, 2, 3, 1).reshape(-1, self.num_classes).unsqueeze(2)
        gru_logits_flat = gru_logits.permute(0, 2, 3, 1).reshape(-1, self.num_classes).unsqueeze(2)
        
        expert_logits = torch.cat([swin_logits_flat, gru_logits_flat], dim=2)

        fused_logits_flat = (final_weights * expert_logits).sum(dim=2)

        fused_logits = fused_logits_flat.reshape(B, H, W, self.num_classes).permute(0, 3, 1, 2)

        return fused_logits, self.expert_preferences, dynamic_weights

    def get_expert_preferences(self):
        preferences = F.softmax(self.expert_preferences, dim=1)
        return preferences
