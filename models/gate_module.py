import torch
import torch.nn as nn
import torch.nn.functional as F

class CategoryAwareGate(nn.Module):
    """
    Category-aware gate module for dynamic fusion of expert outputs
    """
    
    def __init__(self, config): # Modified here to receive config object
        super().__init__()
        
        # Extract parameters from configuration
        self.num_classes = config.num_classes 
        hidden_dim = config.hidden_dim
        dropout = config.dropout
        
        # Create independent gate network for each class
        self.gate_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, hidden_dim), # Input is HLS and CDL probabilities, so it's 2
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2), # Output HLS and CDL weights, so it's 2
                nn.Softmax(dim=1) # Apply Softmax on the last dimension
            ) for _ in range(self.num_classes) # Use self.num_classes
        ])
        
        # Dynamically initialize expert preferences for each class
        # Generate weights dynamically based on class count, not hardcoded
        preferences_data = []
        for i in range(self.num_classes):
            # Generate different weight combinations for each class
            # Here you can adjust weight allocation strategy as needed
            weight_hls = 0.3 + 0.4 * (i / max(1, self.num_classes - 1))  # From 0.3 to 0.7
            weight_cdl = 1.0 - weight_hls
            preferences_data.append([weight_hls, weight_cdl])
        
        self.expert_preferences = nn.Parameter(
            torch.tensor(preferences_data, dtype=torch.float32)
        )
        
        # Add a small epsilon to prevent division by zero
        self.epsilon = 1e-8

    def forward(self, swin_logits, gru_logits):
        """
        Forward pass
        Args:
            swin_logits: logits output from Swin expert (B, num_classes, H, W)
            gru_logits: logits output from GRU expert (B, num_classes, H, W) - assume same dimensions
        Returns:
            fused_logits: logits after fusion
            expert_preferences: current expert preferences (for logging)
            dynamic_weights: dynamic weights (for logging)
        """
        # Ensure logits dimensions are consistent
        assert swin_logits.shape == gru_logits.shape, \
            f"Logits shape mismatch: Swin {swin_logits.shape}, GRU {gru_logits.shape}"
        
        B, C, H, W = swin_logits.shape
        assert C == self.num_classes, \
            f"Number of classes mismatch: Expected {self.num_classes}, got {C}"

        # Convert Logits to probabilities (apply Softmax on class dimension)
        swin_probs = F.softmax(swin_logits, dim=1) # (B, C, H, W)
        gru_probs = F.softmax(gru_logits, dim=1)   # (B, C, H, W)

        # Prepare input for gate network
        # (B, C, H, W) -> (B, H, W, C) -> (B*H*W, C)
        swin_probs_flat = swin_probs.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        gru_probs_flat = gru_probs.permute(0, 2, 3, 1).reshape(-1, self.num_classes)

        # Calculate dynamic weights for each class
        dynamic_weights_list = []
        for i in range(self.num_classes):
            # Extract probabilities for each class
            class_swin_probs = swin_probs_flat[:, i:i+1] # (B*H*W, 1)
            class_gru_probs = gru_probs_flat[:, i:i+1]   # (B*H*W, 1)
            
            # Combine probabilities as input for gate network
            gate_input = torch.cat([class_swin_probs, class_gru_probs], dim=1) # (B*H*W, 2)
            
            # Calculate weights through corresponding gate network
            class_weights = self.gate_networks[i](gate_input) # (B*H*W, 2) -> [weight_swin, weight_gru]
            dynamic_weights_list.append(class_weights.unsqueeze(1)) # (B*H*W, 1, 2)

        # Combine dynamic weights for all classes
        dynamic_weights = torch.cat(dynamic_weights_list, dim=1) # (B*H*W, C, 2)

        # Get current expert preferences (ensure on same device)
        current_preferences = F.softmax(self.expert_preferences, dim=1).to(dynamic_weights.device) # (C, 2)
        
        # Expand expert preferences to match dynamic weights shape
        # (C, 2) -> (1, C, 2) -> (B*H*W, C, 2)
        expanded_preferences = current_preferences.unsqueeze(0).expand(dynamic_weights.size(0), -1, -1)
        
        # Combine dynamic weights and expert preferences (e.g., weighted average)
        # Here simply use dynamic weights, can adjust fusion strategy as needed
        # final_weights = 0.7 * dynamic_weights + 0.3 * expanded_preferences
        # final_weights = F.softmax(final_weights, dim=2) # Ensure sum of two weights for each class equals 1
        
        final_weights = dynamic_weights # Directly use dynamic weights

        # Prepare original Logits for fusion
        # (B, C, H, W) -> (B, H, W, C) -> (B*H*W, C) -> (B*H*W, C, 1)
        swin_logits_flat = swin_logits.permute(0, 2, 3, 1).reshape(-1, self.num_classes).unsqueeze(2)
        gru_logits_flat = gru_logits.permute(0, 2, 3, 1).reshape(-1, self.num_classes).unsqueeze(2)
        
        # Combine Logits from two experts
        expert_logits = torch.cat([swin_logits_flat, gru_logits_flat], dim=2) # (B*H*W, C, 2)

        # Apply weights for fusion
        # (B*H*W, C, 1) = sum((B*H*W, C, 2) * (B*H*W, C, 2), dim=2)
        fused_logits_flat = (final_weights * expert_logits).sum(dim=2) # (B*H*W, C)

        # Restore fused Logits to original shape
        # (B*H*W, C) -> (B, H, W, C) -> (B, C, H, W)
        fused_logits = fused_logits_flat.reshape(B, H, W, self.num_classes).permute(0, 3, 1, 2)

        # Return fusion results, expert preferences and dynamic weights
        return fused_logits, self.expert_preferences, dynamic_weights

    def get_expert_preferences(self):
        """
        Get current expert preferences (normalized)
        """
        # Use softmax to ensure sum of weights for each class equals 1
        preferences = F.softmax(self.expert_preferences, dim=1)
        return preferences
