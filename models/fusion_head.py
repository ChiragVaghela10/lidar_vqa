import torch
import torch.nn as nn

class CLIPFusionHead(nn.Module):
    def __init__(self, input_dim=512*2, hidden_dim=256, num_classes=5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, image_feat, text_feat):
        combined = torch.cat([image_feat, text_feat], dim=1)
        return self.mlp(combined)
