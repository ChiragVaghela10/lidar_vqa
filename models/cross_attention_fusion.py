import torch
import torch.nn as nn

class CrossAttentionFusionHead(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=256, num_heads=4, num_classes=5, dropout=0.1):
        super().__init__()

        # Project CLIP features to same dim
        self.img_proj = nn.Linear(embed_dim, hidden_dim)
        self.txt_proj = nn.Linear(embed_dim, hidden_dim)

        # Cross-attention layers
        self.text_to_image_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.image_to_text_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        # Normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, image_feat, text_feat):
        """
        image_feat: [B, 512]
        text_feat:  [B, 512]
        """

        # Project both to hidden_dim
        img = self.img_proj(image_feat).unsqueeze(1)   # [B, 1, hidden]
        txt = self.txt_proj(text_feat).unsqueeze(1)    # [B, 1, hidden]

        # Text attends to image
        txt_attn_out, txt_weights = self.text_to_image_attn(query=txt, key=img, value=img)
        txt_attn_out = self.norm1(txt + txt_attn_out)  # residual

        # Image attends to text
        img_attn_out, img_weights = self.image_to_text_attn(query=img, key=txt, value=txt)
        img_attn_out = self.norm2(img + img_attn_out)  # residual

        # Combine both (concat)
        fused = torch.cat([txt_attn_out.squeeze(1), img_attn_out.squeeze(1)], dim=-1)  # [B, 2H]

        # Classification
        out = self.classifier(fused)

        return out, txt_weights, img_weights  # return weights for visualization
