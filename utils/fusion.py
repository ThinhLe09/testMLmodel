import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion giữa đặc trưng ảnh và văn bản."""

    def __init__(self, visual_dim, text_dim, embed_dim, num_heads=8):
        super().__init__()
        self.vis_project = nn.Linear(visual_dim, embed_dim)
        self.txt_project = nn.Linear(text_dim, embed_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, visual_features, text_features):
        v_proj = self.vis_project(visual_features)
        t_proj = self.txt_project(text_features)

        attn_output, _ = self.multihead_attn(query=t_proj, key=v_proj, value=v_proj)
        combined = self.norm(attn_output + t_proj)

        avg_pool = combined.mean(dim=1)
        max_pool = combined.max(dim=1)[0]

        return avg_pool + max_pool
