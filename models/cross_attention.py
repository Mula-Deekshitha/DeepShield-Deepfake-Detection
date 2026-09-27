import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    """
    Fuses representations from Spatial, Frequency, Landmark, and Global streams
    using Multi-Head Attention.
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, spatial: torch.Tensor, freq: torch.Tensor, landmark: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        # Stack inputs into sequence tensor of shape [Batch, 4, Embed_Dim]
        tokens = torch.stack([spatial, freq, landmark, global_feat], dim=1)
        
        # Self/Cross attention across modalities
        attn_out, _ = self.mha(tokens, tokens, tokens)
        tokens = self.norm(tokens + attn_out)
        
        # Feed-forward network
        ffn_out = self.ffn(tokens)
        fused = self.norm(tokens + ffn_out)
        
        # Flatten tokens into a unified feature vector: [Batch, 4 * Embed_Dim]
        return fused.view(fused.size(0), -1)