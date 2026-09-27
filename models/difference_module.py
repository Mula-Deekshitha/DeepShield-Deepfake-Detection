import torch
import torch.nn as nn

class AttentionDDM(nn.Module):
    """
    Attention-Based Difference Detection Module (ADDM).
    Computes multi-scale frame differences (t vs t+1, t vs t+2)
    to capture identity drift and frame-to-frame flickering.
    """
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.spatial_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, frame_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_seq: Feature tensor sequence [Batch, Seq_Len, Feature_Dim]
        Returns:
            Anomaly summary vector [Batch, Feature_Dim]
        """
        b, seq_len, dim = frame_seq.shape
        diffs = []
        
        # Multi-scale frame difference strides
        strides = [1, 2]
        for stride in strides:
            if seq_len > stride:
                f_t = frame_seq[:, :-stride, :]
                f_t_k = frame_seq[:, stride:, :]
                
                diff = torch.abs(f_t - f_t_k)
                concat_feat = torch.cat([f_t, f_t_k], dim=-1)
                gate = self.spatial_gate(concat_feat)
                
                gated_diff = torch.mean(diff * gate, dim=1)
                diffs.append(gated_diff)
                
        if len(diffs) > 0:
            return torch.stack(diffs, dim=1).mean(dim=1)
        
        return torch.zeros(b, dim, device=frame_seq.device)