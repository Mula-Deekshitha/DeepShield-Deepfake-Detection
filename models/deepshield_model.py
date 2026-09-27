import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepShieldEngine(nn.Module):
    def __init__(self, embed_dim=256):
        super(DeepShieldEngine, self).__init__()
        
        # Spatial Stream (RGB Texture / Boundary Features)
        self.spatial_backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Frequency Stream (2D FFT Spectral Artifacts)
        self.freq_backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Landmark Graph Stream (468 3D Mesh Nodes)
        self.graph_fc = nn.Sequential(
            nn.Linear(468 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Classification Head: Outputs [Prob_Real, Prob_Fake]
        self.classifier = nn.Sequential(
            nn.Linear(64 + 64 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, seq_spatial, seq_freq, seq_landmarks, seq_global):
        b, seq_len, c, h, w = seq_spatial.shape
        
        # Flatten batch and sequence dimensions for efficient parallel execution
        spatial_flat = seq_spatial.view(b * seq_len, c, h, w)
        freq_flat = seq_freq.view(b * seq_len, c, h, w)
        landmarks_flat = seq_landmarks.view(b * seq_len, -1)
        
        # Extract modality features
        feat_spatial = self.spatial_backbone(spatial_flat).view(b, seq_len, -1)
        feat_freq = self.freq_backbone(freq_flat).view(b, seq_len, -1)
        feat_graph = self.graph_fc(landmarks_flat).view(b, seq_len, -1)
        
        # Aggregate temporal sequence features (Mean Pooling)
        pooled_spatial = torch.mean(feat_spatial, dim=1)
        pooled_freq = torch.mean(feat_freq, dim=1)
        pooled_graph = torch.mean(feat_graph, dim=1)
        
        # Concatenate multi-modal feature representations
        fused = torch.cat([pooled_spatial, pooled_freq, pooled_graph], dim=-1)
        logits = self.classifier(fused)
        
        # Return probability distribution: [Prob_Real, Prob_Fake]
        return F.softmax(logits, dim=-1)