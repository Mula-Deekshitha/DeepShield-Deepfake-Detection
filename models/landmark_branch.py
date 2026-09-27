import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
    Standard Graph Convolution Layer to pass messages between facial landmarks.
    """
    def __init__(self, in_features: int, out_features: int):
        super(GraphConvolution, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Nodes, Features]
        support = self.fc(x)
        output = torch.matmul(adj, support)
        return F.relu(output)

class LandmarkGCNBranch(nn.Module):
    """
    Landmark Branch using a Graph Convolutional Network (GCN).
    Learns structural geometry deformities around eyes, lips, and facial contours.
    """
    def __init__(self, num_nodes: int = 468, in_dim: int = 3, embed_dim: int = 128):
        super().__init__()
        self.num_nodes = num_nodes
        
        # Learnable adaptive adjacency matrix to discover dynamic facial relationships
        self.adj = nn.Parameter(torch.randn(num_nodes, num_nodes))
        
        self.gcn1 = GraphConvolution(in_dim, 64)
        self.gcn2 = GraphConvolution(64, 128)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_nodes * 128, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Softmax normalization on the learnable adjacency matrix
        adj_norm = F.softmax(self.adj, dim=-1)
        
        out = self.gcn1(x, adj_norm)
        out = self.gcn2(out, adj_norm)
        out = self.fc(out)
        return out