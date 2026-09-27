import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DeepShieldDataset(Dataset):
    """
    Sequence dataset loader that returns spatial frames, frequency spectra,
    3D MediaPipe landmarks, and global transformer inputs for video frame sequences.
    """
    def __init__(self, num_samples: int = 100, seq_len: int = 5):
        self.num_samples = num_samples
        self.seq_len = seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        # Generate multi-modal sequence tensors [Seq_Len, Channels, Height, Width]
        seq_spatial = torch.randn(self.seq_len, 3, 224, 224)
        seq_freq = torch.randn(self.seq_len, 3, 224, 224)
        seq_landmarks = torch.randn(self.seq_len, 468, 3)
        seq_global = torch.randn(self.seq_len, 3, 224, 224)
        
        # Binary target label: 0 for Real, 1 for Fake
        label = torch.tensor(np.random.choice([0, 1]), dtype=torch.long)
        
        return {
            'spatial': seq_spatial,
            'freq': seq_freq,
            'landmarks': seq_landmarks,
            'global_img': seq_global,
            'label': label
        }

def get_dataloader(batch_size: int = 2, num_samples: int = 50, seq_len: int = 5):
    dataset = DeepShieldDataset(num_samples=num_samples, seq_len=seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)