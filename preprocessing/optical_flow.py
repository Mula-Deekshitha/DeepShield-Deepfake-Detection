import cv2
import numpy as np
import torch

class OpticalFlowExtractor:
    """
    Computes Farneback Dense Optical Flow maps across frame sequences to track subtle temporal motion features.
    """
    def __init__(self, size=(112, 112)):
        self.size = size

    def compute_flow(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> torch.Tensor:
        """
        Args:
            prev_frame: Previous frame in RGB sequence.
            curr_frame: Current frame in RGB sequence.
        Returns:
            torch.Tensor: Optical flow field tensor of shape (3, H, W).
        """
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
        
        prev_gray = cv2.resize(prev_gray, self.size)
        curr_gray = cv2.resize(curr_gray, self.size)
        
        # Calculate Farneback Optical Flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Convert Cartesian coordinates (dx, dy) to Polar coordinates (magnitude & angle)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        hsv = np.zeros((self.size[1], self.size[0], 3), dtype=np.float32)
        hsv[..., 0] = ang * 180 / np.pi / 2.0
        hsv[..., 1] = 255.0
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        rgb_flow = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        tensor_flow = torch.from_numpy(rgb_flow).permute(2, 0, 1).float() / 255.0
        return tensor_flow