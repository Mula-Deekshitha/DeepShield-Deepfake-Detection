import cv2
import numpy as np

class FrequencyTransform:
    """
    Computes 2D Fast Fourier Transform (FFT) log-magnitude spectrum
    to expose high-frequency GAN/Diffusion synthesis artifacts.
    """
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def compute_fft(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Computes 2D FFT magnitude spectrum for an RGB frame.
        Returns a float32 array normalized between [0, 1] of shape [3, H, W].
        """
        if image_rgb.shape[:2] != self.target_size:
            image_rgb = cv2.resize(image_rgb, self.target_size)

        channels_fft = []
        for c in range(3):
            channel = image_rgb[:, :, c].astype(np.float32)
            # 2D Discrete Fourier Transform
            dft = np.fft.fft2(channel)
            dft_shift = np.fft.fftshift(dft)
            
            # Log-magnitude spectrum to compress dynamic range
            magnitude_spectrum = np.log(np.abs(dft_shift) + 1e-8)
            
            # Normalize to [0, 1]
            min_val, max_val = np.min(magnitude_spectrum), np.max(magnitude_spectrum)
            if max_val - min_val > 1e-5:
                norm_spectrum = (magnitude_spectrum - min_val) / (max_val - min_val)
            else:
                norm_spectrum = np.zeros_like(magnitude_spectrum)
                
            channels_fft.append(norm_spectrum)

        # Shape: [3, 224, 224]
        return np.array(channels_fft, dtype=np.float32)