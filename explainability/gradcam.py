import cv2
import numpy as np
import torch

class GradCAMPlusPlus:
    """
    Grad-CAM++ implementation tailored for spatial feature extractors.
    Produces high-resolution activation heatmaps showing manipulated regions.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to capture forward activations and backward gradients
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, spatial_img: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        self.model.eval()
        
        # Run dummy forward pass for spatial network
        features = self.model.spatial_net(spatial_img)
        self.model.zero_grad()
        
        score = features[0, :] if features.dim() > 1 else features
        score_val = score.sum()
        score_val.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback uniform heatmap if layer didn't expose gradients
            return np.ones((224, 224), dtype=np.float32)

        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]

        # Grad-CAM++ weight calculations
        g2 = gradients ** 2
        g3 = gradients ** 3
        alpha = g2 / (2 * g2 + np.sum(activations * g3, axis=(-2, -1), keepdims=True) + 1e-7)
        weights = np.sum(alpha * np.maximum(gradients, 0), axis=(-2, -1), keepdims=True)

        cam = np.sum(weights * activations, axis=0)
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-7)
        return cam