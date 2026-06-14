import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import Tuple
from .classifier import CIFAR10Classifier

class GradCAM:
    def __init__(self, classifier: CIFAR10Classifier):
        self.classifier = classifier

    def generate_heatmap(self, image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        if self.classifier.gradients is None or self.classifier.features is None:
            return np.zeros(target_size, dtype=np.float32)

        gradients = self.classifier.gradients
        features = self.classifier.features
        
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * features, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, target_size)
        
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        
        return cam

    def apply_heatmap(self, image: Image.Image, heatmap: np.ndarray, 
                      threshold: float = 0.5, alpha: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        original_size = (img_array.shape[1], img_array.shape[0])
        
        heatmap_resized = cv2.resize(heatmap, original_size)
        
        heatmap_normalized = heatmap_resized.copy()
        heatmap_normalized[heatmap_normalized < threshold] = 0
        
        mask = heatmap_normalized > threshold
        
        if not mask.any():
            return img_array.copy(), heatmap_resized
        
        heatmap_uint8 = (heatmap_normalized * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        mask_3ch = np.stack([mask] * 3, axis=-1)
        
        overlay = img_array.copy()
        result = cv2.addWeighted(
            img_array[mask_3ch].reshape(-1, 3),
            1 - alpha,
            heatmap_color[mask_3ch].reshape(-1, 3),
            alpha,
            0
        )
        if result is not None:
            overlay[mask_3ch] = result.flatten()
        
        return overlay, heatmap_resized

    def generate_overlay(self, image: Image.Image, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        target_size = (224, 224)
        heatmap = self.generate_heatmap(image, target_size)
        overlay, heatmap_full = self.apply_heatmap(image, heatmap, threshold)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        original = np.array(image)
        
        return original, overlay, heatmap_full
