import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import Tuple, List

CIFAR10_CLASSES = [
    '飞机', '汽车', '鸟', '猫', '鹿',
    '狗', '青蛙', '马', '船', '卡车'
]

class CIFAR10Classifier:
    def __init__(self, pretrained: bool = True):
        self.device = torch.device('cpu')
        self.model = self._build_model(pretrained)
        self.model.eval()
        self.transform = self._build_transform()
        self.features = None
        self.gradients = None

    def _build_model(self, pretrained: bool) -> nn.Module:
        model = models.mobilenet_v2(pretrained=pretrained)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 10)
        
        if pretrained:
            try:
                state_dict = torch.hub.load_state_dict_from_url(
                    'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar10_mobilenetv2_x1_0-61a44353.pt',
                    map_location=self.device,
                    progress=False
                )
                model.load_state_dict(state_dict)
            except Exception:
                pass
        
        for param in model.parameters():
            param.requires_grad = True
            
        target_layer = model.features[-1]
        target_layer.register_forward_hook(self._save_features)
        target_layer.register_full_backward_hook(self._save_gradients)
        
        return model

    def _build_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])

    def _save_features(self, module, input, output):
        self.features = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return self.transform(image).unsqueeze(0)

    def predict(self, image: Image.Image) -> Tuple[List[str], List[float], torch.Tensor]:
        self.model.zero_grad()
        
        input_tensor = self.preprocess_image(image)
        input_tensor.requires_grad = True
        
        with torch.set_grad_enabled(True):
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            top_probs, top_indices = torch.topk(probabilities, 3)
            top_class_names = [CIFAR10_CLASSES[idx] for idx in top_indices[0].tolist()]
            top_prob_values = [round(p * 100, 2) for p in top_probs[0].tolist()]
            
            predicted_idx = top_indices[0][0]
            target = torch.zeros_like(outputs)
            target[0, predicted_idx] = 1
            outputs.backward(gradient=target)
        
        return top_class_names, top_prob_values, input_tensor
