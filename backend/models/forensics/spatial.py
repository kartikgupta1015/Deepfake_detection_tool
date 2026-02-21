import torch
import torch.nn as nn
from torchvision import models, transforms

class SpatialBranch(nn.Module):
    """
    Forensic Branch: Spatial Analysis using Swin-Base Transformer.
    Advanced training backbone for deep visual artifacts.
    """
    def __init__(self):
        super().__init__()
        # Use EfficientNet-B0 for extreme speed/low memory (only ~20MB)
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        # EfficientNet-B0 feature dim is 1280
        self.classifier = nn.Sequential(
            nn.Linear(1280, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        # x: (Batch, 3, 224, 224)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        if return_features:
            return x
        return self.classifier(x)

def get_spatial_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
