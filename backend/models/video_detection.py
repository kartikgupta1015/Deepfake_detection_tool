import torch
import torch.nn as nn
import torch.fft
import timm
from einops import rearrange

class SpectralAttention(nn.Module):
    """
    Spectral Attention Module (SAM).
    Converts features to frequency domain to highlight GAN/Diffusion artifacts.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # Learnable frequency weights
        self.freq_weights = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.conv_atten = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        
        # 1. Spatial FFT
        x_fft = torch.fft.rfft2(x, norm="ortho") # (B, C, H, W//2 + 1)
        
        # 2. Apply Frequency Weights
        # (Magnitude only for attention)
        mag = torch.abs(x_fft)
        
        # 3. Frequency-to-Spatial Projection for Attention
        # We simplify by taking the mean of frequencies or inverse FFT
        # But per user req: "Pass through 1x1 conv attention"
        # Let's map magnitude back to spatial dimensions to get attention mask
        mag_spatial = torch.fft.irfft2(x_fft * self.freq_weights, s=(h, w), norm="ortho")
        
        # 4. Concatenate Spatial + Spectral info
        combined = torch.cat([x, mag_spatial], dim=1)
        gate = self.conv_atten(combined)
        
        return x * gate

class VideoFeatureExtractor(nn.Module):
    """
    EfficientNet-B4 with Spectral Attention.
    """
    def __init__(self):
        super().__init__()
        # Load pre-trained EfficientNet-B4
        self.backbone = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0, global_pool='')
        
        # Add Spectral Attention at a deep layer (level 3 or 4)
        # For EfficientNet-B4, level 3 features have around 56x56 or 28x28 resol
        self.spectral_atten = SpectralAttention(160) # 160 is a common mid-layer dim for B4

        # Register final spectral attention (features from B4 are 1792)
        self.final_spectral_atten = SpectralAttention(1792)

    def forward(self, x):
        # x: (B, C, H, W)
        # EfficientNet-B4 extraction with spectral modulation
        features = self.backbone.forward_features(x) # (B, 1792, 7, 7) - final features
        
        # Apply spectral attention to final feature map
        features = self.final_spectral_atten(features)
        
        # Global Average Pool
        pooled = torch.mean(features, dim=[2, 3])
        return pooled

class TemporalTransformer(nn.Module):
    """
    4-Layer Transformer Encoder for temporal consistency analysis.
    """
    def __init__(self, input_dim=1792, nhead=8, num_layers=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x: (B, Frames, Dim)
        return self.transformer(x)
