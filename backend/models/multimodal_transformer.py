import torch
import torch.nn as nn
from .video_detection import VideoFeatureExtractor, TemporalTransformer
from .audio_detection import AudioTransformer

class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention: Audio-Visual Alignment.
    L-Sync: Captures lip-sync and modal mismatches.
    """
    def __init__(self, v_dim=1792, a_dim=512, inner_dim=512):
        super().__init__()
        self.q_proj = nn.Linear(v_dim, inner_dim)
        self.k_proj = nn.Linear(a_dim, inner_dim)
        self.v_proj = nn.Linear(a_dim, inner_dim)
        
        self.atten = nn.MultiheadAttention(embed_dim=inner_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(inner_dim)

    def forward(self, v_feat, a_feat):
        # v_feat: (B, T_v, Dim_v)
        # a_feat: (B, T_a, Dim_a)
        
        q = self.q_proj(v_feat)
        k = self.k_proj(a_feat)
        v = self.v_proj(a_feat)
        
        # Video attends to Audio
        attn_out, _ = self.atten(q, k, v)
        out = self.norm(q + attn_out)
        
        return out

class MultimodalDetector(nn.Module):
    """
    The Full Transformer Model:
    Spectral-Video + Audio-Transformer + Cross-Modal Fusion.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        # 1. Branches
        self.video_branch = VideoFeatureExtractor().to(device)
        self.video_temporal = TemporalTransformer(input_dim=1280).to(device)
        
        self.audio_branch = AudioTransformer().to(device)
        
        # 2. Fusion
        self.cross_modal = CrossModalAttention(v_dim=1280, a_dim=512).to(device)
        
        # 3. Final Head (Ensemble Light for Uncertainty)
        # Two heads for "Deep Ensemble" effect
        self.classifier_a = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(device)
        
        self.classifier_b = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, frames, audio_mel):
        # frames: (B, T_v, C, H, W)
        # audio_mel: (B, MelBins, T_a)
        
        b, t, c, h, w = frames.shape
        
        # 1. Video Feature Extraction (Step across frames)
        frames_flat = frames.view(b * t, c, h, w)
        v_feats = self.video_branch(frames_flat) # (B*T, 1792)
        v_seq = v_feats.view(b, t, -1)
        
        # 2. Temporal Modeling
        v_temp = self.video_temporal(v_seq) # (B, T, 1792)
        
        # 3. Audio Modeling
        a_temp = self.audio_branch(audio_mel) # (B, T_a, 512)
        
        # 4. Cross-Modal Fusion (Video attends to Audio)
        fused = self.cross_modal(v_temp, a_temp) # (B, T, 512)
        
        # 5. Global Pooling
        pooled = fused.mean(dim=1)
        
        # 6. Ensemble Classifier (Uncertainty)
        out1 = self.classifier_a(pooled)
        out2 = self.classifier_b(pooled)
        
        # Final prediction is average, variance for uncertainty
        final_prob = (out1 + out2) / 2
        variance = torch.abs(out1 - out2)
        
        return final_prob, variance
