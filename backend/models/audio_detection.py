import torch
import torch.nn as nn
import librosa
import numpy as np

class AudioTransformer(nn.Module):
    """
    Research-Grade Audio Deepfake Detector.
    Log-Mel Spectrogram -> 1D CNN -> Transformer Encoder.
    """
    def __init__(self, input_channels=1, hidden_dim=512, nhead=8, num_layers=4):
        super().__init__()
        
        # 1. 1D CNN for local spectral feature extraction
        # Input shape: (Batch, Channels=1, TimeSteps, MelBins=128)
        # We treat MelBins as the local feature dimension for 1D CNN along time
        self.cnn = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 2. Transformer Encoder for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, mel_spec):
        # mel_spec: (Batch, MelBins=128, Time)
        # 1. Local extraction
        x = self.cnn(mel_spec) # (B, hidden_dim, Time)
        
        # 2. Reshape for Transformer
        x = rearrange_audio(x) # (Batch, Time, hidden_dim)
        
        # 3. Temporal Modeling
        x = self.transformer(x)
        
        return x

def rearrange_audio(x):
    # (B, D, T) -> (B, T, D)
    return x.permute(0, 2, 1)

def extract_log_mel(audio_path, n_mels=128, n_fft=2048, hop_length=512):
    """
    Extract Log-Mel Spectrogram from audio file.
    """
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        # Trim silence
        y, _ = librosa.effects.trim(y)
        
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_log = librosa.power_to_db(S, ref=np.max)
        
        # Normalize
        S_norm = (S_log - np.min(S_log)) / (np.max(S_log) - np.min(S_log) + 1e-9)
        return torch.from_numpy(S_norm).float() # (MelBins, Time)
    except Exception as e:
        print(f"[DeepShield] Audio extraction error: {e}")
        return torch.zeros((n_mels, 100))
