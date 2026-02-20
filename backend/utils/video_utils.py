import cv2
import os
import subprocess
import torch
import numpy as np
import PIL.Image as PILImage
from torchvision import transforms

class VideoProcessor:
    """
    Utilities for Video Deepfake Preprocessing.
    Extracts frames and audio for multimodal analysis.
    """
    def __init__(self, frame_count=16, size=(224, 224)):
        self.frame_count = frame_count
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_frames(self, video_path):
        """Extract 'frame_count' equidistant frames from video."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: return torch.zeros((self.frame_count, 3, self.size[0], self.size[1]))
        
        indices = np.linspace(0, total_frames - 1, self.frame_count, dtype=int)
        frames = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: 
                # Use a zero frame if read fails
                frames.append(torch.zeros((3, self.size[0], self.size[1])))
                continue
            
            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb)
            tensor = self.transform(pil_img)
            frames.append(tensor)
            
        cap.release()
        return torch.stack(frames) # (T, C, H, W)

    def extract_audio(self, video_path, output_audio_path):
        """Extract audio stream from video using FFmpeg."""
        try:
            # -y overwrites, -i input, -vn no video, -acodec pcm_s16le wav
            cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_audio_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except Exception as e:
            print(f"[DeepShield] Audio extraction failed: {e}")
            return False
