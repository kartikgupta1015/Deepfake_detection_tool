import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import PIL.Image as PILImage
from tqdm import tqdm

from models.forensics.spatial import SpatialBranch, get_spatial_transform
from models.forensics.frequency import FrequencyBranch, extract_fft_magnitude
from models.forensics.noise import NoiseBranch

class FocalLoss(nn.Module):
    """Focal Loss to handle hard-to-detect AI fakes (class imbalance)."""
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        f_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        return f_loss.mean()

class ForensicDataset(Dataset):
    """
    Standardizes 'Real' vs 'AI' data for training.
    Assumes directory structure:
    data/real/*.jpg
    data/ai/*.jpg
    """
    def __init__(self, root_dir, transform=None, branch="spatial"):
        self.root_dir = root_dir
        self.transform = transform
        self.branch = branch
        self.samples = []
        
        # Load Real samples
        real_dir = os.path.join(root_dir, "real")
        if os.path.exists(real_dir):
            for f in os.listdir(real_dir):
                self.samples.append((os.path.join(real_dir, f), 0.0))
        
        # Load AI samples
        ai_dir = os.path.join(root_dir, "ai")
        if os.path.exists(ai_dir):
            for f in os.listdir(ai_dir):
                self.samples.append((os.path.join(ai_dir, f), 1.0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        if self.branch == "spatial":
            img = PILImage.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor([label], dtype=torch.float32)
        
        elif self.branch == "frequency":
            mag, _ = extract_fft_magnitude(path)
            return mag.squeeze(0), torch.tensor([label], dtype=torch.float32)
            
        elif self.branch == "noise":
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))
            tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
            return tensor, torch.tensor([label], dtype=torch.float32)

def train_branch(branch_name, model, dataloader, epochs=5, lr=1e-4, device='cpu'):
    """Standard training loop for a forensic branch."""
    print(f"--- Training {branch_name.upper()} Branch ---")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = FocalLoss()
    
    for epoch in range(epochs):
        loop = tqdm(dataloader, leave=True)
        epoch_loss = 0
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())
        
        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(dataloader):.4f}")
    
    # Save Weights
    os.makedirs("weights", exist_ok=True)
    save_path = f"weights/{branch_name}_weights.pt"
    torch.save(model.state_dict(), save_path)
    print(f"✅ {branch_name} weights saved to {save_path}\n")

if __name__ == "__main__":
    # This script is meant to be run manually after data is collected.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Define Augmentations (JPEG Compression, Blur, Noise)
    forensic_augs = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2)], p=0.3),
        transforms.RandomHorizontalFlip(),
        # Simulating web-compression
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Placeholder for data path
    data_path = "forensic_data" 
    if os.path.exists(data_path):
        # Example: Train Spatial Branch (Swin-Base)
        spatial_model = SpatialBranch().to(device)
        ds = ForensicDataset(data_path, transform=forensic_augs, branch="spatial")
        dl = DataLoader(ds, batch_size=16, shuffle=True)
        train_branch("spatial", spatial_model, dl, device=device)
    else:
        print(f"⚠️ Data directory '{data_path}' not found. Please collect AI vs Real images.")
