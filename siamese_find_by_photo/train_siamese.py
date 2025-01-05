import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        
        resnet = models.resnet18(pretrained=True)
        resnet.fc = nn.Identity()
        self.features = resnet

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
        
    def forward_once(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x
    
    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(
                torch.clamp(self.margin - euclidean_distance, min=0.0), 2
            )
        )
        return loss

class SiameseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.imagefolder = datasets.ImageFolder(root=self.root_dir, transform=None)
        
        self.class_to_indices = {}
        for i, (path, cls_idx) in enumerate(self.imagefolder.samples):
            if cls_idx not in self.class_to_indices:
                self.class_to_indices[cls_idx] = []
            self.class_to_indices[cls_idx].append(i)
        
    def __len__(self):
        return len(self.imagefolder.samples)
    
    def __getitem__(self, idx):
        path1, label1 = self.imagefolder.samples[idx]
        same_class = random.random() > 0.5
        if same_class:
            idx2 = random.choice(self.class_to_indices[label1])
        else:
            diff_cls_indices = [
                i for cls_idx, indices in self.class_to_indices.items() if cls_idx != label1 for i in indices
            ]
            idx2 = random.choice(diff_cls_indices)
        
        path2, label2 = self.imagefolder.samples[idx2]
        img1 = Image.open(path1).convert('RGB')
        img2 = Image.open(path2).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        label = torch.FloatTensor([1.0]) if label1 == label2 else torch.FloatTensor([0.0])
        return img1, img2, label


def train_siamese_network(
    data_dir="data",
    batch_size=64,
    lr=3e-4,
    num_epochs=30,
    embedding_dim=128,
    margin=1.0
):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = SiameseDataset(root_dir=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SiameseNetwork(embedding_dim=embedding_dim).to(device)
    criterion = ContrastiveLoss(margin=margin)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (img1, img2, label) in enumerate(dataloader):
            img1, img2, label = img1.to(device, non_blocking=True), img2.to(device, non_blocking=True), label.to(device)
            
            optimizer.zero_grad()
            with autocast():
                out1, out2 = model(img1, img2)
                loss = criterion(out1, out2, label)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
        if epoch_loss <= 0.26:
            print(f"Stopping early as loss reached {epoch_loss:.4f}")
            break
        
        scheduler.step()
    
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/siamese_network.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")

if __name__ == "__main__":
    train_siamese_network(
        data_dir="data",
        batch_size=64,
        lr=3e-4,
        num_epochs=30,
        embedding_dim=128,
        margin=1.0
    )

