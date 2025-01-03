import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import PIL
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
    def forward_once(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
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
        
        self.all_indices = list(range(len(self.imagefolder.samples)))
        
    def __len__(self):
        return len(self.imagefolder.samples)
    
    def __getitem__(self, idx):
        path1, label1 = self.imagefolder.samples[idx]
        
        same_class = random.random() > 0.5
        
        if same_class:
            same_class_indices = self.class_to_indices[label1]
            idx2 = random.choice(same_class_indices)
        else:
            diff_cls_indices = []
            for cls_idx, idx_list in self.class_to_indices.items():
                if cls_idx != label1:
                    diff_cls_indices.extend(idx_list)
            idx2 = random.choice(diff_cls_indices)
        
        path2, label2 = self.imagefolder.samples[idx2]
        
        img1 = PIL.Image.open(path1).convert('RGB')
        img2 = PIL.Image.open(path2).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        label = torch.FloatTensor([1.0]) if label1 == label2 else torch.FloatTensor([0.0])
        
        return img1, img2, label


def train_siamese_network(
    data_dir="data",
    batch_size=16,
    lr=1e-3,
    num_epochs=5,
    margin=2.0
):
# All images are saved at 50x50. If you have a faster PC, consider preparing images at a larger size, 
# as excessive compression and decompression dont make sense.

        transforms.Resize((50, 50)),
        transforms.ToTensor(),
    ])
    
    dataset = SiameseDataset(root_dir=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (img1, img2, label) in enumerate(dataloader):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            optimizer.zero_grad()
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/siamese_network.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")

if __name__ == "__main__":

    train_siamese_network(
        data_dir="data",
        batch_size=16,
        lr=1e-3,
        num_epochs=5,
        margin=2.0
    )

