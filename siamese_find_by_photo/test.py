import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import json


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
            nn.AdaptiveAvgPool2d((1, 1))
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

def load_siamese_model(checkpoint_path="checkpoints/siamese_network.pth", device="cpu"):
    model = SiameseNetwork()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

class SiameseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.transform = transform
        
        for label, class_dir in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                self.image_paths.extend(
                    [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith(('.jpg', '.png'))]
                )
                self.labels.extend([label] * len(os.listdir(class_path)))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label, img_path

def compute_distances(model, query_loader, reference_loader, device="cpu", top_k=5):
    model.eval()
    results = []

    with torch.no_grad():
        for query_img, query_label, query_path in query_loader:
            query_img = query_img.to(device)
            query_emb = model.forward_once(query_img)

            distances = []
            for ref_img, ref_label, ref_path in reference_loader:
                ref_img = ref_img.to(device)
                ref_emb = model.forward_once(ref_img)
                
                distance = nn.functional.pairwise_distance(query_emb, ref_emb).item()
                distances.append({
                    "ref_path": ref_path[0],
                    "ref_label": ref_label.item(),
                    "distance": distance
                })

            distances.sort(key=lambda x: x["distance"])
            results.append({
                "query_path": query_path[0],
                "query_label": query_label.item(),
                "top_k_matches": distances[:top_k]
            })
    
    return results

def plot_average_distances(results, output_dir="test_results"):
    same_distances = []
    diff_distances = []

    for result in results:
        query_label = result["query_label"]
        for match in result["top_k_matches"]:
            if query_label == match["ref_label"]:
                same_distances.append(match["distance"])
            else:
                diff_distances.append(match["distance"])

    avg_same = sum(same_distances) / len(same_distances) if same_distances else 0
    avg_diff = sum(diff_distances) / len(diff_distances) if diff_distances else 0

    # Plot the bar chart
    plt.figure(figsize=(6, 6))
    plt.bar(["Same", "Different"], [avg_same, avg_diff], color=["green", "red"])
    plt.title("Average Distances")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "average_distances.png"))
    plt.close()

def save_results(results, output_dir="test_results"):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Siamese Network by comparing test_data to data.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/siamese_network.pth",
                        help="Path to the trained Siamese model (.pth) file.")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Path to the test dataset directory.")
    parser.add_argument("--ref_dir", type=str, required=True,
                        help="Path to the reference dataset directory.")
    parser.add_argument("--img_size", type=int, default=50, help="Image size (default=50).")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu' or 'cuda'.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top matches to display.")
    parser.add_argument("--output_dir", type=str, default="test_results", help="Directory to save results.")
    args = parser.parse_args()


    model = load_siamese_model(args.checkpoint, device=args.device)
    model.to(args.device)


    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])
    query_dataset = SiameseDataset(root_dir=args.test_dir, transform=transform)
    ref_dataset = SiameseDataset(root_dir=args.ref_dir, transform=transform)

    query_loader = DataLoader(query_dataset, batch_size=1, shuffle=False)
    ref_loader = DataLoader(ref_dataset, batch_size=1, shuffle=False)

    results = compute_distances(model, query_loader, ref_loader, device=args.device, top_k=args.top_k)

    save_results(results, output_dir=args.output_dir)
    plot_average_distances(results, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

