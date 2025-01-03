import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

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


def load_siamese_model(checkpoint_path="checkpoints/siamese_network.pth"):
    model = SiameseNetwork()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def get_embedding(model, image_path, transform):

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  
    
    with torch.no_grad():
        emb = model.forward_once(img_tensor)
    return emb

def compute_distance(emb1, emb2):

    return nn.functional.pairwise_distance(emb1, emb2).item()

def display_results(query_image_path, results, db_path):

    query_img = Image.open(query_image_path).convert('RGB')

    num_matches = len(results)
    fig, axes = plt.subplots(1, num_matches + 1, figsize=(15, 5))
    axes[0].imshow(query_img)
    axes[0].set_title("Query Image")
    axes[0].axis('off')
    
    for i, (filename, dist) in enumerate(results):
        match_img = Image.open(os.path.join(db_path, filename)).convert('RGB')
        axes[i + 1].imshow(match_img)
        axes[i + 1].set_title(f"Match {i+1}\nDist: {dist:.2f}")
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Find top 5 similar images using a Siamese Network.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/siamese_network.pth",
                        help="Path to the trained Siamese model (.pth) file.")
    parser.add_argument("--query", type=str, required=True,
                        help="Path to the query image.")
    parser.add_argument("--db", type=str, required=True,
                        help="Path to the database folder with images.")
    parser.add_argument("--img_size", type=int, default=50,
                        help="Resize images to this size (default=50).")
    
    args = parser.parse_args()
    
    model = load_siamese_model(args.checkpoint)
    
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])
    
    query_emb = get_embedding(model, args.query, transform)
    
    image_files = [
        f for f in os.listdir(args.db)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    if not image_files:
        print("No valid images found.")
        return
    

    results = []
    for f in image_files:
        db_path = os.path.join(args.db, f)
        db_emb = get_embedding(model, db_path, transform)
        
        dist = compute_distance(query_emb, db_emb)
        results.append((f, dist))
    
    results.sort(key=lambda x: x[1])
    
    top_results = results[:5]
    print("Top 5 similar images:")
    for filename, dist in top_results:
        print(f"{filename}: distance={dist:.4f}")
    
    display_results(args.query, top_results, args.db)

if __name__ == "__main__":
    main()

