import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt


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


def load_siamese_model(checkpoint_path="checkpoints/siamese_network.pth"):
    model = SiameseNetwork(embedding_dim=128)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
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
    parser.add_argument("--img_size", type=int, default=224,
                        help="Resize images to this size (default=224).")
    
    args = parser.parse_args()
    
    print("Loading model...")
    model = load_siamese_model(args.checkpoint)
    
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet stats
    ])
    
    print("Computing query embedding...")
    query_emb = get_embedding(model, args.query, transform)
    
    image_files = [
        f for f in os.listdir(args.db)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    if not image_files:
        print("No valid images found in the database.")
        return
    
    print("Comparing against database images...")
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

