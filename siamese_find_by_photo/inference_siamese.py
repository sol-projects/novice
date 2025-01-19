from flask import Flask, request, jsonify
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Flask setup
app = Flask(__name__)

# Path to save user-uploaded images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model
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

# Load the model
def load_siamese_model(checkpoint_path="checkpoints/siamese_network.pth"):
    model = SiameseNetwork()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model

# Helper functions
def get_embedding(model, image_path, transform):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = model.forward_once(img_tensor)
    return emb

def compute_distance(emb1, emb2):
    return nn.functional.pairwise_distance(emb1, emb2).item()

# Initialize the model and transformations
MODEL_PATH = "checkpoints/siamese_network.pth"
model = load_siamese_model(MODEL_PATH)
transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor()
])

@app.route('/compare', methods=['POST'])
def compare_images():
    if 'file' not in request.files:
        return jsonify({"error": "Query image not provided"}), 400

    query_file = request.files['file']
    query_path = os.path.join(UPLOAD_FOLDER, "query.jpg")
    query_file.save(query_path)
    app.logger.info(f"Query image saved at {query_path}")

    # Save news images
    news_files = request.files.getlist("news_images")
    news_image_paths = []
    for news_file in news_files:
        news_path = os.path.join(UPLOAD_FOLDER, news_file.filename)
        news_file.save(news_path)
        news_image_paths.append(news_path)
        app.logger.info(f"Saved news image: {news_path}")

    if not news_image_paths:
        app.logger.warning("No news images provided for comparison")
        return jsonify([])

    # Compare query image with each news image
    query_emb = get_embedding(model, query_path, transform)
    results = []
    for news_path in news_image_paths:
        news_emb = get_embedding(model, news_path, transform)
        distance = compute_distance(query_emb, news_emb)
        results.append({"filename": os.path.basename(news_path), "distance": distance})

    # Sort results by similarity (distance)
    results = sorted(results, key=lambda x: x["distance"])
    app.logger.info(f"Comparison results: {results}")

    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000)

