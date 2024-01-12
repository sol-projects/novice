import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms
import json
from PIL import Image, ImageDraw
import requests
from io import BytesIO
import torch.optim as optim
import numpy as np
from torch.nn.functional import pad
import torchvision.ops as ops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from torch.utils.data import random_split
from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, Resize, GaussNoise, Rotate, RandomScale, RandomGamma, RandomCrop
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms.functional as TF
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import distance_box_iou_loss
from torchvision.ops import nms
from torchvision.ops import box_iou
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils
from torchvision.models.detection import SSD300_VGG16_Weights

class Params:
    TRANSFORMED_IMAGE_SIZE = (300, 300)
    MAX_LABELS = 10
    LR = 0.00001
    EPOCHS = 80
    AUGMENTATIONS_PER_IMAGE = 1
    BATCH_SIZE = 32
    PAD_VALUE = -1e9

ARTICLE_LABELS = {
    "title": 1, # 0 je rezerviran za background
    "content": 2,
    "categories": 3,
    "authors": 4
}

def from_label(label):
    return ARTICLE_LABELS.get(label, None)

def to_label(numeric_label):
    reverse_mapping = {v: k for k, v in ARTICLE_LABELS.items()}
    return reverse_mapping.get(numeric_label, None)

class Articles(Dataset):
    def __init__(self, data, target_size=Params.TRANSFORMED_IMAGE_SIZE, train=True):
        self.train = train

        self.data = data
        self.data = self.filter_data(data)
        self.data = self.augment_data(Params.AUGMENTATIONS_PER_IMAGE)

    def filter_data(self, data):
        filtered_data = [item for item in data if 'label' in item]
        return filtered_data

    def augment_data(self, num_copies):
        augmented_data = []
        images_folder = 'data/images'

        transform = Compose([
            HorizontalFlip(p=0.3),
            VerticalFlip(p=0.3),
            RandomScale(scale_limit=0.3),
            GaussNoise(var_limit=(5, 70), p=0.3),
            Rotate(limit=105, p=0.3), #brez rotacij
            RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.4, p=0.5),
            Resize(width=Params.TRANSFORMED_IMAGE_SIZE[0], height=Params.TRANSFORMED_IMAGE_SIZE[1]),
            ToTensorV2()
        ], bbox_params={'format': 'albumentations', 'label_fields': ['labels']})

        for item in self.data:
            try:
                image_path = os.path.join(images_folder, os.path.basename(item["url"]))

                num_labels = len(item['label'])

                if num_labels > Params.MAX_LABELS:
                    print(f'Skipping item with more than {Params.MAX_LABELS} labels. Item ID: {item["id"]}, Image URL: {item["url"]}')
                    continue

                print(f'Opening image for: {item["url"]}', flush=True)
                image = self.download_image(image_path, item["url"]) if not os.path.isfile(image_path) else Image.open(image_path).convert("RGB")
                original_width = item['label'][0]['original_width']
                original_height = item['label'][0]['original_height']

                bboxes = []
                for label in item['label']:
                    x = original_width * (label['x'] / 100.0)
                    y = original_height * (label['y'] / 100.0)
                    width = original_width * (label['width'] / 100.0)
                    height = original_height * (label['height'] / 100.0)

                    x_max = x + width
                    y_max = y + height
                    bboxes.append([
                        x / original_width,
                        y / original_height,
                        x_max / original_width,
                        y_max / original_height
                    ])

                labels = [from_label(label['labels'][0]) for label in item['label']]

                for _ in range(num_copies if self.train else 1):
                    transformed = transform(image=np.array(image), bboxes=bboxes, labels=labels)
                    transformed_image = transformed['image']
                    transformed_bboxes = transformed['bboxes']
                    transformed_labels = transformed['labels']

                    augmented_data.append({
                        'image': transformed_image,
                        'bboxes': transformed_bboxes,
                        'labels': transformed_labels
                    })

                    print(f'Appending new augmented image for: {item["url"]}')
                    #if len(augmented_data) <= 10:
                    #    print(transformed_image.shape)
                    #    plt.imshow(transformed_image.permute(1, 2, 0))
                    #    plt.title(f'Augmentation {len(augmented_data)} for {item["url"]}')
                    #    plt.show()

            except KeyError as e:
                print(f'KeyError: {e}. Skipping item: {item["url"]}')

        return augmented_data

    def download_image(self, image_path, url):
        response = requests.get(f'http://localhost:8081{url}', headers={"Authorization": "Token 27b24db2231c1840e10d03125a72ae9ecde95565"}, stream=True)
        if response.status_code == 200:
            with open(image_path, 'wb') as f:
                f.write(response.content)
        return Image.open(image_path).convert("RGB")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = self.data[idx]["image"].float()
        #bboxes = np.array([[label[0], label[1], label[2], label[3]] for label in item['bboxes']])
        labels = np.array([label for label in item['labels']])

        original_bboxes = []
        transformed_bboxes = item['bboxes']
        for transformed_bbox in transformed_bboxes:
            x = transformed_bbox[0] * Params.TRANSFORMED_IMAGE_SIZE[0]
            y = transformed_bbox[1] * Params.TRANSFORMED_IMAGE_SIZE[0]
            x_max = transformed_bbox[2] * Params.TRANSFORMED_IMAGE_SIZE[0]
            y_max = transformed_bbox[3] * Params.TRANSFORMED_IMAGE_SIZE[0]

            original_bboxes.append([x, y, x_max, y_max])

        target = {}
        target["boxes"] = torch.tensor(original_bboxes).to(device)
        target["labels"] = torch.tensor(labels, dtype=torch.int64).to(device)

        return image, target

    def max_labels_on_image(self):
        return max(len(item["label"]) for item in self.data)

def ssd300():
    model = torchvision.models.detection.ssd300_vgg16(
        weights=SSD300_VGG16_Weights.COCO_V1
    )

    in_channels = _utils.retrieve_out_channels(model.backbone, (Params.TRANSFORMED_IMAGE_SIZE[0], Params.TRANSFORMED_IMAGE_SIZE[0]))
    num_anchors = model.anchor_generator.num_anchors_per_location()

    # freeze layers requires_grad_
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=len(ARTICLE_LABELS) + 1 # + 1 za background
    )

    model.transform.min_size = (Params.TRANSFORMED_IMAGE_SIZE[0],)
    model.transform.max_size = Params.TRANSFORMED_IMAGE_SIZE[0]
    return model

def collate(batch):
    images = torch.stack([item[0] for item in batch]).to(device)
    targets = [item[1] for item in batch]

    return images, targets

if __name__ == '__main__':
    torch.cuda.empty_cache()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ssd300()
    model = model.to(device)
    with open('data/data.json', 'r') as f:
        data = json.load(f)
        total_images = len(data)
        train_size = int(0.9 * total_images)
        print(train_size)
        val_size = total_images - train_size
        train_data, val_data = random_split(data, [train_size, val_size], generator=torch.Generator().manual_seed(1))

        train_dataset = Articles(train_data)
        val_dataset = Articles(val_data, train=False)

        train_dataloader = DataLoader(train_dataset, batch_size=Params.BATCH_SIZE, shuffle=True, collate_fn=collate)
        val_dataloader = DataLoader(val_dataset, batch_size=Params.BATCH_SIZE, shuffle=False, collate_fn=collate)

        criterion_label = torch.nn.CrossEntropyLoss()
        criterion_bbox = nn.SmoothL1Loss()
        #criterion_bbox = lambda boxes1, boxes2: distance_box_iou_loss(boxes1.double(), boxes2.double())
        optimizer = torch.optim.Adam(model.parameters(), lr=Params.LR)

        train_losses = []
        val_losses = []

        for epoch in range(Params.EPOCHS):
            running_loss = 0.0

            model.train()
            for i, data in enumerate(train_dataloader, 0):
                images, targets = data

                optimizer.zero_grad()

                loss_dict = model(images, targets)
                losses = loss_dict['bbox_regression'] + loss_dict['classification']
                running_loss += losses.item()

                losses.backward()
                optimizer.step()
            average_loss = running_loss / len(train_dataloader)
            train_losses.append(average_loss)

            print(f"[Epoch {epoch + 1}] (Training) Loss: {average_loss:.4f}")
            running_loss = 0.0

            loss_iou = 0.0
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(val_dataloader, 0):
                    images, targets = data

                    outputs = model(images, targets)
                    #https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html
                    #loss_iou += torchvision.ops.distance_box_iou_loss(outputs[0]['boxes'].item().cpu().detach().numpy(), images.float(), reduction='mean')

            average_loss = running_loss / len(val_dataloader)
            val_losses.append(average_loss)
            print(f"[Epoch {epoch + 1}] (Validation) Loss: {loss_iou}")

        model.eval()

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='(Train) Loss')
        plt.plot(val_losses, label='(Validation) Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Bounding Box Losses')
        plt.legend()

        plt.tight_layout()
        plt.show()


        transform = transforms.Compose([
            transforms.Resize(Params.TRANSFORMED_IMAGE_SIZE),
            transforms.ToTensor(),
        ])

        image_path = 'data/test/images/1.jpg'
        image = Image.open(image_path)
        image = image.resize(Params.TRANSFORMED_IMAGE_SIZE)
        image_tensor = transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            outputs = model(image_tensor)

        predicted_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
        predicted_scores = outputs[0]['scores'].detach().cpu().numpy()
        predicted_labels = [to_label(val) for val in outputs[0]['labels'].cpu().numpy()]
        print(predicted_bboxes)
        print(predicted_scores)
        print(predicted_labels)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(image)

        for bbox, label, score in zip(predicted_bboxes, predicted_labels, predicted_scores):
            if label is not None and score >= 0.3:

                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='r', facecolor='none')

                ax.add_patch(rect)

                ax.text(bbox[0], bbox[1], label, verticalalignment='top', color='white', fontsize=8, weight='bold')

        plt.show()
