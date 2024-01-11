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
    TRANSFORMED_IMAGE_SIZE = (256, 256)
    MAX_LABELS = 10
    LR = 0.0000001
    EPOCHS = 200
    BATCH_SIZE = 32
    PAD_VALUE = -1e9

ARTICLE_LABELS = {
    "title": 0,
    "content": 1,
    "categories": 2,
    "authors": 3
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
        self.data = self.augment_data(10)

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
            Rotate(limit=105, p=0.3),
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
        image = self.data[idx]["image"]
        bboxes = np.array([[label[0], label[1], label[2], label[3]] for label in item['bboxes']])
        labels = np.array([label for label in item['labels']])

        if labels.shape[0] < Params.MAX_LABELS:
            labels = np.pad(labels, (0, Params.MAX_LABELS - labels.shape[0]), mode='constant', constant_values=Params.PAD_VALUE)

        if bboxes.shape[0] < Params.MAX_LABELS:
            bboxes = np.pad(bboxes, ((0, Params.MAX_LABELS - bboxes.shape[0]), (0, 0)), mode='constant', constant_values=Params.PAD_VALUE)

        sample = {
            "image": image.float().to(device),
            "bboxes": torch.from_numpy(bboxes).to(device),
            "labels": torch.from_numpy(labels).to(device),
        }

        return sample

    def max_labels_on_image(self):
        return max(len(item["label"]) for item in self.data)

class ArticlesNN(nn.Module):
    def __init__(self):
        super(ArticlesNN, self).__init__()
        self.conv_part = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ELU(),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ELU(),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ELU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(107648, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, Params.MAX_LABELS * len(ARTICLE_LABELS)),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(107648, 2**11),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2**11, 2**10),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2**10, Params.MAX_LABELS * 4),
        )

    def forward(self, x):
        x = self.conv_part(x)
        batch_size = x.size(0)
        outputs_bbox = self.regressor(x).view(batch_size, Params.MAX_LABELS, 4)
        outputs_label_logits = self.classifier(x).view(batch_size, Params.MAX_LABELS, len(ARTICLE_LABELS))
        outputs_label = F.softmax(outputs_label_logits, dim=-1)
        return outputs_bbox, outputs_label

def ssd300():
    model = torchvision.models.detection.ssd300_vgg16(
        weights=SSD300_VGG16_Weights.COCO_V1
    )

    in_channels = _utils.retrieve_out_channels(model.backbone, (size, size))
    num_anchors = model.anchor_generator.num_anchors_per_location()

    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=Params.MAX_LABELS,
    )

    model.transform.min_size = (TRANSFORMED_IMAGE_SIZE[0],)
    model.transform.max_size = TRANSFORMED_IMAGE_SIZE[0]
    return model

def collate(batch):
    images = torch.stack([item['image'] for item in batch]).to(device)
    labels = torch.stack([item['labels'] for item in batch]).to(device)
    bboxes = torch.stack([item['bboxes'] for item in batch]).to(device)

    return images, labels, bboxes

if __name__ == '__main__':
    torch.cuda.empty_cache()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ArticlesNN()
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

        #class_sample_counts = [2, 12, 0.5, 1]
        #weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
        #weights = weights / weights.sum()
        #weights = weights.to(device)

        criterion_label = torch.nn.CrossEntropyLoss()
        #criterion_label = torch.nn.CrossEntropyLoss(weight=weights)
        criterion_bbox = nn.SmoothL1Loss()
        #criterion_bbox = lambda boxes1, boxes2: distance_box_iou_loss(boxes1.double(), boxes2.double())
        optimizer = torch.optim.Adam(model.parameters(), lr=Params.LR)#, weight_decay=1e-4)
        #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

        bbox_train_losses = []
        label_train_losses = []
        bbox_val_losses = []
        label_val_losses = []

        bbox_train_accuracy = []
        label_train_accuracy = []
        bbox_val_accuracy = []
        label_val_accuracy = []

        train_losses = []
        val_losses = []

        for epoch in range(Params.EPOCHS):
            total_labels = 0.0
            correct_labels = 0.0
            total_bboxes = 0.0
            correct_bboxes = 0.0

            running_loss = 0.0
            running_loss_labels = 0.0
            running_loss_bboxes = 0.0

            model.train()
            for i, data in enumerate(train_dataloader, 0):
                inputs, labels, bboxes = data
                optimizer.zero_grad()

                outputs_bboxes, outputs_labels = model(inputs)

                label_mask = (labels != Params.PAD_VALUE)
                outputs_labels_masked = outputs_labels[label_mask]
                labels_masked = labels[label_mask]
                loss_class = criterion_label(outputs_labels_masked, labels_masked.long())

                bbox_mask = (bboxes != Params.PAD_VALUE).all(dim=-1)
                outputs_bboxes_masked = outputs_bboxes[bbox_mask]
                bboxes_masked = bboxes[bbox_mask]
                loss_bbox = criterion_bbox(outputs_bboxes_masked, bboxes_masked)
                #loss_iou = torchvision.ops.distance_box_iou_loss(outputs_bboxes_masked.float(), bboxes_masked.float(), reduction='none')

                _, predicted_labels = torch.max(outputs_labels_masked, 1)
                correct_labels += (predicted_labels == labels_masked).sum().item()
                total_labels += labels_masked.numel()

                loss = loss_class + loss_bbox# + loss_iou.mean()
                loss.backward()
                optimizer.step()

                iou = box_iou(outputs_bboxes_masked, bboxes_masked)
                iou_threshold = 0.5
                correct_bboxes += (iou > iou_threshold).sum().item()
                #predicted_bboxes = torch.sigmoid(outputs_bboxes_masked) > 0.5
                #correct_bboxes += (predicted_bboxes == bboxes_masked.byte()).sum().item()
                total_bboxes += bboxes_masked.numel()

                running_loss_labels += loss_class.item()
                running_loss_bboxes += loss_bbox.item()
                running_loss += loss.item()
            average_loss = running_loss / len(train_dataloader)
            train_losses.append(average_loss)
            average_loss_labels = running_loss_labels / len(train_dataloader)
            average_loss_bboxes = running_loss_bboxes / len(train_dataloader)
            bbox_train_losses.append(average_loss_bboxes)
            label_train_losses.append(average_loss_labels)
            bbox_train_accuracy.append(correct_bboxes / total_bboxes)
            label_train_accuracy.append(correct_labels / total_labels)

            print(f"[Epoch {epoch + 1}] (Training) Loss: {average_loss:.4f}")
            print(f"[Epoch {epoch + 1}] (Training) Loss: {average_loss_labels:.4f} (Labels)")
            print(f"[Epoch {epoch + 1}] (Training) Loss: {average_loss_bboxes:.4f} (Bboxes)")
            print(f"[Epoch {epoch + 1}] (Training) Accuracy: {correct_bboxes / total_bboxes} (Bboxes)")
            print(f"[Epoch {epoch + 1}] (Training) Accuracy: {correct_labels / total_labels} (Labels)")
            total_labels = 0.0
            correct_labels = 0.0
            correct_bboxes = 0.0
            total_bboxes = 0.0
            #scheduler.step(average_loss)
            running_loss = 0.0
            running_loss_bboxes = 0.0
            running_loss_labels = 0.0

            model.eval()
            with torch.no_grad():
                for i, data in enumerate(val_dataloader, 0):
                    inputs, labels, bboxes = data
                    labels = labels.long()

                    outputs_bboxes, outputs_labels = model(inputs)
                    label_mask = (labels != Params.PAD_VALUE)
                    outputs_labels_masked = outputs_labels[label_mask]
                    labels_masked = labels[label_mask]
                    loss_class = criterion_label(outputs_labels_masked, labels_masked)

                    bbox_mask = (bboxes != Params.PAD_VALUE).all(dim=-1)
                    outputs_bboxes_masked = outputs_bboxes[bbox_mask]
                    bboxes_masked = bboxes[bbox_mask]
                    loss_bbox = criterion_bbox(outputs_bboxes_masked, bboxes_masked)
                    #loss_iou = torchvision.ops.distance_box_iou_loss(outputs_bboxes_masked.float(), bboxes_masked.float(), reduction='none')

                    _, predicted_labels = torch.max(outputs_labels_masked, 1)
                    correct_labels += (predicted_labels == labels_masked).sum().item()
                    total_labels += labels_masked.numel()

                    loss = loss_class + loss_bbox# + loss_iou.mean()

                    iou = box_iou(outputs_bboxes_masked, bboxes_masked)
                    iou_threshold = 0.5
                    correct_bboxes += (iou > iou_threshold).sum().item()

                    #predicted_bboxes = torch.sigmoid(outputs_bboxes_masked) > 0.5 #to je verjetno narobe
                    #correct_bboxes += (predicted_bboxes == bboxes_masked.byte()).sum().item()
                    total_bboxes += bboxes_masked.numel()

                    running_loss_labels += loss_class.item()
                    running_loss_bboxes += loss_bbox.item()
                    running_loss += loss.item()
            average_loss = running_loss / len(val_dataloader)
            val_losses.append(average_loss)
            average_loss_labels = running_loss_labels / len(val_dataloader)
            average_loss_bboxes = running_loss_bboxes / len(val_dataloader)
            bbox_val_losses.append(average_loss_bboxes)
            label_val_losses.append(average_loss_labels)
            bbox_val_accuracy.append(correct_bboxes / total_bboxes)
            label_val_accuracy.append(correct_labels / total_labels)
            print(f"[Epoch {epoch + 1}] (Validation) Loss: {average_loss:.4f}")
            print(f"[Epoch {epoch + 1}] (Validation) Loss: {average_loss_labels:.4f} (Labels)")
            print(f"[Epoch {epoch + 1}] (Validation) Loss: {average_loss_bboxes:.4f} (Bboxes)")
            print(f"[Epoch {epoch + 1}] (Training) Accuracy: {correct_bboxes / total_bboxes} (Bboxes)")
            print(f"[Epoch {epoch + 1}] (Training) Accuracy: {correct_labels / total_labels} (Labels)")
            total_labels = 0.0
            correct_labels = 0.0
            correct_bboxes = 0.0
            total_bboxes = 0.0
            running_loss = 0.0
            running_loss_bboxes = 0.0
            running_loss_labels = 0.0

        model.eval()

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(bbox_train_losses, label='Bounding Box (Train) Loss')
        plt.plot(bbox_val_losses, label='Bounding Box (Validation) Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Bounding Box Losses')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(label_train_losses, label='Label (Train) Loss')
        plt.plot(label_val_losses, label='Label (Validation) Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Label Losses')
        plt.legend()

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(bbox_train_accuracy, label='Bounding Box (Train) Accuracy')
        plt.plot(bbox_val_accuracy, label='Bounding Box (Validation) Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Bounding Box Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(label_train_accuracy, label='Label (Train) Accuracy')
        plt.plot(label_val_accuracy, label='Label (Validation) Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Label Accuracy')
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

        predicted_bboxes = outputs[0].detach().cpu()
        predicted_labels_logits = outputs[1].detach().cpu()
        predicted_labels = F.softmax(predicted_labels_logits, dim=-1)
        print(predicted_labels)
        print(predicted_labels[0])
        predicted_labels_str = [[to_label(val) for val in label.argmax(dim=-1)] for label in predicted_labels]
        print(predicted_labels_str)
        print(predicted_bboxes[0])

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(image)

        for bbox, label in zip(predicted_bboxes[0], predicted_labels_str[0]):
            bbox[0] *= Params.TRANSFORMED_IMAGE_SIZE[0]
            bbox[1] *= Params.TRANSFORMED_IMAGE_SIZE[1]
            bbox[2] *= Params.TRANSFORMED_IMAGE_SIZE[0]
            bbox[3] *= Params.TRANSFORMED_IMAGE_SIZE[1]

            #bbox[0] = (bbox[0]/100.0) * Params.TRANSFORMED_IMAGE_SIZE[0]
            #bbox[1] = (bbox[1]/100.0) * Params.TRANSFORMED_IMAGE_SIZE[1]
            #bbox[2] = (bbox[2]/100.0) * Params.TRANSFORMED_IMAGE_SIZE[0]
            #bbox[3] = (bbox[3]/100.0) * Params.TRANSFORMED_IMAGE_SIZE[1]

            print(bbox)

            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='r', facecolor='none')

            ax.add_patch(rect)

            ax.text(bbox[0], bbox[1], label, verticalalignment='top', color='white', fontsize=8, weight='bold')

        plt.show()
