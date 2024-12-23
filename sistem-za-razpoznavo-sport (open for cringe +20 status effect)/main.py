# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
class NEWS(Enum):
    motor = 0
    nogomet= 1
    smucanje = 2
    veslanje = 3

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    resize_image = 64
    testing_transformation = transforms.Compose([
        transforms.Resize((resize_image, resize_image)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    image_path = 'input2.jpg'
    output_file="outputh.txt"
    image = Image.open(image_path).convert("RGB")
    transformed_image = testing_transformation(image)

    shufle = True
    batch_size = 1
    model18 = models.resnet18(pretrained=True)
    num_features = model18.fc.in_features
    model18.fc = nn.Linear(num_features, 4)
    model18.load_state_dict(torch.load("resnet18_zvrst_3_.pth"))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model18.parameters(), lr=0.001, momentum=0.9)
    model18.eval()
    final_loss = 0
    final_correct = 0
    max2 = 1
    print('EVALUATION RESULTS: ')
    with torch.no_grad():
       outputs_test = model18(transformed_image)
       notimp_test, preds_test = torch.max(outputs_test, max2)
       print("Predicted value:", preds_test.item())
       with open(output_file, 'w') as file:
           file.write(NEWS(preds_test.item()).name)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
