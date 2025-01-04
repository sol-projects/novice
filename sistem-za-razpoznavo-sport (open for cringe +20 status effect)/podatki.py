import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

def load_images(data_dir, categories, output_size=(128, 128)):
    """
    Dynamically loads images based on categories.

    Parameters:
    - data_dir (str): Path to the directory containing category folders.
    - categories (list): List of category folder names.
    - output_size (tuple): Target size for resizing images.

    Returns:
    - data (numpy array): Array of image data.
    - labels (numpy array): Array of corresponding labels.
    """
    data = []
    labels = []
    for category in categories:
        path = os.path.join(data_dir, category)
        if not os.path.exists(path):
            print(f"Category folder '{category}' does not exist. Skipping.")
            continue
        for img_file in os.listdir(path):
            try:
                img_path = os.path.join(path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, output_size)
                data.append(img)
                labels.append(category)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
    label_mapping = {label: idx for idx, label in enumerate(categories)}
    labels = np.array([label_mapping[label] for label in labels])
    return np.array(data), labels

def preprocess_data(data_dir, categories):
    """
    Splits training data into train/test sets.

    Parameters:
    - data_dir (str): Path to the training dataset directory.
    - categories (list): List of category folder names.

    Returns:
    - X_train, X_val, y_train, y_val: Preprocessed training and validation data.
    """
    images, labels = load_images(data_dir, categories)
    images = images / 255.0  # Normalize images
    labels = to_categorical(labels, num_classes=len(categories))
    from sklearn.model_selection import train_test_split
    return train_test_split(images, labels, test_size=0.2, random_state=42)

def preprocess_test_data(test_dir, categories):
    """
    Preprocesses the test dataset.

    Parameters:
    - test_dir (str): Path to the test dataset directory.
    - categories (list): List of category folder names.

    Returns:
    - X_test, y_test: Preprocessed test data and labels.
    """
    images, labels = load_images(test_dir, categories)
    images = images / 255.0  # Normalize images
    labels = to_categorical(labels, num_classes=len(categories))
    return images, labels
