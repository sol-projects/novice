import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import cv2

def evaluate_model(model, X_test, y_test, categories):
    """
    Evaluates the model on the test dataset.

    Parameters:
    - model (Model): Trained Keras model.
    - X_test, y_test: Test data and labels.
    - categories (list): List of category names.
    """
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    print("Classification Report:")
    print(classification_report(y_test_classes, y_pred_classes, target_names=categories))

    conf_mat = confusion_matrix(y_test_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def predict_image(img_path, model, categories, output_size=(128, 128)):
    """
    Predicts the class of a single image.

    Parameters:
    - img_path (str): Path to the image.
    - model (Model): Trained Keras model.
    - categories (list): List of category names.
    - output_size (tuple): Target size for resizing the image.

    Returns:
    - str: Predicted category name.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, output_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return categories[np.argmax(prediction)]
