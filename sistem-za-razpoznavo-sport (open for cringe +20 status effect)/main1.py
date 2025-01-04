from podatki import preprocess_data, preprocess_test_data
from model import build_model, train_model
from test import evaluate_model, predict_image

# Paths and categories
data_dir = r'C:\Users\legen\OneDrive\Dokumenti\GitHub\novice\sistem-za-razpoznavo-sport (open for cringe +20 status effect)\slike'
test_dir = r'C:\Users\legen\OneDrive\Dokumenti\GitHub\novice\sistem-za-razpoznavo-sport (open for cringe +20 status effect)\test'
categories = ['accidents', 'apple', 'apple_red', 'pear']  # Include all categories

# 1. Load and preprocess training data
print("Loading and preprocessing training data...")
X_train, X_val, y_train, y_val = preprocess_data(data_dir, categories)
print("Training data loaded and preprocessed.")

# 2. Load and preprocess test data
print("Loading and preprocessing test data...")
X_test, y_test = preprocess_test_data(test_dir, categories)
print("Test data loaded and preprocessed.")

# 3. Build the model
print("Building the model...")
model = build_model(input_shape=(128, 128, 3), num_classes=len(categories))
print("Model built successfully.")

# 4. Train the model
print("Training the model...")
history = train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=16)
print("Model training completed.")

# 5. Evaluate the model on the test dataset
print("Evaluating the model on test data...")
evaluate_model(model, X_test, y_test, categories)
print("Test data evaluation completed.")

# 6. Save the model
model.save('multi_class_model.keras')
print("Model saved as 'multi_class_model.keras'.")


