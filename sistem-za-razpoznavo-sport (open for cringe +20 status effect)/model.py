from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Disable oneDNN optimizations to suppress warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def build_model(input_shape, num_classes):
    """
    Builds and compiles a MobileNetV2-based model.

    Parameters:
    - input_shape (tuple): Shape of the input images, e.g., (128, 128, 3).
    - num_classes (int): Number of output classes.

    Returns:
    - model (Model): Compiled Keras model.
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    
    if num_classes == 2:  # Binary classification
        predictions = Dense(1, activation='sigmoid')(x)
    else:  # Multiclass classification
        predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=16, save_path='best_model.keras'):
    """
    Trains the model on the given data.

    Parameters:
    - model (Model): Compiled Keras model.
    - X_train, y_train: Training data and labels.
    - X_val, y_val: Validation data and labels.
    - epochs (int): Number of training epochs.
    - batch_size (int): Batch size.
    - save_path (str): Path to save the best model during training.

    Returns:
    - history: Training history object.
    """
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # Save the best model during training
    model_checkpoint = ModelCheckpoint(
        save_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    print(f"Training completed. Best model saved to {save_path}.")
    return history
