"""
model.py - Core Functions for Handwritten Letter Recognition (A-J)
Modular functions for data loading, model building, training, and prediction.
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

# Set environment to suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

# Use tf.keras instead of standalone keras for compatibility
layers = tf.keras.layers
models = tf.keras.models
to_categorical = tf.keras.utils.to_categorical

# Alphabet mapping for A-J
ALPHABET = list("ABCDEFGHIJ")


def load_and_preprocess_data(filepath: str) -> tuple:
    """
    Load CSV, filter A-J, normalize, balance, and split data.
    
    Args:
        filepath: Path to the A_Z Handwritten Data.csv file
        
    Returns:
        tuple: (x_train, x_test, y_train, y_test, labels_test)
    """
    # Load data
    data = pd.read_csv(filepath).astype("float32")
    
    # Filter to only A-J (labels 0-9)
    raw_labels = data.iloc[:, 0].to_numpy()
    mask = raw_labels < 10
    data_filtered = data[mask]
    
    labels = data_filtered.iloc[:, 0].to_numpy().astype(int)
    pixels = data_filtered.iloc[:, 1:].to_numpy()
    
    # Reshape and normalize
    x = pixels.reshape(-1, 28, 28, 1) / 255.0
    
    # One-hot encoding
    y = to_categorical(labels, num_classes=10)
    
    # Balance data
    min_count = min(Counter(labels).values())
    
    x_balanced = []
    y_balanced = []
    labels_balanced = []
    
    for i in range(10):
        idx = np.where(labels == i)[0][:min_count]
        x_balanced.append(x[idx])
        y_balanced.append(y[idx])
        labels_balanced.append(labels[idx])
    
    x = np.concatenate(x_balanced)
    y = np.concatenate(y_balanced)
    labels = np.concatenate(labels_balanced)
    
    # Train-test split
    x_train, x_test, y_train, y_test, labels_train, labels_test = train_test_split(
        x, y, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    return x_train, x_test, y_train, y_test, labels_test


def build_cnn_model() -> tf.keras.Model:
    """
    Build CNN architecture for handwritten letter recognition.
    
    Returns:
        tf.keras.Model: Compiled CNN model
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')  # 10 classes: A-J
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(model, x_train, y_train, epochs: int, batch_size: int = 128, 
                validation_split: float = 0.1, callbacks=None):
    """
    Train the model and return history.
    
    Args:
        model: Keras model to train
        x_train: Training data
        y_train: Training labels (one-hot encoded)
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of training data for validation
        callbacks: Optional list of Keras callbacks
        
    Returns:
        History: Training history object
    """
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=0,  # Suppress output for Streamlit
        callbacks=callbacks
    )
    
    return history


def predict_single(model, x_test, labels_test, index: int) -> dict:
    """
    Make prediction for a single test sample.
    
    Args:
        model: Trained Keras model
        x_test: Test data
        labels_test: True labels for test data
        index: Index of sample to predict
        
    Returns:
        dict: Prediction results with confidence scores
    """
    img = x_test[index]
    pred = model.predict(img.reshape(1, 28, 28, 1), verbose=0)
    pred_label = np.argmax(pred)
    confidence = pred[0][pred_label]
    
    # Get all confidence scores
    confidence_scores = {ALPHABET[i]: float(pred[0][i]) for i in range(10)}
    
    return {
        'image': img,
        'true_label': ALPHABET[labels_test[index]],
        'predicted_label': ALPHABET[pred_label],
        'confidence': float(confidence),
        'confidence_scores': confidence_scores,
        'all_probabilities': pred[0]
    }


def evaluate_model(model, x_test, y_test) -> tuple:
    """
    Evaluate model on test data.
    
    Args:
        model: Trained Keras model
        x_test: Test data
        y_test: Test labels (one-hot encoded)
        
    Returns:
        tuple: (test_loss, test_accuracy)
    """
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    return test_loss, test_acc


def get_data_info(x_train, x_test, labels_test) -> dict:
    """
    Get information about the dataset.
    
    Args:
        x_train: Training data
        x_test: Test data
        labels_test: Test labels
        
    Returns:
        dict: Dataset information
    """
    unique, counts = np.unique(labels_test, return_counts=True)
    class_distribution = {ALPHABET[int(label)]: int(count) for label, count in zip(unique, counts)}
    
    return {
        'train_samples': len(x_train),
        'test_samples': len(x_test),
        'image_shape': x_train[0].shape,
        'num_classes': 10,
        'classes': ALPHABET,
        'class_distribution': class_distribution
    }
