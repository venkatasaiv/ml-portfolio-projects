# MNIST Digit Classification

A machine learning project for classifying handwritten digits (0-9) using the MNIST dataset and Keras deep learning framework.

## Project Overview

This project implements a convolutional neural network (CNN) to classify handwritten digits from the famous MNIST dataset. The implementation uses only public data and original, self-written code following ethical AI practices.

## Dataset

- **Source**: MNIST Database of Handwritten Digits
- **Type**: Public domain dataset
- **Size**: 60,000 training images + 10,000 test images
- **Format**: 28x28 pixel grayscale images
- **Classes**: 10 digits (0-9)

## Project Structure

```
mnist-classification/
├── README.md              # Project documentation
├── data_preprocessing.py  # Data loading and preprocessing
├── model_training.py      # CNN model training
├── evaluate.py            # Model evaluation and testing
└── requirements.txt       # Python dependencies
```

## Sample Code

Here's how to load and display MNIST images using Keras:

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Display basic dataset information
print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")

# Display sample images
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f'Label: {y_train[i]}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Normalize pixel values to 0-1 range
X_train_normalized = X_train.astype('float32') / 255.0
X_test_normalized = X_test.astype('float32') / 255.0

print("\nData preprocessing completed!")
print(f"Pixel value range: [{X_train_normalized.min()}, {X_train_normalized.max()}]")
```

## Model Architecture

The CNN model includes:
- Convolutional layers with ReLU activation
- MaxPooling layers for dimensionality reduction
- Dropout layers for regularization
- Dense layers for final classification
- Softmax activation for probability output

## Requirements

- TensorFlow/Keras
- NumPy
- Matplotlib
- Scikit-learn

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Preprocess data: `python data_preprocessing.py`
3. Train model: `python model_training.py`
4. Evaluate results: `python evaluate.py`

## Compliance

✅ **Original Code**: All implementations are self-written
✅ **Public Dataset**: MNIST is freely available and widely used
✅ **Copyright Compliant**: No proprietary content used
✅ **Educational Purpose**: Demonstrates CNN concepts for digit classification

## Results

Expected performance:
- Training accuracy: >99%
- Test accuracy: >98%
- Fast training time on modern hardware

## License

This project is licensed under the MIT License - see the main repository LICENSE file for details.
