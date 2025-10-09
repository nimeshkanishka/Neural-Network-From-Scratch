"""
Training a convolutional neural network on MNIST dataset to classify images of digits 0 and 1.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import nn


# Parent directory of the dataset
DATASET_DIR = r"D:\Datasets\MNIST-PNG"
# Categories to use for training
# Here we will do binary classification of digits 0 and 1
CATEGORIES = ["0", "1"]
# Number of images per category used for training
NUM_TRAIN_IMAGES_PER_CATEGORY = 5000
# Number of images per category used for testing (validation)
NUM_TEST_IMAGES_PER_CATEGORY = 750
# Training parameters
NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


if __name__ == "__main__":
    # Training dataset
    X_train, y_train = [], []
    # Test dataset
    X_test, y_test = [], []

    for dataset in ["train", "test"]:
        dataset_folder = os.path.join(DATASET_DIR, dataset)

        # Add the required number of images from each category to the dataset
        for label, category in enumerate(CATEGORIES):
            image_folder = os.path.join(dataset_folder, category)
            image_files = os.listdir(image_folder)

            # Number of images to be added from this category
            num_images_to_add = NUM_TRAIN_IMAGES_PER_CATEGORY if dataset == "train" else NUM_TEST_IMAGES_PER_CATEGORY
            # Current index in the list of image files
            i = -1

            while num_images_to_add > 0:
                i += 1
                # Break if all images in the directory are processed
                # even if we don't have the required number of images
                if i >= len(image_files):
                    break

                image_path = os.path.join(image_folder, image_files[i])

                try:
                    # Load image in grayscale mode
                    # Shape: (Height, Width)
                    # For MNIST, Height = Width = 28
                    image_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)                    

                # Skip any image that cannot be loaded
                except Exception as e:
                    print(f"Error processing image '{os.path.join(image_folder, image_files[i])}': {e}")
                    continue

                # Add pixel value array and label to the correct dataset
                if dataset == "train":
                    X_train.append(image_array)
                    y_train.append(label)
                else:
                    X_test.append(image_array)
                    y_test.append(label)

                num_images_to_add -= 1

    # Convert datasets to numpy arrays, reshape to match the shapes expected by the model
    # and normalize pixel values to [0, 1]
    # Shape: (Number of training images, Channels, Height, Width)
    X_train = np.array(X_train, dtype=np.float32).reshape(-1, 1, 28, 28) / 255.0
    # Shape: (Number of training images, 1)
    y_train = np.array(y_train, dtype=np.float32).reshape(-1, 1)
    # Shape: (Number of test images, Channels, Height, Width)
    X_test = np.array(X_test, dtype=np.float32).reshape(-1, 1, 28, 28) / 255.0
    # Shape: (Number of test images, 1)
    y_test = np.array(y_test, dtype=np.float32).reshape(-1, 1)

    # Define CNN architecture
    model = nn.Sequential(
        nn.Conv2d((1, 28, 28), 8, kernel_size=3),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * 26 * 26, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    loss_fn = nn.BCELoss()

    # Lists to keep track of training and test loss and accuracy over epochs
    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []

    # Training loop
    for k in range(NUM_EPOCHS):
        print(f"--- Epoch {k + 1}/{NUM_EPOCHS} ---")

        # Shuffle training data each epoch
        indices = np.random.permutation(X_train.shape[0])
        X_train, y_train = X_train[indices], y_train[indices]

        # Training
        total_loss = 0.0
        correct_preds = 0
        for i in tqdm(range(0, X_train.shape[0], BATCH_SIZE)):
            # Get batch of images and labels
            X = X_train[i : min(i + BATCH_SIZE, X_train.shape[0])]
            y = y_train[i : min(i + BATCH_SIZE, y_train.shape[0])]

            # Forward pass
            y_pred = model.forward(X)

            loss = loss_fn.forward(y_pred, y)
            total_loss += loss
            correct_preds += np.sum((y_pred >= 0.5) == y)

            # Backward pass
            output_gradient = loss_fn.backward()
            output_gradient = model.backward(output_gradient, LEARNING_RATE)

        train_loss = total_loss / np.ceil(X_train.shape[0] / BATCH_SIZE)
        train_loss_history.append(train_loss)
        train_accuracy = correct_preds / X_train.shape[0]
        train_acc_history.append(train_accuracy)
        print(f"Training loss: {train_loss:.4f} - Training accuracy: {(train_accuracy * 100):.2f}%")

        # Testing (Validation)
        total_loss = 0.0
        correct_preds = 0
        for i in range(0, X_test.shape[0], BATCH_SIZE):
            # Get batch of images and labels
            X = X_test[i : min(i + BATCH_SIZE, X_test.shape[0])]
            y = y_test[i : min(i + BATCH_SIZE, y_test.shape[0])]

            y_pred = model.forward(X)

            loss = loss_fn.forward(y_pred, y)
            total_loss += loss
            correct_preds += np.sum((y_pred >= 0.5) == y)

        test_loss = total_loss / np.ceil(X_test.shape[0] / BATCH_SIZE)
        test_loss_history.append(test_loss)
        test_accuracy = correct_preds / X_test.shape[0]
        test_acc_history.append(test_accuracy)
        print(f"Test loss: {test_loss:.4f} - Test accuracy: {(test_accuracy * 100):.2f}%")

    # Plot training and test loss and accuracy graphs
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    epochs = [i for i in range(1, NUM_EPOCHS + 1)]

    # Loss vs Epoch graph
    axes[0].plot(epochs, train_loss_history, label="Training Loss")
    axes[0].plot(epochs, test_loss_history, label="Test Loss")
    axes[0].set_title("Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy vs Epoch graph
    axes[1].plot(epochs, train_acc_history, label="Training Accuracy")
    axes[1].plot(epochs, test_acc_history, label="Test Accuracy")
    axes[1].set_title("Accuracy vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.show()
