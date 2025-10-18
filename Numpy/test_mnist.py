"""
Training a convolutional neural network on the full MNIST dataset (all 10 categories; 2.5k training images per class).
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import nn


# Parent directory of the dataset
# Dataset from: https://www.kaggle.com/datasets/ben519/mnist-as-png
DATASET_PARENT_DIR = r"D:\Datasets\MNIST-PNG"
# Number of images per category used for training
NUM_TRAIN_IMAGES_PER_CATEGORY = 2500
# Number of images per category used for testing (validation)
NUM_TEST_IMAGES_PER_CATEGORY = 500
# Training parameters
NUM_EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


if __name__ == "__main__":
    print("Creating datasets...")

    # Training dataset
    X_train, y_train = [], []
    # Test dataset
    X_test, y_test = [], []

    for dataset in ["train", "test"]:
        # Directory containing the current dataset
        dataset_dir = os.path.join(DATASET_PARENT_DIR, dataset)

        # Add the required number of images from each category to the dataset
        for label, category in enumerate(sorted(os.listdir(dataset_dir))):
            # Directory containing images from the current category
            image_dir = os.path.join(dataset_dir, category)

            # Number of images to be added from the current category
            num_images_to_add = NUM_TRAIN_IMAGES_PER_CATEGORY if dataset == "train" else NUM_TEST_IMAGES_PER_CATEGORY
            
            for image_file in os.listdir(image_dir):
                image_path = os.path.join(image_dir, image_file)

                try:
                    # Load image in grayscale mode
                    # Shape: (Height, Width)
                    # For MNIST, Height = Width = 28
                    image_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                # Skip any image that cannot be loaded
                except Exception as e:
                    print(f"Warning: Could not process file '{image_path}': {e}")
                    continue

                # Skip any files that do not contain a valid image
                if image_array is None:
                    print(f"Warning: File '{image_path}' could not be read as an image.")
                    continue

                # Add pixel value array and label to the correct dataset
                if dataset == "train":
                    X_train.append(image_array)
                    y_train.append(label)
                else:
                    X_test.append(image_array)
                    y_test.append(label)

                # Decrement the counter and break the loop if we have processed the required number of images
                num_images_to_add -= 1
                if num_images_to_add <= 0:
                    break

    print("Finished creating datasets!")

    # Convert image pixel value lists to numpy arrays
    # Reshape to match the shapes expected by the model
    # Normalize pixel values to [0, 1]
    # Shape: (Number of training images, Channels, Height, Width)
    X_train = np.array(X_train, dtype=np.float32).reshape(-1, 1, 28, 28) / 255.0
    # Shape: (Number of test images, Channels, Height, Width)
    X_test = np.array(X_test, dtype=np.float32).reshape(-1, 1, 28, 28) / 255.0

    # Convert label lists to numpy arrays and one-hot encode
    # Shape: (Number of training images,)
    y_train = np.array(y_train, dtype=np.uint8)
    # Shape: (Number of training images, Number of classes)
    y_train_one_hot = np.eye(10, dtype=np.float32)[y_train]
    # Shape: (Number of test images,)
    y_test = np.array(y_test, dtype=np.uint8)
    # Shape: (Number of test images, Number of classes)
    y_test_one_hot = np.eye(10, dtype=np.float32)[y_test]

    # Define CNN architecture
    model = nn.Sequential(
        nn.Conv2d((1, 28, 28), 8, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d((8, 26, 26), 16, kernel_size=3),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16 * 24 * 24, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
        nn.Softmax()
    )
    loss_fn = nn.CrossEntropyLoss()

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
        X_train, y_train, y_train_one_hot = X_train[indices], y_train[indices], y_train_one_hot[indices]

        # Training
        total_loss = 0.0
        correct_preds = 0
        for batch_start in tqdm(range(0, X_train.shape[0], BATCH_SIZE)):
            # If the last batch is smaller than BATCH_SIZE, batch end index can't be batch_start + BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, X_train.shape[0])

            # Get batch of images and labels
            # Shape: (Batch size, Channels, Height, Width)
            X = X_train[batch_start:batch_end]
            # Shape: (Batch size,)
            y = y_train[batch_start:batch_end]
            # Shape: (Batch size, Classes)
            y_one_hot = y_train_one_hot[batch_start:batch_end]

            # Forward pass
            # Shape: (Batch size, Classes)
            y_pred = model.forward(X)

            loss = loss_fn.forward(y_pred, y_one_hot)
            total_loss += loss
            correct_preds += np.sum(np.argmax(y_pred, axis=-1) == y)

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
        for batch_start in range(0, X_test.shape[0], BATCH_SIZE):
            # If the last batch is smaller than BATCH_SIZE, batch end index can't be batch_start + BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, X_test.shape[0])

            # Get batch of images and labels
            # Shape: (Batch size, Channels, Height, Width)
            X = X_test[batch_start:batch_end]
            # Shape: (Batch size,)
            y = y_test[batch_start:batch_end]
            # Shape: (Batch size, Classes)
            y_one_hot = y_test_one_hot[batch_start:batch_end]

            # Shape: (Batch size, Classes)
            y_pred = model.forward(X)

            loss = loss_fn.forward(y_pred, y_one_hot)
            total_loss += loss
            correct_preds += np.sum(np.argmax(y_pred, axis=-1) == y)

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
