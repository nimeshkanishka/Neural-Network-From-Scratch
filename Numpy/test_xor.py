"""
Learning XOR function using a very simple custom neural network.
"""

import numpy as np
import matplotlib.pyplot as plt
import nn


# Training parameters
NUM_EPOCHS = 10_000
LEARNING_RATE = 0.5


if __name__ == "__main__":
    # XOR dataset
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    # Define network architecture
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
        nn.Sigmoid()
    )
    loss_fn = nn.BCELoss()

    loss_history = []

    for i in range(NUM_EPOCHS):
        # Forward pass
        y_pred = model.forward(X)

        loss = loss_fn.forward(y_pred, y)

        # Backward pass
        output_gradient = loss_fn.backward()
        output_gradient = model.backward(output_gradient, LEARNING_RATE)

        loss_history.append(loss)
        if (i + 1) % 1000 == 0:
            print(f"Epoch: {i + 1} - Loss: {loss:.4f}")

    # Generate loss vs epoch graph
    plt.plot(range(1, NUM_EPOCHS + 1), loss_history)
    plt.title("MSE Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()
