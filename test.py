"""
Learn XOR function using a very simple custom neural network.
"""

import matplotlib.pyplot as plt
import nn


if __name__ == "__main__":
    X = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    y = [
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0]
    ]

    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.Sigmoid(),
        nn.Linear(4, 2),
        nn.Sigmoid()
    )
    loss_fn = nn.MSELoss()

    num_epochs = 20_000
    learning_rate = 0.1

    loss_history = []

    for i in range(num_epochs):
        total_loss = 0.0

        for j in range(len(X)):
            # Forward pass
            y_pred = model.forward(X[j])

            loss = loss_fn.forward(y_pred, y[j])
            total_loss += loss

            # Backward pass
            output_gradient = loss_fn.backward()
            output_gradient = model.backward(output_gradient, learning_rate)

        epoch_loss = total_loss / len(X)
        loss_history.append(epoch_loss)

        if (i + 1) % 1000 == 0:
            print(f"Epoch: {i + 1} - Loss: {epoch_loss:.4f}")

    # Generate loss vs epoch graph
    plt.plot(range(1, num_epochs + 1), loss_history)
    plt.title("MSE Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()