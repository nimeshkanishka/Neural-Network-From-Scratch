from typing import Callable
import numpy as np
from scipy import signal


class Module:
    def __init__(self) -> None:
        pass
    
    def forward(self):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError
    

class Layer(Module):
    def __init__(self) -> None:
        self.input = None

    def forward(self, input: np.ndarray):
        raise NotImplementedError

    def backward(self, output_gradient: np.ndarray, lr: float):
        raise NotImplementedError


class Linear(Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int
    ) -> None:
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        limit = np.sqrt(1.0 / in_features)
        self.weights = np.random.uniform(low=-limit, high=limit, size=(in_features, out_features))
        self.biases = np.random.uniform(low=-limit, high=limit, size=(out_features,))

    def forward(
        self,
        input: np.ndarray
    ) -> np.ndarray:
        self.input = input
    
        return input @ self.weights + self.biases

    def backward(
        self,
        output_gradient: np.ndarray,
        lr: float
    ) -> np.ndarray:
        # Calculate gradients
        weight_gradient = self.input.T @ output_gradient
        bias_gradient = np.sum(output_gradient, axis=0)
        input_gradient = output_gradient @ self.weights.T

        # Update weights and biases
        self.weights -= weight_gradient * lr
        self.biases -= bias_gradient * lr

        return input_gradient


class Conv2d(Layer):
    def __init__(self,
        input_shape: tuple[int, int, int],
        out_channels: int,
        kernel_size: int
    ) -> None:
        super(Conv2d, self).__init__()

        # Input shape: (Input channels, Input height, Input width)
        self.in_channels, in_height, in_width = input_shape
        self.out_channels = out_channels

        # Output size = (Input size - Kernel size + 2 * Padding) // Stride + 1
        # Because stride = 1 and padding = 0,
        # Output size = Input size - Kernel size + 1
        # Output shape: (Output channels, Output height, Output width)
        self.output_shape = (out_channels, in_height - kernel_size + 1, in_width - kernel_size + 1)

        limit = np.sqrt(1.0 / (self.in_channels * kernel_size * kernel_size))
        # Shape: (Out channels, In channels, Kernel height, Kernel width)
        self.kernels = np.random.uniform(low=-limit, high=limit,
                                         size=(out_channels, self.in_channels, kernel_size, kernel_size))
        # Shape: (Out channels,)
        self.biases = np.random.uniform(low=-limit, high=limit,
                                        size=(out_channels,))
        
        self.batch_size = None
        
    def forward(
        self,
        input: np.ndarray
    ) -> np.ndarray:
        # Shape: (Batch size, Input channels, Input height, Input width)
        self.input = input
        self.batch_size = input.shape[0]

        # Shape: (Batch size, Output channels, Output height, Output width)
        output = np.full(shape=(self.batch_size, *self.output_shape), fill_value=self.biases[:, None, None])
        for b in range(self.batch_size):
            for i in range(self.out_channels):
                for j in range(self.in_channels):
                    output[b, i] += signal.correlate2d(input[b, j], self.kernels[i, j], mode="valid")
        return output
    
    def backward(
        self,
        output_gradient: np.ndarray,
        lr: float
    ) -> np.ndarray:
        # Shape of output_gradient: (Batch size, Output channels, Output height, Output width)

        # Initialize gradients
        # Shape: (Out channels, In channels, Kernel height, Kernel width)
        kernel_gradient = np.zeros_like(self.kernels)
        # Shape: (Out channels,)
        bias_gradient = np.sum(output_gradient, axis=(0, 2, 3))
        # Shape: (Batch size, Input channels, Input height, Input width)
        input_gradient = np.zeros_like(self.input)

        # Calculate gradients
        for b in range(self.batch_size):
            for i in range(self.out_channels):
                for j in range(self.in_channels):
                    kernel_gradient[i, j] += signal.correlate2d(self.input[b, j], output_gradient[b, i], mode="valid")
                    input_gradient[b, j] += signal.convolve2d(output_gradient[b, i], self.kernels[i, j], mode="full")

        # Update kernels and biases
        self.kernels -= kernel_gradient * lr
        self.biases -= bias_gradient * lr

        return input_gradient


class Flatten(Layer):
    def __init__(self) -> None:
        super(Flatten, self).__init__()

    def forward(self,
        input: np.ndarray
    ) -> np.ndarray:
        # Shape: (Batch size, Channels, Height, Width)
        self.input = input

        # Shape: (Batch size, Channels * Height * Width)
        return np.reshape(input, (input.shape[0], -1))

    def backward(
        self,
        output_gradient: np.ndarray,
        lr: float
    ) -> np.ndarray:
        # (Batch size, Channels * Height * Width) -> (Batch size, Channels, Height, Width)
        return np.reshape(output_gradient, self.input.shape)
    

class Activation(Layer):
    def __init__(
        self,
        activation_fn: Callable[[np.ndarray], np.ndarray],
        activation_prime_fn: Callable[[np.ndarray], np.ndarray]
    ) -> None:
        super(Activation, self).__init__()

        self.activation_fn = activation_fn
        self.activation_prime_fn = activation_prime_fn

    def forward(
        self,
        input: np.ndarray
    ) -> np.ndarray:
        self.input = input

        return self.activation_fn(input)

    def backward(
        self,
        output_gradient: np.ndarray,
        lr: float
    ) -> np.ndarray:
        return self.activation_prime_fn(self.input) * output_gradient
    

class ReLU(Activation):
    def __init__(self) -> None:
        relu = lambda x: np.maximum(x, 0.0)

        relu_prime = lambda x: np.where(x > 0.0, 1.0, 0.0)
        
        super(ReLU, self).__init__(relu, relu_prime)


class Sigmoid(Activation):
    def __init__(self) -> None:
        sigmoid = lambda x: 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)
        
        super(Sigmoid, self).__init__(sigmoid, sigmoid_prime)


class Softmax(Layer):
    def __init__(self):
        super(Softmax, self).__init__()

        self.output = None

    def forward(
        self,
        input: np.ndarray
    ) -> np.ndarray:
        # Shape: (Batch size, Classes)
        self.input = input

        # Shape: (Batch size, Classes)
        input_exponential = np.exp(input)

        # Shape: (Batch size, Classes) / (Batch size, 1)
        #     -> (Batch size, Classes) / (Batch size, Classes)
        #     -> (Batch size, Classes)
        self.output = input_exponential / np.sum(input_exponential, axis=-1, keepdims=True)
        return self.output
    
    def backward(
        self,
        output_gradient: np.ndarray,
        lr: float
    ) -> np.ndarray:
        # For each sample in the batch:
        # dL/dx_i = y_i * (dL/dy_i - sum_j(y_j * dL/dy_j))

        # dot = sum(y * dL/dy)
        #     = sum(output * output_gradient)
        # Shape: (Batch size, 1)
        dot = np.sum(self.output * output_gradient, axis=-1, keepdims=True)

        # dL/dx = y * (dL/dy - sum(y * dL/dy))
        #       = output * (output_gradient - dot)
        # This uses the Jacobian of the softmax function: J = diag(y) - y yᵀ.
        # Shape: (Batch size, Classes) * ((Batch size, Classes) - (Batch size, 1))
        #     -> (Batch size, Classes) * ((Batch size, Classes) - (Batch size, Classes))
        #     -> (Batch size, Classes)
        return self.output * (output_gradient - dot)


class Loss(Module):
    def __init__(
        self,
        loss_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        loss_prime_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> None:
        self.loss_fn = loss_fn
        self.loss_prime_fn = loss_prime_fn

        self.input = None
        self.target = None
        self.batch_size = None

    def forward(
        self,
        input: np.ndarray,
        target: np.ndarray
    ) -> float:
        # Shape: (Batch size, Features)
        self.input = input
        self.target = target
        self.batch_size = self.input.shape[0]

        return np.mean(self.loss_fn(input, target))

    def backward(self) -> np.ndarray:
        # Normalized by batch size
        return self.loss_prime_fn(self.input, self.target) / self.batch_size


class MSELoss(Loss):
    """
    Read more: https://neuralthreads.medium.com/mean-square-error-the-most-used-regression-loss-2f684ec4ca04
    """

    def __init__(self) -> None:
        mse = lambda i, t: (t - i) ** 2

        mse_prime = lambda i, t: 2 * (i - t) / i.shape[-1]

        super(MSELoss, self).__init__(mse, mse_prime)
    

class BCELoss(Loss):
    """
    Read more: https://neuralthreads.medium.com/binary-cross-entropy-loss-special-case-of-categorical-cross-entropy-loss-95c0c338d183
    """

    def __init__(self) -> None:
        bce = lambda i, t: -(t * np.log(i) + (1.0 - t) * np.log(1.0 - i))

        bce_prime = lambda i, t: -(t / i - (1.0 - t) / (1.0 - i)) / i.shape[-1]

        super(BCELoss, self).__init__(bce, bce_prime)


class CrossEntropyLoss(Loss):
    """
    Read more: https://neuralthreads.medium.com/categorical-cross-entropy-loss-the-most-important-loss-function-d3792151d05b
    """

    def __init__(self) -> None:
        cross_entropy = lambda i, t: -np.sum(t * np.log(i), axis=-1)

        cross_entropy_prime = lambda i, t: -t / i

        super(CrossEntropyLoss, self).__init__(cross_entropy, cross_entropy_prime)


class Sequential(Module):
    def __init__(
        self,
        *args: Module
    ) -> None:
        self.layers = []
        for layer in args:
            if not isinstance(layer, Module):
                raise TypeError(
                    f"Sequential can only contain objects of type Module, got {type(layer).__name__}"
                )
            if isinstance(layer, Loss):
                raise TypeError(
                    f"Sequential cannot contain Loss objects, got {type(layer).__name__}"
                )
            self.layers.append(layer)

    def forward(
        self,
        input: np.ndarray
    ) -> np.ndarray:
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(
        self,
        output_gradient: np.ndarray,
        lr: float
    ) -> np.ndarray:
        input_gradient = output_gradient
        for layer in self.layers[::-1]:
            input_gradient = layer.backward(input_gradient, lr)
        return input_gradient
