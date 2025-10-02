from typing import Callable
import numpy as np


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

    def forward(
        self,
        input: np.ndarray,
        target: np.ndarray
    ) -> float:        
        self.input = input
        self.target = target

        return np.mean(self.loss_fn(input, target))

    def backward(self) -> np.ndarray:
        return self.loss_prime_fn(self.input, self.target) / np.size(self.input)


class MSELoss(Loss):
    def __init__(self) -> None:
        mse = lambda i, t: (i - t) ** 2

        mse_prime = lambda i, t: 2 * (i - t)

        super(MSELoss, self).__init__(mse, mse_prime)
    

class BCELoss(Loss):
    def __init__(self) -> None:
        bce = lambda i, t: -(t * np.log(i) + (1.0 - t) * np.log(1.0 - i))

        bce_prime = lambda i, t: -(t / i - (1.0 - t) / (1.0 - i))

        super(BCELoss, self).__init__(bce, bce_prime)


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