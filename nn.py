import math
import random
from typing import Callable


class Module:
    def __init__(self) -> None:
        pass
    
    def forward(self):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError
    

class Sequential(Module):
    def __init__(self, *args: Module) -> None:
        self.layers = [*args]

    def forward(self, input: list[list[float]]) -> list[list[float]]:
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def backward(self, output_gradient: list[list[float]], lr: float) -> list[list[float]]:
        for layer in self.layers[::-1]:
            output_gradient = layer.backward(output_gradient, lr)
        return output_gradient
    

class Layer(Module):
    def __init__(self) -> None:
        self.input = None
        self.batch_size = None

    def forward(self, input: list[list[float]]):
        raise NotImplementedError

    def backward(self, output_gradient: list[list[float]], lr: float):
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        limit = math.sqrt(1.0 / in_features)
        self.weights = [
            [random.uniform(-limit, limit) for _ in range(out_features)]
            for _ in range(in_features)
        ]
        self.biases = [random.uniform(-limit, limit) for _ in range(out_features)]

    def forward(self, input: list[list[float]]) -> list[list[float]]:
        self.input = input
        self.batch_size = len(input)

        output = [
            [0.0 for _ in range(self.out_features)]
            for _ in range(self.batch_size)
        ]
        for b in range(self.batch_size):
            for j in range(self.out_features):
                out = self.biases[j]
                for i in range(self.in_features):
                    out += input[b][i] * self.weights[i][j]
                output[b][j] = out
        return output

    def backward(self, output_gradient: list[list[float]], lr: float) -> list[list[float]]:
        weight_gradient = [
            [0.0 for _ in range(self.out_features)]
            for _ in range(self.in_features)
        ]
        bias_gradient = [0.0 for _ in range(self.out_features)]
        input_gradient = [
            [0.0 for _ in range(self.in_features)]
            for _ in range(self.batch_size)
        ]

        # Calculate gradients
        for b in range(self.batch_size):
            for j in range(self.out_features):
                bias_gradient[j] += output_gradient[b][j]
                for i in range(self.in_features):
                    weight_gradient[i][j] += self.input[b][i] * output_gradient[b][j]
                    input_gradient[b][i] += self.weights[i][j] * output_gradient[b][j]

        # Update weights and biases
        for j in range(self.out_features):
            self.biases[j] -= bias_gradient[j] * lr
            for i in range(self.in_features):
                self.weights[i][j] -= weight_gradient[i][j] * lr

        return input_gradient
    

class Activation(Layer):
    def __init__(self, activation_fn: Callable[[float], float], activation_prime_fn: Callable[[float], float]) -> None:
        super(Activation, self).__init__()

        self.activation_fn = activation_fn
        self.activation_prime_fn = activation_prime_fn

        self.out_features = None

    def forward(self, input: list[list[float]]) -> list[list[float]]:
        self.input = input
        self.batch_size = len(input)
        self.out_features = len(input[0])

        output = [
            [0.0 for _ in range(self.out_features)]
            for _ in range(self.batch_size)
        ]
        for b in range(self.batch_size):
            for j in range(self.out_features):
                output[b][j] = self.activation_fn(input[b][j])
        return output

    def backward(self, output_gradient: list[list[float]], lr: float) -> list[list[float]]:
        input_gradient = [
            [0.0 for _ in range(self.out_features)]
            for _ in range(self.batch_size)
        ]
        for b in range(self.batch_size):
            for i in range(self.out_features):
                input_gradient[b][i] = self.activation_prime_fn(self.input[b][i]) * output_gradient[b][i]
        return input_gradient


class ReLU(Activation):
    def __init__(self) -> None:
        relu = lambda x: max(0.0, x)

        relu_prime = lambda x: 1.0 if x > 0.0 else 0.0
        
        super(ReLU, self).__init__(relu, relu_prime)


class Sigmoid(Activation):
    def __init__(self) -> None:
        sigmoid = lambda x: 1 / (1 + math.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)
        
        super(Sigmoid, self).__init__(sigmoid, sigmoid_prime)


class Loss(Module):
    def __init__(self, loss_fn: Callable[[float, float], float], loss_prime_fn: Callable[[float, float], float]) -> None:
        self.loss_fn = loss_fn
        self.loss_prime_fn = loss_prime_fn

        self.input = None
        self.target = None
        self.batch_size = None
        self.out_features = None

    def forward(self, input: list[list[float]], target: list[list[float]]) -> float:        
        self.input = input
        self.target = target
        self.batch_size = len(input)
        self.out_features = len(input[0])

        total_loss = 0.0
        for b in range(self.batch_size):
            for i in range(self.out_features):
                total_loss += self.loss_fn(input[b][i], target[b][i])
        return total_loss / (self.batch_size * self.out_features)

    def backward(self) -> list[list[float]]:
        input_gradient = [
            [0.0 for _ in range(self.out_features)]
            for _ in range(self.batch_size)
        ]
        for b in range(self.batch_size):
            for i in range(self.out_features):
                input_gradient[b][i] = self.loss_prime_fn(self.input[b][i], self.target[b][i]) / (self.batch_size * self.out_features)
        return input_gradient


class MSELoss(Loss):
    def __init__(self) -> None:
        mse = lambda i, t: (i - t) ** 2

        mse_prime = lambda i, t: 2 * (i - t)

        super(MSELoss, self).__init__(mse, mse_prime)
    

class BCELoss(Loss):
    def __init__(self) -> None:
        bce = lambda i, t: -(t * math.log(i) + (1 - t) * math.log(1 - i))

        bce_prime = lambda i, t: -(t / i - (1 - t) / (1 - i))

        super(BCELoss, self).__init__(bce, bce_prime)
