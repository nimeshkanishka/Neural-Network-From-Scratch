import math
import random


class Module:
    def __init__(self):
        pass
    
    def forward(self):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError
    

class Sequential(Module):
    def __init__(self, *args: Module):
        self.layers = [*args]

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def backward(self, output_gradient, lr):
        for layer in self.layers[::-1]:
            output_gradient = layer.backward(output_gradient, lr)
        return output_gradient
    

class Layer(Module):
    def __init__(self):
        self.input = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient, lr):
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int):
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Get random weights and between -1 and +1 and scale them by 1 / sqrt(in_features)
        self.weights = [
            [
                random.uniform(-1.0, 1.0) / math.sqrt(self.in_features)
                for _ in range(self.out_features)
            ]
            for _ in range(self.in_features)
        ]

        self.biases = [
            random.uniform(-1.0, 1.0) / math.sqrt(self.in_features)
            for _ in range(out_features)
        ]

    def forward(self, input):
        self.input = input

        output = [0.0 for _ in range(self.out_features)]
        for j in range(self.out_features):
            out = self.biases[j]
            for i in range(self.in_features):
                out += input[i] * self.weights[i][j]
            output[j] = out
        return output

    def backward(self, output_gradient, lr):
        weight_gradient = [[0.0 for _ in range(self.out_features)] for _ in range(self.in_features)]
        bias_gradient = output_gradient
        input_gradient = [0.0 for _ in range(self.in_features)]

        for j in range(self.out_features):
            for i in range(self.in_features):
                weight_gradient[i][j] = self.input[i] * output_gradient[j]
                input_gradient[i] += self.weights[i][j] * output_gradient[j]

        for j in range(self.out_features):
            for i in range(self.in_features):
                self.weights[i][j] -= weight_gradient[i][j] * lr

            self.biases[j] -= bias_gradient[j] * lr

        return input_gradient
    

class Activation(Layer):
    def __init__(self, activation_fn, activation_prime_fn):
        super(Activation, self).__init__()

        self.activation_fn = activation_fn
        self.activation_prime_fn = activation_prime_fn

    def forward(self, input):
        self.input = input
        out_features = len(input)

        output = [0.0 for _ in range(out_features)]
        for j in range(out_features):
            output[j] = self.activation_fn(input[j])
        return output

    def backward(self, output_gradient, lr):
        in_features = len(output_gradient)

        input_gradient = [0.0 for _ in range(in_features)]
        for i in range(in_features):
            input_gradient[i] = self.activation_prime_fn(self.input[i]) * output_gradient[i]
        return input_gradient


class Sigmoid(Activation):
    def __init__(self):
        sigmoid = lambda x: 1 / (1 + math.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)
        
        super(Sigmoid, self).__init__(sigmoid, sigmoid_prime)


class MSELoss(Module):
    def __init__(self):
        self.input = None
        self.target = None

    def forward(self, input, target):
        self.input = input
        self.target = target

        loss = 0.0
        for i in range(len(input)):
            loss += (target[i] - input[i]) ** 2
        return loss

    def backward(self):
        out_features = len(self.input)

        input_gradient = [0.0 for _ in range(out_features)]
        for i in range(out_features):
            input_gradient[i] = 2 * (self.input[i] - self.target[i])
        return input_gradient