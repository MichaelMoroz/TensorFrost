import TensorFrost as tf
import math
import numpy as np

tf.initialize(tf.opengl)


class Parameter:
    def __init__(self, shape, dtype, random_init=True):
        self.shape = shape
        self.dtype = dtype
        self.random_init = random_init

class ParameterArray:
    def __init__(self, prefix):
        self._prefix = prefix
        self._parameters = {}

    def __getitem__(self, index):
        key = f"{self._prefix}_{index}"
        if key not in self._parameters:
            raise IndexError(f"Parameter {key} not found")
        return self._parameters[key]

    def __setitem__(self, index, value):
        key = f"{self._prefix}_{index}"
        self._parameters[key] = value

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self._parameter_arrays = {}

    def register_parameter(self, name, param):
        self._parameters[name] = param

    def register_module(self, name, module):
        self._modules[name] = module

    def register_parameter_array(self, name):
        array = ParameterArray(name)
        self._parameter_arrays[name] = array
        return array

    def __getattr__(self, name):
        if name in self._parameters:
            return self._parameters[name]
        elif name in self._modules:
            return self._modules[name]
        elif name in self._parameter_arrays:
            return self._parameter_arrays[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, ParameterArray):
            self._parameter_arrays[name] = value
        else:
            super().__setattr__(name, value)

    def assert_parameters(self):
        pass

    def initialize_input(self):
        for module in self._modules.values():
            module.initialize_input()

        for name, param in self._parameters.items():
            tensor = tf.input(param.shape, param.dtype)
            self._parameters[name] = tensor
        
        for array in self._parameter_arrays.values():
            for key, param in array._parameters.items():
                tensor = tf.input(param.shape, param.dtype)
                array._parameters[key] = tensor
        
        self.assert_parameters()

    def initialize_parameter(self, param):
        if param.random_init:
            return tf.tensor(np.random.randn(*param.shape).astype(np.float32) * math.sqrt(2.0 / param.shape[0]))
        else:
            return tf.tensor(np.zeros(param.shape, np.float32))

    def initialize_parameters(self):
        for module in self._modules.values():
            module.initialize_parameters()

        for name, param in self._parameters.items():
            self._parameters[name] = self.initialize_parameter(param)
        
        for array in self._parameter_arrays.values():
            for key, param in array._parameters.items():
                array._parameters[key] = self.initialize_parameter(param)

    def get_all_parameters(self):
        params = []
        for module in self._modules.values():
            params += module.get_all_parameters()
        params += list(self._parameters.values())
        for array in self._parameter_arrays.values():
            params += list(array._parameters.values())
        return params
    
    def create_input(self, X, Y):
        inputs = self.get_all_parameters()
        inputs.append(X)
        inputs.append(Y)
        return inputs

    def update_parameters(self, parameter_values):
        index = 0
        
        def update_params(obj):
            nonlocal index
            for module in obj._modules.values():
                update_params(module)
            
            for param_name in obj._parameters:
                obj._parameters[param_name] = parameter_values[index]
                index += 1
            
            for array in obj._parameter_arrays.values():
                for param_name in array._parameters:
                    array._parameters[param_name] = parameter_values[index]
                    index += 1
        
        update_params(self)

def softmax(X):
    exp = tf.exp(X)
    return exp / tf.unsqueeze(tf.sum(exp))

def leaky_relu(X):
    return tf.select(X > 0.0, X, 0.01 * X)

def mul_bias(X, W):
    ids = tf.indices(list(X.shape[:-1]) + [W.shape[-2]])
    return tf.select(ids[-1] == X.shape[-1], 1.0, X[ids]) @ W

class MNIST_net(Module):
    def __init__(self, N1, N2, N3):
        super().__init__()
        self.register_parameter('W1', Parameter([N1, N2], tf.float32))
        self.register_parameter('W2', Parameter([N2, N3], tf.float32))
        self.register_parameter('W3', Parameter([N3, 10], tf.float32))

    def assert_parameters(self):
        N1, N2 = self.W1.shape
        _, N3 = self.W2.shape
        # Additional assertions can be added here if needed

    def forward(self, X):
        L1 = leaky_relu(mul_bias(X, self.W1))
        L2 = leaky_relu(mul_bias(L1, self.W2))
        return softmax(mul_bias(L2, self.W3))

    def loss(self, X, Y):
        Yhat = self.forward(X)
        return tf.mean(tf.mean( - Y * tf.log(Yhat + 1e-3) - (1.0 - Y) * tf.log(1.0 - Yhat + 1e-3)))
    
class MNIST_Momentum_opt(Module):
    def __init__(self, net: MNIST_net, learning_rate: float, momentum: float):
        super().__init__()
        self.register_module('net', net)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.m = self.register_parameter_array('m')
        for i, (name, param) in enumerate(net._parameters.items()):
            self.m[i] = Parameter(param.shape, param.dtype, False)

    def assert_parameters(self):
        for i, (name, param) in enumerate(self.net._parameters.items()):
            self.m[i] = tf.assert_tensor(self.m[i], param.shape, tf.float32)

    def step(self, X, Y):
        L = self.net.loss(X, Y)
        for i, (name, param) in enumerate(self.net._parameters.items()):
            m = self.m[i]
            grad = tf.grad(L, param)
            m = self.momentum * m - self.learning_rate * grad
            param += m
            self.m[i] = m
            self.net._parameters[name] = param
        return L

def OptimizerStep():
    model = MNIST_net(-1, -1, -1)
    opt = MNIST_Momentum_opt(model, 0.02, 0.99)

    opt.initialize_input()

    X = tf.input([-1, -1], tf.float32)
    Y = tf.input([X.shape[0], 10], tf.float32)

    L = opt.step(X, Y)
    params = opt.get_all_parameters()
    for param in params:
        print(param)
    return params

train_step = tf.compile(OptimizerStep)

def ComputeLoss():
    model = MNIST_net(-1, -1, -1)

    model.initialize_input()

    X = tf.input([-1, -1], tf.float32)
    Y = tf.input([X.shape[0], 10], tf.float32)

    L = model.loss(X, Y)
    return L

compute_loss = tf.compile(ComputeLoss)

# Usage example:
model = MNIST_net(784, 64, 64)
opt = MNIST_Momentum_opt(model, 0.02, 0.99)

opt.initialize_parameters()

X = tf.tensor(np.random.randn(100, 784).astype(np.float32))
Y = tf.tensor(np.random.randn(100, 10).astype(np.float32))

loss_history = []
for i in range(1000):
    #optimization step
    new_params = train_step(*opt.create_input(X, Y))
    opt.update_parameters(new_params)
    #compute loss
    loss = compute_loss(*model.create_input(X, Y))
    print(loss.numpy)
    loss_history.append(loss.numpy)

# Plot loss history
import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()