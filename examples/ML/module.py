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
            return tf.tensor(np.random.randn(*param.shape).astype(np.float32) / math.sqrt(param.shape[0]))
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

    def create_input(self, *args):
        inputs = self.get_all_parameters()
        inputs += args
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

    def loss(self, X, Y):
        raise NotImplementedError
    
    def forward(self, X):
        raise NotImplementedError

def softmax(X):
    exp = tf.exp(X)
    return exp / tf.unsqueeze(tf.sum(exp))

def log_softmax(X):
    Xmax = tf.max(X, axis=-1)
    X = X - tf.unsqueeze(Xmax)
    return X - tf.log(tf.unsqueeze(tf.sum(tf.exp(X))))

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
        return mul_bias(L2, self.W3)
    
    def smax(self, X):
        return softmax(self.forward(X))

    def loss(self, X, Y):
        Yhat = self.forward(X)
        #return tf.sum(tf.sum( - Y * tf.log(Yhat + 1e-3) - (1.0 - Y) * tf.log(1.0 - Yhat + 1e-3))) / tf.float(math.prod(Y.shape))
        return tf.sum(tf.sum(-Y * log_softmax(Yhat))) / tf.float(math.prod(Y.shape))
    
class MomentumOpt(Module):
    def __init__(self, net: Module, learning_rate: float, momentum: float):
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

class RMSPropOpt(Module):
    def __init__(self, net: Module, learning_rate: float, decay: float, epsilon: float = 1e-6):
        super().__init__()
        self.register_module('net', net)
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.v = self.register_parameter_array('v')
        for i, (name, param) in enumerate(net._parameters.items()):
            self.v[i] = Parameter(param.shape, param.dtype, False)

    def assert_parameters(self):
        for i, (name, param) in enumerate(self.net._parameters.items()):
            self.v[i] = tf.assert_tensor(self.v[i], param.shape, tf.float32)

    def step(self, X, Y):
        L = self.net.loss(X, Y)
        for i, (name, param) in enumerate(self.net._parameters.items()):
            v = self.v[i]
            grad = tf.grad(L, param)
            grad = tf.clamp(grad, -0.1, 0.1)
            v = self.decay * v + (1.0 - self.decay) * (grad * grad)
            param -= self.learning_rate * grad / (tf.sqrt(v) + self.epsilon)
            self.v[i] = v
            self.net._parameters[name] = param
        return L

lr = 0.01
decay = 0.99

def OptimizerStep():
    model = MNIST_net(-1, -1, -1)
    opt = RMSPropOpt(model, lr, decay)

    opt.initialize_input()

    X = tf.input([-1, -1], tf.float32)
    Y = tf.input([X.shape[0], 10], tf.float32)

    L = opt.step(X, Y)
    params = opt.get_all_parameters()
    for param in params:
        print(param)

    params.append(L)
    return params

train_step = tf.compile(OptimizerStep)

def ComputeForward():
    model = MNIST_net(-1, -1, -1)

    model.initialize_input()

    X = tf.input([-1, -1], tf.float32)

    Y = model.smax(X)
    return Y

compute_forward = tf.compile(ComputeForward)

# Usage example:
model = MNIST_net(784, 64, 64)
opt = RMSPropOpt(model, lr, decay)

opt.initialize_parameters()

# Load MNIST data
data = np.load('mnist.npz')

def image_to_vector(X):
    X = np.reshape(X, (len(X), -1))         # Flatten: (N x 28 x 28) -> (N x 784)
    return X

data = np.load('mnist.npz')
Xtrain = image_to_vector(data['train_x'])   # (60000 x 784)
Ytrain = data['train_y']                    # (60000)
Xtest = image_to_vector(data['test_x'])     # (10000 x 784)
Ytest = data['test_y']        

Xsamples = Xtrain
Ysamples = np.zeros((Xsamples.shape[0], 10))
Ysamples[np.arange(Xsamples.shape[0]), Ytrain] = 1.0
Xtf = tf.tensor(Xsamples)
Ytf = tf.tensor(Ysamples)

def test_accuracy(model, X, Y):
    Yhat = compute_forward(*model.create_input(tf.tensor(X)))
    Yhatnp = Yhat.numpy
    Predict = np.argmax(Yhatnp, axis = 1)
    correct_tf = np.sum(Predict == Y)
    return correct_tf * 100.0 / len(Y)

batch_size = 1024
epochs = 100
iterations = Xsamples.shape[0] // batch_size

loss_curve = []

for i in range(epochs):
    avg_loss_tf = 0.0

    #shuffle offsets
    offsets = np.random.permutation(Xsamples.shape[0] // batch_size) * batch_size 

    for j in range(iterations):
        offset = offsets[j]
        Xbatch = Xsamples[offset:offset + batch_size]
        Ybatch = Ysamples[offset:offset + batch_size]
        Xbatch_tf = tf.tensor(Xbatch)
        Ybatch_tf = tf.tensor(Ybatch)

        res = train_step(*opt.create_input(Xbatch_tf, Ybatch_tf))
        new_params = res[:-1]
        opt.update_parameters(new_params)
        loss = res[-1]
        avg_loss_tf += loss.numpy
        loss_curve.append(loss.numpy)    

    print("Epoch: ", i, " Tf Loss: ", avg_loss_tf / iterations)

test_accuracy_tf = test_accuracy(model, Xtest, Ytest)
print("Final Tf test accuracy: ", test_accuracy_tf, "%")

# Plot loss history
import matplotlib.pyplot as plt
plt.plot(loss_curve)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()

#fashion mnist
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#digits mnist
#class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

