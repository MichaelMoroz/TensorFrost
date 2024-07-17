import TensorFrost as tf
import math
import numpy as np

tf.initialize(tf.opengl)


def softmax(X):
    exp = tf.exp(X)
    return exp / tf.unsqueeze(tf.sum(exp))

def leaky_relu(X):
    return tf.select(X > 0.0, X, 0.01 * X)

def mul_bias(X, W):
    ids = tf.indices(list(X.shape[:-1]) + [W.shape[-2]])
    return tf.select(ids[-1] == X.shape[-1], 1.0, X[ids]) @ W

class MNIST_net(tf.Module):
    def __init__(self, N1: tf.argument, N2: tf.argument, N3: tf.argument):
        self.W1 = tf.parameter([N1, N2], tf.float32)
        self.W2 = tf.parameter([N2, N3], tf.float32)
        self.W3 = tf.parameter([N3, 10], tf.float32)

    def forward(self, X):
        L1 = leaky_relu(mul_bias(X, self.W1))
        L2 = leaky_relu(mul_bias(L1, self.W2))
        return softmax(mul_bias(L2, self.W3))

    def loss(self, X, Y):
        Yhat = self.forward(X)
        return tf.mean(tf.mean( - Y * tf.log(Yhat + 1e-3) - (1.0 - Y) * tf.log(1.0 - Yhat + 1e-3)))

class MNIST_ADAM_opt(tf.Module):
    def __init__(self, net: MNIST_net, learning_rate: float):
        self.net = net
        self.learning_rate = learning_rate
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m = []
        self.v = []
        for W in self.net.parameters():
            self.m.append(tf.parameter(W.shape, tf.float32))
            self.v.append(tf.parameter(W.shape, tf.float32))

    def step(self, X, Y):
        L = self.net.loss(X, Y)
        for i, W in enumerate(self.net.parameters()):
            grad = tf.grad(L, W)
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad * grad
            m_hat = self.m[i] / (1.0 - self.beta1)
            v_hat = self.v[i] / (1.0 - self.beta2)
            W -= self.learning_rate * m_hat / (tf.sqrt(v_hat) + self.epsilon)
        return L


def OptimizerStep():
    model = tf.module_input(MNIST_ADAM_opt)

    X = tf.input([-1, 784], tf.float32)
    Y = tf.input([-1, 10], tf.float32)

    return model.step(X, Y)

train_step = tf.compile(OptimizerStep)