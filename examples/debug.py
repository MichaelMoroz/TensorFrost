import TensorFrost as tf
import numpy as np
import matplotlib.pyplot as plt
import time

tf.initialize(tf.cpu)

def leaky_relu(X):
	return tf.select(X > 0.0, X, 0.01 * X)

def forward(W, X, b):
    L1 = tf.matmul(W, X)
    L1 = L1 + b[L1.indices[1]]
    L2 = leaky_relu(L1)
    return L2

def loss(Y, Yhat):
    return tf.sum(tf.sum((Y - Yhat) ** 2.0))

def backward(W, X, b, Y, Yhat):
    dL2 = 2.0 * (Yhat - Y)
    dL1 = tf.select(Yhat > 0.0, dL2, 0.01 * dL2)
    dW = tf.matmul(dL1, X.T)
    db = tf.sum(dL1, axis = 1)
    return dW, db

def update(W, X, b, dW, db, lr):
    W -= lr * dW
    b -= lr * db
    return W, b

def step():
    W = tf.input([-1, -1], tf.float32)
    Out, In = W.shape
    X = tf.input([In, -1], tf.float32)
    b = tf.input([Out], tf.float32)
    Samples = X.shape[1]
    Y = tf.input([Out, Samples], tf.float32)
    Yhat = forward(W, X, b)
    L = loss(Y, Yhat)
    dW, db = backward(W, X, b, Y, Yhat)
    W, b = update(W, X, b, dW, db, 0.01)

    return [Yhat, L, W, b]

fwd = tf.compile(step)

W = np.random.randn(2, 2)
b = np.random.randn(2)
X = np.random.randn(2, 2)
Y = np.random.randn(2, 2)

Wtf = tf.tensor(W)
btf = tf.tensor(b)
Xtf = tf.tensor(X)
Ytf = tf.tensor(Y)

Yhat, L, Wtf, btf = fwd(Wtf, Xtf, btf, Ytf)

print(Yhat.numpy)
print(L.numpy)
