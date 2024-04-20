import numpy as np
import TensorFrost as tf

tf.initialize(tf.cpu)
def leaky_relu(X):
	return tf.select(X > 0.0, X, 0.01 * X)

def softmax(X):
    exp = tf.exp(X)
    norm = tf.sum(exp, axis = 1)
    return exp / tf.reshape(norm, [norm.shape[0], 1])

def forward(W, X, b):
    L1 = leaky_relu(tf.matmul(X, W) + b)
    return L1

def loss(Y, Yhat):
    Y *= 1.0 #tensor view bug
    return tf.sum(tf.sum((Y - Yhat) ** 2.0)) / tf.float(Y.shape[1] * Y.shape[0])

def backward(W, X, b, Y, Yhat):
    dL2 = 2.0 * (Yhat - Y)
    dL1 = dL2 * tf.select(Yhat > 0.0, 1.0, 0.01)
    dW = tf.matmul(tf.transpose(X), dL1)
    db = tf.sum(dL1, axis = 0)
    return dL2, dL1, dW, db

def update(W, b,  dW, db, lr):
    W -= lr * dW
    b -= lr * db
    return W, b

def step():
    W = tf.input([-1, -1], tf.float32)
    In, Out = W.shape
    b = tf.input([Out], tf.float32)

    X = tf.input([-1, In], tf.float32)
    Y = tf.input([-1, Out], tf.float32)
    
    info = tf.input([2], tf.int32)
    offset = info[0]
    batch_size = info[1]

    #TODO: implement slicing instead of this crap
    i, j = tf.indices([batch_size, In])
    Xbatch = X[i + offset, j]
    i, j = tf.indices([batch_size, Out])
    Ybatch = Y[i + offset, j]

    Yhat = forward(W, Xbatch, b)
    L = loss(Ybatch, Yhat)
    dL2, dL1, dW, db = backward(W, Xbatch, b, Ybatch, Yhat)
    W, b, = update(W, b, dW, db, 0.0001)

    return [L, W, b, dW, db, dL2, dL1]

train_step = tf.compile(step)