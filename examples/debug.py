import numpy as np
import TensorFrost as tf

tf.initialize(tf.cpu)

learning_rate = 5.0

def leaky_relu(X):
	return tf.select(X > 0.0, X, 0.01 * X)

def softmax(X):
    exp = tf.exp(X)
    norm = tf.sum(exp, axis = 1)
    return exp / tf.reshape(norm, [exp.shape[0], 1])

def leaky_relu(X):
	return tf.select(X > 0.0, X, 0.01 * X)

def forward(W1, W2, b1, b2, X):
    L1 = leaky_relu(tf.matmul(X, W1) + b1)
    L2 = tf.matmul(L1, W2) + b2
    L3 = softmax(L2)
    return L1, L2, L3

def loss(Y, Yhat):
    Y *= 1.0 #tensor view bug
    #return tf.sum(tf.sum((Y - Yhat) ** 2.0)) / tf.float(Y.shape[1] * Y.shape[0])
    #cross entropy loss
    return tf.sum(tf.sum(Y * tf.log(Yhat) + (1.0 - Y) * tf.log(1.0 - Yhat))) / tf.float(Y.shape[1] * Y.shape[0])

def backward(W1, W2, b1, b2, L1, L2, L3, X, Y):
    #derivative of loss with respect to L3
    #dL3 = 2.0 * (L3 - Y) / tf.float(L3.shape[1] * L3.shape[0])
    ##derivative of loss with respect to softmax (L2)
    #i, j = dL3.indices
    #dij = tf.select(i == j, 1.0, 0.0)
    #dL2 = L3 

    #derivative of cross entropy loss with respect to L2
    dL2 = (L3 - Y) / tf.float(L3.shape[1] * L3.shape[0])
    dW2 = tf.matmul(tf.transpose(L1), dL2)
    db2 = tf.sum(dL2, axis = 0)
    dL1 = tf.matmul(dL2, tf.transpose(W2))
    dLeaky = tf.select(L1 > 0.0, 1.0, 0.01) * dL1
    dW1 = tf.matmul(tf.transpose(X), dLeaky)
    db1 = tf.sum(dLeaky, axis = 0)
    return dW1, dW2, db1, db2

def update(W1, W2, b1, b2, dW1, dW2, db1, db2, lr):
    W1 -= lr * dW1
    W2 -= lr * dW2
    b1 -= lr * db1
    b2 -= lr * db2
    return W1, W2, b1, b2

def step():
    #input weights and biases
    W1 = tf.input([-1, -1], tf.float32)
    In, Hidden = W1.shape
    W2 = tf.input([Hidden, -1], tf.float32)
    Out = W2.shape[1]
    b1 = tf.input([Hidden], tf.float32)
    b2 = tf.input([Out], tf.float32)

    #input data
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

    #Yhat = forward(W, Xbatch, b)
    #L = loss(Ybatch, Yhat)
    #dL2, dL1, dW, db = backward(W, Xbatch, b, Ybatch, Yhat)
    #W, b, = update(W, b, dW, db, learning_rate)

    L1, L2, Yhat = forward(W1, W2, b1, b2, Xbatch)
    L = loss(Ybatch, Yhat)
    dW1, dW2, db1, db2 = backward(W1, W2, b1, b2, L1, L2, Yhat, Xbatch, Ybatch)
    W1, W2, b1, b2 = update(W1, W2, b1, b2, dW1, dW2, db1, db2, learning_rate)

    return [L, W1, W2, b1, b2]

train_step = tf.compile(step)