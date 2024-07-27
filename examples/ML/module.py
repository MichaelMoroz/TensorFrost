import TensorFrost as tf
import math
import numpy as np

tf.initialize(tf.opengl)

def softmax(X):
    exp = tf.exp(X)
    return exp / tf.unsqueeze(tf.sum(exp))

def log_softmax(X):
    X = X - tf.unsqueeze(tf.max(X))
    return X - tf.log(tf.unsqueeze(tf.sum(tf.exp(X))))

def leaky_relu(X):
    return tf.select(X > 0.0, X, 0.01 * X)

def mul_bias(X, W):
    ids = tf.indices(list(X.shape[:-1]) + [W.shape[-2]])
    return tf.select(ids[-1] == X.shape[-1], 1.0, X[ids]) @ W

class MNIST_net(tf.Module):
    def __init__(self, N1, N2, N3):
        super().__init__()
        self.W1 = tf.Parameter([N1, N2], tf.float32)
        self.W2 = tf.Parameter([N2, N3], tf.float32)
        self.W3 = tf.Parameter([N3, 10], tf.float32)

    def forward(self, X):
        L1 = leaky_relu(mul_bias(X, self.W1))
        L2 = leaky_relu(mul_bias(L1, self.W2))
        return mul_bias(L2, self.W3)

    def prob_distr(self, X):
        return softmax(self.forward(X))

    def loss(self, X, Y):
        Yhat = self.forward(X)
        return tf.sum(tf.sum(-Y * log_softmax(Yhat))) / tf.float(math.prod(Y.shape))


lr = 0.002
decay = 0.99

def OptimizerStep():
    model = MNIST_net(-1, -1, -1)
    opt = tf.optimizers.adam(model, lr)
    opt.initialize_input()

    X = tf.input([-1, -1], tf.float32)
    Y = tf.input([X.shape[0], 10], tf.float32)

    L = opt.step(X, Y)

    params = opt.parameters()
    params.append(L)
    return params

train_step = tf.compile(OptimizerStep)

def ComputeForward():
    model = MNIST_net(-1, -1, -1)
    model.initialize_input()
    X = tf.input([-1, -1], tf.float32)
    return model.prob_distr(X)

compute_forward = tf.compile(ComputeForward)

# Load MNIST data
data = np.load('mnist.npz')

def image_to_vector(X):
    return np.reshape(X, (len(X), -1))         # Flatten: (N x 28 x 28) -> (N x 784)

Xtrain = image_to_vector(data['train_x'])  
Ytrain = np.zeros((Xtrain.shape[0], 10))
Ytrain[np.arange(Xtrain.shape[0]), data['train_y']] = 1.0

Xtest = image_to_vector(data['test_x'])     
Ytest = data['test_y']

batch_size = 1024
epochs = 100
iterations = Xtrain.shape[0] // batch_size

loss_curve = []

model = MNIST_net(784, 64, 64)
opt = tf.optimizers.adam(model, lr)
opt.initialize_parameters()

for i in range(epochs):
    avg_loss_tf = 0.0

    #shuffle offsets
    offsets = np.random.permutation(Xtrain.shape[0] // batch_size) * batch_size

    for j in range(iterations):
        offset = offsets[j]
        Xbatch = Xtrain[offset:offset + batch_size]
        Ybatch = Ytrain[offset:offset + batch_size]
        res = train_step(opt, Xbatch, Ybatch)
        new_params = res[:-1]
        opt.update_parameters(new_params)
        loss = res[-1]
        avg_loss_tf += loss.numpy
        loss_curve.append(loss.numpy)
        #print("Epoch: ", i, " Iteration: ", j, " Tf Loss: ", loss.numpy)

    print("Epoch: ", i, " Tf Loss: ", avg_loss_tf / iterations)

def test_accuracy(model, X, Y):
    Yhat = compute_forward(model, X)
    Yhatnp = Yhat.numpy
    Predict = np.argmax(Yhatnp, axis = 1)
    correct_tf = np.sum(Predict == Y)
    return correct_tf * 100.0 / len(Y)

test_accuracy_tf = test_accuracy(model, Xtest, Ytest)
print("Final Tf test accuracy: ", test_accuracy_tf, "%")

# Plot loss history
import matplotlib.pyplot as plt
plt.plot(loss_curve)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()
