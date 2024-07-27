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


lr = 0.004
decay = 0.99

def OptimizerStep():
    model = MNIST_net(-1, -1, -1)
    opt = tf.optimizers.sgd(model, lr)
    #opt = tf.optimizers.adam(model, lr)
    #opt = tf.optimizers.rmsprop(model, lr, decay)
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

# Usage example:
model = MNIST_net(784, 64, 64)
opt = tf.optimizers.sgd(model, lr)

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
epochs = 30
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
        #print("Epoch: ", i, " Iteration: ", j, " Tf Loss: ", loss.numpy)

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

