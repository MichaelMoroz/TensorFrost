import TensorFrost as tf
import numpy as np
import time

tf.initialize(tf.opengl)

def log_softmax(X):
    X = X - tf.unsqueeze(tf.max(X))
    return X - tf.log(tf.unsqueeze(tf.sum(tf.exp(X))))

def leaky_relu(X):
    return tf.select(X > 0.0, X, 0.01 * X)

class MNIST_net(tf.Module):
    def __init__(self, input_resolution = 28, output_size = 10, is_compiler = False):
        super().__init__()
        self.resolution = input_resolution
        self.kernel_size = 5
        self.kernel_rad = self.kernel_size // 2
        self.res1 = (self.resolution - self.kernel_size + 1)
        self.res1p = self.res1 // 2
        self.res2 = (self.res1p - self.kernel_size + 1)
        self.res2p = self.res2 // 2
        self.kernels1 = 4
        self.kernels2 = 8
        self.layer1 = 64  
        self.conv1 = tf.Parameter([self.kernels1, 1, self.kernel_size, self.kernel_size], tf.float32, random_scale = np.sqrt(2.0 / (self.kernel_size ** 2 * 1)))
        self.conv1_bias = tf.Parameter([self.kernels1], tf.float32, random_scale = 0.0)
        self.conv2 = tf.Parameter([self.kernels2, self.kernels1, self.kernel_size, self.kernel_size], tf.float32, random_scale = np.sqrt(2.0 / (self.kernel_size ** 2 * self.kernels1)))
        self.conv2_bias = tf.Parameter([self.kernels2], tf.float32, random_scale = 0.0)
        self.fc1 = tf.Parameter([self.kernels2 * self.res2p ** 2, self.layer1], tf.float32)
        self.fc1_bias = tf.Parameter([self.layer1], tf.float32, random_scale = 0.0)
        self.fc2 = tf.Parameter([self.layer1, output_size], tf.float32)
        self.fc2_bias = tf.Parameter([output_size], tf.float32, random_scale = 0.0)

    def assert_parameters(self):
        self.fc2 = tf.assert_tensor(self.fc2, [self.fc1.shape[1], self.fc2.shape[1]], tf.float32)

    def conv2d(self, X, W, b):
        bi, wi, hi, cout, cin, it = tf.indices([X.shape[0], X.shape[1] - W.shape[2] + 1, X.shape[2] - W.shape[3] + 1, W.shape[0], W.shape[1], W.shape[2] * W.shape[3]])
        i, j = it%W.shape[2], it/W.shape[2]
        conv = tf.sum(tf.sum(X[bi, wi + i, hi + j, cin] * W[cout, cin, i, j]))
        return conv + b
    
    def max_pool2d(self, X):
        bi, wi, hi, ci, i, j = tf.indices([X.shape[0], X.shape[1] / 2, X.shape[2] / 2, X.shape[3], 2, 2])
        return tf.max(tf.max(X[bi, 2 * wi + i, 2 * hi + j, ci]))
    
    def forward(self, X):
        X = tf.reshape(X, [X.shape[0], self.resolution, self.resolution, 1])
        X = self.max_pool2d(self.conv2d(X, self.conv1, self.conv1_bias))
        X = leaky_relu(X)
        X = self.max_pool2d(self.conv2d(X, self.conv2, self.conv2_bias))
        X = leaky_relu(X)
        X = tf.reshape(X, [X.shape[0], self.fc1.shape[0]])
        X = leaky_relu(X @ self.fc1 + self.fc1_bias)
        X = X @ self.fc2 + self.fc2_bias
        return X

    def loss(self, X, Y):
        Yhat = self.forward(X)
        return tf.mean(tf.sum(-Y * log_softmax(Yhat)))

lr = 0.0005

def OptimizerStep():
    X = tf.input([-1, -1], tf.float32)
    Y = tf.input([-1, 10], tf.float32)
    
    info = tf.input([-1], tf.float32)
    offset = tf.int(info[0])
    batch_size = tf.int(info[1])
    learning_rate = info[2]

    model = MNIST_net(is_compiler = True)
    opt = tf.optimizers.adam(model, learning_rate)
    opt.initialize_input()

    #TODO: implement slicing instead of this crap
    i, j = tf.indices([batch_size, X.shape[1]])
    Xbatch = X[i + offset, j]
    i, j = tf.indices([batch_size, Y.shape[1]])
    Ybatch = Y[i + offset, j]

    L = opt.step(Xbatch, Ybatch)

    params = opt.parameters()
    params.append(L)
    return params

train_step = tf.compile(OptimizerStep)

def ComputeForward():
    model = MNIST_net(is_compiler = True)
    model.initialize_input()
    X = tf.input([-1, -1], tf.float32)
    return model.forward(X)

compute_forward = tf.compile(ComputeForward)

# Load MNIST data
data = np.load('mnist.npz')

def image_to_vector(X):
    return np.reshape(X, (len(X), -1))         # Flatten: (N x 28 x 28) -> (N x 784)

Xtrain = image_to_vector(data['train_x'])  
Ytrain = np.zeros((Xtrain.shape[0], 10))
Ytrain[np.arange(Xtrain.shape[0]), data['train_y']] = 1.0

Xtest = image_to_vector(data['test_x'])#[0:3000]
Ytest = data['test_y']#[0:3000]

batch_size = 128
epochs = 50
iterations = Xtrain.shape[0] // batch_size
print("Iterations per epoch: ", iterations)

model = MNIST_net()
opt = tf.optimizers.adam(model, lr)
opt.initialize_parameters()

Xtf = tf.tensor(Xtrain)
Ytf = tf.tensor(Ytrain)
Xtest = tf.tensor(Xtest)

init_time = time.time()

def test_accuracy(model, X, Y):
    Yhat = compute_forward(model, X)
    Predict = np.argmax(Yhat.numpy, axis = 1)
    correct_tf = np.sum(Predict == Y)
    return correct_tf * 100.0 / len(Y)

from tqdm import tqdm

loss_curve = []
accuracy_curve = []
for i in range(epochs):
    avg_loss_tf = 0.0

    #shuffle offsets
    offsets = np.random.permutation(Xtrain.shape[0] // batch_size) * batch_size + np.random.randint(batch_size)

    for j in range(iterations):
        if(j == 0): tf.renderdoc_start_capture()
        res = train_step(Xtf, Ytf, [offsets[j], batch_size, lr], opt)
        opt.update_parameters(res[:-1])
        loss = res[-1].numpy
        avg_loss_tf += loss
        loss_curve.append(loss)
        #print("Epoch: ", i, " Iteration: ", j, " Loss: ", loss)
        if(j == 0): tf.renderdoc_end_capture()

    accuracy = test_accuracy(model, Xtest, Ytest)
    accuracy_curve.append(accuracy)
    print("Epoch: ", i, " Loss: ", avg_loss_tf / iterations, " Test accuracy: ", accuracy, "%")

test_accuracy_tf = test_accuracy(model, Xtest, Ytest)
print("Final Tf test accuracy: ", test_accuracy_tf, "%")

#accuracy_on_train = test_accuracy(model, Xtrain, data['train_y'])
#print("Final Tf train accuracy: ", accuracy_on_train, "%")

print("Iterations per second: ", iterations * epochs / (time.time() - init_time))

# Plot loss history
import matplotlib.pyplot as plt
plt.plot(loss_curve)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid()
plt.show()

#Plot accuracy history
plt.plot(accuracy_curve)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.ylim(70, 95)
plt.show()
