import TensorFrost as tf
import numpy as np
import time
from tqdm import tqdm

tf.initialize(tf.opengl)

def log_softmax(X):
    X = X - tf.unsqueeze(tf.max(X))
    return X - tf.log(tf.unsqueeze(tf.sum(tf.exp(X))))

def leaky_relu(X):
    return tf.select(X > 0.0, X, 0.01 * X)

def GELU(X):
    return 0.5*X*(1.0 + tf.tanh(np.sqrt(2.0/np.pi) * (X + 0.044715 * (X * X * X))))

def GLIN(X):
    return 0.5*X + 0.5*X/(1.0 + tf.exp(-X))

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
        self.kernels1 = 16
        self.kernels2 = 64
        self.layer1 = 256
        self.conv1 = tf.Parameter([self.kernels1, 1, self.kernel_size, self.kernel_size], tf.float32, random_scale = np.sqrt(0.1 / (self.kernel_size ** 2 * 1)))
        self.conv1_bias = tf.Parameter([self.kernels1], tf.float32, random_scale = 0.0)
        self.conv2 = tf.Parameter([self.kernels2, self.kernels1, self.kernel_size, self.kernel_size], tf.float32, random_scale = np.sqrt(0.1 / (self.kernel_size ** 2 * self.kernels1)))
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
        tf.region_begin('Forward')
        X = tf.reshape(X, [X.shape[0], self.resolution, self.resolution, 1])
        X = self.max_pool2d(self.conv2d(X, self.conv1, self.conv1_bias))
        X = GELU(X)
        X = self.max_pool2d(self.conv2d(X, self.conv2, self.conv2_bias))
        X = GELU(X)
        X = tf.reshape(X, [X.shape[0], self.fc1.shape[0]])
        X = GELU(X @ self.fc1 + self.fc1_bias)
        X = X @ self.fc2 + self.fc2_bias
        tf.region_end('Forward')
        return X

    def loss(self, X, Y):
        Yhat = self.forward(X)
        loss = tf.mean(tf.sum(-Y * log_softmax(Yhat)))
        return loss
        
lr = 0.0005

def GetModelOptimizer(is_compiler = False, learning_rate = 0.0005):
    model = MNIST_net(is_compiler = is_compiler)
    opt = tf.optimizers.adam(model, learning_rate = learning_rate)
    return model, opt

def OptimizerStep():
    X = tf.input([-1, -1], tf.float32)
    Y = tf.input([-1, 10], tf.float32)
    
    info = tf.input([-1], tf.float32)
    offset = tf.int(info[0])
    batch_size = tf.int(info[1])
    learning_rate = info[2]

    model, opt = GetModelOptimizer(is_compiler = True, learning_rate = learning_rate)
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

Xtest = image_to_vector(data['test_x'])[0:1000]
Ytest = data['test_y'][0:1000]

batch_size = 128
epochs = 10
iterations = Xtrain.shape[0] // batch_size
smoothing = 5.0 / iterations
print("Iterations per epoch: ", iterations)

model, opt = GetModelOptimizer(is_compiler = False, learning_rate = lr)
opt.initialize_parameters()

Xtf = tf.tensor(Xtrain)
Ytf = tf.tensor(Ytrain)
Xtest = tf.tensor(Xtest)

def test_accuracy(model, X, Y):
    Yhat = compute_forward(model, X)
    Predict = np.argmax(Yhat.numpy, axis = 1)
    correct_tf = np.sum(Predict == Y)
    return correct_tf * 100.0 / len(Y)

curves = {'loss': [], 'avg_loss': [], 'accuracy': []}
progress_bar = tqdm(range(epochs * iterations))
avg_loss = 0.0
for i in progress_bar:
    batch = i % iterations
    if(batch == 0):
        #shuffle offsets
        offsets = np.random.permutation(Xtrain.shape[0] // batch_size) * batch_size + np.random.randint(batch_size)
        accuracy = test_accuracy(model, Xtest, Ytest)
        curves['accuracy'].append([i, accuracy])

    if(i == 0): tf.renderdoc_start_capture()

    res = train_step(Xtf, Ytf, [offsets[batch], batch_size, lr], opt)
    opt.update_parameters(res[:-1])
    loss = float(res[-1].numpy[0])
    curves['loss'].append([i, loss])
    if(i == 0): avg_loss = loss
    avg_loss = (1.0 - smoothing) * avg_loss + smoothing * loss
    curves['avg_loss'].append([i, avg_loss])
    progress_bar.set_postfix(loss = avg_loss, accuracy = accuracy)

    if(i == 0): tf.renderdoc_end_capture()
    

test_accuracy_tf = test_accuracy(model, Xtest, Ytest)
print("Final Tf test accuracy: ", test_accuracy_tf, "%")

#accuracy_on_train = test_accuracy(model, Xtrain, data['train_y'])
#print("Final Tf train accuracy: ", accuracy_on_train, "%")

# Plot loss history
import matplotlib.pyplot as plt

curves = {k: np.array(v) for k, v in curves.items()}
plt.plot(curves['loss'][:, 0], curves['loss'][:, 1], label = 'Loss')
plt.plot(curves['avg_loss'][:, 0], curves['avg_loss'][:, 1], label = 'Avg Loss')
plt.xlabel('Iteration')
plt.legend()
plt.grid()
plt.yscale('log')
plt.show()

#Plot accuracy history
plt.plot(curves['accuracy'][:, 0], curves['accuracy'][:, 1])
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.grid()