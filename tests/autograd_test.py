import TensorFrost as tf
import numpy as np
import torch
import torch.nn.functional as F
import unittest

tf.initialize(tf.opengl)

class ADGrad(tf.Module):
    def __init__(self, net: tf.Module):
        super().__init__()
        self.net = net
        self.grad = tf.ParameterArray()
        for i, param in enumerate(self.net.parameters()):
            self.grad[i] = tf.Parameter(param.shape, param.dtype)
           
    def assert_parameters(self):
        for i, param in enumerate(self.net.parameters()):
            self.grad[i] = tf.assert_tensor(self.grad[i], param.shape, tf.float32)
        
    def compute(self, L):
        for i, param in enumerate(self.net.parameters()):
            self.grad[i] = tf.grad(L, param)
    

def GELU(X):
    return 0.5*X*(1.0 + tf.tanh(np.sqrt(2.0/np.pi) * (X + 0.044715 * (X * X * X))))

def log_softmax(X):
    X = X - tf.unsqueeze(tf.max(X))
    return X - tf.log(tf.unsqueeze(tf.sum(tf.exp(X))))

#tensorfrost module
class TestModule1(tf.Module):
    def __init__(self, input_resolution = 28, output_size = 10):
        super().__init__()
        self.resolution = input_resolution
        self.kernel_size = 5
        self.kernel_rad = self.kernel_size // 2
        self.res1 = (self.resolution - self.kernel_size + 1)
        self.res1p = self.res1 // 2
        self.res2 = (self.res1p - self.kernel_size + 1)
        self.res2p = self.res2 // 2
        self.dense_res = self.res2p ** 2
        #self.dense_res = self.res1 ** 2
        self.kernels1 = 16
        self.kernels2 = 64
        self.layer1 = 64
        self.conv1 = tf.Parameter([self.kernels1, 1, self.kernel_size, self.kernel_size], tf.float32, random_scale = np.sqrt(0.1 / (self.kernel_size ** 2 * 1)))
        #self.conv1_bias = tf.Parameter([self.kernels1], tf.float32, random_scale = 0.0)
        self.conv2 = tf.Parameter([self.kernels2, self.kernels1, self.kernel_size, self.kernel_size], tf.float32, random_scale = np.sqrt(0.1 / (self.kernel_size ** 2 * self.kernels1)))
        #self.conv2_bias = tf.Parameter([self.kernels2], tf.float32, random_scale = 0.0)
        self.fc1 = tf.Parameter([self.kernels2 * self.dense_res, self.layer1], tf.float32)
        #self.fc1 = tf.Parameter([self.kernels1 * self.dense_res, self.layer1], tf.float32)
        self.fc1_bias = tf.Parameter([self.layer1], tf.float32, random_scale = 0.0)
        self.fc2 = tf.Parameter([self.layer1, output_size], tf.float32)
        self.fc2_bias = tf.Parameter([output_size], tf.float32, random_scale = 0.0)

        self.sum_layer1 = tf.Parameter([16, 8], tf.float32, random_scale = 0.1)
        self.sum_layer1_bias = tf.Parameter([8], tf.float32, random_scale = 0.0)

        self.sum_layer2 = tf.Parameter([8, 16], tf.float32, random_scale = 0.1)
        self.sum_layer2_bias = tf.Parameter([16], tf.float32, random_scale = 0.0)

    def assert_parameters(self):
        self.fc2 = tf.assert_tensor(self.fc2, [self.fc1.shape[1], self.fc2.shape[1]], tf.float32)

    def conv2d(self, X, W):
        N, CIN, HI, WI = X.shape
        COUT, CIN, h, w = W.shape
        bi, cout, wi, hi, cin, it = tf.indices([N, COUT, HI - h + 1, WI - w + 1, CIN, h * w])
        i, j = it%w, it/w
        conv = tf.sum(tf.sum(X[bi, cin, wi + i, hi + j] * W[cout, cin, i, j]))
        return conv
    
    def max_pool2d(self, X):
        bi, ci, wi, hi, i, j = tf.indices([X.shape[0], X.shape[1], X.shape[2] / 2, X.shape[3] / 2, 2, 2])
        return tf.max(tf.max(X[bi, ci, 2 * wi + i, 2 * hi + j]))
    
    def forward(self, x):
        tf.region_begin('Forward')
        x = tf.reshape(x, [x.shape[0], 1, self.resolution, self.resolution])
        x = self.conv2d(x, self.conv1)
        x = self.max_pool2d(x)
        x = GELU(x)
        x = self.max_pool2d(self.conv2d(x, self.conv2))
        x = GELU(x)
        x = tf.reshape(x, [x.shape[0], self.fc1.shape[0]])
        x = GELU(x @ self.fc1 + self.fc1_bias)
        x = tf.reshape(x, [x.shape[0], 64//16, 16])
        r = tf.tanh(x @ self.sum_layer1 + self.sum_layer1_bias)
        r = tf.sum(r, axis=1)
        r = tf.unsqueeze(r, axis=1)
        x = x + tf.tanh(r @ self.sum_layer2 + self.sum_layer2_bias)
        x = tf.reshape(x, [x.shape[0], 64])
        x = x @ self.fc2 + self.fc2_bias
        tf.region_end('Forward')
        return x

    def loss(self, X, Y):
        Yhat = self.forward(X)
        loss = tf.mean(tf.sum(-Y * log_softmax(Yhat)))
        #loss = tf.mean(tf.mean((Y - Yhat)**2.0))
        return loss, Yhat

def log_softmax_torch(X):
    X = X - torch.unsqueeze(torch.max(X), dim=-1)
    return X - torch.log(torch.unsqueeze(torch.sum(torch.exp(X), dim=-1), dim=-1))

def GELU_torch(X):
    return 0.5*X*(1.0 + torch.tanh(np.sqrt(2.0/np.pi) * (X + 0.044715 * (X * X * X))))

#pytorch module
class TestModule2(torch.nn.Module):
    def __init__(self, input_resolution = 28, output_size = 10):
        super().__init__()
        self.resolution = input_resolution
        self.kernel_size = 5
        self.kernel_rad = self.kernel_size // 2
        self.res1 = (self.resolution - self.kernel_size + 1)
        self.res1p = self.res1 // 2
        self.res2 = (self.res1p - self.kernel_size + 1)
        self.res2p = self.res2 // 2
        self.dense_res = self.res2p ** 2
        #self.dense_res = self.res1 ** 2
        self.kernels1 = 16
        self.kernels2 = 64
        self.layer1 = 64
        self.conv1 = torch.nn.Parameter(0.1*torch.randn(self.kernels1, 1, self.kernel_size, self.kernel_size))
        #self.conv1_bias = torch.nn.Parameter(torch.randn(self.kernels1))
        self.conv2 = torch.nn.Parameter(0.1*torch.randn(self.kernels2, self.kernels1, self.kernel_size, self.kernel_size))
        #self.conv2_bias = torch.nn.Parameter(torch.randn(self.kernels2))
        self.fc1 = torch.nn.Parameter(0.1*torch.randn(self.kernels2 * self.res2p ** 2, self.layer1))
        #self.fc1 = torch.nn.Parameter(0.1*torch.randn(self.kernels1 * self.dense_res, self.layer1))
        self.fc1_bias = torch.nn.Parameter(torch.randn(self.layer1))
        self.fc2 = torch.nn.Parameter(0.1*torch.randn(self.layer1, output_size))
        self.fc2_bias = torch.nn.Parameter(torch.randn(output_size))

        self.sum_layer1 = torch.nn.Parameter(0.1*torch.randn(16, 8))
        self.sum_layer1_bias = torch.nn.Parameter(torch.randn(8))
        self.sum_layer2 = torch.nn.Parameter(0.1*torch.randn(8, 16))
        self.sum_layer2_bias = torch.nn.Parameter(torch.randn(16))


    def conv2d(self, X, W):
        conv = torch.nn.functional.conv2d(X, W)
        return conv

    def forward(self, x):
        x = x.reshape([-1, 1, self.resolution, self.resolution])
        x = self.conv2d(x, self.conv1)
        x = F.max_pool2d(x, 2)
        x = GELU_torch(x)
        x = F.max_pool2d(self.conv2d(x, self.conv2), 2)
        x = GELU_torch(x)
        x = x.reshape(-1, self.fc1.shape[0])
        x = GELU_torch(x @ self.fc1 + self.fc1_bias)
        x = x.reshape(-1, 64//16, 16)
        r = torch.tanh(x @ self.sum_layer1 + self.sum_layer1_bias)
        r = r.sum(dim=1)
        r = torch.unsqueeze(r, dim=1)
        x = x + torch.tanh(r @ self.sum_layer2 + self.sum_layer2_bias)
        x = x.reshape(-1, 64)
        x = x @ self.fc2 + self.fc2_bias
        return x
    
    def loss(self, X, Y):
        Y_pred = self.forward(X)
        #return torch.mean((Y - Y_pred)**2.0)
        return torch.mean(torch.sum(-Y * log_softmax_torch(Y_pred), dim = -1)), Y_pred
        #return torch.mean(torch.mean((Y - Y_pred)**2.0)), Y_pred
    
#create the modules
# pytorch
torch.manual_seed(1)
model_torch = TestModule2()

# tensorfrost
model_tf = TestModule1()
tf_grads = ADGrad(model_tf)
tf_grads.initialize_parameters()

#copy torch weights to tensorfrost
params = tf_grads.parameters()
for i, param in enumerate(model_torch.parameters()):
    params[i] = param.detach().numpy()
    params[i] = tf.tensor(params[i])
tf_grads.update_parameters(params)

#create random MNIST data
ImSize = 28
N = 64

#compute tensorfrost gradients
def grad_computer():
    model = TestModule1()
    grads = ADGrad(model)
    grads.initialize_input()

    x = tf.input([N, ImSize**2], tf.float32)
    y = tf.input([N, 10], tf.float32)
    loss, yhat = model.loss(x, y)
    grads.compute(loss)
    params = grads.parameters()
    params.append(loss)
    params.append(yhat)
    return params


class TestAutograd(unittest.TestCase):
    def test_autograd(self):
        grad_compute = tf.compile(grad_computer)

        x = np.random.randn(N, ImSize**2).astype(np.float32)
        x_tf = tf.tensor(x)
        x_torch = torch.tensor(x)

        #random categorical data for 10 classes using numpy
        y = np.random.randint(0, 10, N).astype(np.int32)
        y = np.eye(10)[y]
        y_tf = tf.tensor(y)
        y_torch = torch.tensor(y)

        all_params = grad_compute(tf_grads, x_tf, y_tf)
        tf_grads.update_parameters(all_params[:-2])
        loss_tf = all_params[-2]
        yhat_tf = all_params[-1]

        model_torch.zero_grad()
        loss_torch, yhat = model_torch.loss(x_torch, y_torch)
        loss_torch.backward()
        
        self.assertTrue(np.abs(loss_tf.numpy - loss_torch.item()) < 1e-5)
        self.assertTrue(np.allclose(yhat_tf.numpy, yhat.detach().numpy(), atol=1e-5))
        for i, param in enumerate(model_torch.parameters()):
            self.assertTrue(np.allclose(tf_grads.grad[i].numpy, param.grad.detach().numpy(), atol=1e-4))
            self.assertTrue(np.allclose(tf_grads.net.parameters()[i].numpy, param.detach().numpy(), atol=1e-4))