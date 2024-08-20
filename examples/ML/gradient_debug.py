import TensorFrost as tf
import numpy as np
import torch
import time
from tqdm import tqdm

tf.initialize(tf.cpu)

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
    

N = 10

#tensorfrost module
class TestModule1(tf.Module):
    def __init__(self):
        super().__init__()
        # self.fc1 = tf.Parameter([N, N], tf.float32) 
        # self.fc1_bias = tf.Parameter([N], tf.float32)
        self.fc2 = tf.Parameter([N, N], tf.float32)
        self.fc2_bias = tf.Parameter([N], tf.float32)
   
    def forward(self, x):
        # x = x @ self.fc1 + self.fc1_bias
        # x = tf.tanh(x)
        x = x @ self.fc2 + self.fc2_bias
        return x

    def loss(self, X, Y):
        Y_pred = self.forward(X)
        return tf.sum((Y - Y_pred)**2.0)
    
#pytorch module
class TestModule2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.fc1 = torch.nn.Parameter(torch.randn(N, N))
        # self.fc1_bias = torch.nn.Parameter(torch.randn(N))
        self.fc2 = torch.nn.Parameter(torch.randn(N, N))
        self.fc2_bias = torch.nn.Parameter(torch.randn(N))

    def forward(self, x):
        # x = x @ self.fc1 + self.fc1_bias
        # x = torch.tanh(x)
        x = x @ self.fc2 + self.fc2_bias
        return x
    
    def loss(self, X, Y):
        Y_pred = self.forward(X)
        return torch.sum((Y - Y_pred)**2.0)
    
#create the modules
# pytorch
torch.manual_seed(0)
model_torch = TestModule2()

# tensorfrost
model_tf = TestModule1()
tf_grads = ADGrad(model_tf)
tf_grads.initialize_parameters()
# model_tf.fc1 = tf.tensor(model_torch.fc1.detach().numpy())
# model_tf.fc1_bias = tf.tensor(model_torch.fc1_bias.detach().numpy())
model_tf.fc2 = tf.tensor(model_torch.fc2.detach().numpy())
model_tf.fc2_bias = tf.tensor(model_torch.fc2_bias.detach().numpy())

x = np.random.randn(N).astype(np.float32)
x_tf = tf.tensor(x)
x_torch = torch.tensor(x)
y = np.random.randn(N).astype(np.float32)
y_tf = tf.tensor(y)
y_torch = torch.tensor(y)

#compute tensorfrost gradients
def grad_computer():
    model = TestModule1()
    grads = ADGrad(model)
    grads.initialize_input()

    x = tf.input([N], tf.float32)
    y = tf.input([N], tf.float32)
    loss = model.loss(x, y)
    grads.compute(loss)
    params = grads.parameters()
    params.append(loss)
    return params

grad_compute = tf.compile(grad_computer)

#compute tensorfrost gradients
all_params = grad_compute(tf_grads, x_tf, y_tf)
tf_grads.update_parameters(all_params[:-1])
loss_tf = all_params[-1]
print("Tensorfrost loss: ", loss_tf.numpy)

#compute pytorch gradients
model_torch.zero_grad()
loss_torch = model_torch.loss(x_torch, y_torch)
loss_torch.backward()
print("Pytorch loss: ", loss_torch.item())

#compare the gradients
for i, param in enumerate(model_torch.parameters()):
    print("Param ", i)
    # print("Torhch grad: ")
    # print(param.grad)
    # print("Tensorfrost grad: ")
    # print(tf_grads.grad[i].numpy)
    # print("Torch param: ")
    # print(param)
    # print("Tensorfrost param: ")
    # print(tf_grads.net.parameters()[i].numpy)
    print("Gradient error: ", np.mean(np.abs(param.grad.detach().numpy() - tf_grads.grad[i].numpy)))
    print("Parameter error: ", np.mean(np.abs(param.detach().numpy() - tf_grads.net.parameters()[i].numpy)))