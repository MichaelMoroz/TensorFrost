# 🔢🥶 TensorFrost
[![PyPI Build and Deploy](https://github.com/MichaelMoroz/TensorFrost/actions/workflows/build-and-deploy-to-pypi.yml/badge.svg)](https://github.com/MichaelMoroz/TensorFrost/actions/workflows/build-and-deploy-to-pypi.yml)


A static optimizing tensor compiler with a Python frontend, autodifferentiation, and a more "shader-like" syntax.

Currently working platforms:
| Backend/OS | CPU | OpenGL | CUDA | Vulkan |
|------------|-----|--------|------|--------|
| Windows    | 🚧  |  🚧   |  ⛔  |  ⛔   |
| Linux      | 🚧  |  🚧   |  ⛔  |  ⛔   |


For more detail about this project, please read my blog post!
[Writing an optimizing tensor compiler from scratch](https://michaelmoroz.github.io/WritingAnOptimizingTensorCompilerFromScratch/)

## Examples

<a href="https://github.com/MichaelMoroz/TensorFrost/blob/main/examples/Simulation/wave_simulation.ipynb"><img src="https://github.com/MichaelMoroz/TensorFrost/blob/main/examples/sin_gordon.gif?raw=true" height="192px"></a>
<a href="https://github.com/MichaelMoroz/TensorFrost/blob/main/examples/Simulation/fluid_simulation.ipynb"><img src="https://github.com/MichaelMoroz/TensorFrost/blob/main/examples/fluid_sim.gif?raw=true" height="192px"></a>
<a href="https://github.com/MichaelMoroz/TensorFrost/blob/main/examples/GUI/buddhabrot.py"><img src="https://github.com/MichaelMoroz/TensorFrost/blob/main/examples/buddhabrot.gif?raw=true" height="192px"></a>
<a href="https://github.com/MichaelMoroz/TensorFrost/blob/main/examples/GUI/interactive_path_tracer.py"><img src="https://github.com/MichaelMoroz/TensorFrost/blob/main/examples/path_tracer.gif?raw=true" height="192px"></a>
<a href="https://github.com/MichaelMoroz/TensorFrost/blob/main/examples/Simulation/n-body.ipynb"><img src="https://github.com/MichaelMoroz/TensorFrost/blob/main/examples/n_body.gif?raw=true" height="192px"></a>
<a href="https://github.com/MichaelMoroz/TensorFrost/blob/main/examples/Rendering/neural_embed.ipynb"><img src="https://github.com/MichaelMoroz/TensorFrost/blob/main/examples/neural_embed.gif?raw=true" height="192px"></a>


## Installation

## From PyPI

You can install the latest version of the library from [PyPI](https://pypi.org/project/tensorfrost/):

```bash
pip install tensorfrost
```

## From source

You need to have CMake installed to build the library.

First clone the repository:
```bash
git clone --recurse-submodules https://github.com/MichaelMoroz/TensorFrost.git
cd TensorFrost
```

Then run cmake to build the library.

```bash
cmake -S . -B build && cmake --build build
```

> [!TIP]
> If you are using a Linux distribution that doesn't support installing packages through pip (e.g. Arch Linux), read **[Using a Virtual Environment](#using-a-virtual-environment)**.

The cmake script will automatically install the compiled python module into your python environment.

### Using a Virtual Environment

Certain Linux distributions (e.g. Arch Linux) want you to use their package manager to manage system-wide Python packages instead of pip. TensorFrost uses pip to install itself once built, so before running CMake you will need to activate a Virtual Environment.

1. From the TensorFrost directory, create a venv:

    ```sh
    python -m venv ./venv
    ```

2. Activate the venv:

    ```sh
    source venv/bin/activate
    ```

3. Install `jinja` (required for building)

    ```sh
    pip install jinja
    ```

4. Now, you can use CMake as usual.

    ```bash
    cmake -S . -B build && cmake --build build
    ```

> [!TIP]
> The newly-created venv is treated like a fresh Python installation, so you may need to reinstall any needed packages such as `numpy`, `matplotlib`, and `tqdm` if you are trying out the examples. `pip` works fine once the venv is active (e.g. `pip install numpy`).

## Usage

### Setup
For the library to work you need a C++ compiler that supports C++17 (Currently only Microsoft Visual Studio Compiler on Windows, and gcc on Linux)

First you need to import the library:
```python
import TensorFrost as tf
```

Then you need to initialize the library with the device you want to use and the kernel compiler flags (different for each platform):
```python
tf.initialize(tf.cpu) # or tf.opengl
```

TensorFrost will find any available MSVC(Windows) or GCC(Linux) compiler and use it to compile the main code and the kernels. In OpenGL mode the driver compiles the kernels. (TODO: compile the main code into python for faster compile times, MSVC is super slow, 1.5 seconds for a single function)

You can have TensorFrost in code generation mode instead (you cant run tensor programs here), it is much faster, but you would need to use the code manually afterwards:

```python
tf.initialize(tf.codegen, kernel_lang = tf.hlsl_lang) # or tf.glsl_lang for OpenGL, or tf.cpp_lang for C++
```

After you compiled all the tensor programs you need, you can get all the generated code and save it to a file:
```python
# Save all the compiled functions
cpp_header = tf.get_cpp_header()
all_main_functions = tf.get_all_generated_main_functions() #always in C++
with open('tensorfrost_main.cpp', 'w') as f:
    f.write(cpp_header)
    for func in all_main_functions:
        f.write(func)

# Save all the compiled kernels
all_kernels = tf.get_all_generated_kernels() #depends on the kernel_lang
for i, kernel in enumerate(all_kernels):
    with open('generated_kernels/kernel_{}.hlsl'.format(i), 'w') as f:
        f.write(kernel)
```

Right now you cant just compile the code and run it, since it also requires a Kernel compiler and executor as well as memory manager for tensors. In the future I plan to add all the required functions for that too, for better portability.

### Basic usage

Now you can create and compile functions, for example here is a very simple function does a wave simulation:
```python
def WaveEq():
    #shape is not specified -> shape is inferred from the input tensor (can result in slower execution)
    u = tf.input([-1, -1], tf.float32)
    #shape must match 
    v = tf.input(u.shape, tf.float32)

    i,j = u.indices
    laplacian = u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] - u * 4.0
    v_new = v + dt*laplacian
    u_new = u + dt*v_new

    return v_new, u_new

wave_eq = tf.compile(WaveEq)
```

As you can see, inputs are not arguments to the function, but are created inside the function. This is because some inputs can be constrained by the shape of other inputs, and the shape of the input tensor is not known at compile time. You can give shape arguments to the input function, constants for exactly matching shapes, or -1 for any shape. If you want to constrain the shape of the input tensor, you need to get the shape of the other tensor and use it as an argument to the input function.

The tensor programs take and output tensor memory buffers, which can be created from numpy arrays:
```python
A = tf.tensor(np.zeros([100, 100], dtype=np.float32))
B = tf.tensor(np.zeros([100, 100], dtype=np.float32))
```

Then you can run the program:
```python
A, B = wave_eq(A, B)
```
As you can see the inputs are given to the compiled function in the same order as they are created in the function.

To get the result back into a numpy array, you can use the `numpy` property:
```python
Anp = A.numpy
```

### Operations

TensorFrost supports most of the basic numpy operations, including indexing, arithmetic, and broadcasting.
The core operation is the indexing operation, which is used to specify indices for accessing the tensor data. Depending on the dimensinality of the tensor there can be N indices. This operation is similar to numpy's `np.ogrid` and `np.mgrid` functions, but it is basically free due to fusion.

```python
#can be created either from a provided shape or from a tensor
i,j = tf.indices([8, 8]) 
i,j = A.indices
```

For example `i` contains:

```
[[0, 0, 0, ..., 0, 0, 0],
 [1, 1, 1, ..., 1, 1, 1],
 [2, 2, 2, ..., 2, 2, 2],
    ...,
 [7, 7, 7, ..., 7, 7, 7]]
```

And analogously for `j`.

These indices can then be used to index into the tensor data, to either read or write data:
```python
#set elements [16:32, 16:32] to 1.0
i,j = tf.indices([16, 16]) 
B[i+16, j+16] = 1.0

#read elements [8:24, 8:24]
i,j = tf.indices([16, 16])
C = B[i+8, j+8]
```

Here we can see that the shape of the "computation" is not the same as the shape of the tensor, and one thread is spawned for each given index. Then all sequential computations of the same shape are fused into a single kernel, if their computaion is not dependent on each other.

When doing out-of-bounds indexing, the index is currently clamped to the tensor shape. This is required to avoid undefined behaviour, in the future I plan to give the user the option to specify the behaviour of out-of-bounds indexing.

You can also use the index_grid operation which is similar to numpy's `np.meshgrid` function and provides a grid of indices for each dimension:

```python
p, k = tf.index_grid([0, i + 1], [m, n])
```

Which is equivalent to numpy's `np.meshgrid` function (only for ints with step 1 for now):

```python
p, k = np.meshgrid(np.arange(0, m), np.arange(i + 1, n))
```

Slicing is still not implemented, as that would require better shape comparison for undefined shapes, without it, you would get a lot of errors where there should not be any.

### Currently supported operations

All the default arithmetic operations are supported:

`+`, `-`, `*`, `/`, `**`, `==`, `!=`, `>`, `<`, `>=`, `<=`, `&`, `|`, `~`, `neg`

Note that the boolean operations `and`, `or`, `not` are not overloaded yet, and you should use `&`, `|`, `~` instead on boolean tensors. (Might be changed in the future)

Also there are these provided functions: 

`abs`, `sign`, `ceil`, `floor`, `round`, `frac`, `exp`, `exp2`, `log`, `log2`, `sqrt`, `rsqrt`, `rcp`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `reversebits`, `pow`, `atan2`, `modf`, `step`, `clamp`, `lerp`, `fma`, `smoothstep`, `ternary`, `const`.

Additionally you can use `uint`, `int`, `float`, `bool` to cast between types, and `asuint`, `asint`, `asfloat`, `asbool` to reinterpret the bits of the number.

If needed, you can copy a value with the `copy` operation which is useful as you can not assign a tensor to another tensor directly.

For random number generation you can either implement your own hashing function, or use the provided pcg32 hash.

```python
#generate a random number between 0 and 1
value = tf.pcgf(seed)

#generate a random uint32 number
value = tf.pcg(seed)
```

TensorFrost does not have a built-in seed, so its similar to JAX where you need to provide your own seed. This is useful for reproducibility, as you can just provide the same seed to the program and get the same results.

### Scatter operations

These operations allow implementing non-trivial reduction operations, and are basically equivalent to atomics in compute shaders. For example, here is a simple example of a scatter operation:

```python
def ScatterMatrixMultiplication():
    A = tf.input([-1, -1], tf.float32)
    N, M = A.shape
    B = tf.input([M, -1], tf.float32) #M must match
    K = B.shape[1]

    C = tf.zeros([N, K])
    i, j, k = tf.indices([N, K, M])
    tf.scatterAdd(C[i, j], A[i, k] * B[k, j])

    return C

matmul = tf.compile(ScatterMatrixMultiplication)
```

Here the 3D nature of the matrix multiplication is apparent. The scatter operation is used to accumulate the results of the row-column dot products into the elements of the resulting matrix.

The compiler will optimize the scatter operation into a loop in this particular case, so this will not be too slow, but you should prefer to just use `A @ B` for matrix multiplication.

### Reduction operations

Reduction operations are used to reduce the tensor data along one dimension. For example, here is a simple example of a sum reduction:

```python
def MatrixMultiplication():
    A = tf.input([-1, -1], tf.float32)
    N, M = A.shape
    B = tf.input([M, -1], tf.float32) #M must match
    K = B.shape[1]

    i, j, k = tf.indices([N, K, M])
    C = tf.sum(A[i, k] * B[k, j], axis=2) #by default axis is -1 (last axis)

    return C

matmul = tf.compile(MatrixMultiplication)
```

Here the `sum` operation is used to sum the dot products of the rows and columns of the input matrices along the `k` axis.
This is much more efficient than the scatter operation, and in fact this compiles to a single N*K kernel.

The following reduction operations are supported: `sum`, `mean`, `max`, `min`, `all`, `any`, `prod` and `norm`

In the future I plan to add support for multiple reduction axes.

> [!TIP]
> If the shape is specified explicitly, for reductions >= 1024 elements, the reduction will be split into stages and will have much better performance.

### Scan operations

Right now only `prefix_sum` is supported (numpy's `np.cumsum`).

An automatic optimization pass that does staged prefix sum is planned for the future, but right now you can use:

```python
def PrefixSum(A, axis = -1):
    axis = len(A.shape) + axis if axis < 0 else axis
    group_size = 64
    grouped = tf.split_dim(A, group_size, axis)
    group_scan = tf.prefix_sum(tf.sum(grouped, axis = axis + 1), axis = axis)
    ids = grouped.indices
    gid, eid = ids[axis], ids[axis + 1]
    ids = [ids[i] for i in range(len(ids)) if i != axis + 1]
    ids[axis] = gid - 1
    group_scan = tf.prefix_sum(grouped + tf.select((gid == 0) | (eid != 0), 0, group_scan[tuple(ids)]), axis = axis + 1)
    full_scan = tf.merge_dim(group_scan, target_size = A.shape[axis], axis = axis + 1)
    return full_scan
```

### Sorting operations

Sort is not yet built-in the library, but you can use a custom implemented one from the [sorting test in examples folder](https://github.com/MichaelMoroz/TensorFrost/blob/main/examples/Algorithms/sorting_tests.py). There is a relatively optimized histogram radix sort as well as a simple bitonic sort.

### Broadcasting

Broadcasting is used to make the shapes of the input tensors compatible. For example, here is a simple example of a broadcasting operation:

```python
def Broadcasting():
    A = tf.input([1, 3], tf.float32)
    B = tf.input([3, 1], tf.float32)

    C = A + B

    return C
```

Here the `+` operation is used to add the two input tensors. The shapes of the input tensors are `[1, 3]` and `[3, 1]`, and the shape of the output tensor is `[3, 3]`. The `+` operation is broadcasted over the input tensors, and the result is a tensor with the shape `[3, 3]`.
The rules are the same as in numpy essentially.

### Reshape

Reshape operation is used to change the shape of the tensor. For example, here is a simple example of a reshape operation:

```python
def Reshape():
    A = tf.input([2, 3], tf.float32)

    B = tf.reshape(A, [3, 2])

    return B
```

Here the `reshape` operation is used to change the shape of the input tensor from `[2, 3]` to `[3, 2]`.
At the moment this is implemented in a very crude way, so doing this will always halt kernel fusion, so use it only when you are sure things are unfusable
(usually at the beginning or end of the program).

Additionally, you can also use `transpose`, `unsqueeze` and `squeeze` operations to change the shape of the tensor, which work fine with fusion.

```python
def Transpose():
    A = tf.input([2, 3], tf.float32)

    B = tf.transpose(A) #shape is [3, 2]
    C = B.T #shape is [2, 3]

    return C
```

```python
def Unsqueeze():
    A = tf.input([2, 3], tf.float32)

    B = tf.unsqueeze(A, 1) #shape is [2, 1, 3]

    return B
```

Additionally there are `merge_dim` and `split_dim` operations that can be used to merge or split dimensions of the tensor.

```python
A = tf.input([2, 3, 4], tf.float32)
B = tf.merge_dim(A, axis = 1) #shape is [2, 12]
```

```python
A = tf.input([2, 12], tf.float32)
B = tf.split_dim(A, 4, axis = 1) #shapes are [2, 3, 4]
```

> [!TIP]
> If you want the compiler to be able to merge kernels with reshape, you should try using `merge_dim` and `split_dim` instead.

### Matrix operations

Matrix operations are used to perform matrix operations on the tensor data. For example, here is a simple example of a matrix multiplication:

```python
def MatrixMultiplication():
    A = tf.input([-1, -1], tf.float32)
    N, M = A.shape
    B = tf.input([M, -1], tf.float32) #M must match

    C = tf.matmul(A, B) #or A @ B

    return C

matmul = tf.compile(MatrixMultiplication)

A = tf.tensor(np.zeros([100, 100], dtype=np.float32))
B = tf.tensor(np.zeros([100, 100], dtype=np.float32))

C = matmul(A, B)
```

Here the `matmul` operation is used to multiply the input matrices `A` and `B`. The shapes of the input tensors are `[N, M]` and `[M, K]`, and the shape of the output tensor is `[N, K]`.
The inputs can have any shape of the form [A, B, ..., N, M], and as long as they are broadcastable, the operation will work.

### Loops and conditionals

```python
#Mandelbrot set
z_re = tf.const(0.0)
z_im = tf.const(0.0)
with tf.loop(128) as k: #or tf.loop(0, 128) for a range loop, or tf.loop(0, 128, 2) for a range loop with step
    z_re_new = z_re*z_re - z_im*z_im + c_re
    z_im_new = 2.0*z_re*z_im + c_im
    z_re.val = z_re_new
    z_im.val = z_im_new
    with tf.if_cond(z_re*z_re + z_im*z_im > 256.0):
        tf.break_loop()
```

Scopes in TensorFrost are implemented through python context managers. There are `tf.loop` and `tf.if_cond` context managers that can be used to create loops and conditionals. The loop context manager takes the number of iterations as an argument, and the if_cond context manager takes a condition as an argument. The condition can be any tensor operation that returns a boolean tensor.
Also since the setting operation can not be overloaded in python, the `set` method must be used to update the tensor data outside of this scope, or alternatively the `val` property can be used to set the value of the tensor. 

```python
z_re = tf.const(0.0)
with tf.loop(128):
    z_re.set(z_re_new) #this is fine
    z_re.val = z_re_new #this is also fine
    z_re = z_re_new #this is not fine
```

Just setting the tensor to a new value will actually create a new tensor on top of the old one, and the old one will not be updated.

Loops and conditionals can be stacked and nested. Usually they are compiled into a single kernel with the scopes inside it, but they can be compiled into separate kernels if the data dependencies are not local (look at the QR decomposition example in the examples folder). Not all possible loop and conditional can be valid here, if the loop iteration count has a shape incompatible with the shapes of the tensors in the loop body, the program will not compile correctly.

PS: You can also provide a function instead of using a context manager, but it is not recommended, as it is less readable.

```python
def loop_body(k):
    z_re_new = z_re*z_re - z_im*z_im + c_re
    z_im_new = 2.0*z_re*z_im + c_im
    z_re.val = z_re_new
    z_im.val = z_im_new
    with tf.if_cond(z_re*z_re + z_im*z_im > 256.0):
        tf.break_loop()

tf.loop(0, 128, 1, loop_body)
```

### Autodifferentiation

Currently only backward mode autodifferentiation is supported, and can not properly be applied at control flow operations.

```python
y_pred = x @ W + b
loss = tf.mean((y - y_pred)**2)
dW = tf.grad(loss, W)
db = tf.grad(loss, b)
```
In this example, the `grad` function is used to compute the gradients of the loss with respect to the weights `W` and the bias `b`. If the gradient is taken from the same "loss" tensor, the compiler will still only do one backward pass. At the moment doing gradients from gradients might not work correctly.

Additionally, if the loss is not a scalar, the initial gradient tensor will be assumed to be the same shape as the loss tensor and equal to 1.0. For most cases this is quite useful, as you can compute the gradients of multiple outputs at the same time, as long as they are not dependent on each other. Like doing a gradient of a potential for N particles at the same time.

```python
dx = x1 - x2
dist = tf.sqrt(tf.sum(dx**2))
pot = 1.0 / dist
force = - tf.grad(pot, dx)
```

In this example, the `grad` function is used to compute the gradient of the potential with respect to the distance between two particles. The force is then computed as the negative gradient of the potential with respect to the distance.

Giving a custom gradient tensor is not supported yet, but it is planned for the future.

You can also stop the gradient computation for some tensors by `tensor.detach_grad()`. In that case the autograd algorithm will stop at this tensor.

Or if you want to force the gradient through a operation without applying the operation gradient you can do `tensor.pass_grad()`. This is useful for example when you want to optimize discrete parameters like a quantized weight.
### Modules 

TensorFrost has a simple module system similar to PyTorch, where you can define a module with trainable parameters and a forward function that computes the output of the module as well as a loss function. 

```python
class SmolNet(tf.Module):
    def __init__(self):
        #specify a custom random scale and offset for the weights when initializing
        self.W = tf.Parameter([16, -1], tf.float32, random_scale=0.01, random_offset=0.0)
        #dont compute gradients for the bias
        self.b = tf.Parameter([-1], tf.float32, requires_grad=False)
        
    def assert_parameters(self):
        #makes sure that the compiler knows that b has shape compatible with W
        self.b = tf.assert_tensor(self.b, [self.W.shape[1]], tf.float32)
        
    def forward(self, x):
        return x @ self.W + self.b
    
    def loss(self, x, y):
        y_pred = self.forward(x, y)
        return tf.mean((y - y_pred)**2)
```

When initializing the module you can add 3 types of TensorFrost accessible parameters:
- `tf.Parameter` - a tensor that will be passed to the TensorProgram as an argument
- `tf.ParameterArray` - a dynamic list of parameters, all of them will be passed to the TensorProgram as arguments
- `tf.Module` - another module, all of its parameters will be passed to the TensorProgram as arguments

The shape argument of the parameter can be a list of integers, where -1 means that the shape is not specified yet, and will be inferred from the input tensor. If you need to compute an operation over several tensors of unspecified shape, you need to assert the shapes in the `assert_parameters` function.
`random_scale` and `random_offset` are used to initialize the weights with random values, and are optional, by default the weights are initialized with Xavier initialization for normal random values.
`requires_grad` is used to specify if the parameter should be trained or not, by default all parameters are trainable. This argument does not stop you from computing `tf.grad` manually, it is just used to specify if the parameter should be updated by the optimizer module.

By itself the module does not do anything, you need to do a second initialization step to either use it inside a TensorProgram, or initialize it as a container for the tensors outside of the program.

```python

def ComputeForward():
    model = SmolNet()
    #creates tf.input tensors from all the parameters of the module
    model.initialize_input()
    X = tf.input([-1, -1], tf.float32)
    return model.forward(X)

forward = tf.compile(ComputeForward)

model_container = SmolNet()
#creates tf.tensor tensors from all the parameters of the module and initializes them
model_container.initialize_parameters()
#you can change them afterwards too
model_container.W = tf.tensor(np.zeros([16, 100], dtype=np.float32))

X = tf.tensor(np.zeros([100, 100], dtype=np.float32))
#the module is passed as an argument to the compiled function, in the same order as they are created in the function
Y = forward(model_container, X)
```

`model.initialize_input()` creates put `tf.input()` tensors for all the parameters of the module. Afterwards `assert_parameters` is automatically called for this and all child modules. This is useful if you want to use the module inside a TensorProgram, as you can just pass the module as an argument to the compiled function, and all the parameters will be automatically created and the shapes will be asserted.
`model.initialize_parameters()` creates `tf.tensor()` tensors for all the parameters of the module and initializes them with random values. This is useful if you want to use the module outside of a TensorProgram, as you can just pass the module as an argument to the compiled function.

You can not, however, do both at the same time, as the module will not know if it is used inside or outside of a TensorProgram.

### Optimizer modules

TensorFrost has a set of built-in optimizer modules that can be used to train the parameters of the module. 
- `tf.optimizers.sgd` - Stochastic Gradient Descent, has a `learning_rate` and `grad_clip` parameters, default values are 0.001 and 0.0 respectively.
- `tf.optimizers.adam` - Adam optimizer, has a `learning_rate`, `beta1`, `beta2` and `grad_clip` parameters, default values are 0.001, 0.9, 0.999 and 0.0 respectively.
- `tf.optimizers.rmsprop` - RMSProp optimizer, has a `learning_rate`, `decay` and `grad_clip` parameters, default values are 0.001, 0.9 and 0.0 respectively.

All optimizer modules are initialized with the module as the first argument, and the training hyperparameters as the rest of the arguments.

```python
def OptimizerStep():
    X = tf.input([-1, -1], tf.float32)
    Y = tf.input([-1, 10], tf.float32)

    model = SmolNet()
    opt = tf.optimizers.adam(model, learning_rate=0.001, beta1=0.9, beta2=0.999)
    opt.initialize_input()
    
    #do a single step of the optimizer (automatically computes gradients and updates the parameters)
    L = opt.step(X, Y) 
    #or 
    #L = model.loss(X, Y)
    #opt.step(L)

    params = opt.parameters()
    params.append(L)
    return params

step = tf.compile(OptimizerStep)

model_container = SmolNet()
opt = tf.optimizers.adam(model_container)
opt.initialize_parameters()

X = tf.tensor(np.zeros([100, 100], dtype=np.float32))
Y = tf.tensor(np.zeros([100, 10], dtype=np.float32))
out = step(X, Y, opt)
opt.update_parameters(res[:-1])
loss = res[-1].numpy[0]
```

Outputting the optimizer state is somewhat inconvenient at the moment, as you can only output a list of tensors from the compiled function, so you need to append the loss to the list of parameters and then extract it from the list afterwards. The optimizer state is not saved in the module, so you need to pass it as an argument to the compiled function, and then update the parameters of the module with the updated parameters from the optimizer.

Optionally you can also enable regularization for the parameters of the module, by specifying the `l1` and `l2` regularization parameters in the `initialize_parameters` function. This will apply regularization to the parameters after the optimizer step, meaning the `adam` optimizer will behave like `adamw` optimizer.

```py
optimizer = tf.optimizers.adam(model_container, beta1 = 0.0, beta2 = 0.999, reg_type = tf.regularizers.l2, reg = 0.02, clip = 0.01)
optimizer.set_clipping_type(tf.clipping.norm)
```

You can also specify the clipping type for the gradients, by default the value of `clip` is zero which turns it off. The clipping type can be `tf.clipping.norm` or `tf.clipping.clamp`.

### Debugging

For debugging convenience there are 2 function types that you can call inside a tensor program:

```python
tf.renderdoc_start_capture()
tf.renderdoc_end_capture()
```

These functions will start and end a RenderDoc capture, only if python is started from the RenderDoc GUI. This is useful for debugging the OpenGL backend, as it allows you to inspect compiled kernel execution, its code and buffers.

```python
tf.region_begin('Region name')
tf.region_end('Region name')
```

When debugging from RenderDoc (or any other OpenGL debugger), these functions will create a region in the RenderDoc capture, which can be useful for profiling and seeing what parts of the program are slow.
The placement of these functions might not reflect their position in the code, as the code is heavily optimized and fused, so if you placed a region in the middle of a generated kernel, it will be placed at the beginning or end of the kernel. Placing them in a scoped operation might make the compilation fail or unfuse kernels, so be careful with that.

To debug the generated code you can either look at the generated code in the Temp folder with `tf.cpu` backend enabled if you need kernel code. If you want to debug the GPU kernel code, you can use RenderDoc.

> [!TIP]
> You can print out tensors at compilation time in the main function by just doing `print(tensor)`. This will output its debug information, its shape, its data type, what operation it is, shape (inverted), its arguments, etc.

### GUI and visualization

TensorFrost has simple bindings for the GLFW window library, and some ImGui bindings for GUI. You can render tensors as images (only [-1, -1, 3] float32 tensors for now) and display them in a window. You can also use ImGui to create simple GUIs for your programs. Do note that this only works in the OpenGL backend.

```python

#creates a single global window (can only be one at the moment)
tf.window.show(1280, 720, "a window")

while not tf.window.should_close(): #window will close if you press the close button and this will return True
    mx, my = tf.window.get_mouse_position()
    wx, wy = tf.window.get_size()

    #simple input example
    if tf.window.is_mouse_button_pressed(tf.window.MOUSE_BUTTON_0):
        tf.imgui.text("Mouse button 0 is pressed")

    if tf.window.is_key_pressed(tf.window.KEY_W):
        tf.imgui.text("W is pressed")

    #ImGui example
    tf.imgui.begin("an imgui window")
    tf.imgui.text("some text")
    value = tf.imgui.slider("slider", value, 0.0, 10.0)
    if(tf.imgui.button("a button")):
        print("button pressed")
    tf.imgui.end()

    #exectute a tensorfrost program that outputs a [-1, -1, 3] float32 tensor
    img = render_image(...)

    #display the image (will be stretched to the window size with nearest neighbor interpolation)
    tf.window.render_frame(img)
    
```

Currently provided `window` functions are:
- `show(width, height, title)` - creates a window
- `should_close()` - returns `True` if the window should close
- `get_mouse_position()` - returns the mouse position
- `get_size()` - returns the window size
- `is_mouse_button_pressed(button)` - returns `True` if the mouse button is pressed
- `is_key_pressed(key)` - returns `True` if the key is pressed
- `render_frame(tensor)` - renders the tensor as an image

Currently provided `imgui` functions are:
- `begin(name)` - begins an ImGui window
- `end()` - ends an ImGui window
- `text(text)` - displays text
- `slider(name, value, min, max)` - displays a slider
- `button(text)` - displays a button, returns `True` if the button is pressed
- `checkbox(text, value)` - displays a checkbox
- `plotlines(label, values, values_offset, overlay_text, scale_min, scale_max, graph_size, stride)` - displays a plot
- `scale_all_sizes(scale)` - scales all ImGui sizes by a factor
- `add_background_text(text, pos, color)` - adds background text at the specified position with the specified color

### Usage tips

- Using an explicit shape for the input tensors can help the compiler to optimize the program better, as it can infer the shapes of the tensors in the program better. On top of that some optimizations like loop unrolls or staged reductions only happen if the shape is known at compile time.
- Large matrix multiplications are currently very much not optimized, as the compiler does not use groupshared memory or any other optimizations for matrix multiplication. This is planned for the future. For now using TensorFrost mostly makes sense for small to medium sized architectures where cache hits are high.
- Complex operations like convolutions can be implemented through sum + indexing operaitons, example below (taken from [here](https://github.com/MichaelMoroz/TensorFrost/blob/main/examples/ML/module.py))

  While this might seem less optimal than a hand optimized convolution kernel especially when computing its gradient, but it is much more flexible and is actually optimized quite well by the compiler. While the gradient of the indexing operations is an atomicAdd operation, in this case, several of the dimensions of the gradient kernel are not used in the index of the tensors, and get unrolled into sums removing the atomics from the kernel.
  In such a way you can implement any operation you want, even matrix multiplication works fine (`tf.sum(A[i, k] * B[k, j])`), and the compiler will optimize it and its gradient quite well.
  Not all atomics will get optimized out however, so be careful when taking gradients of indexed tensors, as the current atomicAdd for floats is an emulated operation and is can get extremely slow with high write contention.
```python
def conv2d(self, X, W, b):
        bi, wi, hi, cout, cin, it = tf.indices([X.shape[0], X.shape[1] - W.shape[2] + 1, X.shape[2] - W.shape[3] + 1, W.shape[0], W.shape[1], W.shape[2] * W.shape[3]])
        i, j = it%W.shape[2], it/W.shape[2]
        conv = tf.sum(tf.sum(X[bi, wi + i, hi + j, cin] * W[cout, cin, i, j]))
        return conv + b 
```

- Inplace operation gradients simply don't work, even though it does compile, the gradients are not computed correctly. This is planned to be fixed in the future.
- You can check the compiled code in the Temp folder in `generated_lib_*.cpp` files, it is not very readable, but you can see the operations and the memory allocations, the kernel code is in the same file, only on CPU backend.
## Roadmap 

Core features:
- [x] Basic operations (memory, indexing, arithmetic, etc.)
- [x] Basic kernel fusion and compilation
- [x] Advanced built-in functions (random, special functions, etc.)
- [x] Advanced operations (loops, conditionals, etc.)
- [x] Kernel code and execution graph export and editing
- [x] Backward mode autodifferentiation
- [x] Module system
- [x] Optimizer modules (SGD, Adam, RMSProp)
- [x] GUI and visualization
- [ ] Compiled `TensorProgram` export and import
- [ ] Forward mode autodifferentiation
- [ ] Gradients of control flow operations and gradients from gradients
- [ ] Advanced data types and quantization
- [ ] Compile from Python AST instead of tracing
- [ ] Groupshared memory support
- [ ] Automatic data caching and reuse
  
Algorithm library:
- [x] Scan, reduction, etc.
- [x] Module system
- [x] Optimizer modules (SGD, Adam, RMSProp)
- [x] Matrix operations (matrix multiplication, etc.)
- [ ] Sorting algorithms (some examples already in the examples folder)
- [ ] Advanced matrix operations (QR, SVD, eigenvalues, etc.) (some examples already in the examples folder)
- [ ] Fast Fourier Transform (some examples already in the examples folder)
- [ ] High-level neural network layers (convolution, etc.) (some examples already in the examples folder)

Platforms:
- [x] Windows
- [x] Linux
- [ ] MacOS

Backends:
- [x] CPU (C++ OpenMP backend)
- [x] OpenGL (most basic GPU backend, has a lot of driver bugs)
- [ ] CUDA
- [ ] Vulkan
- [ ] ISPC (for better CPU utilization)
- [ ] WGPU (for web)

## Contributing
Contributions are welcome! If you want to contribute, please open an issue first to discuss the changes you want to make.
